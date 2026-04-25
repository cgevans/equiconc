//! `equiconc-coffee-cli` — a CLI front-end that accepts the same inputs as
//! COFFEE's `coffee_cli` (NUPACK-style `.ocx`/`.cfe` complex tables and
//! `.con` monomer-concentration files) and produces the same results payload
//! so users can swap the binary without changing surrounding tooling.
//!
//! This is part of the BSD-3-Clause `equiconc` crate; it contains no code
//! from COFFEE (which is Apache-2.0). File formats, CLI shape, and the
//! `{:.2e}` results-payload format are reproduced from public
//! specifications — behavior, not source.
//!
//! Hard-coded defaults match COFFEE's non-configurable defaults so that
//! dropping in this binary gives numerically-equivalent output on unchanged
//! invocations:
//!
//! * Temperature: 37 °C.
//! * Mole-fraction scaling of inputs/outputs via the water density
//!   (COFFEE calls this "scalarity"; always on).
//! * Energy clamp: `log_q = -ΔG / (RT) ≤ 230`.
//!
//! No flag exposes those — any divergence from COFFEE would defeat the
//! purpose of the compatibility surface.

use std::fs;
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use clap::Parser;
use equiconc::{SolverOptions, System};
use ndarray::{Array1, Array2};

// ---------------------------------------------------------------------------
// Physical constants
// ---------------------------------------------------------------------------

/// Molar gas constant in kcal/(mol·K), derived from CODATA 2018
/// (`R = 8.314 462 618... J/(mol·K)`) divided by `4184 J/kcal`.
const R_KCAL_PER_MOL_K: f64 = 1.987_204_258_640_832e-3;

/// Default operating temperature in Celsius. Hard-coded to match COFFEE's
/// non-configurable default so output matches drop-in.
const T_CELSIUS: f64 = 37.0;

/// Lower bound on ΔG (kcal/mol) before computing `log_q = -ΔG/(RT)`.
/// Any species with `ΔG < -ΔG_CLAMP_KCAL_PER_MOL` is treated as having
/// `ΔG = -ΔG_CLAMP_KCAL_PER_MOL`. COFFEE applies the same clamp in its
/// internal `polymers_q = exp(-max(ΔG, -230)/RT)` line, which keeps the
/// effective problem identical between the two solvers. Expressed in
/// energy units (kcal/mol) rather than unitless `log_q` so the clamp
/// corresponds to a fixed physical cutoff regardless of temperature.
const DG_CLAMP_KCAL_PER_MOL: f64 = 230.0;

/// Molar density of liquid water at `t_c` degrees Celsius, returned in
/// mol/L (= mol/dm³).
///
/// Implemented from the empirical mass-density formula of Tanaka et al.
/// 2001, *Metrologia* 38, 301–309 (VSMOW water, 0–40 °C), converted to
/// molar units via the IUPAC-recommended molar mass of H₂O,
/// `M = 18.015 28 g/mol` (Meija et al. 2016, *Pure Appl. Chem.* 88, 265).
///
/// Coffee's `density_water` helper is the same math expressed differently;
/// this implementation was written against the published Tanaka formula
/// directly, not ported from coffee's source.
fn density_water_molar(t_c: f64) -> f64 {
    // Tanaka et al. 2001, Table 1 (constants for VSMOW).
    const A1: f64 = -3.983_035_f64;
    const A2: f64 = 301.797_f64;
    const A3: f64 = 522_528.9_f64;
    const A4: f64 = 69.348_81_f64;
    const RHO_MAX_KG_PER_M3: f64 = 999.974_950_f64;

    // Molar mass of water (g/mol).
    const M_WATER_G_PER_MOL: f64 = 18.015_28_f64;

    let offset = t_c + A1;
    let mass_density = RHO_MAX_KG_PER_M3
        * (1.0 - offset * offset * (t_c + A2) / (A3 * (t_c + A4)));
    // kg/m³ ÷ g/mol = mol/L (1000 mol/m³ ÷ 1000 L/m³ · g/kg).
    mass_density / M_WATER_G_PER_MOL
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

/// Split a COFFEE-style row into fields.
///
/// COFFEE accepts whitespace, comma, semicolon, and pipe separators for
/// complex tables. Reuse the same delimiter set here so `.csv`/`.tsv`
/// extensions behave as advertised; empty fields are ignored to match
/// repeated whitespace behavior.
fn split_fields(line: &str) -> Vec<&str> {
    line.split(|c: char| c.is_whitespace() || matches!(c, ',' | ';' | '|'))
        .filter(|field| !field.is_empty())
        .collect()
}

/// Read monomer initial concentrations from a `.con` file.
///
/// Format: one f64 per non-blank line, in scientific or decimal notation.
/// Lines that split into more than one delimited field are rejected —
/// this preserves the single-column concentration-file contract.
fn parse_con(content: &str) -> Result<Array1<f64>, String> {
    let mut values: Vec<f64> = Vec::new();
    for (lineno, raw) in content.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        let tokens = split_fields(line);
        if tokens.len() != 1 {
            return Err(format!(
                ".con line {} has {} delimited fields; expected 1",
                lineno + 1,
                tokens.len()
            ));
        }
        let c: f64 = tokens[0].parse().map_err(|_| {
            format!(
                ".con line {}: could not parse {:?} as a number",
                lineno + 1,
                tokens[0]
            )
        })?;
        values.push(c);
    }
    if values.is_empty() {
        return Err(".con file contained no numeric entries".to_string());
    }
    Ok(Array1::from_vec(values))
}

/// Read the stoichiometry matrix and reference free energies from an
/// `.ocx` / `.cfe` file.
///
/// Returns `(A, ΔG)` where `A` is `n_species × n_mon` and `ΔG` is the
/// per-species reference free energy in kcal/mol. The first `n_mon` rows
/// must be the identity block (species = monomer).
///
/// The NUPACK layout is auto-detected by inspecting the first `min(n, 20)`
/// rows: if every row has `col0 == row_num + 1` and `col1 == 1`, those two
/// leading columns are treated as bookkeeping and dropped. Otherwise the
/// row is taken to be `[stoich_1, .., stoich_{n_mon}, ΔG]` directly.
/// Fields may be separated by whitespace, comma, semicolon, or pipe.
fn parse_ocx(content: &str, n_mon: usize) -> Result<(Array2<f64>, Array1<f64>), String> {
    let rows: Vec<Vec<String>> = content
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .map(|l| split_fields(l).into_iter().map(str::to_owned).collect())
        .collect();

    if rows.is_empty() {
        return Err("ocx/cfe file contained no data rows".into());
    }

    let nupack = detect_nupack_header(&rows);
    let cols_expected_raw = if nupack { n_mon + 3 } else { n_mon + 1 };

    let n_species = rows.len();
    let mut stoich = Array2::<f64>::zeros((n_species, n_mon));
    let mut dg = Array1::<f64>::zeros(n_species);

    for (i, row) in rows.iter().enumerate() {
        if row.len() != cols_expected_raw {
            return Err(format!(
                "ocx/cfe row {} has {} fields, expected {}",
                i + 1,
                row.len(),
                cols_expected_raw
            ));
        }

        let stoich_start = if nupack { 2 } else { 0 };
        for j in 0..n_mon {
            stoich[[i, j]] = row[stoich_start + j].parse::<f64>().map_err(|_| {
                format!(
                    "ocx/cfe row {}: could not parse stoichiometry {:?} as a number",
                    i + 1,
                    row[stoich_start + j]
                )
            })?;
        }
        let dg_idx = row.len() - 1;
        dg[i] = row[dg_idx].parse::<f64>().map_err(|_| {
            format!(
                "ocx/cfe row {}: could not parse ΔG {:?} as a number",
                i + 1,
                row[dg_idx]
            )
        })?;
    }

    Ok((stoich, dg))
}

/// Return `true` iff the leading columns of each row look like NUPACK
/// bookkeeping (`col0 = row+1`, `col1 = 1`) for every sampled row.
fn detect_nupack_header(rows: &[Vec<String>]) -> bool {
    let sample = rows.len().min(20);
    for (i, row) in rows.iter().take(sample).enumerate() {
        if row.len() < 3 {
            return false;
        }
        let c0 = row[0].parse::<i64>().ok();
        let c1 = row[1].parse::<i64>().ok();
        if c0 != Some((i + 1) as i64) || c1 != Some(1) {
            return false;
        }
    }
    true
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

/// Validate that a filename has one of the allowed extensions. The CLI
/// mirrors COFFEE's extension gating to catch obvious swaps (con ↔ cfe).
fn validate_extension(name: &str, role: &str, allowed: &[&str]) -> Result<String, String> {
    if allowed.iter().any(|ext| name.ends_with(ext)) {
        Ok(name.to_string())
    } else {
        Err(format!(
            "{role} file must end in one of: {}",
            allowed.join(", ")
        ))
    }
}

fn cfe_parser(s: &str) -> Result<String, String> {
    validate_extension(s, "cfe/ocx", &[".cfe", ".ocx", ".txt", ".csv", ".tsv"])
}

fn con_parser(s: &str) -> Result<String, String> {
    validate_extension(s, "con", &[".con", ".txt", ".csv", ".tsv"])
}

fn log_parser(s: &str) -> Result<String, String> {
    validate_extension(s, "log/output", &[".txt", ".log"])
}

/// Drop-in COFFEE-CLI front-end for equiconc.
#[derive(Parser, Debug)]
#[command(
    name = "equiconc-coffee-cli",
    version,
    about = "equiconc solver behind a coffee_cli-compatible interface"
)]
struct Cli {
    /// Complex table (`.cfe`/`.ocx`/`.txt`/`.csv`/`.tsv`): stoichiometry
    /// + reference free energies.
    #[arg(value_parser = cfe_parser)]
    cfe: String,

    /// Monomer concentrations (`.con`/`.txt`/`.csv`/`.tsv`): one value
    /// per line.
    #[arg(value_parser = con_parser)]
    con: String,

    /// Where to write the log (preamble + results payload). If omitted,
    /// the log is printed to stdout.
    #[arg(short = 'l', long = "log", value_parser = log_parser)]
    log: Option<String>,

    /// Where to write the results payload on its own. Independent of
    /// `--log`; both may be given.
    #[arg(short = 'o', long = "output", value_parser = log_parser)]
    output: Option<String>,

    /// Print extra progress detail to the log.
    #[arg(short = 'v', long = "verbose")]
    verbose: bool,
}

// ---------------------------------------------------------------------------
// Output formatting
// ---------------------------------------------------------------------------

/// Concatenate `format!("{:.2e} ", c)` across all species, matching the
/// COFFEE `results_message` wire format byte-for-byte (space-separated
/// 2-decimal scientific with a single trailing space, no newline).
fn format_results(concentrations: &[f64]) -> String {
    let mut s = String::with_capacity(concentrations.len() * 10);
    for c in concentrations {
        s.push_str(&format!("{:.2e} ", c));
    }
    s
}

/// Run statistics gathered between parse and output.
struct LogStats {
    n_mon: usize,
    n_species: usize,
    iterations: usize,
    converged_fully: bool,
    residual: f64,
    elapsed_ms: f64,
}

/// Compose the per-invocation log text. Original prose — no phrases
/// borrowed from COFFEE's `format.rs`.
fn format_log(cli: &Cli, stats: &LogStats, results: &str) -> String {
    let n_mon = stats.n_mon;
    let n_species = stats.n_species;
    let iterations = stats.iterations;
    let converged_fully = stats.converged_fully;
    let residual = stats.residual;
    let elapsed_ms = stats.elapsed_ms;
    let mut s = String::new();
    s.push_str(&format!(
        "equiconc-coffee-cli v{} — COFFEE-compatible equilibrium solve\n",
        env!("CARGO_PKG_VERSION")
    ));
    if cli.verbose {
        s.push_str(&format!("  complex table : {}\n", cli.cfe));
        s.push_str(&format!("  concentrations: {}\n", cli.con));
    }
    s.push_str(&format!(
        "  monomers: {n_mon}, species: {n_species}, T = {T_CELSIUS:.2} °C, mole-fraction mode\n"
    ));
    s.push_str(&format!(
        "  solved in {iterations} iterations ({}), {elapsed_ms:.2} ms\n",
        if converged_fully {
            "full precision"
        } else {
            "relaxed precision"
        }
    ));
    s.push_str(&format!(
        "  mass-balance residual (max |c0 - Aᵀ·c|): {residual:.3e}\n"
    ));
    s.push('\n');
    s.push_str(results);
    s.push('\n');
    s
}

/// Largest absolute per-monomer mass-balance deviation:
/// `max_i |c0_i − Σ_j A_{ji} · c_j|`.
///
/// Inputs are in the user-facing (pre-scaling) units: `c0_original` is the
/// un-scaled concentrations from `.con`, `conc` are the solved
/// concentrations already rescaled back out of mole-fraction space.
fn mass_balance_residual(a: &Array2<f64>, c0_original: &Array1<f64>, conc: &[f64]) -> f64 {
    let n_mon = c0_original.len();
    let n_species = a.nrows();
    debug_assert_eq!(conc.len(), n_species);

    let mut worst: f64 = 0.0;
    for i in 0..n_mon {
        let mut total = 0.0;
        for j in 0..n_species {
            total += a[[j, i]] * conc[j];
        }
        let diff = (c0_original[i] - total).abs();
        if diff > worst {
            worst = diff;
        }
    }
    worst
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    let cfe_content = read_file(&cli.cfe, "complex table")?;
    let con_content = read_file(&cli.con, "concentrations")?;

    let c0 = parse_con(&con_content)?;
    let n_mon = c0.len();

    let (stoich, dg) = parse_ocx(&cfe_content, n_mon)?;
    let n_species = stoich.nrows();
    if n_species < n_mon {
        return Err(format!(
            "need at least n_mon={n_mon} species in ocx (got {n_species})"
        )
        .into());
    }

    let rt = R_KCAL_PER_MOL_K * (T_CELSIUS + 273.15);
    // Clamp ΔG in energy units before converting to log_q, so the effective
    // physical cutoff matches COFFEE's `(-x.max(-230) / k_t).exp()`.
    // Monomer rows carry ΔG = 0 and are unaffected.
    let log_q = dg.mapv(|g| -g.max(-DG_CLAMP_KCAL_PER_MOL) / rt);

    let rho_water = density_water_molar(T_CELSIUS);
    let c0_fraction = c0.mapv(|c| c / rho_water);

    // No `log_q_clamp` in `SolverOptions` — we've already applied the
    // ΔG-space clamp above, which is the physically-meaningful one.
    // Allow more outer iterations than the library default: COFFEE-scale
    // inputs can have very large `q` values that put the dual objective
    // in a regime where equiconc's dog-leg solver takes many outer steps.
    let opts = SolverOptions {
        max_iterations: 5000,
        ..SolverOptions::default()
    };

    let t0 = Instant::now();
    let mut sys = System::from_arrays_with_options(stoich.clone(), log_q, c0_fraction, opts)?;
    let eq = sys.solve()?;
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let iterations = eq.iterations();
    let converged_fully = eq.converged_fully();

    let concentrations: Vec<f64> = eq.concentrations().iter().map(|c| c * rho_water).collect();

    let residual = mass_balance_residual(&stoich, &c0, &concentrations);
    let results_str = format_results(&concentrations);
    let stats = LogStats {
        n_mon,
        n_species,
        iterations,
        converged_fully,
        residual,
        elapsed_ms,
    };
    let log_str = format_log(&cli, &stats, &results_str);

    if let Some(log_path) = &cli.log {
        fs::write(log_path, &log_str)
            .map_err(|e| format!("failed to write log {log_path}: {e}"))?;
    } else {
        print!("{log_str}");
    }

    if let Some(out_path) = &cli.output {
        fs::write(out_path, &results_str)
            .map_err(|e| format!("failed to write output {out_path}: {e}"))?;
    }

    Ok(())
}

fn read_file(path: &str, role: &str) -> Result<String, Box<dyn std::error::Error>> {
    fs::read_to_string(Path::new(path)).map_err(|e| {
        format!("failed to read {role} file {path}: {e}").into()
    })
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("equiconc-coffee-cli: {e}");
            ExitCode::FAILURE
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn density_water_matches_reference_values() {
        // Reference values from Tanaka et al. 2001 Table 2 (kg/m³)
        // divided by M = 18.01528 g/mol to convert to mol/L.
        // At 0°C:   999.8395 / 18.01528 = 55.498...
        // At 25°C:  997.0479 / 18.01528 = 55.343...
        // At 37°C:  993.3317 / 18.01528 = 55.137...
        let cases = [
            (0.0_f64, 55.498_f64),
            (25.0_f64, 55.343_f64),
            (37.0_f64, 55.137_f64),
        ];
        for (t, expected) in cases {
            let got = density_water_molar(t);
            assert!(
                (got - expected).abs() / expected < 1e-3,
                "ρ(T={t} °C) = {got}, expected ≈ {expected}"
            );
        }
    }

    #[test]
    fn parse_con_rejects_multi_column() {
        let content = "1.0e-6\n2.0e-6,oops\n";
        assert!(parse_con(content).is_err());
    }

    #[test]
    fn parse_con_accepts_blank_lines_and_whitespace() {
        let content = "\n  1.0e-6\n\n2.0e-6\n";
        let got = parse_con(content).unwrap();
        assert_eq!(got.len(), 2);
        assert!((got[0] - 1.0e-6).abs() < 1e-18);
        assert!((got[1] - 2.0e-6).abs() < 1e-18);
    }

    #[test]
    fn parse_ocx_nupack_layout() {
        // 2 monomers, 3 species (2 identity + 1 AB complex).
        let content = "\
1\t1\t1\t0\t0.0
2\t1\t0\t1\t0.0
3\t1\t1\t1\t-5.0
";
        let (a, dg) = parse_ocx(content, 2).unwrap();
        assert_eq!(a.shape(), &[3, 2]);
        assert_eq!(a[[0, 0]], 1.0);
        assert_eq!(a[[0, 1]], 0.0);
        assert_eq!(a[[1, 0]], 0.0);
        assert_eq!(a[[1, 1]], 1.0);
        assert_eq!(a[[2, 0]], 1.0);
        assert_eq!(a[[2, 1]], 1.0);
        assert_eq!(dg[0], 0.0);
        assert_eq!(dg[1], 0.0);
        assert_eq!(dg[2], -5.0);
    }

    #[test]
    fn parse_ocx_raw_layout() {
        // Same system without NUPACK bookkeeping columns.
        let content = "\
1\t0\t0.0
0\t1\t0.0
1\t1\t-5.0
";
        let (a, dg) = parse_ocx(content, 2).unwrap();
        assert_eq!(a.shape(), &[3, 2]);
        assert_eq!(a[[2, 0]], 1.0);
        assert_eq!(a[[2, 1]], 1.0);
        assert_eq!(dg[2], -5.0);
    }

    #[test]
    fn parse_ocx_raw_layout_accepts_coffee_delimiters() {
        let content = "\
1,0,0.0
0;1;0.0
1|1|-5.0
";
        let (a, dg) = parse_ocx(content, 2).unwrap();
        assert_eq!(a.shape(), &[3, 2]);
        assert_eq!(a[[0, 0]], 1.0);
        assert_eq!(a[[1, 1]], 1.0);
        assert_eq!(a[[2, 0]], 1.0);
        assert_eq!(a[[2, 1]], 1.0);
        assert_eq!(dg[2], -5.0);
    }

    #[test]
    fn parse_ocx_nupack_layout_accepts_csv() {
        let content = "\
1,1,1,0,0.0
2,1,0,1,0.0
3,1,1,1,-5.0
";
        let (a, dg) = parse_ocx(content, 2).unwrap();
        assert_eq!(a.shape(), &[3, 2]);
        assert_eq!(a[[2, 0]], 1.0);
        assert_eq!(a[[2, 1]], 1.0);
        assert_eq!(dg[2], -5.0);
    }

    #[test]
    fn parse_ocx_rejects_wrong_width() {
        let content = "1\t1\t1\t0\t0\t0.0\n";
        assert!(parse_ocx(content, 2).is_err());
    }

    #[test]
    fn results_format_is_byte_compatible() {
        // Two values, 2-sf scientific, single trailing space, no newline.
        let s = format_results(&[1.23e-8, 4.56e-6]);
        assert_eq!(s, "1.23e-8 4.56e-6 ");
    }

    #[test]
    fn mass_balance_residual_is_zero_on_identity() {
        // If A is identity and conc = c0, residual = 0.
        let a = ndarray::arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        let c0 = ndarray::arr1(&[1e-6, 2e-6]);
        let r = mass_balance_residual(&a, &c0, &[1e-6, 2e-6]);
        assert!(r < 1e-20);
    }
}
