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
use equiconc::io::{parse_cfe, parse_concentrations};
use equiconc::{SolverOptions, System, water_molar_density};
use ndarray::Array1;

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

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    let cfe_content = read_file(&cli.cfe, "complex table")?;
    let con_content = read_file(&cli.con, "concentrations")?;

    let c0 = parse_concentrations(&con_content)?;
    let n_mon = c0.len();

    let (stoich, dg) = parse_cfe(&cfe_content, n_mon)?;
    let n_species = stoich.nrows();
    if n_species < n_mon {
        return Err(format!("need at least n_mon={n_mon} species in ocx (got {n_species})").into());
    }

    let rt = R_KCAL_PER_MOL_K * (T_CELSIUS + 273.15);
    // Clamp ΔG in energy units before converting to log_q, so the effective
    // physical cutoff matches COFFEE's `(-x.max(-230) / k_t).exp()`.
    // Monomer rows carry ΔG = 0 and are unaffected.
    let log_q = dg.mapv(|g| -g.max(-DG_CLAMP_KCAL_PER_MOL) / rt);

    let rho_water = water_molar_density(T_CELSIUS);
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

    let concentrations: Array1<f64> = eq.concentrations().iter().map(|c| c * rho_water).collect();

    let residual = equiconc::mass_balance_residual(stoich.view(), c0.view(), concentrations.view());
    let results_str = format_results(concentrations.as_slice().expect("contiguous"));
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
    fs::read_to_string(Path::new(path))
        .map_err(|e| format!("failed to read {role} file {path}: {e}").into())
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
    fn results_format_is_byte_compatible() {
        let s = format_results(&[1.23e-8, 4.56e-6]);
        assert_eq!(s, "1.23e-8 4.56e-6 ");
    }
}
