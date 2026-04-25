//! Parsers for NUPACK-style complex tables (`.cfe` / `.ocx`) and
//! one-value-per-line concentration files (`.con`).
//!
//! These are general primitives — they encode the NUPACK and COFFEE
//! file formats, not anything COFFEE-specific about how the solver is
//! configured.
//!
//! Both parsers accept whitespace, comma, semicolon, or pipe as field
//! separators so `.csv` / `.tsv` extensions behave as advertised. Empty
//! fields are ignored, matching the repeated-whitespace convention.

use ndarray::{Array1, Array2};

/// Split a row into delimited fields. Whitespace, `,`, `;`, and `|` all
/// separate; consecutive separators collapse.
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
pub fn parse_concentrations(content: &str) -> Result<Array1<f64>, String> {
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
/// per-species reference free energy (in whatever units the caller is
/// using — kcal/mol in NUPACK files).
///
/// The NUPACK layout is auto-detected by inspecting the first `min(n, 20)`
/// rows: if every row has `col0 == row_num + 1` and `col1 == 1`, those two
/// leading columns are treated as bookkeeping and dropped. Otherwise the
/// row is taken to be `[stoich_1, .., stoich_{n_mon}, ΔG]` directly.
pub fn parse_cfe(content: &str, n_mon: usize) -> Result<(Array2<f64>, Array1<f64>), String> {
    let rows: Vec<Vec<String>> = content
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && !l.starts_with('%') && !l.starts_with('#'))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_concentrations_rejects_multi_column() {
        let content = "1.0e-6\n2.0e-6,oops\n";
        assert!(parse_concentrations(content).is_err());
    }

    #[test]
    fn parse_concentrations_accepts_blank_lines_and_whitespace() {
        let content = "\n  1.0e-6\n\n2.0e-6\n";
        let got = parse_concentrations(content).unwrap();
        assert_eq!(got.len(), 2);
        assert!((got[0] - 1.0e-6).abs() < 1e-18);
        assert!((got[1] - 2.0e-6).abs() < 1e-18);
    }

    #[test]
    fn parse_cfe_nupack_layout() {
        let content = "\
1\t1\t1\t0\t0.0
2\t1\t0\t1\t0.0
3\t1\t1\t1\t-5.0
";
        let (a, dg) = parse_cfe(content, 2).unwrap();
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
    fn parse_cfe_raw_layout() {
        let content = "\
1\t0\t0.0
0\t1\t0.0
1\t1\t-5.0
";
        let (a, dg) = parse_cfe(content, 2).unwrap();
        assert_eq!(a.shape(), &[3, 2]);
        assert_eq!(a[[2, 0]], 1.0);
        assert_eq!(a[[2, 1]], 1.0);
        assert_eq!(dg[2], -5.0);
    }

    #[test]
    fn parse_cfe_raw_layout_accepts_alt_delimiters() {
        let content = "\
1,0,0.0
0;1;0.0
1|1|-5.0
";
        let (a, dg) = parse_cfe(content, 2).unwrap();
        assert_eq!(a.shape(), &[3, 2]);
        assert_eq!(a[[0, 0]], 1.0);
        assert_eq!(a[[1, 1]], 1.0);
        assert_eq!(a[[2, 0]], 1.0);
        assert_eq!(a[[2, 1]], 1.0);
        assert_eq!(dg[2], -5.0);
    }

    #[test]
    fn parse_cfe_nupack_layout_csv() {
        let content = "\
1,1,1,0,0.0
2,1,0,1,0.0
3,1,1,1,-5.0
";
        let (a, dg) = parse_cfe(content, 2).unwrap();
        assert_eq!(a.shape(), &[3, 2]);
        assert_eq!(a[[2, 0]], 1.0);
        assert_eq!(a[[2, 1]], 1.0);
        assert_eq!(dg[2], -5.0);
    }

    #[test]
    fn parse_cfe_rejects_wrong_width() {
        let content = "1\t1\t1\t0\t0\t0.0\n";
        assert!(parse_cfe(content, 2).is_err());
    }

    #[test]
    fn parse_cfe_skips_comment_lines() {
        let content = "\
% NUPACK comment
# also a comment
1\t1\t1\t0\t0.0
2\t1\t0\t1\t0.0
3\t1\t1\t1\t-5.0
";
        let (a, _dg) = parse_cfe(content, 2).unwrap();
        assert_eq!(a.shape(), &[3, 2]);
    }
}
