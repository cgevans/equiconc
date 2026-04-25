//! Render solve results into TSV / CSV / JSON for export.
//!
//! Stays string-only — actually pushing bytes into the clipboard or a
//! download blob is in `outputs.rs`, where we have access to `web-sys`.

use crate::state::SolveResult;

/// Render the per-species table as TSV with a header row.
///
/// Columns: `index, name, ΔG_kcal_mol, c_M, ` then one column per
/// monomer-share-of-mass (`share_M1`, `share_M2`, …). `name` is
/// `M{i}` for monomers and `S{i}` for species, indexed from 1 to
/// match the .ocx file layout. We don't have user names in this
/// front end yet (the `parse_cfe` path doesn't carry them).
pub fn results_tsv(result: &SolveResult) -> String {
    render(result, '\t')
}

/// Same as [`results_tsv`], with `,` as the separator.
pub fn results_csv(result: &SolveResult) -> String {
    render(result, ',')
}

fn render(result: &SolveResult, sep: char) -> String {
    let n_species = result.concentrations.len();
    let n_mon = result.n_mon;
    let mut s = String::new();

    s.push_str("index");
    s.push(sep);
    s.push_str("name");
    s.push(sep);
    s.push_str("dG_kcal_mol");
    s.push(sep);
    s.push_str("concentration_M");
    for i in 0..n_mon {
        s.push(sep);
        s.push_str(&format!("share_M{}", i + 1));
    }
    s.push('\n');

    let shares: Vec<_> = (0..n_mon).map(|i| result.share_of_monomer(i)).collect();
    for j in 0..n_species {
        let name = if j < n_mon {
            format!("M{}", j + 1)
        } else {
            format!("S{}", j + 1)
        };
        s.push_str(&(j + 1).to_string());
        s.push(sep);
        s.push_str(&name);
        s.push(sep);
        s.push_str(&format!("{:.6}", result.dg_kcal_used[j]));
        s.push(sep);
        s.push_str(&format!("{:.6e}", result.concentrations[j]));
        for share in shares.iter() {
            s.push(sep);
            s.push_str(&format!("{:.6}", share[j]));
        }
        s.push('\n');
    }
    s
}

/// Reproducibility bundle: the inputs, the options, and the result.
/// Hand-rolled JSON so we don't have to pull in `serde_json` for what
/// is essentially a flat dump.
pub fn report_json(cfe_text: &str, con_text: &str, result: &SolveResult) -> String {
    let mut s = String::new();
    s.push_str("{\n");
    s.push_str("  \"version\": 1,\n");
    s.push_str("  \"inputs\": {\n");
    s.push_str("    \"cfe\": ");
    push_json_string(&mut s, cfe_text);
    s.push_str(",\n    \"con\": ");
    push_json_string(&mut s, con_text);
    s.push_str("\n  },\n");

    s.push_str("  \"options\": {\n");
    s.push_str(&format!(
        "    \"energy_unit\": \"{}\",\n",
        match result.options.energy_unit {
            crate::state::EnergyUnit::KcalPerMol => "kcal_per_mol",
            crate::state::EnergyUnit::RT => "RT",
        }
    ));
    s.push_str(&format!(
        "    \"temperature_K\": {:.6},\n",
        result.options.temperature_kelvin()
    ));
    s.push_str(&format!(
        "    \"scalarity\": {},\n",
        result.options.scalarity
    ));
    s.push_str(&format!(
        "    \"dg_clamp_on\": {},\n",
        result.options.dg_clamp_on
    ));
    s.push_str(&format!(
        "    \"dg_clamp_kcal\": {:.6},\n",
        result.options.dg_clamp_kcal
    ));
    s.push_str(&format!(
        "    \"max_iterations\": {},\n",
        result.options.solver.max_iterations
    ));
    s.push_str(&format!(
        "    \"gradient_abs_tol\": {:e},\n",
        result.options.solver.gradient_abs_tol
    ));
    s.push_str(&format!(
        "    \"gradient_rel_tol\": {:e},\n",
        result.options.solver.gradient_rel_tol
    ));
    s.push_str(&format!(
        "    \"objective\": \"{:?}\"\n",
        result.options.solver.objective
    ));
    s.push_str("  },\n");

    s.push_str("  \"diagnostics\": {\n");
    s.push_str(&format!("    \"iterations\": {},\n", result.iterations));
    s.push_str(&format!(
        "    \"converged_fully\": {},\n",
        result.converged_fully
    ));
    s.push_str(&format!(
        "    \"mass_balance_residual\": {:e},\n",
        result.residual
    ));
    s.push_str(&format!("    \"elapsed_ms\": {:.6}\n", result.elapsed_ms));
    s.push_str("  },\n");

    s.push_str("  \"c0_M\": [");
    for (i, c) in result.c0.iter().enumerate() {
        if i > 0 {
            s.push_str(", ");
        }
        s.push_str(&format!("{:e}", c));
    }
    s.push_str("],\n");

    s.push_str("  \"concentrations_M\": [");
    for (j, c) in result.concentrations.iter().enumerate() {
        if j > 0 {
            s.push_str(", ");
        }
        s.push_str(&format!("{:e}", c));
    }
    s.push_str("]\n");

    s.push_str("}\n");
    s
}

fn push_json_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::solve;
    use crate::state::UiOptions;

    fn ab_solve() -> SolveResult {
        let cfe = "1\t1\t1\t0\t0.0\n2\t1\t0\t1\t0.0\n3\t1\t1\t1\t-10.0\n";
        let con = "1.0e-6\n1.0e-6\n";
        solve(cfe, con, &UiOptions::equiconc_default()).unwrap()
    }

    #[test]
    fn tsv_has_header_and_one_row_per_species() {
        let r = ab_solve();
        let tsv = results_tsv(&r);
        let lines: Vec<&str> = tsv.lines().collect();
        assert_eq!(lines.len(), 1 + r.concentrations.len());
        assert!(lines[0].starts_with("index\tname\t"));
    }

    #[test]
    fn json_round_trip_includes_inputs() {
        let r = ab_solve();
        let cfe = "1\t1\t1\t0\t0.0\n";
        let con = "1.0e-6\n";
        let s = report_json(cfe, con, &r);
        assert!(s.contains("\"version\": 1"));
        assert!(s.contains("\"cfe\": \"1\\t1\\t1\\t0\\t0.0\\n\""));
        assert!(s.contains("\"converged_fully\""));
    }
}
