//! Glue between the UI state and the `equiconc` solver. Lives in plain
//! Rust — no wasm/JS interop here — so the same code runs natively in
//! `cargo test`.

use equiconc::io::{parse_cfe, parse_concentrations};
use equiconc::{IterationStatus, R, SolveControl, System, water_molar_density};
use ndarray::Array1;

use crate::state::{EnergyUnit, ErrSource, SolveError, SolveResult, UiOptions};

/// Wall-clock timer that uses `performance.now()` in the browser and
/// `std::time::Instant` natively. Lets the same module run in
/// `cargo test` without dragging in `web_sys::window()`, which only
/// resolves on wasm32.
struct Timer {
    #[cfg(target_arch = "wasm32")]
    perf: Option<web_sys::Performance>,
    #[cfg(target_arch = "wasm32")]
    start: f64,
    #[cfg(not(target_arch = "wasm32"))]
    start: std::time::Instant,
}

impl Timer {
    #[cfg(target_arch = "wasm32")]
    fn start() -> Self {
        let perf = web_sys::window().and_then(|w| w.performance());
        let start = perf.as_ref().map(|p| p.now()).unwrap_or(0.0);
        Self { perf, start }
    }
    #[cfg(not(target_arch = "wasm32"))]
    fn start() -> Self {
        Self {
            start: std::time::Instant::now(),
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn elapsed_ms(&self) -> f64 {
        self.perf
            .as_ref()
            .map(|p| p.now() - self.start)
            .unwrap_or(0.0)
    }
    #[cfg(not(target_arch = "wasm32"))]
    fn elapsed_ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1000.0
    }
}

/// Run the full pipeline: parse the textareas, configure the solver,
/// solve, package the result.
pub fn solve(cfe_text: &str, con_text: &str, opts: &UiOptions) -> Result<SolveResult, SolveError> {
    solve_with_progress(cfe_text, con_text, opts, |_| SolveControl::Continue)
}

/// As [`solve`], but invokes `on_iter` once per outer trust-region
/// iteration. Used by the worker to forward progress messages to the
/// main thread.
pub fn solve_with_progress<F>(
    cfe_text: &str,
    con_text: &str,
    opts: &UiOptions,
    mut on_iter: F,
) -> Result<SolveResult, SolveError>
where
    F: FnMut(&IterationStatus) -> SolveControl,
{
    if let Err(e) = opts.solver.validate() {
        return Err(SolveError::from_options(format!("{e}")));
    }

    let c0_molar = parse_concentrations(con_text).map_err(SolveError::from_concentrations)?;
    let n_mon = c0_molar.len();

    let (stoich, dg_kcal) = parse_cfe(cfe_text, n_mon).map_err(SolveError::from_composition)?;
    if stoich.nrows() < n_mon {
        return Err(SolveError::from_composition(format!(
            "need at least n_mon = {n_mon} species in the composition table (got {})",
            stoich.nrows()
        )));
    }

    // Temperature is required when energy is in kcal/mol (used to
    // build RT) or when scalarity is on (used for water density).
    // With RT-units inputs and scalarity off it isn't read at all.
    let needs_temperature = opts.energy_unit == EnergyUnit::KcalPerMol || opts.scalarity;
    let t_kelvin = opts.temperature_kelvin();
    if needs_temperature && (!t_kelvin.is_finite() || t_kelvin <= 0.0) {
        return Err(SolveError::from_options(format!(
            "temperature must be positive (got {t_kelvin} K)"
        )));
    }

    // Apply the optional ΔG clamp before converting to log_q. The
    // clamp value is in the same units as the input (kcal/mol or RT).
    let dg_used = if opts.dg_clamp_on {
        let bound = -opts.dg_clamp_kcal.abs();
        dg_kcal.mapv(|g| g.max(bound))
    } else {
        dg_kcal.clone()
    };
    let log_q = match opts.energy_unit {
        EnergyUnit::KcalPerMol => {
            let rt = R * t_kelvin;
            dg_used.mapv(|g| -g / rt)
        }
        EnergyUnit::RT => dg_used.mapv(|g| -g),
    };
    let dg_kcal_used = dg_used;

    let (c0_for_solver, rho_water) = if opts.scalarity {
        let rho = water_molar_density(opts.temperature_celsius());
        (c0_molar.mapv(|c| c / rho), rho)
    } else {
        (c0_molar.clone(), 1.0_f64)
    };

    let solver_opts = opts.solver.clone();

    let timer = Timer::start();
    let mut sys =
        System::from_arrays_with_options(stoich.clone(), log_q, c0_for_solver, solver_opts)
            .map_err(|e| translate_solver_err(&e))?;
    let eq = sys
        .solve_with_progress(&mut on_iter)
        .map_err(|e| translate_solver_err(&e))?;
    let elapsed_ms = timer.elapsed_ms();

    let iterations = eq.iterations();
    let converged_fully = eq.converged_fully();

    let concentrations: Array1<f64> = if opts.scalarity {
        eq.concentrations().iter().map(|c| c * rho_water).collect()
    } else {
        eq.concentrations().to_owned()
    };

    let residual =
        equiconc::mass_balance_residual(stoich.view(), c0_molar.view(), concentrations.view());

    Ok(SolveResult {
        concentrations,
        c0: c0_molar,
        stoich,
        dg_kcal_used,
        n_mon,
        iterations,
        converged_fully,
        residual,
        elapsed_ms,
        options: opts.clone(),
    })
}

fn translate_solver_err(e: &equiconc::EquilibriumError) -> SolveError {
    let msg = format!("{e}");
    SolveError {
        message: msg,
        source: ErrSource::Solver,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ab_inputs() -> (&'static str, &'static str) {
        (
            "1\t1\t1\t0\t0.0\n2\t1\t0\t1\t0.0\n3\t1\t1\t1\t-10.0\n",
            "1.0e-6\n1.0e-6\n",
        )
    }

    #[test]
    fn solve_runs_on_ab_dimer() {
        let (cfe, con) = ab_inputs();
        let mut opts = UiOptions::equiconc_default();
        opts.solver.max_iterations = 1000;
        let res = solve(cfe, con, &opts).unwrap();
        assert_eq!(res.n_mon, 2);
        assert_eq!(res.concentrations.len(), 3);
        // Mass balance: residual should be tiny relative to c0 = 1e-6.
        assert!(res.residual < 1e-12, "residual={}", res.residual);
        // AB should have absorbed most of the monomers at -10 kcal/mol.
        let ab = res.concentrations[2];
        let a_free = res.concentrations[0];
        assert!(ab > a_free, "ab={ab}, a_free={a_free}");
    }

    #[test]
    fn coffee_compatible_preset_runs() {
        let (cfe, con) = ab_inputs();
        let opts = UiOptions::coffee_compatible();
        let res = solve(cfe, con, &opts).unwrap();
        assert!(res.iterations > 0);
        assert!(res.residual < 1e-9);
    }

    #[test]
    fn bad_concentrations_surface_as_concentration_error() {
        let (cfe, _) = ab_inputs();
        let con = "1.0e-6\nnotanumber\n";
        let err = solve(cfe, con, &UiOptions::equiconc_default()).unwrap_err();
        assert_eq!(err.source, ErrSource::Concentrations);
    }

    #[test]
    fn rt_units_matches_kcal_input_pre_divided_by_rt() {
        // Solve the same system two ways and compare:
        //   (a) ΔG = −10 kcal/mol, kcal/mol units, T = 25 °C
        //   (b) ΔG = −10 / RT (dimensionless), RT units (no temperature)
        // The solver should see identical log_q in both, so the
        // equilibrium concentrations must agree to round-off.
        let con = "1.0e-6\n1.0e-6\n";
        let cfe_kcal = "1\t1\t1\t0\t0.0\n2\t1\t0\t1\t0.0\n3\t1\t1\t1\t-10.0\n";
        let mut opts_kcal = UiOptions::equiconc_default();
        opts_kcal.solver.max_iterations = 1000;
        let res_kcal = solve(cfe_kcal, con, &opts_kcal).unwrap();

        let rt = R * (25.0 + 273.15);
        let dg_rt = -10.0_f64 / rt;
        let cfe_rt = format!("1\t1\t1\t0\t0.0\n2\t1\t0\t1\t0.0\n3\t1\t1\t1\t{dg_rt}\n");
        let mut opts_rt = UiOptions::equiconc_default();
        opts_rt.energy_unit = EnergyUnit::RT;
        opts_rt.solver.max_iterations = 1000;
        let res_rt = solve(&cfe_rt, con, &opts_rt).unwrap();

        for (a, b) in res_kcal
            .concentrations
            .iter()
            .zip(res_rt.concentrations.iter())
        {
            let denom = a.abs().max(1e-30);
            assert!((a - b).abs() / denom < 1e-9, "kcal={a:e} vs RT={b:e}");
        }
    }

    #[test]
    fn rt_units_with_scalarity_off_ignores_temperature_changes() {
        // With energy_unit = RT and scalarity off, temperature is not
        // read by the pipeline. Solving at two wildly different T's
        // should produce the same concentrations.
        let con = "1.0e-6\n1.0e-6\n";
        let cfe = "1\t1\t1\t0\t0.0\n2\t1\t0\t1\t0.0\n3\t1\t1\t1\t-15.0\n";

        let mut opts_low = UiOptions::equiconc_default();
        opts_low.energy_unit = EnergyUnit::RT;
        opts_low.temperature_value = 0.0;
        opts_low.solver.max_iterations = 1000;
        let res_low = solve(cfe, con, &opts_low).unwrap();

        let mut opts_high = opts_low.clone();
        opts_high.temperature_value = 95.0;
        let res_high = solve(cfe, con, &opts_high).unwrap();

        for (a, b) in res_low
            .concentrations
            .iter()
            .zip(res_high.concentrations.iter())
        {
            assert_eq!(a, b, "concentrations differed across T with RT units");
        }
    }
}
