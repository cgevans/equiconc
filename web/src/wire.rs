//! Serializable types crossing the worker `postMessage` boundary.
//!
//! `ndarray::Array1` / `Array2` aren't directly `Serialize`-compatible
//! through the default gloo-worker bincode codec, so we ferry flat
//! `Vec<f64>` plus shape and reconstruct on the main side. This is the
//! only place that knows about the wire shape; the UI works in terms
//! of [`crate::state::SolveResult`] and [`crate::state::SolveError`].

use equiconc::{SolverObjective, SolverOptions};
use serde::{Deserialize, Serialize};

use crate::state::{EnergyUnit, ErrSource, SolveError, SolveResult, TempUnit, UiOptions};

/// Outbound from the main thread to the worker. One per Solve click.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolveRequest {
    pub cfe: String,
    pub con: String,
    pub options: WireUiOptions,
    /// Maximum interval (in milliseconds) between forwarded
    /// [`SolveResponse::Progress`] messages. The worker always emits
    /// the first iteration and the last-pre-completion iteration in
    /// addition to the throttled stream so the UI sees both endpoints.
    pub progress_throttle_ms: f64,
}

/// Serializable mirror of [`UiOptions`]. Shape matches one-to-one;
/// the only reason for this type to exist is that `equiconc::SolverOptions`
/// doesn't itself derive serde and we don't want to add a serde feature
/// to the numerical crate just for the web bundler.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WireUiOptions {
    pub temperature_value: f64,
    pub temperature_unit: TempUnit,
    #[serde(default = "default_energy_unit")]
    pub energy_unit: EnergyUnit,
    pub scalarity: bool,
    pub dg_clamp_on: bool,
    pub dg_clamp_kcal: f64,
    pub solver: WireSolverOptions,
}

fn default_energy_unit() -> EnergyUnit {
    EnergyUnit::KcalPerMol
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum WireObjective {
    Linear,
    Log,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WireSolverOptions {
    pub max_iterations: usize,
    pub gradient_abs_tol: f64,
    pub gradient_rel_tol: f64,
    pub relaxed_gradient_abs_tol: f64,
    pub relaxed_gradient_rel_tol: f64,
    pub stagnation_threshold: u32,
    pub initial_trust_region_radius: f64,
    pub max_trust_region_radius: f64,
    pub step_accept_threshold: f64,
    pub trust_region_shrink_rho: f64,
    pub trust_region_grow_rho: f64,
    pub trust_region_shrink_scale: f64,
    pub trust_region_grow_scale: f64,
    pub log_c_clamp: f64,
    pub log_q_clamp: Option<f64>,
    pub objective: WireObjective,
}

impl From<&SolverOptions> for WireSolverOptions {
    fn from(o: &SolverOptions) -> Self {
        Self {
            max_iterations: o.max_iterations,
            gradient_abs_tol: o.gradient_abs_tol,
            gradient_rel_tol: o.gradient_rel_tol,
            relaxed_gradient_abs_tol: o.relaxed_gradient_abs_tol,
            relaxed_gradient_rel_tol: o.relaxed_gradient_rel_tol,
            stagnation_threshold: o.stagnation_threshold,
            initial_trust_region_radius: o.initial_trust_region_radius,
            max_trust_region_radius: o.max_trust_region_radius,
            step_accept_threshold: o.step_accept_threshold,
            trust_region_shrink_rho: o.trust_region_shrink_rho,
            trust_region_grow_rho: o.trust_region_grow_rho,
            trust_region_shrink_scale: o.trust_region_shrink_scale,
            trust_region_grow_scale: o.trust_region_grow_scale,
            log_c_clamp: o.log_c_clamp,
            log_q_clamp: o.log_q_clamp,
            objective: match o.objective {
                SolverObjective::Linear => WireObjective::Linear,
                SolverObjective::Log => WireObjective::Log,
            },
        }
    }
}

impl From<WireSolverOptions> for SolverOptions {
    fn from(w: WireSolverOptions) -> Self {
        SolverOptions {
            max_iterations: w.max_iterations,
            gradient_abs_tol: w.gradient_abs_tol,
            gradient_rel_tol: w.gradient_rel_tol,
            relaxed_gradient_abs_tol: w.relaxed_gradient_abs_tol,
            relaxed_gradient_rel_tol: w.relaxed_gradient_rel_tol,
            stagnation_threshold: w.stagnation_threshold,
            initial_trust_region_radius: w.initial_trust_region_radius,
            max_trust_region_radius: w.max_trust_region_radius,
            step_accept_threshold: w.step_accept_threshold,
            trust_region_shrink_rho: w.trust_region_shrink_rho,
            trust_region_grow_rho: w.trust_region_grow_rho,
            trust_region_shrink_scale: w.trust_region_shrink_scale,
            trust_region_grow_scale: w.trust_region_grow_scale,
            log_c_clamp: w.log_c_clamp,
            log_q_clamp: w.log_q_clamp,
            objective: match w.objective {
                WireObjective::Linear => SolverObjective::Linear,
                WireObjective::Log => SolverObjective::Log,
            },
        }
    }
}

impl From<&UiOptions> for WireUiOptions {
    fn from(u: &UiOptions) -> Self {
        Self {
            temperature_value: u.temperature_value,
            temperature_unit: u.temperature_unit,
            energy_unit: u.energy_unit,
            scalarity: u.scalarity,
            dg_clamp_on: u.dg_clamp_on,
            dg_clamp_kcal: u.dg_clamp_kcal,
            solver: WireSolverOptions::from(&u.solver),
        }
    }
}

impl From<WireUiOptions> for UiOptions {
    fn from(w: WireUiOptions) -> Self {
        UiOptions {
            temperature_value: w.temperature_value,
            temperature_unit: w.temperature_unit,
            energy_unit: w.energy_unit,
            scalarity: w.scalarity,
            dg_clamp_on: w.dg_clamp_on,
            dg_clamp_kcal: w.dg_clamp_kcal,
            solver: w.solver.into(),
        }
    }
}

/// Inbound to the main thread from the worker. The worker emits zero
/// or more `Progress` messages, then exactly one of `Done` or `Error`.
/// `Done` boxes the result because [`SolveResultWire`] is much larger
/// than the other variants (carries the full concentration / stoich
/// vectors).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SolveResponse {
    Progress(ProgressMsg),
    Done(Box<SolveResultWire>),
    Error(SolveErrorWire),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressMsg {
    pub iteration: usize,
    pub gradient_norm: f64,
    pub objective: f64,
    pub trust_radius: f64,
    /// Wall-clock milliseconds since the worker received the
    /// `SolveRequest`, captured by `performance.now()` on the worker
    /// thread. Useful for the convergence chart's secondary axis.
    pub elapsed_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolveResultWire {
    pub concentrations: Vec<f64>,
    pub c0: Vec<f64>,
    pub stoich_flat: Vec<f64>,
    pub stoich_shape: (usize, usize),
    pub dg_kcal_used: Vec<f64>,
    pub n_mon: usize,
    pub iterations: usize,
    pub converged_fully: bool,
    pub residual: f64,
    pub elapsed_ms: f64,
    pub options: WireUiOptions,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolveErrorWire {
    pub message: String,
    pub source: ErrSource,
}

impl SolveResultWire {
    /// Reconstruct the `ndarray`-backed [`SolveResult`] used by the UI.
    pub fn into_result(self) -> SolveResult {
        let SolveResultWire {
            concentrations,
            c0,
            stoich_flat,
            stoich_shape,
            dg_kcal_used,
            n_mon,
            iterations,
            converged_fully,
            residual,
            elapsed_ms,
            options,
        } = self;
        SolveResult {
            concentrations: ndarray::Array1::from_vec(concentrations),
            c0: ndarray::Array1::from_vec(c0),
            stoich: ndarray::Array2::from_shape_vec(stoich_shape, stoich_flat)
                .expect("stoich_flat length matches stoich_shape"),
            dg_kcal_used: ndarray::Array1::from_vec(dg_kcal_used),
            n_mon,
            iterations,
            converged_fully,
            residual,
            elapsed_ms,
            options: options.into(),
        }
    }
}

impl SolveResult {
    /// Project onto the wire form for `postMessage` transport.
    pub fn to_wire(&self) -> SolveResultWire {
        let stoich_shape = (self.stoich.nrows(), self.stoich.ncols());
        SolveResultWire {
            concentrations: self.concentrations.iter().copied().collect(),
            c0: self.c0.iter().copied().collect(),
            stoich_flat: self.stoich.iter().copied().collect(),
            stoich_shape,
            dg_kcal_used: self.dg_kcal_used.iter().copied().collect(),
            n_mon: self.n_mon,
            iterations: self.iterations,
            converged_fully: self.converged_fully,
            residual: self.residual,
            elapsed_ms: self.elapsed_ms,
            options: WireUiOptions::from(&self.options),
        }
    }
}

impl SolveErrorWire {
    pub fn into_error(self) -> SolveError {
        SolveError {
            message: self.message,
            source: self.source,
        }
    }
}

impl SolveError {
    pub fn to_wire(&self) -> SolveErrorWire {
        SolveErrorWire {
            message: self.message.clone(),
            source: self.source,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ui_options_wire_roundtrip_preserves_all_fields() {
        let mut original = UiOptions::coffee_compatible();
        original.solver.gradient_rel_tol = 1.234e-9;
        original.solver.objective = SolverObjective::Log;
        let wire = WireUiOptions::from(&original);
        let back: UiOptions = wire.into();
        assert_eq!(back, original);
    }

    #[test]
    fn solve_request_serializes_via_serde_json() {
        let req = SolveRequest {
            cfe: "1\t1\t1\t0\t0.0".into(),
            con: "1e-6".into(),
            options: WireUiOptions::from(&UiOptions::default()),
            progress_throttle_ms: 50.0,
        };
        let s = serde_json::to_string(&req).unwrap();
        let parsed: SolveRequest = serde_json::from_str(&s).unwrap();
        assert_eq!(parsed.cfe, req.cfe);
        assert_eq!(parsed.con, req.con);
        assert_eq!(parsed.progress_throttle_ms, req.progress_throttle_ms);
    }

    #[test]
    fn solve_result_wire_roundtrip_preserves_arrays() {
        use crate::solver::solve;
        let cfe = "1\t1\t1\t0\t0.0\n2\t1\t0\t1\t0.0\n3\t1\t1\t1\t-10.0\n";
        let con = "1.0e-6\n1.0e-6\n";
        let result = solve(cfe, con, &UiOptions::default()).unwrap();
        let wire = result.to_wire();
        let s = serde_json::to_string(&wire).unwrap();
        let parsed: SolveResultWire = serde_json::from_str(&s).unwrap();
        let recovered = parsed.into_result();
        assert_eq!(recovered.n_mon, result.n_mon);
        assert_eq!(recovered.iterations, result.iterations);
        assert_eq!(recovered.concentrations.len(), result.concentrations.len());
        assert_eq!(recovered.stoich.shape(), result.stoich.shape());
        // serde_json's f64 round-trip can lose one ulp; the actual
        // worker channel uses bincode which is bit-exact, but the
        // assertion here is weakened so the test stays portable.
        for (got, want) in recovered
            .concentrations
            .iter()
            .zip(result.concentrations.iter())
        {
            assert!(
                (got - want).abs() <= 1e-12 * want.abs().max(1e-30),
                "concentration mismatch: got={got:e} want={want:e}"
            );
        }
    }
}
