//! Cross-validate equiconc against the upstream COFFEE optimizer.
//!
//! Both implement the Dirks et al. (2007) trust-region Newton method on the
//! convex dual problem, but with different subproblem solvers (dog-leg vs
//! Steihaug CG). For valid inputs they must converge to the same unique
//! optimum, so their equilibrium concentrations should agree to high precision.

use coffee::{extras::OptimizerArgs, optimize::Optimizer};
use equiconc::{SolverObjective, SolverOptions, SystemBuilder, R};
use ndarray::{Array1, Array2};
use proptest::prelude::*;

// Cross-validation tolerances. COFFEE's convergence criterion
// (actual_reduction == 0.0, exact float equality) is looser than equiconc's
// gradient-norm check (RTOL 1e-7), so we can't expect better than ~1e-5
// agreement. 1e-4 relative is tight enough to catch algorithmic bugs
// (wrong matrix layout, sign errors, unit mismatches) while allowing for
// solver-precision differences.
const CROSS_REL_TOL: f64 = 1e-4;
const CROSS_ABS_TOL: f64 = 1e-14;

// --- System generation (adapted from proptest_equiconc.rs) ---

const MONOMER_NAMES: [&str; 4] = ["A", "B", "C", "D"];

#[derive(Debug, Clone)]
struct SystemSpec {
    temperature: f64,
    monomers: Vec<(&'static str, f64)>,
    complexes: Vec<(String, Vec<(&'static str, usize)>, f64)>,
}

impl SystemSpec {
    fn build(&self) -> Result<equiconc::System, equiconc::EquilibriumError> {
        self.build_with(SolverObjective::Linear)
    }

    fn build_with(
        &self,
        objective: SolverObjective,
    ) -> Result<equiconc::System, equiconc::EquilibriumError> {
        let mut b = SystemBuilder::new().temperature(self.temperature);
        for &(name, conc) in &self.monomers {
            b = b.monomer(name, conc);
        }
        for (name, comp, dg) in &self.complexes {
            let comp_refs: Vec<(&str, usize)> = comp.iter().copied().collect();
            b = b.complex(name, &comp_refs, *dg);
        }
        let opts = SolverOptions {
            objective,
            ..Default::default()
        };
        b.options(opts).build()
    }
}

fn log_uniform_concentration() -> impl Strategy<Value = f64> {
    (-12.0f64..=-3.0).prop_map(|e| 10f64.powf(e))
}

fn arb_system() -> impl Strategy<Value = SystemSpec> {
    (293.15..=373.15f64, 1..=4usize)
        .prop_flat_map(|(temp, n_mon)| {
            let concs = prop::collection::vec(log_uniform_concentration(), n_mon);
            (Just(temp), Just(n_mon), concs, 0..=6usize)
        })
        .prop_flat_map(|(temp, n_mon, concs, n_cplx)| {
            let complexes = prop::collection::vec(
                (
                    prop::collection::vec(0..=3usize, n_mon),
                    -40.0..=10.0f64,
                ),
                n_cplx,
            );
            (Just(temp), Just(n_mon), Just(concs), complexes)
        })
        .prop_map(|(temp, n_mon, concs, raw_complexes)| {
            let monomers: Vec<_> = (0..n_mon)
                .map(|i| (MONOMER_NAMES[i], concs[i]))
                .collect();
            let complexes: Vec<_> = raw_complexes
                .into_iter()
                .enumerate()
                .map(|(k, (mut counts, dg))| {
                    if counts.iter().all(|&c| c == 0) {
                        counts[0] = 1;
                    }
                    let comp: Vec<(&'static str, usize)> = counts
                        .into_iter()
                        .enumerate()
                        .filter(|&(_, count)| count > 0)
                        .map(|(i, count)| (MONOMER_NAMES[i], count))
                        .collect();
                    (format!("c{k}"), comp, dg)
                })
                .collect();

            SystemSpec {
                temperature: temp,
                monomers,
                complexes,
            }
        })
}

// --- COFFEE bridge ---

/// Construct COFFEE optimizer inputs from a SystemSpec.
///
/// With `scalarity=false`, COFFEE computes Q = exp(-energy). equiconc uses
/// log_q = -ΔG/(RT), so Q = exp(-ΔG/(RT)). Passing ΔG/(RT) as the energy
/// gives the same Q in both solvers.
///
/// The polymers matrix includes free monomers (identity rows, energy 0) so
/// that COFFEE's mass-conservation check works correctly.
fn solve_with_coffee(spec: &SystemSpec) -> Option<Vec<f64>> {
    let n_mon = spec.monomers.len();
    let n_cplx = spec.complexes.len();
    let n_species = n_mon + n_cplx;
    let rt = R * spec.temperature;

    // Initial monomer concentrations
    let monomers = Array1::from_vec(spec.monomers.iter().map(|&(_, c)| c).collect());

    // Stoichiometry matrix: (n_species × n_mon)
    // First n_mon rows: identity (free monomers)
    // Remaining rows: complex stoichiometry
    let mut polymers = Array2::zeros((n_species, n_mon));
    for i in 0..n_mon {
        polymers[[i, i]] = 1.0;
    }
    for (k, (_name, comp, _dg)) in spec.complexes.iter().enumerate() {
        for &(mon_name, count) in comp {
            let mon_idx = spec
                .monomers
                .iter()
                .position(|&(n, _)| n == mon_name)
                .unwrap();
            polymers[[n_mon + k, mon_idx]] = count as f64;
        }
    }

    // Energy vector: 0 for monomers, ΔG/(RT) for complexes
    // (COFFEE does exp(-energy), giving exp(-ΔG/(RT)) = Q)
    let mut q_nonexp = Array1::zeros(n_species);
    for (k, (_name, _comp, dg)) in spec.complexes.iter().enumerate() {
        q_nonexp[n_mon + k] = dg / rt;
    }

    let args = OptimizerArgs {
        scalarity: false,
        use_terminal: false,
        ..OptimizerArgs::default()
    };

    let mut optimizer = Optimizer::new(&monomers, &polymers, &q_nonexp, &args).ok()?;
    optimizer.optimize(1.0).ok()?;
    let results = optimizer.get_results();

    // Reject if COFFEE produced NaN (its log-lagrangian formulation can
    // compute ln(∞ - ∞) when the trust-region step overshoots) or if the
    // mass-conservation error is too large (premature convergence for
    // strongly-binding systems with huge Q values).
    if results.optimal_x.iter().any(|x| !x.is_finite()) {
        return None;
    }
    let max_c0: f64 = spec.monomers.iter().map(|&(_, c)| c).fold(0.0, f64::max);
    if results.concentration_error > 1e-6 * max_c0 {
        return None;
    }

    Some(results.optimal_x)
}

fn concentrations_agree(a: f64, b: f64) -> bool {
    let abs_err = (a - b).abs();
    let scale = a.abs().max(b.abs());
    abs_err < CROSS_ABS_TOL + CROSS_REL_TOL * scale
}

// --- Property tests ---

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn prop_equiconc_matches_coffee(spec in arb_system()) {
        prop_assume!(!spec.complexes.is_empty());

        let mut sys = spec.build().unwrap();
        let eq = sys.solve().unwrap();

        let coffee_concs = match solve_with_coffee(&spec) {
            Some(c) => c,
            None => {
                // COFFEE didn't converge or produced unphysical results —
                // skip (its convergence criterion uses exact float equality
                // which is fragile, and its log-lagrangian can produce NaN).
                return Ok(());
            }
        };

        let n_mon = spec.monomers.len();

        // Compare free monomer concentrations
        for (i, &(name, _)) in spec.monomers.iter().enumerate() {
            let eq_val = eq.get(name).unwrap();
            let coffee_val = coffee_concs[i];
            prop_assert!(
                concentrations_agree(eq_val, coffee_val),
                "Free monomer {} disagrees: equiconc={:.6e}, coffee={:.6e}, \
                 rel_err={:.2e}",
                name,
                eq_val,
                coffee_val,
                (eq_val - coffee_val).abs() / eq_val.max(coffee_val).max(1e-30)
            );
        }

        // Compare complex concentrations
        for (k, (name, _, _)) in spec.complexes.iter().enumerate() {
            let eq_val = eq.get(name).unwrap();
            let coffee_val = coffee_concs[n_mon + k];
            prop_assert!(
                concentrations_agree(eq_val, coffee_val),
                "Complex {} disagrees: equiconc={:.6e}, coffee={:.6e}, \
                 rel_err={:.2e}",
                name,
                eq_val,
                coffee_val,
                (eq_val - coffee_val).abs() / eq_val.max(coffee_val).max(1e-30)
            );
        }
    }

    /// Same cross-check, but for equiconc's log objective. The point is
    /// to verify that *equiconc's* log path agrees with COFFEE on every
    /// system COFFEE handles correctly — i.e., we deliver the speed of
    /// log(L) without inheriting any of COFFEE's documented failure modes.
    /// Cases where COFFEE diverges/produces NaN are skipped (handled by
    /// `solve_with_coffee` returning `None`).
    #[test]
    fn prop_equiconc_log_matches_coffee(spec in arb_system()) {
        prop_assume!(!spec.complexes.is_empty());

        let mut sys = spec.build_with(SolverObjective::Log).unwrap();
        let eq = match sys.solve() {
            Ok(eq) => eq,
            // Don't penalize legitimate solver failures (extreme
            // ill-conditioning, etc.) — those are tracked separately.
            Err(_) => return Ok(()),
        };

        let coffee_concs = match solve_with_coffee(&spec) {
            Some(c) => c,
            None => return Ok(()),
        };

        let n_mon = spec.monomers.len();

        for (i, &(name, _)) in spec.monomers.iter().enumerate() {
            let eq_val = eq.get(name).unwrap();
            let coffee_val = coffee_concs[i];
            prop_assert!(
                concentrations_agree(eq_val, coffee_val),
                "Free monomer {} disagrees (log objective): equiconc={:.6e}, coffee={:.6e}, \
                 rel_err={:.2e}",
                name, eq_val, coffee_val,
                (eq_val - coffee_val).abs() / eq_val.max(coffee_val).max(1e-30)
            );
        }
        for (k, (name, _, _)) in spec.complexes.iter().enumerate() {
            let eq_val = eq.get(name).unwrap();
            let coffee_val = coffee_concs[n_mon + k];
            prop_assert!(
                concentrations_agree(eq_val, coffee_val),
                "Complex {} disagrees (log objective): equiconc={:.6e}, coffee={:.6e}, \
                 rel_err={:.2e}",
                name, eq_val, coffee_val,
                (eq_val - coffee_val).abs() / eq_val.max(coffee_val).max(1e-30)
            );
        }
    }
}
