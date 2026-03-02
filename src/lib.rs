//! Equilibrium concentration solver for nucleic acid strand systems.
//!
//! Computes equilibrium concentrations for systems of interacting nucleic acid
//! strands that form multi-strand complexes. Implements the trust-region Newton
//! method on the convex dual problem from
//! [Dirks et al. (2007), *SIAM Review* 49(1), 65–88](https://doi.org/10.1137/060651100).
//!
//! # Conventions
//!
//! - **Free energies** are standard free energies of formation (ΔG°) in
//!   **kcal/mol** at a **1 M standard state**.
//! - **Concentrations** are in **molar** (mol/L).
//! - **Temperature** is in **kelvin**.
//! - The gas constant [`R`] is in kcal/(mol·K).
//!
//! ## Symmetry
//!
//! Homodimer and higher homo-oligomer symmetry corrections are **not**
//! applied automatically. If your complex contains identical strands,
//! include the symmetry correction in the ΔG° value you provide
//! (e.g., add +RT·ln(σ) where σ is the symmetry number).
//!
//! # Example
//!
//! ```
//! use equiconc::System;
//!
//! // A + B ⇌ AB with ΔG° = -10 kcal/mol at 37 °C
//! let eq = System::new()
//!     .monomer("A", 100e-9)      // 100 nM
//!     .monomer("B", 100e-9)
//!     .complex("AB", &[("A", 1), ("B", 1)], -10.0)
//!     .equilibrium()?;
//!
//! let free_a = eq.concentration("A").unwrap();
//! let free_b = eq.concentration("B").unwrap();
//! let ab = eq.concentration("AB").unwrap();
//!
//! // Mass conservation: [A] + [AB] = 100 nM
//! assert!((free_a + ab - 100e-9).abs() < 1e-6 * 100e-9);
//! # Ok::<(), equiconc::EquilibriumError>(())
//! ```

use nalgebra::{DMatrix, DVector};

/// Gas constant in kcal/(mol·K).
pub const R: f64 = 1.987204e-3;

/// Maximum log-concentration before clamping to prevent f64 overflow.
///
/// `exp(709.78)` ≈ `f64::MAX`; we clamp below that to stay safely in range.
const LOG_C_MAX: f64 = 700.0;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug)]
#[non_exhaustive]
pub enum EquilibriumError {
    NoMonomers,
    UnknownMonomer(String),
    EmptyComposition,
    InvalidConcentration(f64),
    InvalidTemperature(f64),
    DuplicateMonomer(String),
    DuplicateComplex(String),
    ZeroCount(String),
    DuplicateMonomerInComposition(String),
    InvalidDeltaG(f64),
    EmptyName,
    DuplicateSpeciesName(String),
    ConvergenceFailure {
        iterations: usize,
        gradient_norm: f64,
    },
}

impl std::fmt::Display for EquilibriumError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoMonomers => write!(f, "system has no monomers"),
            Self::UnknownMonomer(name) => write!(f, "unknown monomer: {name}"),
            Self::EmptyComposition => write!(f, "complex has empty composition"),
            Self::InvalidConcentration(c) => {
                write!(f, "invalid concentration: {c} (must be finite and positive)")
            }
            Self::InvalidTemperature(t) => {
                write!(f, "invalid temperature: {t} (must be finite and positive)")
            }
            Self::DuplicateMonomer(name) => write!(f, "duplicate monomer: {name}"),
            Self::DuplicateComplex(name) => write!(f, "duplicate complex: {name}"),
            Self::ZeroCount(name) => {
                write!(f, "zero stoichiometric count for monomer: {name}")
            }
            Self::DuplicateMonomerInComposition(name) => {
                write!(f, "duplicate monomer in composition: {name}")
            }
            Self::InvalidDeltaG(dg) => {
                write!(f, "invalid delta_g: {dg} (must be finite)")
            }
            Self::EmptyName => write!(f, "species name must not be empty"),
            Self::DuplicateSpeciesName(name) => write!(
                f,
                "species name already in use: {name} (monomer and complex names must be unique)"
            ),
            Self::ConvergenceFailure {
                iterations,
                gradient_norm,
            } => write!(
                f,
                "did not converge after {iterations} iterations (‖∇‖ = {gradient_norm:.2e})"
            ),
        }
    }
}

impl std::error::Error for EquilibriumError {}

// ---------------------------------------------------------------------------
// Public builder / system
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct System {
    monomers: Vec<(String, f64)>,
    complexes: Vec<(String, Vec<(String, usize)>, f64)>,
    temperature: f64, // Kelvin
}

impl Default for System {
    fn default() -> Self {
        Self::new()
    }
}

impl System {
    /// Create an empty system at 37 °C (310.15 K).
    #[must_use]
    pub fn new() -> Self {
        System {
            monomers: Vec::new(),
            complexes: Vec::new(),
            temperature: 310.15,
        }
    }

    /// Set the temperature in Kelvin.
    #[must_use]
    pub fn temperature(mut self, t: f64) -> Self {
        self.temperature = t;
        self
    }

    /// Add a monomer species with a given total concentration (molar).
    #[must_use]
    pub fn monomer(mut self, name: &str, total_concentration: f64) -> Self {
        self.monomers
            .push((name.to_string(), total_concentration));
        self
    }

    /// Add a complex with the given composition and standard free energy of
    /// formation.
    ///
    /// # Arguments
    ///
    /// * `name` — unique identifier for this complex (must not collide with
    ///   monomer names)
    /// * `composition` — slice of `(monomer_name, count)` pairs specifying
    ///   how many of each monomer strand appear in the complex
    /// * `delta_g` — ΔG° in **kcal/mol** at a **1 M standard state**. This is
    ///   the standard free energy of complex formation from its constituent
    ///   monomers. Values from NUPACK or other tools that use a water-molarity
    ///   standard state need a correction of `+(n-1)·RT·ln(c_water)` where
    ///   `n` is the number of strands.
    #[must_use]
    pub fn complex(
        mut self,
        name: &str,
        composition: &[(&str, usize)],
        delta_g: f64,
    ) -> Self {
        self.complexes.push((
            name.to_string(),
            composition
                .iter()
                .map(|&(n, c)| (n.to_string(), c))
                .collect(),
            delta_g,
        ));
        self
    }

    /// Current temperature in Kelvin.
    #[must_use]
    pub fn get_temperature(&self) -> f64 {
        self.temperature
    }

    /// Number of monomers added so far.
    #[must_use]
    pub fn monomer_count(&self) -> usize {
        self.monomers.len()
    }

    /// Number of complexes added so far.
    #[must_use]
    pub fn complex_count(&self) -> usize {
        self.complexes.len()
    }

    /// Validate inputs, resolve monomer references, and build the internal
    /// problem representation.
    fn validate_and_build(
        &self,
    ) -> Result<
        (
            Vec<String>,                // monomer names
            Vec<f64>,                   // monomer concentrations
            Vec<String>,                // complex names
            Vec<Vec<(usize, usize)>>,   // resolved compositions
            Vec<f64>,                   // complex delta_g values
        ),
        EquilibriumError,
    > {
        // Temperature validation
        if !(self.temperature > 0.0 && self.temperature.is_finite()) {
            return Err(EquilibriumError::InvalidTemperature(self.temperature));
        }

        // Must have at least one monomer
        if self.monomers.is_empty() {
            return Err(EquilibriumError::NoMonomers);
        }

        // Validate monomers
        let mut monomer_names: Vec<String> = Vec::new();
        let mut monomer_concs: Vec<f64> = Vec::new();
        for (name, conc) in &self.monomers {
            if name.is_empty() {
                return Err(EquilibriumError::EmptyName);
            }
            if !(conc > &0.0 && conc.is_finite()) {
                return Err(EquilibriumError::InvalidConcentration(*conc));
            }
            if monomer_names.iter().any(|n| n == name) {
                return Err(EquilibriumError::DuplicateMonomer(name.clone()));
            }
            monomer_names.push(name.clone());
            monomer_concs.push(*conc);
        }

        // Validate complexes and resolve monomer references
        let mut complex_names: Vec<String> = Vec::new();
        let mut resolved_comps: Vec<Vec<(usize, usize)>> = Vec::new();
        let mut complex_dgs: Vec<f64> = Vec::new();

        for (name, composition, delta_g) in &self.complexes {
            if name.is_empty() {
                return Err(EquilibriumError::EmptyName);
            }
            if composition.is_empty() {
                return Err(EquilibriumError::EmptyComposition);
            }
            if !delta_g.is_finite() {
                return Err(EquilibriumError::InvalidDeltaG(*delta_g));
            }
            if complex_names.iter().any(|n| n == name) {
                return Err(EquilibriumError::DuplicateComplex(name.clone()));
            }
            if monomer_names.iter().any(|n| n == name) {
                return Err(EquilibriumError::DuplicateSpeciesName(name.clone()));
            }

            let mut comp = Vec::new();
            for (monomer_name, count) in composition {
                if *count == 0 {
                    return Err(EquilibriumError::ZeroCount(monomer_name.clone()));
                }
                let idx = monomer_names
                    .iter()
                    .position(|n| n == monomer_name)
                    .ok_or_else(|| {
                        EquilibriumError::UnknownMonomer(monomer_name.clone())
                    })?;
                if comp.iter().any(|&(existing_idx, _): &(usize, usize)| existing_idx == idx) {
                    return Err(EquilibriumError::DuplicateMonomerInComposition(
                        monomer_name.clone(),
                    ));
                }
                comp.push((idx, *count));
            }

            complex_names.push(name.clone());
            resolved_comps.push(comp);
            complex_dgs.push(*delta_g);
        }

        Ok((monomer_names, monomer_concs, complex_names, resolved_comps, complex_dgs))
    }

    /// Build the stoichiometry matrix **A**, the log-partition-function vector
    /// **log_q**, and the total-concentration vector **c⁰**.
    fn build_problem(
        n_mon: usize,
        monomer_concs: &[f64],
        resolved_comps: &[Vec<(usize, usize)>],
        complex_dgs: &[f64],
        temperature: f64,
    ) -> (DMatrix<f64>, DVector<f64>, DVector<f64>) {
        let n_cplx = resolved_comps.len();
        let n_species = n_mon + n_cplx;

        // A: n_monomers × n_species
        let mut a = DMatrix::zeros(n_mon, n_species);
        // Identity block for free monomers
        for i in 0..n_mon {
            a[(i, i)] = 1.0;
        }
        // Complex columns
        for (k, comp) in resolved_comps.iter().enumerate() {
            for &(mi, count) in comp {
                a[(mi, n_mon + k)] = count as f64;
            }
        }

        // log_q: 0 for free monomers, -ΔG°/(RT) for complexes
        let mut log_q = DVector::zeros(n_species);
        let rt = R * temperature;
        for (k, &dg) in complex_dgs.iter().enumerate() {
            log_q[n_mon + k] = -dg / rt;
        }

        // c⁰
        let c0 = DVector::from_iterator(n_mon, monomer_concs.iter().copied());

        (a, log_q, c0)
    }

    /// Solve for equilibrium concentrations.
    ///
    /// Validates all inputs and returns an error if any are invalid.
    pub fn equilibrium(&self) -> Result<Equilibrium, EquilibriumError> {
        let (monomer_names, monomer_concs, complex_names, resolved_comps, complex_dgs) =
            self.validate_and_build()?;

        let n_mon = monomer_names.len();

        // Short-circuit: no complexes
        if complex_names.is_empty() {
            return Ok(Equilibrium {
                monomer_names,
                complex_names,
                free_monomer_concentrations: monomer_concs,
                complex_concentrations: Vec::new(),
                converged_fully: true,
            });
        }

        let (a, log_q, c0) =
            Self::build_problem(n_mon, &monomer_concs, &resolved_comps, &complex_dgs, self.temperature);
        let solution = solve_dual(&a, &log_q, &c0)?;

        // Recover concentrations: c_j = exp(log_q_j + (Aᵀλ)_j)
        let log_c = &log_q + a.transpose() * &solution.lambda;
        let concentrations: Vec<f64> =
            log_c.iter().map(|&lc| lc.min(LOG_C_MAX).exp()).collect();

        // Debug-mode validation
        #[cfg(debug_assertions)]
        {
            let rt = R * self.temperature;
            // Mass conservation
            for i in 0..n_mon {
                let total: f64 =
                    (0..concentrations.len()).map(|j| a[(i, j)] * concentrations[j]).sum();
                debug_assert!(
                    (total - c0[i]).abs() < 1e-3 * c0[i] + 1e-30,
                    "mass conservation violated for monomer {}: {total} != {}",
                    monomer_names[i],
                    c0[i]
                );
            }
            // Equilibrium condition (in log-space to avoid overflow)
            for (k, comp) in resolved_comps.iter().enumerate() {
                let log_keq = -complex_dgs[k] / rt;
                let mut log_expected = log_keq;
                for &(mi, count) in comp {
                    log_expected += count as f64 * concentrations[mi].ln();
                }
                let log_actual = concentrations[n_mon + k].ln();
                let rel_err = (log_actual - log_expected).abs()
                    / (log_expected.abs() + 1e-30);
                debug_assert!(
                    rel_err < 1e-3,
                    "equilibrium condition violated for {}: log(actual)={log_actual} != log(expected)={log_expected} (rel err: {rel_err})",
                    complex_names[k]
                );
            }
        }

        let converged_fully = matches!(
            solution.convergence,
            SolverConvergence::Full { .. }
        );

        Ok(Equilibrium {
            monomer_names,
            complex_names,
            free_monomer_concentrations: concentrations[..n_mon].to_vec(),
            complex_concentrations: concentrations[n_mon..].to_vec(),
            converged_fully,
        })
    }
}

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Equilibrium {
    monomer_names: Vec<String>,
    complex_names: Vec<String>,
    free_monomer_concentrations: Vec<f64>,
    complex_concentrations: Vec<f64>,
    converged_fully: bool,
}

impl Equilibrium {
    /// Look up a concentration by species name (monomer or complex).
    #[must_use]
    pub fn concentration(&self, name: &str) -> Option<f64> {
        if let Some(i) = self.monomer_names.iter().position(|n| n == name) {
            return Some(self.free_monomer_concentrations[i]);
        }
        if let Some(i) = self.complex_names.iter().position(|n| n == name) {
            return Some(self.complex_concentrations[i]);
        }
        None
    }

    /// Monomer names in the order they were added.
    #[must_use]
    pub fn monomer_names(&self) -> &[String] {
        &self.monomer_names
    }

    /// Complex names in the order they were added.
    #[must_use]
    pub fn complex_names(&self) -> &[String] {
        &self.complex_names
    }

    /// Free monomer concentrations (same order as [`monomer_names`](Self::monomer_names)).
    #[must_use]
    pub fn free_monomer_concentrations(&self) -> &[f64] {
        &self.free_monomer_concentrations
    }

    /// Complex concentrations (same order as [`complex_names`](Self::complex_names)).
    #[must_use]
    pub fn complex_concentrations(&self) -> &[f64] {
        &self.complex_concentrations
    }

    /// Whether the solver achieved full convergence (relative tolerance 1e-7).
    ///
    /// Returns `false` if the solver accepted the result at a relaxed tolerance
    /// (1e-4) due to stagnation at f64 precision limits. In practice, results
    /// with relaxed convergence are still accurate for most purposes, but
    /// callers doing high-precision work should check this flag.
    #[must_use]
    pub fn converged_fully(&self) -> bool {
        self.converged_fully
    }
}

// ---------------------------------------------------------------------------
// Trust-region Newton solver on the dual
// ---------------------------------------------------------------------------

struct DualEval {
    f: f64,
    grad: DVector<f64>,
    hessian: DMatrix<f64>,
}

/// Evaluate the dual objective, gradient, and Hessian at λ.
fn evaluate(
    a: &DMatrix<f64>,
    log_q: &DVector<f64>,
    c0: &DVector<f64>,
    lambda: &DVector<f64>,
) -> DualEval {
    let n_mon = a.nrows();
    let n_species = a.ncols();

    // log_c = log_q + Aᵀλ
    let log_c = log_q + a.transpose() * lambda;

    // c_j = exp(log_c_j), clamped to avoid overflow
    let c = DVector::from_iterator(
        n_species,
        log_c.iter().map(|&lc| lc.min(LOG_C_MAX).exp()),
    );

    // f = -λᵀc⁰ + Σ_j c_j
    let f = -lambda.dot(c0) + c.sum();

    // grad = -c⁰ + A c
    let grad = -c0 + a * &c;

    // H = A diag(c) Aᵀ  (symmetric, computed directly)
    let mut hessian = DMatrix::zeros(n_mon, n_mon);
    for k in 0..n_species {
        let ck = c[k];
        for i in 0..n_mon {
            let aik = a[(i, k)];
            if aik == 0.0 {
                continue;
            }
            for j in i..n_mon {
                let ajk = a[(j, k)];
                if ajk == 0.0 {
                    continue;
                }
                let val = aik * ck * ajk;
                hessian[(i, j)] += val;
                if i != j {
                    hessian[(j, i)] += val;
                }
            }
        }
    }

    DualEval { f, grad, hessian }
}

/// Compute the dog-leg step within a trust region of radius `delta`.
fn dogleg_step(grad: &DVector<f64>, hessian: &DMatrix<f64>, delta: f64) -> DVector<f64> {
    // Cholesky decomposition
    let chol = match nalgebra::linalg::Cholesky::new(hessian.clone()) {
        Some(c) => c,
        None => {
            // Defensive: H = A diag(c) Aᵀ with all c > 0, so H is always
            // positive-definite and Cholesky cannot fail in practice.
            let g_norm = grad.norm();
            if g_norm == 0.0 {
                return DVector::zeros(grad.len());
            }
            return -(delta / g_norm) * grad;
        }
    };

    // Newton step: p_n = -H⁻¹g
    let p_n = -chol.solve(grad);
    let p_n_norm = p_n.norm();

    if p_n_norm <= delta {
        return p_n;
    }

    // Cauchy step: p_c = -(gᵀg / gᵀHg) g
    let gtg = grad.dot(grad);
    let hg = hessian * grad;
    let gthg = grad.dot(&hg);

    // Guard against near-zero curvature: fall back to steepest descent
    // clipped to trust region, same as the Cholesky failure path.
    if gthg < 1e-30 * gtg {
        let g_norm = grad.norm();
        return -(delta / g_norm) * grad;
    }

    let p_c = -(gtg / gthg) * grad;
    let p_c_norm = p_c.norm();

    if p_c_norm >= delta {
        return (delta / p_c_norm) * &p_c;
    }

    // Dog-leg interpolation: find τ ∈ [0,1] s.t. ‖p_c + τ(p_n − p_c)‖ = δ
    let d = &p_n - &p_c;
    let dd = d.dot(&d);
    let pd = p_c.dot(&d);
    let pp = p_c.dot(&p_c);
    // Solve dd·τ² + 2·pd·τ + (pp − δ²) = 0
    let discriminant = (pd * pd - dd * (pp - delta * delta)).max(0.0);
    let tau = ((-pd + discriminant.sqrt()) / dd).clamp(0.0, 1.0);

    &p_c + tau * &d
}

/// Convergence quality returned by the dual solver.
#[derive(Debug, Clone, Copy, PartialEq)]
enum SolverConvergence {
    /// Converged to full tolerance (RTOL = 1e-7).
    Full { iterations: usize },
    /// Accepted at relaxed tolerance (RTOL = 1e-4) due to stagnation.
    Relaxed { iterations: usize, gradient_norm: f64 },
}

struct DualSolution {
    lambda: DVector<f64>,
    convergence: SolverConvergence,
}

/// Solve the dual problem, returning the optimal λ* and convergence info.
fn solve_dual(
    a: &DMatrix<f64>,
    log_q: &DVector<f64>,
    c0: &DVector<f64>,
) -> Result<DualSolution, EquilibriumError> {
    const MAX_ITER: usize = 1000;
    const ATOL: f64 = 1e-14;
    const RTOL: f64 = 1e-7;
    const ETA: f64 = 1e-4;
    const DELTA_MAX: f64 = 1e10;

    // Initial guess: λ_i = ln(c⁰_i)
    let mut lambda = DVector::from_iterator(c0.len(), c0.iter().map(|&c| c.ln()));
    let mut delta = 1.0;
    let mut stagnation = 0u32;

    for iter in 0..MAX_ITER {
        let eval = evaluate(a, log_q, c0, &lambda);

        // Per-component convergence: each gradient entry (= mass balance
        // error for that monomer) must be small relative to its c⁰.
        if eval
            .grad
            .iter()
            .zip(c0.iter())
            .all(|(&g, &c)| g.abs() < ATOL + RTOL * c)
        {
            return Ok(DualSolution {
                lambda,
                convergence: SolverConvergence::Full { iterations: iter + 1 },
            });
        }

        let p = dogleg_step(&eval.grad, &eval.hessian, delta);
        let p_norm = p.norm();

        // Evaluate at candidate point
        let lambda_new = &lambda + &p;
        let eval_new = evaluate(a, log_q, c0, &lambda_new);

        let actual_reduction = eval.f - eval_new.f;
        let predicted_reduction = -(eval.grad.dot(&p) + 0.5 * p.dot(&(&eval.hessian * &p)));

        // Track stagnation: objective can't decrease further in f64.
        if actual_reduction == 0.0 {
            stagnation += 1;
        } else {
            stagnation = 0;
        }

        // When the trust region has collapsed and steps produce no
        // measurable reduction, try a full Newton step. The objective
        // is at f64 precision, but lambda can still improve, yielding
        // better gradient (mass-balance) accuracy.
        if stagnation >= 3 {
            let p_full = dogleg_step(&eval.grad, &eval.hessian, DELTA_MAX);
            let lambda_full = &lambda + &p_full;
            let eval_full = evaluate(a, log_q, c0, &lambda_full);
            if eval_full.f <= eval.f {
                lambda = lambda_full;
                delta = p_full.norm();
                stagnation = 0;
                continue;
            }
            // Recovery failed — accept if close enough with relaxed tolerance
            // (the solver is at f64 precision limits), otherwise report failure.
            const STAG_RTOL: f64 = 1e-4;
            if eval
                .grad
                .iter()
                .zip(c0.iter())
                .all(|(&g, &c)| g.abs() < ATOL + STAG_RTOL * c)
            {
                return Ok(DualSolution {
                    lambda,
                    convergence: SolverConvergence::Relaxed {
                        iterations: iter + 1,
                        gradient_norm: eval.grad.norm(),
                    },
                });
            }
            return Err(EquilibriumError::ConvergenceFailure {
                iterations: iter + 1,
                gradient_norm: eval.grad.norm(),
            });
        }

        let rho = if predicted_reduction.abs() < 1e-30 {
            if actual_reduction >= 0.0 {
                1.0
            } else {
                0.0
            }
        } else {
            actual_reduction / predicted_reduction
        };

        // Update trust radius
        if rho < 0.25 {
            delta *= 0.25;
        } else if rho > 0.75 && (p_norm - delta).abs() < 1e-10 * delta {
            delta = (2.0 * delta).min(DELTA_MAX);
        }

        // Accept or reject step
        if rho > ETA {
            lambda = lambda_new;
        }
    }

    // Defensive: the convex dual always converges for valid inputs,
    // so this path is unreachable in practice.
    let final_eval = evaluate(a, log_q, c0, &lambda);
    Err(EquilibriumError::ConvergenceFailure {
        iterations: MAX_ITER,
        gradient_norm: final_eval.grad.norm(),
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const NM: f64 = 1e-9;

    #[test]
    fn no_complexes() {
        let sys = System::new()
            .monomer("A", 50e-9)
            .monomer("B", 100e-9);
        let eq = sys.equilibrium().unwrap();
        assert_eq!(eq.free_monomer_concentrations()[0], 50e-9);
        assert_eq!(eq.free_monomer_concentrations()[1], 100e-9);
        assert!(eq.complex_concentrations().is_empty());
    }

    #[test]
    fn simple_dimerization() {
        let c0 = 100.0 * NM;
        let dg = -10.0;
        let sys = System::new()
            .monomer("A", c0)
            .monomer("B", c0)
            .complex("AB", &[("A", 1), ("B", 1)], dg);
        let eq = sys.equilibrium().unwrap();

        // Analytical solution
        let rt = R * 310.15;
        let k = (-dg / rt).exp();
        // K(c₀ − x)² = x  ⟹  Kx² − (2Kc₀+1)x + Kc₀² = 0
        let x = ((2.0 * k * c0 + 1.0) - (4.0 * k * c0 + 1.0).sqrt()) / (2.0 * k);
        let free = c0 - x;

        assert!((eq.concentration("A").unwrap() - free).abs() < 1e-12);
        assert!((eq.concentration("B").unwrap() - free).abs() < 1e-12);
        assert!((eq.concentration("AB").unwrap() - x).abs() < 1e-12);
    }

    #[test]
    fn homotrimerization() {
        let c0 = 1e-6;
        let sys = System::new()
            .monomer("A", c0)
            .complex("AAA", &[("A", 3)], -15.0);
        let eq = sys.equilibrium().unwrap();
        let free_a = eq.concentration("A").unwrap();
        let aaa = eq.concentration("AAA").unwrap();
        assert!(
            (free_a + 3.0 * aaa - c0).abs() < 1e-8 * c0,
            "mass conservation: {} + 3·{} = {} (expected {c0})",
            free_a,
            aaa,
            free_a + 3.0 * aaa
        );
    }

    #[test]
    fn competing_complexes() {
        let c0 = 100.0 * NM;
        let sys = System::new()
            .monomer("a", c0)
            .monomer("b", c0)
            .monomer("c", c0)
            .complex("ab", &[("a", 1), ("b", 1)], -10.0)
            .complex("aaa", &[("a", 3)], -15.0)
            .complex("bc", &[("b", 1), ("c", 1)], -12.0);
        let eq = sys.equilibrium().unwrap();

        let a = eq.concentration("a").unwrap();
        let b = eq.concentration("b").unwrap();
        let c = eq.concentration("c").unwrap();
        let ab = eq.concentration("ab").unwrap();
        let aaa = eq.concentration("aaa").unwrap();
        let bc = eq.concentration("bc").unwrap();

        // Mass conservation for a: [a] + [ab] + 3[aaa] = c₀
        assert!((a + ab + 3.0 * aaa - c0).abs() < 1e-5 * c0);
        // Mass conservation for b: [b] + [ab] + [bc] = c₀
        assert!((b + ab + bc - c0).abs() < 1e-5 * c0);
        // Mass conservation for c: [c] + [bc] = c₀
        assert!((c + bc - c0).abs() < 1e-5 * c0);
    }

    #[test]
    fn strong_binding() {
        let c0 = 100.0 * NM;
        let sys = System::new()
            .monomer("A", c0)
            .monomer("B", c0)
            .complex("AB", &[("A", 1), ("B", 1)], -30.0);
        let eq = sys.equilibrium().unwrap();

        let ab = eq.concentration("AB").unwrap();
        assert!(
            (ab - c0).abs() < 1e-6 * c0,
            "[AB] = {ab}, expected ≈ {c0}"
        );
        assert!(eq.concentration("A").unwrap() < 1e-12);
        assert!(eq.concentration("B").unwrap() < 1e-12);
    }

    #[test]
    fn asymmetric_concentrations() {
        let sys = System::new()
            .monomer("A", 200.0 * NM)
            .monomer("B", 100.0 * NM)
            .complex("AB", &[("A", 1), ("B", 1)], -10.0);
        let eq = sys.equilibrium().unwrap();

        let a = eq.concentration("A").unwrap();
        let b = eq.concentration("B").unwrap();
        let ab = eq.concentration("AB").unwrap();

        assert!(ab <= 100.0 * NM + 1e-12);
        // Mass conservation
        assert!((a + ab - 200.0 * NM).abs() < 1e-8 * 200.0 * NM);
        assert!((b + ab - 100.0 * NM).abs() < 1e-8 * 100.0 * NM);
    }
    #[test]
    fn negative_concentration() {
        let err = System::new().monomer("A", -1e-9).equilibrium().unwrap_err();
        assert!(matches!(err, EquilibriumError::InvalidConcentration(c) if c == -1e-9));
    }

    #[test]
    fn zero_concentration() {
        let err = System::new().monomer("A", 0.0).equilibrium().unwrap_err();
        assert!(matches!(err, EquilibriumError::InvalidConcentration(c) if c == 0.0));
    }

    #[test]
    fn nan_concentration() {
        let err = System::new().monomer("A", f64::NAN).equilibrium().unwrap_err();
        assert!(matches!(err, EquilibriumError::InvalidConcentration(_)));
    }

    #[test]
    fn inf_concentration() {
        let err = System::new().monomer("A", f64::INFINITY).equilibrium().unwrap_err();
        assert!(matches!(err, EquilibriumError::InvalidConcentration(_)));
    }

    #[test]
    fn zero_temperature() {
        let err = System::new().temperature(0.0).monomer("A", 1e-9).equilibrium().unwrap_err();
        assert!(matches!(err, EquilibriumError::InvalidTemperature(t) if t == 0.0));
    }

    #[test]
    fn negative_temperature() {
        let err = System::new().temperature(-100.0).monomer("A", 1e-9).equilibrium().unwrap_err();
        assert!(matches!(err, EquilibriumError::InvalidTemperature(_)));
    }

    #[test]
    fn duplicate_monomer() {
        let err = System::new()
            .monomer("A", 1e-9)
            .monomer("A", 2e-9)
            .equilibrium().unwrap_err();
        assert!(matches!(err, EquilibriumError::DuplicateMonomer(ref n) if n == "A"));
    }

    #[test]
    fn duplicate_complex() {
        let err = System::new()
            .monomer("A", 1e-9)
            .monomer("B", 1e-9)
            .complex("AB", &[("A", 1), ("B", 1)], -10.0)
            .complex("AB", &[("A", 1), ("B", 1)], -12.0)
            .equilibrium().unwrap_err();
        assert!(matches!(err, EquilibriumError::DuplicateComplex(ref n) if n == "AB"));
    }

    #[test]
    fn complex_name_collides_with_monomer() {
        let err = System::new()
            .monomer("A", 1e-9)
            .monomer("B", 1e-9)
            .complex("A", &[("A", 1), ("B", 1)], -10.0)
            .equilibrium().unwrap_err();
        assert!(matches!(err, EquilibriumError::DuplicateSpeciesName(ref n) if n == "A"));
    }

    #[test]
    fn zero_count_stoichiometry() {
        let err = System::new()
            .monomer("A", 1e-9)
            .monomer("B", 1e-9)
            .complex("AB", &[("A", 0), ("B", 1)], -10.0)
            .equilibrium().unwrap_err();
        assert!(matches!(err, EquilibriumError::ZeroCount(ref n) if n == "A"));
    }

    #[test]
    fn empty_composition() {
        let err = System::new()
            .monomer("A", 1e-9)
            .complex("X", &[], -10.0)
            .equilibrium().unwrap_err();
        assert!(matches!(err, EquilibriumError::EmptyComposition));
    }

    #[test]
    fn unknown_monomer() {
        let err = System::new()
            .monomer("A", 1e-9)
            .complex("AB", &[("A", 1), ("Z", 1)], -10.0)
            .equilibrium().unwrap_err();
        assert!(matches!(err, EquilibriumError::UnknownMonomer(ref n) if n == "Z"));
    }

    #[test]
    fn concentration_unknown_name() {
        let eq = System::new()
            .monomer("A", 1e-9)
            .equilibrium().unwrap();
        assert!(eq.concentration("nonexistent").is_none());
    }

    #[test]
    fn error_display() {
        let cases: Vec<(EquilibriumError, &str)> = vec![
            (EquilibriumError::NoMonomers, "no monomers"),
            (EquilibriumError::UnknownMonomer("X".into()), "unknown monomer"),
            (EquilibriumError::EmptyComposition, "empty composition"),
            (EquilibriumError::InvalidConcentration(-1.0), "invalid concentration"),
            (EquilibriumError::InvalidTemperature(-1.0), "invalid temperature"),
            (EquilibriumError::DuplicateMonomer("A".into()), "duplicate monomer"),
            (EquilibriumError::DuplicateComplex("AB".into()), "duplicate complex"),
            (EquilibriumError::ZeroCount("A".into()), "zero stoichiometric count"),
            (EquilibriumError::DuplicateMonomerInComposition("A".into()), "duplicate monomer in composition"),
            (EquilibriumError::InvalidDeltaG(f64::NAN), "invalid delta_g"),
            (EquilibriumError::EmptyName, "must not be empty"),
            (EquilibriumError::DuplicateSpeciesName("A".into()), "species name already in use"),
            (EquilibriumError::ConvergenceFailure { iterations: 100, gradient_norm: 1.0 }, "did not converge"),
        ];
        for (err, expected_substr) in &cases {
            let msg = err.to_string();
            assert!(
                msg.contains(expected_substr),
                "expected {:?} to contain {:?}, got {:?}",
                err, expected_substr, msg,
            );
        }
    }

    #[test]
    fn duplicate_monomer_in_composition() {
        let err = System::new()
            .monomer("A", 1e-9)
            .monomer("B", 1e-9)
            .complex("AB", &[("A", 1), ("A", 2)], -10.0)
            .equilibrium().unwrap_err();
        assert!(matches!(err, EquilibriumError::DuplicateMonomerInComposition(ref n) if n == "A"));
    }

    #[test]
    fn nan_delta_g() {
        let err = System::new()
            .monomer("A", 1e-9)
            .monomer("B", 1e-9)
            .complex("AB", &[("A", 1), ("B", 1)], f64::NAN)
            .equilibrium().unwrap_err();
        assert!(matches!(err, EquilibriumError::InvalidDeltaG(_)));
    }

    #[test]
    fn inf_delta_g() {
        let err = System::new()
            .monomer("A", 1e-9)
            .monomer("B", 1e-9)
            .complex("AB", &[("A", 1), ("B", 1)], f64::INFINITY)
            .equilibrium().unwrap_err();
        assert!(matches!(err, EquilibriumError::InvalidDeltaG(_)));
    }

    #[test]
    fn neg_inf_delta_g() {
        let err = System::new()
            .monomer("A", 1e-9)
            .monomer("B", 1e-9)
            .complex("AB", &[("A", 1), ("B", 1)], f64::NEG_INFINITY)
            .equilibrium().unwrap_err();
        assert!(matches!(err, EquilibriumError::InvalidDeltaG(_)));
    }

    #[test]
    fn empty_monomer_name() {
        let err = System::new().monomer("", 1e-9).equilibrium().unwrap_err();
        assert!(matches!(err, EquilibriumError::EmptyName));
    }

    #[test]
    fn empty_complex_name() {
        let err = System::new()
            .monomer("A", 1e-9)
            .complex("", &[("A", 1)], -10.0)
            .equilibrium().unwrap_err();
        assert!(matches!(err, EquilibriumError::EmptyName));
    }

    #[test]
    fn accessor_methods() {
        let eq = System::new()
            .monomer("A", 1e-9)
            .monomer("B", 1e-9)
            .complex("AB", &[("A", 1), ("B", 1)], -10.0)
            .equilibrium().unwrap();
        assert_eq!(eq.monomer_names(), &["A", "B"]);
        assert_eq!(eq.complex_names(), &["AB"]);
        assert_eq!(eq.free_monomer_concentrations().len(), 2);
        assert_eq!(eq.complex_concentrations().len(), 1);
    }

    #[test]
    fn dogleg_newton_within_trust_region() {
        // When Newton step is within trust region, dogleg should return it exactly
        let h = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 2.0]);
        let g = DVector::from_vec(vec![1.0, 1.0]);
        // Newton step = -H⁻¹g = [-0.5, -0.5], norm ≈ 0.707
        let p = dogleg_step(&g, &h, 10.0);
        assert!((p[0] - (-0.5)).abs() < 1e-12);
        assert!((p[1] - (-0.5)).abs() < 1e-12);
    }

    #[test]
    fn dogleg_cauchy_clipped() {
        // When trust region is very small, should return scaled gradient direction
        let h = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 2.0]);
        let g = DVector::from_vec(vec![1.0, 1.0]);
        let delta = 0.01; // Much smaller than Cauchy step norm
        let p = dogleg_step(&g, &h, delta);
        let p_norm = p.norm();
        assert!(
            (p_norm - delta).abs() < 1e-12,
            "step norm {} should equal delta {}",
            p_norm, delta,
        );
        // Direction should be opposite to gradient
        let cos_angle = p.dot(&g) / (p_norm * g.norm());
        assert!(cos_angle < -0.99, "step should be opposite gradient");
    }

    #[test]
    fn dogleg_interpolation() {
        // Trust region between Cauchy and Newton norms → dog-leg interpolation
        let h = DMatrix::from_row_slice(2, 2, &[4.0, 0.0, 0.0, 1.0]);
        let g = DVector::from_vec(vec![2.0, 2.0]);
        // Newton step = -H⁻¹g = [-0.5, -2.0], norm ≈ 2.06
        // Cauchy: gtg=8, Hg=[8,2], gHg=20, alpha=8/20=0.4, p_c = -0.4*g = [-0.8,-0.8], norm≈1.13
        let p_c_norm = 0.4 * g.norm();
        let p_n_norm = (0.25 + 4.0_f64).sqrt();
        let delta = (p_c_norm + p_n_norm) / 2.0; // Between Cauchy and Newton
        let p = dogleg_step(&g, &h, delta);
        let p_norm = p.norm();
        assert!(
            (p_norm - delta).abs() < 1e-10,
            "step norm {} should equal delta {}",
            p_norm, delta,
        );
    }

    #[test]
    fn dogleg_cholesky_failure() {
        // Singular Hessian (not positive definite) triggers the Cholesky failure path.
        let h = DMatrix::from_row_slice(2, 2, &[0.0, 0.0, 0.0, 0.0]);
        let g = DVector::from_vec(vec![1.0, 1.0]);
        let p = dogleg_step(&g, &h, 1.0);
        // Should return steepest descent clipped to trust region
        let p_norm = p.norm();
        assert!(
            (p_norm - 1.0).abs() < 1e-12,
            "step norm {} should equal delta 1.0",
            p_norm,
        );
        // Direction should be opposite to gradient
        let cos_angle = p.dot(&g) / (p_norm * g.norm());
        assert!(cos_angle < -0.99, "step should be opposite gradient");
    }

    #[test]
    fn dogleg_cholesky_failure_zero_gradient() {
        // Singular Hessian + zero gradient: should return zero step.
        let h = DMatrix::from_row_slice(2, 2, &[0.0, 0.0, 0.0, 0.0]);
        let g = DVector::from_vec(vec![0.0, 0.0]);
        let p = dogleg_step(&g, &h, 1.0);
        assert!(p.norm() < 1e-15, "zero gradient should give zero step");
    }

    #[test]
    fn dogleg_near_zero_curvature() {
        // gᵀHg ≈ 0 but gradient nonzero: triggers the near-zero curvature guard.
        // Use a Hessian with extremely small eigenvalues (but still PD for Cholesky).
        let eps = 1e-40;
        let h = DMatrix::from_row_slice(2, 2, &[eps, 0.0, 0.0, eps]);
        let g = DVector::from_vec(vec![1.0, 1.0]);
        let delta = 0.5;
        let p = dogleg_step(&g, &h, delta);
        // Newton step would be huge (-g/eps), so it's outside trust region.
        // Cauchy: gᵀHg = 2*eps ≈ 0 < 1e-30 * gᵀg = 2e-30, so near-zero curvature fires.
        let p_norm = p.norm();
        assert!(
            (p_norm - delta).abs() < 1e-12,
            "step norm {} should equal delta {}",
            p_norm, delta,
        );
    }

    #[test]
    fn trust_region_adjustment() {
        // A system with many competing complexes and asymmetric concentrations
        // to exercise trust region shrink/expand during iteration.
        let sys = System::new()
            .monomer("A", 1e-6)
            .monomer("B", 1e-8)
            .complex("AB", &[("A", 1), ("B", 1)], -15.0)
            .complex("A2B", &[("A", 2), ("B", 1)], -20.0)
            .complex("AB2", &[("A", 1), ("B", 2)], -18.0);
        let eq = sys.equilibrium().unwrap();

        // Verify mass conservation
        let a = eq.concentration("A").unwrap();
        let b = eq.concentration("B").unwrap();
        let ab = eq.concentration("AB").unwrap();
        let a2b = eq.concentration("A2B").unwrap();
        let ab2 = eq.concentration("AB2").unwrap();
        assert!((a + ab + 2.0 * a2b + ab2 - 1e-6).abs() < 1e-5 * 1e-6);
        assert!((b + ab + a2b + 2.0 * ab2 - 1e-8).abs() < 1e-5 * 1e-8);
    }
}

// ---------------------------------------------------------------------------
// Python bindings (behind "python" feature)
// ---------------------------------------------------------------------------

#[cfg(feature = "python")]
mod python;
