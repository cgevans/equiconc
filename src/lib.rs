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
//! // A + B ⇌ AB with ΔG° = -10 kcal/mol at 25 °C (default)
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

use ndarray::{Array1, Array2};

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
    /// Create an empty system at 25 °C (298.15 K).
    #[must_use]
    pub fn new() -> Self {
        System {
            monomers: Vec::new(),
            complexes: Vec::new(),
            temperature: 298.15,
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
        use std::collections::HashMap;

        // Temperature validation
        if !(self.temperature > 0.0 && self.temperature.is_finite()) {
            return Err(EquilibriumError::InvalidTemperature(self.temperature));
        }

        // Must have at least one monomer
        if self.monomers.is_empty() {
            return Err(EquilibriumError::NoMonomers);
        }

        // Validate monomers and build index
        let mut monomer_idx: HashMap<&str, usize> = HashMap::with_capacity(self.monomers.len());
        let mut monomer_names: Vec<String> = Vec::with_capacity(self.monomers.len());
        let mut monomer_concs: Vec<f64> = Vec::with_capacity(self.monomers.len());
        for (name, conc) in &self.monomers {
            if name.is_empty() {
                return Err(EquilibriumError::EmptyName);
            }
            if !(conc > &0.0 && conc.is_finite()) {
                return Err(EquilibriumError::InvalidConcentration(*conc));
            }
            if monomer_idx.contains_key(name.as_str()) {
                return Err(EquilibriumError::DuplicateMonomer(name.clone()));
            }
            monomer_idx.insert(name, monomer_names.len());
            monomer_names.push(name.clone());
            monomer_concs.push(*conc);
        }

        // Validate complexes and resolve monomer references
        let mut all_names: HashMap<&str, ()> = monomer_idx.keys().map(|&k| (k, ())).collect();
        let mut complex_names: Vec<String> = Vec::with_capacity(self.complexes.len());
        let mut resolved_comps: Vec<Vec<(usize, usize)>> = Vec::with_capacity(self.complexes.len());
        let mut complex_dgs: Vec<f64> = Vec::with_capacity(self.complexes.len());

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
            if all_names.contains_key(name.as_str()) {
                // Could be a duplicate complex or collision with a monomer
                if monomer_idx.contains_key(name.as_str()) {
                    return Err(EquilibriumError::DuplicateSpeciesName(name.clone()));
                }
                return Err(EquilibriumError::DuplicateComplex(name.clone()));
            }

            let mut comp = Vec::new();
            for (monomer_name, count) in composition {
                if *count == 0 {
                    return Err(EquilibriumError::ZeroCount(monomer_name.clone()));
                }
                let &idx = monomer_idx
                    .get(monomer_name.as_str())
                    .ok_or_else(|| {
                        EquilibriumError::UnknownMonomer(monomer_name.clone())
                    })?;
                if let Some(entry) = comp.iter_mut().find(|(existing_idx, _): &&mut (usize, usize)| *existing_idx == idx) {
                    entry.1 += count;
                } else {
                    comp.push((idx, *count));
                }
            }

            all_names.insert(name, ());
            complex_names.push(name.clone());
            resolved_comps.push(comp);
            complex_dgs.push(*delta_g);
        }

        Ok((monomer_names, monomer_concs, complex_names, resolved_comps, complex_dgs))
    }

    /// Build the stoichiometry matrix **Aᵀ** (transposed for cache-friendly
    /// access in the hot Aᵀλ multiply), the log-partition-function vector
    /// **log_q**, and the total-concentration vector **c⁰**.
    fn build_problem(
        n_mon: usize,
        monomer_concs: &[f64],
        resolved_comps: &[Vec<(usize, usize)>],
        complex_dgs: &[f64],
        temperature: f64,
    ) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
        let n_cplx = resolved_comps.len();
        let n_species = n_mon + n_cplx;

        // at: n_species × n_monomers  (= Aᵀ, stored row-major so that
        // at.dot(λ) — the dominant O(mn) operation — reads contiguous rows)
        let mut at = Array2::zeros((n_species, n_mon));
        // Identity block for free monomers
        for i in 0..n_mon {
            at[[i, i]] = 1.0;
        }
        // Complex rows
        for (k, comp) in resolved_comps.iter().enumerate() {
            for &(mi, count) in comp {
                at[[n_mon + k, mi]] = count as f64;
            }
        }

        // log_q: 0 for free monomers, -ΔG°/(RT) for complexes
        let mut log_q = Array1::zeros(n_species);
        let rt = R * temperature;
        for (k, &dg) in complex_dgs.iter().enumerate() {
            log_q[n_mon + k] = -dg / rt;
        }

        // c⁰
        let c0 = Array1::from_vec(monomer_concs.to_vec());

        (at, log_q, c0)
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
                iterations: 0,
            });
        }

        let (at, log_q, c0) =
            Self::build_problem(n_mon, &monomer_concs, &resolved_comps, &complex_dgs, self.temperature);
        let solution = solve_dual(&at, &log_q, &c0)?;

        // Recover concentrations: c_j = exp(log_q_j + (Aᵀλ)_j)
        let log_c = &log_q + &at.dot(&solution.lambda);
        let concentrations: Vec<f64> =
            log_c.iter().map(|&lc| lc.min(LOG_C_MAX).exp()).collect();

        // Debug-mode validation
        #[cfg(debug_assertions)]
        {
            let rt = R * self.temperature;
            for i in 0..n_mon {
                let total: f64 =
                    (0..concentrations.len()).map(|j| at[[j, i]] * concentrations[j]).sum();
                debug_assert!(
                    (total - c0[i]).abs() < 1e-2 * c0[i] + 1e-30,
                    "mass conservation violated for monomer {}: {total} != {}",
                    monomer_names[i],
                    c0[i]
                );
            }
            for (k, comp) in resolved_comps.iter().enumerate() {
                let log_keq = -complex_dgs[k] / rt;
                let mut log_expected = log_keq;
                for &(mi, count) in comp {
                    log_expected += count as f64 * concentrations[mi].ln();
                }
                let log_actual = concentrations[n_mon + k].ln();
                let log_err = (log_actual - log_expected).abs();
                debug_assert!(
                    log_err < 1e-2 * (1.0 + log_expected.abs()),
                    "equilibrium condition violated for {}: log(actual)={log_actual} != log(expected)={log_expected} (log_err: {log_err})",
                    complex_names[k]
                );
            }
        }

        let converged_fully = matches!(
            solution.convergence,
            SolverConvergence::Full { .. }
        );
        let iterations = match solution.convergence {
            SolverConvergence::Full { iterations } | SolverConvergence::Relaxed { iterations, .. } => iterations,
        };

        Ok(Equilibrium {
            monomer_names,
            complex_names,
            free_monomer_concentrations: concentrations[..n_mon].to_vec(),
            complex_concentrations: concentrations[n_mon..].to_vec(),
            converged_fully,
            iterations,
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
    iterations: usize,
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

    /// Number of solver iterations used.
    #[must_use]
    pub fn iterations(&self) -> usize {
        self.iterations
    }
}

// ---------------------------------------------------------------------------
// Linear algebra helpers (small m×m operations)
// ---------------------------------------------------------------------------

/// L2 norm of a vector.
fn norm(v: &Array1<f64>) -> f64 {
    v.dot(v).sqrt()
}

/// Cholesky factorization of a symmetric positive-definite matrix.
/// Returns the Newton step p = -H⁻¹g, or None if H is not positive definite.
fn cholesky_solve(h: &Array2<f64>, g: &Array1<f64>) -> Option<Array1<f64>> {
    let n = g.len();
    // Compute L (lower triangular, H = LLᵀ)
    let mut l = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut sum = h[[i, j]];
            for k in 0..j {
                sum -= l[[i, k]] * l[[j, k]];
            }
            if i == j {
                if sum <= 0.0 {
                    return None;
                }
                l[[i, j]] = sum.sqrt();
            } else {
                l[[i, j]] = sum / l[[j, j]];
            }
        }
    }
    // Forward solve: Lz = -g
    let mut z = Array1::zeros(n);
    for i in 0..n {
        let mut sum = -g[i];
        for k in 0..i {
            sum -= l[[i, k]] * z[k];
        }
        z[i] = sum / l[[i, i]];
    }
    // Back solve: Lᵀp = z
    let mut p = Array1::zeros(n);
    for i in (0..n).rev() {
        let mut sum = z[i];
        for k in (i + 1)..n {
            sum -= l[[k, i]] * p[k];
        }
        p[i] = sum / l[[i, i]];
    }
    Some(p)
}

// ---------------------------------------------------------------------------
// Trust-region Newton solver on the dual
// ---------------------------------------------------------------------------

/// Pre-allocated working buffers for the solver, reused across iterations.
struct WorkBuffers {
    c: Array1<f64>,        // n_species: concentrations
    grad: Array1<f64>,     // n_mon: gradient
    hessian: Array2<f64>,  // n_mon × n_mon: Hessian
}

impl WorkBuffers {
    fn new(n_mon: usize, n_species: usize) -> Self {
        WorkBuffers {
            c: Array1::zeros(n_species),
            grad: Array1::zeros(n_mon),
            hessian: Array2::zeros((n_mon, n_mon)),
        }
    }
}

/// Evaluate the dual objective, gradient, and Hessian at λ,
/// writing into pre-allocated buffers. Returns f.
///
/// `at` is the transposed stoichiometry matrix Aᵀ (n_species × n_mon),
/// stored row-major so that the dominant `Aᵀλ` multiply reads contiguous rows.
fn evaluate_into(
    at: &Array2<f64>,
    log_q: &Array1<f64>,
    c0: &Array1<f64>,
    lambda: &Array1<f64>,
    w: &mut WorkBuffers,
) -> f64 {
    let n_species = at.nrows();
    let n_mon = at.ncols();

    // c = exp(min(log_q + Aᵀλ, LOG_C_MAX))
    // Compute Aᵀλ into w.c, then fuse the add + clamp + exp in-place
    ndarray::linalg::general_mat_vec_mul(1.0, at, lambda, 0.0, &mut w.c);
    w.c += log_q;
    w.c.mapv_inplace(|lc| lc.min(LOG_C_MAX).exp());

    // f = -λᵀc⁰ + Σ_j c_j
    let f = -lambda.dot(c0) + w.c.sum();

    // grad = -c⁰ + Aᵀᵀ·c
    ndarray::linalg::general_mat_vec_mul(1.0, &at.t(), &w.c, 0.0, &mut w.grad);
    w.grad -= c0;

    // H = A diag(c) Aᵀ  (sparse loop, writing into pre-allocated hessian)
    w.hessian.fill(0.0);
    for k in 0..n_species {
        let ck = w.c[k];
        for i in 0..n_mon {
            let aik = at[[k, i]];
            if aik == 0.0 {
                continue;
            }
            for j in i..n_mon {
                let ajk = at[[k, j]];
                if ajk == 0.0 {
                    continue;
                }
                let val = aik * ck * ajk;
                w.hessian[[i, j]] += val;
                if i != j {
                    w.hessian[[j, i]] += val;
                }
            }
        }
    }

    f
}

/// Evaluate only the dual objective at λ (no gradient or Hessian).
///
/// Uses the `c` buffer from WorkBuffers to avoid allocation.
fn evaluate_objective_into(
    at: &Array2<f64>,
    log_q: &Array1<f64>,
    c0: &Array1<f64>,
    lambda: &Array1<f64>,
    c_buf: &mut Array1<f64>,
) -> f64 {
    ndarray::linalg::general_mat_vec_mul(1.0, at, lambda, 0.0, c_buf);
    *c_buf += log_q;
    let c_sum: f64 = c_buf.iter().map(|&lc| lc.min(LOG_C_MAX).exp()).sum();

    -lambda.dot(c0) + c_sum
}

/// Compute the dog-leg step within a trust region of radius `delta`.
fn dogleg_step(grad: &Array1<f64>, hessian: &Array2<f64>, delta: f64) -> Array1<f64> {
    // Cholesky solve for Newton step: p_n = -H⁻¹g
    let p_n = match cholesky_solve(hessian, grad) {
        Some(p) => p,
        None => {
            // Defensive: H = A diag(c) Aᵀ with all c > 0, so H is always
            // positive-definite and Cholesky cannot fail in practice.
            let g_norm = norm(grad);
            if g_norm == 0.0 {
                return Array1::zeros(grad.len());
            }
            return -(delta / g_norm) * grad;
        }
    };

    let p_n_norm = norm(&p_n);

    if p_n_norm <= delta {
        return p_n;
    }

    // Cauchy step: p_c = -(gᵀg / gᵀHg) g
    let gtg = grad.dot(grad);
    let hg = hessian.dot(grad);
    let gthg = grad.dot(&hg);

    // Guard against near-zero curvature: fall back to steepest descent
    // clipped to trust region, same as the Cholesky failure path.
    if gthg < 1e-30 * gtg {
        let g_norm = norm(grad);
        return -(delta / g_norm) * grad;
    }

    let p_c = -(gtg / gthg) * grad;
    let p_c_norm = norm(&p_c);

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
    lambda: Array1<f64>,
    convergence: SolverConvergence,
}

/// Solve the dual problem, returning the optimal λ* and convergence info.
fn solve_dual(
    at: &Array2<f64>,
    log_q: &Array1<f64>,
    c0: &Array1<f64>,
) -> Result<DualSolution, EquilibriumError> {
    const MAX_ITER: usize = 1000;
    const ATOL: f64 = 1e-22;
    const RTOL: f64 = 1e-7;
    const ETA: f64 = 1e-4;
    const DELTA_MAX: f64 = 1e10;

    let n_mon = at.ncols();
    let n_species = at.nrows();
    let mut w = WorkBuffers::new(n_mon, n_species);
    let mut lambda = c0.mapv(|c| c.ln());
    let mut lambda_new = Array1::zeros(n_mon);
    let mut delta = 1.0;
    let mut stagnation = 0u32;

    for iter in 0..MAX_ITER {
        let f = evaluate_into(at, log_q, c0, &lambda, &mut w);

        if w.grad
            .iter()
            .zip(c0.iter())
            .all(|(&g, &c)| g.abs() < ATOL + RTOL * c)
        {
            return Ok(DualSolution {
                lambda,
                convergence: SolverConvergence::Full { iterations: iter + 1 },
            });
        }

        let p = dogleg_step(&w.grad, &w.hessian, delta);
        let p_norm = norm(&p);

        lambda_new.assign(&lambda);
        lambda_new += &p;
        let f_new = evaluate_objective_into(at, log_q, c0, &lambda_new, &mut w.c);

        let actual_reduction = f - f_new;
        let predicted_reduction = -(w.grad.dot(&p) + 0.5 * p.dot(&w.hessian.dot(&p)));

        // Track stagnation: objective is effectively at its f64 minimum.
        if actual_reduction < 4.0 * f64::EPSILON * f.abs().max(1.0) {
            stagnation += 1;
        } else {
            stagnation = 0;
        }

        // When the trust region has collapsed and steps produce no
        // measurable reduction, try a full Newton step.
        const STAG_ATOL: f64 = 1e-14;
        const STAG_RTOL: f64 = 1e-4;
        if stagnation >= 3 {
            let p_full = dogleg_step(&w.grad, &w.hessian, DELTA_MAX);
            lambda_new.assign(&lambda);
            lambda_new += &p_full;
            let f_full = evaluate_into(at, log_q, c0, &lambda_new, &mut w);
            if f_full <= f {
                std::mem::swap(&mut lambda, &mut lambda_new);

                if w.grad
                    .iter()
                    .zip(c0.iter())
                    .all(|(&g, &c)| g.abs() < ATOL + RTOL * c)
                {
                    return Ok(DualSolution {
                        lambda,
                        convergence: SolverConvergence::Full {
                            iterations: iter + 1,
                        },
                    });
                }
                if w.grad
                    .iter()
                    .zip(c0.iter())
                    .all(|(&g, &c)| g.abs() < STAG_ATOL + STAG_RTOL * c)
                {
                    return Ok(DualSolution {
                        lambda,
                        convergence: SolverConvergence::Relaxed {
                            iterations: iter + 1,
                            gradient_norm: norm(&w.grad),
                        },
                    });
                }

                delta = norm(&p_full);
                stagnation = 0;
                continue;
            }
            // Recovery failed — check relaxed tolerance on pre-step gradient.
            // Need to re-evaluate at lambda since w was overwritten.
            evaluate_into(at, log_q, c0, &lambda, &mut w);
            if w.grad
                .iter()
                .zip(c0.iter())
                .all(|(&g, &c)| g.abs() < STAG_ATOL + STAG_RTOL * c)
            {
                return Ok(DualSolution {
                    lambda,
                    convergence: SolverConvergence::Relaxed {
                        iterations: iter + 1,
                        gradient_norm: norm(&w.grad),
                    },
                });
            }
            return Err(EquilibriumError::ConvergenceFailure {
                iterations: iter + 1,
                gradient_norm: norm(&w.grad),
            });
        }

        let rho = if predicted_reduction.abs() < 1e-30 {
            if actual_reduction >= 0.0 { 1.0 } else { 0.0 }
        } else {
            actual_reduction / predicted_reduction
        };

        if rho < 0.25 {
            delta *= 0.25;
        } else if rho > 0.75 && (p_norm - delta).abs() < 1e-10 * delta {
            delta = (2.0 * delta).min(DELTA_MAX);
        }

        if rho > ETA {
            std::mem::swap(&mut lambda, &mut lambda_new);
        }
    }

    evaluate_into(at, log_q, c0, &lambda, &mut w);
    Err(EquilibriumError::ConvergenceFailure {
        iterations: MAX_ITER,
        gradient_norm: norm(&w.grad),
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

        let rt = R * 298.15;
        let k = (-dg / rt).exp();
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
            free_a, aaa, free_a + 3.0 * aaa
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

        assert!((a + ab + 3.0 * aaa - c0).abs() < 1e-5 * c0);
        assert!((b + ab + bc - c0).abs() < 1e-5 * c0);
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
        assert!((ab - c0).abs() < 1e-6 * c0, "[AB] = {ab}, expected ≈ {c0}");
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
    fn duplicate_monomer_in_composition_sums() {
        let eq_dup = System::new()
            .monomer("A", 1e-6)
            .complex("A3", &[("A", 1), ("A", 2)], -15.0)
            .equilibrium().unwrap();
        let eq_merged = System::new()
            .monomer("A", 1e-6)
            .complex("A3", &[("A", 3)], -15.0)
            .equilibrium().unwrap();
        assert!((eq_dup.concentration("A3").unwrap() - eq_merged.concentration("A3").unwrap()).abs() < 1e-20);
        assert!((eq_dup.concentration("A").unwrap() - eq_merged.concentration("A").unwrap()).abs() < 1e-20);
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
        let h = Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 0.0, 2.0]).unwrap();
        let g = Array1::from_vec(vec![1.0, 1.0]);
        let p = dogleg_step(&g, &h, 10.0);
        assert!((p[0] - (-0.5)).abs() < 1e-12);
        assert!((p[1] - (-0.5)).abs() < 1e-12);
    }

    #[test]
    fn dogleg_cauchy_clipped() {
        let h = Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 0.0, 2.0]).unwrap();
        let g = Array1::from_vec(vec![1.0, 1.0]);
        let delta = 0.01;
        let p = dogleg_step(&g, &h, delta);
        let p_norm = norm(&p);
        assert!(
            (p_norm - delta).abs() < 1e-12,
            "step norm {} should equal delta {}", p_norm, delta,
        );
        let cos_angle = p.dot(&g) / (p_norm * norm(&g));
        assert!(cos_angle < -0.99, "step should be opposite gradient");
    }

    #[test]
    fn dogleg_interpolation() {
        let h = Array2::from_shape_vec((2, 2), vec![4.0, 0.0, 0.0, 1.0]).unwrap();
        let g = Array1::from_vec(vec![2.0, 2.0]);
        let p_c_norm = 0.4 * norm(&g);
        let p_n_norm = (0.25 + 4.0_f64).sqrt();
        let delta = (p_c_norm + p_n_norm) / 2.0;
        let p = dogleg_step(&g, &h, delta);
        let p_norm = norm(&p);
        assert!(
            (p_norm - delta).abs() < 1e-10,
            "step norm {} should equal delta {}", p_norm, delta,
        );
    }

    #[test]
    fn dogleg_cholesky_failure() {
        let h = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 0.0, 0.0]).unwrap();
        let g = Array1::from_vec(vec![1.0, 1.0]);
        let p = dogleg_step(&g, &h, 1.0);
        let p_norm = norm(&p);
        assert!(
            (p_norm - 1.0).abs() < 1e-12,
            "step norm {} should equal delta 1.0", p_norm,
        );
        let cos_angle = p.dot(&g) / (p_norm * norm(&g));
        assert!(cos_angle < -0.99, "step should be opposite gradient");
    }

    #[test]
    fn dogleg_cholesky_failure_zero_gradient() {
        let h = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 0.0, 0.0]).unwrap();
        let g = Array1::from_vec(vec![0.0, 0.0]);
        let p = dogleg_step(&g, &h, 1.0);
        assert!(norm(&p) < 1e-15, "zero gradient should give zero step");
    }

    #[test]
    fn dogleg_near_zero_curvature() {
        let eps = 1e-40;
        let h = Array2::from_shape_vec((2, 2), vec![eps, 0.0, 0.0, eps]).unwrap();
        let g = Array1::from_vec(vec![1.0, 1.0]);
        let delta = 0.5;
        let p = dogleg_step(&g, &h, delta);
        let p_norm = norm(&p);
        assert!(
            (p_norm - delta).abs() < 1e-12,
            "step norm {} should equal delta {}", p_norm, delta,
        );
    }

    #[test]
    fn trust_region_adjustment() {
        let sys = System::new()
            .monomer("A", 1e-6)
            .monomer("B", 1e-8)
            .complex("AB", &[("A", 1), ("B", 1)], -15.0)
            .complex("A2B", &[("A", 2), ("B", 1)], -20.0)
            .complex("AB2", &[("A", 1), ("B", 2)], -18.0);
        let eq = sys.equilibrium().unwrap();

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
