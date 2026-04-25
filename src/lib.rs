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
//! # API overview
//!
//! The high-level entry point is [`SystemBuilder`], which accepts
//! name-keyed monomers and complexes and validates the specification.
//! Calling [`SystemBuilder::build`] produces a [`System`] — a stateful
//! solver handle that owns the numerical inputs, work buffers, and the
//! most recent solution. Call [`System::solve`] to obtain a borrowed
//! [`Equilibrium`] view of the concentrations.
//!
//! For callers that already have indexed numerical data (typically from
//! another library or a tight sweep loop), [`System::from_arrays`] and
//! [`System::from_arrays_with_names`] skip the string-keyed front end
//! entirely.
//!
//! # Example
//!
//! ```
//! use equiconc::SystemBuilder;
//!
//! // A + B ⇌ AB with ΔG° = -10 kcal/mol at 25 °C (default)
//! let mut sys = SystemBuilder::new()
//!     .monomer("A", 100e-9)      // 100 nM
//!     .monomer("B", 100e-9)
//!     .complex("AB", &[("A", 1), ("B", 1)], -10.0)
//!     .build()?;
//!
//! let eq = sys.solve()?;
//! let free_a = eq.get("A").unwrap();
//! let free_b = eq.get("B").unwrap();
//! let ab = eq.get("AB").unwrap();
//!
//! // Mass conservation: [A] + [AB] = 100 nM
//! assert!((free_a + ab - 100e-9).abs() < 1e-6 * 100e-9);
//! # Ok::<(), equiconc::EquilibriumError>(())
//! ```
//!
//! # Sweeps and titrations
//!
//! `System` is designed for tight sweep loops: mutate inputs in place
//! with [`System::set_c0`] / [`System::c0_mut`] (or the `log_q`
//! counterparts) and call [`System::solve`] again. The warm-start λ is
//! preserved across calls, and no new allocations happen in the loop
//! body.
//!
//! ```
//! use equiconc::SystemBuilder;
//!
//! let mut sys = SystemBuilder::new()
//!     .monomer("A", 0.0_f64.max(1e-20))
//!     .monomer("B", 1e-7)
//!     .complex("AB", &[("A", 1), ("B", 1)], -10.0)
//!     .build()?;
//!
//! let a_idx = sys.monomer_index("A").unwrap();
//! let ab_idx = sys.species_index("AB").unwrap();
//!
//! let mut titration = Vec::with_capacity(10);
//! for i in 1..=10 {
//!     let c_a = 1e-7 * (i as f64) / 10.0;
//!     sys.set_c0(a_idx, c_a);
//!     let eq = sys.solve()?;
//!     titration.push((c_a, eq.at(ab_idx)));
//! }
//! # Ok::<(), equiconc::EquilibriumError>(())
//! ```
//!
//! # Borrow-checker-enforced freshness
//!
//! The borrowed [`Equilibrium`] view prevents "stale read" bugs at compile time:
//!
//! ```compile_fail
//! # use equiconc::SystemBuilder;
//! # let mut sys = SystemBuilder::new()
//! #     .monomer("A", 1e-7)
//! #     .monomer("B", 1e-7)
//! #     .complex("AB", &[("A", 1), ("B", 1)], -10.0)
//! #     .build().unwrap();
//! let eq = sys.solve().unwrap();
//! sys.set_c0(0, 2e-7);             // ERROR: cannot borrow `sys` as mutable
//! let ab = eq.get("AB").unwrap();  // while `eq` is still alive
//! ```

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1, s};
use std::collections::HashMap;
use std::ops::Index;

/// Gas constant in kcal/(mol·K).
pub const R: f64 = 1.987204e-3;

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
    /// Structural or numerical invariant violated on a direct
    /// [`System::from_arrays`] / [`System::from_arrays_with_names`]
    /// construction, or detected by [`System::validate`].
    InvalidInputs(String),
    /// [`SolverOptions`] held an internally inconsistent combination
    /// (e.g. non-positive tolerance, `shrink_rho >= grow_rho`).
    InvalidOptions(String),
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
            Self::InvalidInputs(msg) => write!(f, "invalid inputs: {msg}"),
            Self::InvalidOptions(msg) => write!(f, "invalid solver options: {msg}"),
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
// Solver options
// ---------------------------------------------------------------------------

/// Configuration for the trust-region Newton solver.
///
/// Every field has a sensible default matching the solver's original
/// hard-coded constants — `SolverOptions::default()` reproduces
/// pre-configuration behavior bit-for-bit. Tweak individual fields to
/// trade speed for precision, cap iteration budget, or clamp extreme
/// inputs.
///
/// Attach options either at build time with
/// [`SystemBuilder::options`] / [`SystemBuilder::options_ref`] or per
/// [`System`] via [`System::set_options`] / [`System::options_mut`].
///
/// # Example
///
/// ```
/// use equiconc::{SolverOptions, SystemBuilder};
///
/// let opts = SolverOptions {
///     max_iterations: 200,
///     gradient_rel_tol: 1e-9,
///     ..Default::default()
/// };
///
/// let mut sys = SystemBuilder::new()
///     .monomer("A", 1e-7)
///     .monomer("B", 1e-7)
///     .complex("AB", &[("A", 1), ("B", 1)], -10.0)
///     .options(opts)
///     .build()?;
/// sys.solve()?;
/// # Ok::<(), equiconc::EquilibriumError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct SolverOptions {
    // --- Convergence -----------------------------------------------------
    /// Maximum outer Newton iterations before the solver gives up.
    pub max_iterations: usize,
    /// Full-convergence gradient tolerance (absolute component).
    ///
    /// Convergence is declared when, for every monomer `i`,
    /// `|g_i| < gradient_abs_tol + gradient_rel_tol · c0_i`.
    pub gradient_abs_tol: f64,
    /// Full-convergence gradient tolerance relative to each `c0_i`.
    pub gradient_rel_tol: f64,
    /// Relaxed-convergence absolute tolerance, used by the stagnation
    /// recovery path when the objective can no longer be reduced at f64
    /// precision.
    pub relaxed_gradient_abs_tol: f64,
    /// Relaxed-convergence relative tolerance.
    pub relaxed_gradient_rel_tol: f64,
    /// Number of consecutive non-reducing iterations before the
    /// stagnation recovery path fires (which tries one full Newton step).
    pub stagnation_threshold: u32,

    // --- Trust region ----------------------------------------------------
    /// Initial trust-region radius δ₀.
    pub initial_trust_region_radius: f64,
    /// Upper bound on δ.
    pub max_trust_region_radius: f64,
    /// Minimum ρ for a step to be accepted. Steps with `ρ ≤
    /// step_accept_threshold` are rejected and `λ` is not advanced.
    pub step_accept_threshold: f64,
    /// ρ below which δ shrinks.
    pub trust_region_shrink_rho: f64,
    /// ρ above which δ may grow (when the step is on the trust-region
    /// boundary).
    pub trust_region_grow_rho: f64,
    /// Factor applied to δ on shrink. Must be in (0, 1).
    pub trust_region_shrink_scale: f64,
    /// Factor applied to δ on grow. Must be > 1.
    pub trust_region_grow_scale: f64,

    // --- Numerical clamps ------------------------------------------------
    /// Maximum allowed `log_q_j + (Aᵀλ)_j` before `exp()`. Entries
    /// above this are clamped to prevent f64 overflow. Default `700.0`
    /// sits just below `f64::MAX`; lower values introduce approximation
    /// but never worsen stability.
    pub log_c_clamp: f64,
    /// Optional upper bound on `log_q = -ΔG/RT`, applied at
    /// construction / `set_options` time (modifies the stored `log_q`).
    /// `None` preserves the user's inputs unchanged.
    ///
    /// Useful when input energies have extreme magnitudes (|ΔG/RT| ≫ 100)
    /// that would otherwise drive the linear-space dual objective into
    /// a regime where the solver takes hundreds of bulk iterations
    /// before refinement. A clamp at ~230 matches COFFEE's internal
    /// behavior and keeps numerics well-scaled.
    pub log_q_clamp: Option<f64>,
}

impl Default for SolverOptions {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            gradient_abs_tol: 1e-22,
            gradient_rel_tol: 1e-7,
            relaxed_gradient_abs_tol: 1e-14,
            relaxed_gradient_rel_tol: 1e-4,
            stagnation_threshold: 3,
            initial_trust_region_radius: 1.0,
            max_trust_region_radius: 1e10,
            step_accept_threshold: 1e-4,
            trust_region_shrink_rho: 0.25,
            trust_region_grow_rho: 0.75,
            trust_region_shrink_scale: 0.25,
            trust_region_grow_scale: 2.0,
            log_c_clamp: 700.0,
            log_q_clamp: None,
        }
    }
}

impl SolverOptions {
    /// Check internal consistency of the options.
    ///
    /// Returns [`EquilibriumError::InvalidOptions`] on:
    /// - non-positive `max_iterations`;
    /// - negative or non-finite tolerances;
    /// - `trust_region_shrink_rho >= trust_region_grow_rho`;
    /// - `trust_region_shrink_scale` not in (0, 1), or
    ///   `trust_region_grow_scale <= 1`;
    /// - non-positive `initial_trust_region_radius` or
    ///   `max_trust_region_radius`;
    /// - non-finite `log_c_clamp` or non-finite `log_q_clamp`.
    ///
    /// Does *not* reject pathological-but-legal combinations (e.g.
    /// `gradient_rel_tol = 0.5`). Those will still run, just probably
    /// not produce what the user wants.
    pub fn validate(&self) -> Result<(), EquilibriumError> {
        fn check_nonneg_finite(name: &str, v: f64) -> Result<(), EquilibriumError> {
            if !v.is_finite() || v < 0.0 {
                return Err(EquilibriumError::InvalidOptions(format!(
                    "{name} = {v} (must be finite and ≥ 0)"
                )));
            }
            Ok(())
        }
        fn check_pos_finite(name: &str, v: f64) -> Result<(), EquilibriumError> {
            if !(v.is_finite() && v > 0.0) {
                return Err(EquilibriumError::InvalidOptions(format!(
                    "{name} = {v} (must be finite and > 0)"
                )));
            }
            Ok(())
        }

        if self.max_iterations == 0 {
            return Err(EquilibriumError::InvalidOptions(
                "max_iterations = 0".into(),
            ));
        }
        check_nonneg_finite("gradient_abs_tol", self.gradient_abs_tol)?;
        check_nonneg_finite("gradient_rel_tol", self.gradient_rel_tol)?;
        check_nonneg_finite("relaxed_gradient_abs_tol", self.relaxed_gradient_abs_tol)?;
        check_nonneg_finite("relaxed_gradient_rel_tol", self.relaxed_gradient_rel_tol)?;
        check_nonneg_finite("step_accept_threshold", self.step_accept_threshold)?;
        check_pos_finite(
            "initial_trust_region_radius",
            self.initial_trust_region_radius,
        )?;
        check_pos_finite("max_trust_region_radius", self.max_trust_region_radius)?;
        if self.initial_trust_region_radius > self.max_trust_region_radius {
            return Err(EquilibriumError::InvalidOptions(format!(
                "initial_trust_region_radius ({}) exceeds max_trust_region_radius ({})",
                self.initial_trust_region_radius, self.max_trust_region_radius
            )));
        }
        if !(self.trust_region_shrink_rho < self.trust_region_grow_rho) {
            return Err(EquilibriumError::InvalidOptions(format!(
                "trust_region_shrink_rho ({}) must be < trust_region_grow_rho ({})",
                self.trust_region_shrink_rho, self.trust_region_grow_rho
            )));
        }
        if !(self.trust_region_shrink_scale > 0.0 && self.trust_region_shrink_scale < 1.0) {
            return Err(EquilibriumError::InvalidOptions(format!(
                "trust_region_shrink_scale = {} (must be in (0, 1))",
                self.trust_region_shrink_scale
            )));
        }
        if !(self.trust_region_grow_scale > 1.0 && self.trust_region_grow_scale.is_finite()) {
            return Err(EquilibriumError::InvalidOptions(format!(
                "trust_region_grow_scale = {} (must be > 1 and finite)",
                self.trust_region_grow_scale
            )));
        }
        if !self.log_c_clamp.is_finite() {
            return Err(EquilibriumError::InvalidOptions(format!(
                "log_c_clamp = {} (must be finite)",
                self.log_c_clamp
            )));
        }
        if let Some(c) = self.log_q_clamp
            && !c.is_finite()
        {
            return Err(EquilibriumError::InvalidOptions(format!(
                "log_q_clamp = {c} (must be finite when Some)"
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// SystemBuilder — high-level, name-based, validating
// ---------------------------------------------------------------------------

/// High-level, name-keyed builder for an equilibrium-concentration problem.
///
/// Chain [`SystemBuilder::monomer`] and [`SystemBuilder::complex`] calls
/// to describe the system, then call [`SystemBuilder::build`] to validate
/// the specification and obtain a [`System`] solver handle.
#[derive(Debug, Clone)]
pub struct SystemBuilder {
    monomers: Vec<(String, f64)>,
    complexes: Vec<(String, Vec<(String, usize)>, f64)>,
    temperature: f64, // Kelvin
    options: SolverOptions,
}

impl Default for SystemBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl SystemBuilder {
    /// Create an empty builder at 25 °C (298.15 K) with default
    /// [`SolverOptions`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            monomers: Vec::new(),
            complexes: Vec::new(),
            temperature: 298.15,
            options: SolverOptions::default(),
        }
    }

    /// Set the temperature in Kelvin.
    #[must_use]
    pub fn temperature(mut self, t_k: f64) -> Self {
        self.temperature = t_k;
        self
    }

    /// Add a monomer species with a given total concentration (molar).
    #[must_use]
    pub fn monomer(mut self, name: &str, c0: f64) -> Self {
        self.monomers.push((name.to_string(), c0));
        self
    }

    /// Add a complex with the given composition and standard free energy of
    /// formation.
    ///
    /// # Arguments
    ///
    /// * `name` — unique identifier for this complex (must not collide with
    ///   monomer names)
    /// * `composition` — slice of `(monomer_name, count)` pairs
    /// * `dg` — ΔG° in **kcal/mol** at a **1 M standard state**
    #[must_use]
    pub fn complex(
        mut self,
        name: &str,
        composition: &[(&str, usize)],
        dg: f64,
    ) -> Self {
        self.complexes.push((
            name.to_string(),
            composition
                .iter()
                .map(|&(n, c)| (n.to_string(), c))
                .collect(),
            dg,
        ));
        self
    }

    /// Temperature in Kelvin.
    #[must_use]
    pub fn temperature_k(&self) -> f64 {
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

    /// Replace the solver options used by the [`System`] this builder
    /// produces.
    #[must_use]
    pub fn options(mut self, options: SolverOptions) -> Self {
        self.options = options;
        self
    }

    /// Read the currently-configured solver options.
    #[must_use]
    pub fn options_ref(&self) -> &SolverOptions {
        &self.options
    }

    /// Validate the specification, compile it to numerical form, and
    /// return a [`System`] with names attached for ergonomic lookup.
    pub fn build(self) -> Result<System, EquilibriumError> {
        self.options.validate()?;
        let compiled = compile(&self)?;
        let species_index: HashMap<String, usize> = compiled
            .species_names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.clone(), i))
            .collect();
        let names = Names {
            monomer_names: compiled.monomer_names,
            species_names: compiled.species_names,
            species_index,
        };
        let mut inputs = compiled.inputs;
        apply_log_q_clamp(&mut inputs.log_q, self.options.log_q_clamp);
        Ok(System::from_inputs(inputs, Some(names), self.options))
    }

    /// As [`SystemBuilder::build`], but discard names (smaller footprint,
    /// no name-indexed lookup).
    pub fn build_anonymous(self) -> Result<System, EquilibriumError> {
        self.options.validate()?;
        let compiled = compile(&self)?;
        let mut inputs = compiled.inputs;
        apply_log_q_clamp(&mut inputs.log_q, self.options.log_q_clamp);
        Ok(System::from_inputs(inputs, None, self.options))
    }
}

/// Helper: clamp `log_q` entries to `clamp` if `Some`. No-op if `None`.
fn apply_log_q_clamp(log_q: &mut Array1<f64>, clamp: Option<f64>) {
    if let Some(c) = clamp {
        log_q.mapv_inplace(|v| v.min(c));
    }
}

fn stoichiometry_nonzeros(at: &Array2<f64>) -> Vec<Vec<(usize, f64)>> {
    let mut rows = Vec::with_capacity(at.nrows());
    for row in at.rows() {
        let mut nz = Vec::new();
        for (i, &v) in row.iter().enumerate() {
            if v != 0.0 {
                nz.push((i, v));
            }
        }
        rows.push(nz);
    }
    rows
}

// ---------------------------------------------------------------------------
// Internal data structures
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct ProblemInputs {
    /// Stoichiometry in transposed layout, `n_species × n_mon`
    /// (= Aᵀ, stored row-major so that the dominant `Aᵀλ` multiply
    /// reads contiguous rows).
    at: Array2<f64>,
    /// Sparse row view of `at`: for each species, the non-zero
    /// `(monomer_index, stoichiometric_count)` entries. Used by Hessian
    /// assembly to avoid scanning dense zero columns for sparse complexes.
    at_nz: Vec<Vec<(usize, f64)>>,
    /// `n_species`. Zero for the monomer block, `-ΔG/(RT)` for complexes.
    log_q: Array1<f64>,
    /// `n_mon`. Total monomer concentrations.
    c0: Array1<f64>,
}

#[derive(Debug, Clone)]
struct SolutionStorage {
    /// Dual-problem iterate. On a fresh [`System`] this is seeded with
    /// `ln(c0)`; after [`System::solve`] it holds the last converged λ
    /// (used as a warm start for the next solve).
    lambda: Array1<f64>,
    /// `n_species`. Primal concentrations, filled by the solver.
    concentrations: Array1<f64>,
    iterations: usize,
    converged_fully: bool,
}

#[derive(Debug, Clone)]
struct Names {
    monomer_names: Vec<String>,
    species_names: Vec<String>,
    /// Built once at construction time; lookup is O(1).
    /// Monomer-scope lookup is the same index restricted to `[0, n_mon)`.
    species_index: HashMap<String, usize>,
}

/// Pre-allocated working buffers for the solver, reused across iterations
/// and across consecutive [`System::solve`] calls.
#[derive(Debug, Clone)]
struct WorkBuffers {
    c: Array1<f64>,        // n_species: concentrations at current λ
    grad: Array1<f64>,     // n_mon: gradient
    hessian: Array2<f64>,  // n_mon × n_mon: Hessian
    lambda_new: Array1<f64>, // n_mon: candidate iterate
    step: Array1<f64>,     // n_mon: trust-region step
    full_step: Array1<f64>, // n_mon: stagnation-recovery step
    dogleg: DoglegBuffers,
}

#[derive(Debug, Clone)]
struct DoglegBuffers {
    chol_l: Array2<f64>, // n_mon × n_mon: Cholesky factor
    chol_z: Array1<f64>, // n_mon: forward-substitution scratch
    p_n: Array1<f64>,   // n_mon: Newton step
    p_c: Array1<f64>,   // n_mon: Cauchy step
    d: Array1<f64>,     // n_mon: p_n - p_c
    hg: Array1<f64>,    // n_mon: H·g
}

impl WorkBuffers {
    fn new(n_mon: usize, n_species: usize) -> Self {
        Self {
            c: Array1::zeros(n_species),
            grad: Array1::zeros(n_mon),
            hessian: Array2::zeros((n_mon, n_mon)),
            lambda_new: Array1::zeros(n_mon),
            step: Array1::zeros(n_mon),
            full_step: Array1::zeros(n_mon),
            dogleg: DoglegBuffers {
                chol_l: Array2::zeros((n_mon, n_mon)),
                chol_z: Array1::zeros(n_mon),
                p_n: Array1::zeros(n_mon),
                p_c: Array1::zeros(n_mon),
                d: Array1::zeros(n_mon),
                hg: Array1::zeros(n_mon),
            },
        }
    }
}

/// Output of [`compile`]: validated `ProblemInputs` plus the names we
/// resolved along the way (the builder owns them and the caller decides
/// whether to keep them).
struct CompiledSystem {
    inputs: ProblemInputs,
    monomer_names: Vec<String>,
    species_names: Vec<String>,
}

/// Validate a [`SystemBuilder`] and produce its numerical representation.
fn compile(b: &SystemBuilder) -> Result<CompiledSystem, EquilibriumError> {
    if !(b.temperature > 0.0 && b.temperature.is_finite()) {
        return Err(EquilibriumError::InvalidTemperature(b.temperature));
    }

    if b.monomers.is_empty() {
        return Err(EquilibriumError::NoMonomers);
    }

    let mut monomer_idx: HashMap<&str, usize> = HashMap::with_capacity(b.monomers.len());
    let mut monomer_names: Vec<String> = Vec::with_capacity(b.monomers.len());
    let mut monomer_concs: Vec<f64> = Vec::with_capacity(b.monomers.len());
    for (name, conc) in &b.monomers {
        if name.is_empty() {
            return Err(EquilibriumError::EmptyName);
        }
        if !(*conc > 0.0 && conc.is_finite()) {
            return Err(EquilibriumError::InvalidConcentration(*conc));
        }
        if monomer_idx.contains_key(name.as_str()) {
            return Err(EquilibriumError::DuplicateMonomer(name.clone()));
        }
        monomer_idx.insert(name, monomer_names.len());
        monomer_names.push(name.clone());
        monomer_concs.push(*conc);
    }

    let mut all_names: HashMap<&str, ()> = monomer_idx.keys().map(|&k| (k, ())).collect();
    let mut complex_names: Vec<String> = Vec::with_capacity(b.complexes.len());
    let mut resolved_comps: Vec<Vec<(usize, usize)>> =
        Vec::with_capacity(b.complexes.len());
    let mut complex_dgs: Vec<f64> = Vec::with_capacity(b.complexes.len());

    for (name, composition, delta_g) in &b.complexes {
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
            if monomer_idx.contains_key(name.as_str()) {
                return Err(EquilibriumError::DuplicateSpeciesName(name.clone()));
            }
            return Err(EquilibriumError::DuplicateComplex(name.clone()));
        }

        let mut comp: Vec<(usize, usize)> = Vec::new();
        for (monomer_name, count) in composition {
            if *count == 0 {
                return Err(EquilibriumError::ZeroCount(monomer_name.clone()));
            }
            let &idx = monomer_idx
                .get(monomer_name.as_str())
                .ok_or_else(|| EquilibriumError::UnknownMonomer(monomer_name.clone()))?;
            if let Some(entry) = comp.iter_mut().find(
                |(existing_idx, _): &&mut (usize, usize)| *existing_idx == idx,
            ) {
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

    let n_mon = monomer_names.len();
    let n_cplx = complex_names.len();
    let n_species = n_mon + n_cplx;

    let mut at = Array2::zeros((n_species, n_mon));
    for i in 0..n_mon {
        at[[i, i]] = 1.0;
    }
    for (k, comp) in resolved_comps.iter().enumerate() {
        for &(mi, count) in comp {
            at[[n_mon + k, mi]] = count as f64;
        }
    }

    let mut log_q = Array1::zeros(n_species);
    let rt = R * b.temperature;
    for (k, &dg) in complex_dgs.iter().enumerate() {
        log_q[n_mon + k] = -dg / rt;
    }

    let c0 = Array1::from_vec(monomer_concs);

    let mut species_names = Vec::with_capacity(n_species);
    species_names.extend(monomer_names.iter().cloned());
    species_names.extend(complex_names);

    Ok(CompiledSystem {
        inputs: ProblemInputs {
            at_nz: stoichiometry_nonzeros(&at),
            at,
            log_q,
            c0,
        },
        monomer_names,
        species_names,
    })
}

/// Check structural invariants on a raw [`ProblemInputs`].
///
/// Requires: `at` is `n_species × n_mon` with `n_species >= n_mon`; the
/// first `n_mon` rows form the identity matrix; the first `n_mon`
/// entries of `log_q` are zero; all entries are finite; `c0` is
/// strictly positive and finite.
fn validate_inputs(inputs: &ProblemInputs) -> Result<(), EquilibriumError> {
    let n_species = inputs.at.nrows();
    let n_mon = inputs.at.ncols();

    if inputs.log_q.len() != n_species {
        return Err(EquilibriumError::InvalidInputs(format!(
            "log_q length {} does not match stoichiometry rows {n_species}",
            inputs.log_q.len()
        )));
    }
    if inputs.c0.len() != n_mon {
        return Err(EquilibriumError::InvalidInputs(format!(
            "c0 length {} does not match stoichiometry columns {n_mon}",
            inputs.c0.len()
        )));
    }
    if n_mon == 0 {
        return Err(EquilibriumError::NoMonomers);
    }
    if n_species < n_mon {
        return Err(EquilibriumError::InvalidInputs(format!(
            "stoichiometry must have at least n_mon={n_mon} rows (has {n_species})"
        )));
    }
    for i in 0..n_mon {
        for j in 0..n_mon {
            let expected = if i == j { 1.0 } else { 0.0 };
            let v = inputs.at[[i, j]];
            if v != expected {
                return Err(EquilibriumError::InvalidInputs(format!(
                    "stoichiometry[{i}, {j}] = {v} (monomer rows must be identity)"
                )));
            }
        }
    }
    for i in 0..n_mon {
        if inputs.log_q[i] != 0.0 {
            return Err(EquilibriumError::InvalidInputs(format!(
                "log_q[{i}] = {} (monomer entries must be zero)",
                inputs.log_q[i]
            )));
        }
    }
    for i in 0..n_species {
        if !inputs.log_q[i].is_finite() {
            return Err(EquilibriumError::InvalidInputs(format!(
                "log_q[{i}] is not finite"
            )));
        }
        for j in 0..n_mon {
            if !inputs.at[[i, j]].is_finite() {
                return Err(EquilibriumError::InvalidInputs(format!(
                    "stoichiometry[{i}, {j}] is not finite"
                )));
            }
        }
    }
    for i in 0..n_mon {
        let c = inputs.c0[i];
        if !(c > 0.0 && c.is_finite()) {
            return Err(EquilibriumError::InvalidConcentration(c));
        }
    }
    Ok(())
}

fn build_species_index(
    n_mon: usize,
    n_species: usize,
    monomer_names: &[String],
    species_names: &[String],
) -> Result<HashMap<String, usize>, EquilibriumError> {
    if monomer_names.len() != n_mon {
        return Err(EquilibriumError::InvalidInputs(format!(
            "monomer_names length {} != n_mon {n_mon}",
            monomer_names.len()
        )));
    }
    if species_names.len() != n_species {
        return Err(EquilibriumError::InvalidInputs(format!(
            "species_names length {} != n_species {n_species}",
            species_names.len()
        )));
    }
    for i in 0..n_mon {
        if species_names[i] != monomer_names[i] {
            return Err(EquilibriumError::InvalidInputs(format!(
                "species_names[{i}] {:?} != monomer_names[{i}] {:?}",
                species_names[i], monomer_names[i]
            )));
        }
    }
    let mut index = HashMap::with_capacity(n_species);
    for (i, name) in species_names.iter().enumerate() {
        if name.is_empty() {
            return Err(EquilibriumError::EmptyName);
        }
        if index.insert(name.clone(), i).is_some() {
            return Err(EquilibriumError::DuplicateSpeciesName(name.clone()));
        }
    }
    Ok(index)
}

// ---------------------------------------------------------------------------
// System — low-level, indexed, repeatable solver handle
// ---------------------------------------------------------------------------

/// A solver handle owning numerical inputs, work buffers, and the most
/// recent solution.
///
/// Obtain one via [`SystemBuilder::build`], [`SystemBuilder::build_anonymous`],
/// [`System::from_arrays`], or [`System::from_arrays_with_names`]. Call
/// [`System::solve`] to compute concentrations; the returned
/// [`Equilibrium`] borrows `self` and prevents concurrent mutation at
/// compile time.
#[derive(Debug, Clone)]
pub struct System {
    inputs: ProblemInputs,
    work: WorkBuffers,
    solution: SolutionStorage,
    names: Option<Names>,
    options: SolverOptions,
    /// `true` when `solution` reflects the current `inputs`. Flipped off
    /// by any mutating accessor; flipped on by a successful `solve`.
    fresh: bool,
}

impl System {
    /// Assemble a `System` from a validated [`ProblemInputs`] and
    /// optional [`Names`]. Work buffers are allocated here.
    fn from_inputs(
        inputs: ProblemInputs,
        names: Option<Names>,
        options: SolverOptions,
    ) -> Self {
        let n_mon = inputs.c0.len();
        let n_species = inputs.log_q.len();
        let work = WorkBuffers::new(n_mon, n_species);
        let lambda = inputs.c0.mapv(|c| c.ln());
        let solution = SolutionStorage {
            lambda,
            concentrations: Array1::zeros(n_species),
            iterations: 0,
            converged_fully: false,
        };
        Self {
            inputs,
            work,
            solution,
            names,
            options,
            fresh: false,
        }
    }

    /// Construct directly from numerical arrays, using default
    /// [`SolverOptions`].
    ///
    /// `stoichiometry` is `n_species × n_mon` (Aᵀ in the solver's
    /// convention). **Required:** the first `n_mon` rows must be the
    /// identity matrix (free monomers), and the first `n_mon` entries
    /// of `log_q` must be zero. `c0` is `n_mon`-long.
    ///
    /// Validates: shapes consistent, identity block present, monomer
    /// `log_q` zero, all entries finite, `c0` strictly positive.
    pub fn from_arrays(
        stoichiometry: Array2<f64>,
        log_q: Array1<f64>,
        c0: Array1<f64>,
    ) -> Result<Self, EquilibriumError> {
        Self::from_arrays_with_options(stoichiometry, log_q, c0, SolverOptions::default())
    }

    /// As [`System::from_arrays`], with custom [`SolverOptions`]. If
    /// `options.log_q_clamp` is `Some`, the provided `log_q` is
    /// clamped in place before storage.
    pub fn from_arrays_with_options(
        stoichiometry: Array2<f64>,
        mut log_q: Array1<f64>,
        c0: Array1<f64>,
        options: SolverOptions,
    ) -> Result<Self, EquilibriumError> {
        options.validate()?;
        apply_log_q_clamp(&mut log_q, options.log_q_clamp);
        let inputs = ProblemInputs {
            at_nz: stoichiometry_nonzeros(&stoichiometry),
            at: stoichiometry,
            log_q,
            c0,
        };
        validate_inputs(&inputs)?;
        Ok(Self::from_inputs(inputs, None, options))
    }

    /// As [`System::from_arrays`], with names attached for ergonomic
    /// lookup.
    ///
    /// `monomer_names.len()` must equal `n_mon`, `species_names.len()`
    /// must equal `n_species`, and the first `n_mon` species names must
    /// equal `monomer_names`.
    pub fn from_arrays_with_names(
        stoichiometry: Array2<f64>,
        log_q: Array1<f64>,
        c0: Array1<f64>,
        monomer_names: Vec<String>,
        species_names: Vec<String>,
    ) -> Result<Self, EquilibriumError> {
        Self::from_arrays_with_names_and_options(
            stoichiometry,
            log_q,
            c0,
            monomer_names,
            species_names,
            SolverOptions::default(),
        )
    }

    /// Full low-level constructor: arrays + names + options.
    pub fn from_arrays_with_names_and_options(
        stoichiometry: Array2<f64>,
        mut log_q: Array1<f64>,
        c0: Array1<f64>,
        monomer_names: Vec<String>,
        species_names: Vec<String>,
        options: SolverOptions,
    ) -> Result<Self, EquilibriumError> {
        options.validate()?;
        apply_log_q_clamp(&mut log_q, options.log_q_clamp);
        let inputs = ProblemInputs {
            at_nz: stoichiometry_nonzeros(&stoichiometry),
            at: stoichiometry,
            log_q,
            c0,
        };
        validate_inputs(&inputs)?;
        let n_mon = inputs.c0.len();
        let n_species = inputs.log_q.len();
        let species_index =
            build_species_index(n_mon, n_species, &monomer_names, &species_names)?;
        let names = Names {
            monomer_names,
            species_names,
            species_index,
        };
        Ok(Self::from_inputs(inputs, Some(names), options))
    }

    /// Solve. Returns a borrowed view of the current solution.
    ///
    /// If `self` is already fresh (no mutation since the last successful
    /// solve), this returns the cached result with no recomputation.
    pub fn solve(&mut self) -> Result<Equilibrium<'_>, EquilibriumError> {
        if !self.fresh {
            self.options.validate()?;
            validate_inputs(&self.inputs)?;
            solve_inputs_into(
                &self.inputs,
                &mut self.work,
                &mut self.solution,
                &self.options,
            )?;
            self.fresh = true;
        }
        Ok(Equilibrium { sys: self })
    }

    /// Read-only access to the current solver options.
    #[must_use]
    pub fn options(&self) -> &SolverOptions {
        &self.options
    }

    /// Mutable access to the solver options. The freshness flag is
    /// flipped off, since changes to tolerances or clamps can alter
    /// what the next `solve()` would return, even when `inputs` are
    /// unchanged.
    ///
    /// This does **not** apply `log_q_clamp` retroactively — changing
    /// the clamp through `options_mut()` has no effect on the stored
    /// `log_q` values. Use [`System::set_options`] if you want the
    /// clamp re-applied.
    pub fn options_mut(&mut self) -> &mut SolverOptions {
        self.fresh = false;
        &mut self.options
    }

    /// Replace the solver options in one shot.
    ///
    /// Validates the new options, re-applies `log_q_clamp` to the
    /// stored `log_q` (if `Some`), and flips the freshness flag off.
    ///
    /// Note on clamping: re-applying a tighter clamp is destructive —
    /// entries already clamped to a higher bound cannot be recovered.
    /// If you need to swap between clamp settings, rebuild from source
    /// inputs rather than toggling `log_q_clamp`.
    pub fn set_options(&mut self, options: SolverOptions) -> Result<(), EquilibriumError> {
        options.validate()?;
        apply_log_q_clamp(&mut self.inputs.log_q, options.log_q_clamp);
        self.options = options;
        self.fresh = false;
        Ok(())
    }

    /// Access the last solution, if the system is still fresh.
    #[must_use]
    pub fn last_solution(&self) -> Option<Equilibrium<'_>> {
        if self.fresh {
            Some(Equilibrium { sys: self })
        } else {
            None
        }
    }

    /// Mutable view into `c0`. Obtaining it flips the freshness flag
    /// unconditionally; callers that want to read without mutating
    /// should use [`System::c0`] instead.
    pub fn c0_mut(&mut self) -> ArrayViewMut1<'_, f64> {
        self.fresh = false;
        self.inputs.c0.view_mut()
    }

    /// Mutable view into `log_q`. See [`System::c0_mut`] for the
    /// freshness note.
    pub fn log_q_mut(&mut self) -> ArrayViewMut1<'_, f64> {
        self.fresh = false;
        self.inputs.log_q.view_mut()
    }

    /// Set a single `c0` entry and flip the freshness flag.
    ///
    /// The value is validated by [`System::solve`] before numerical work
    /// begins. Call [`System::validate`] explicitly if you want to check
    /// mutated inputs before solving.
    pub fn set_c0(&mut self, idx: usize, value: f64) {
        self.fresh = false;
        self.inputs.c0[idx] = value;
    }

    /// Set a single `log_q` entry and flip the freshness flag.
    ///
    /// The value is validated by [`System::solve`] before numerical work
    /// begins. Call [`System::validate`] explicitly if you want to check
    /// mutated inputs before solving.
    pub fn set_log_q(&mut self, idx: usize, value: f64) {
        self.fresh = false;
        self.inputs.log_q[idx] = value;
    }

    /// Read-only view of `c0`.
    #[must_use]
    pub fn c0(&self) -> ArrayView1<'_, f64> {
        self.inputs.c0.view()
    }

    /// Read-only view of `log_q`.
    #[must_use]
    pub fn log_q(&self) -> ArrayView1<'_, f64> {
        self.inputs.log_q.view()
    }

    /// Read-only view of the stoichiometry matrix (transposed layout,
    /// `n_species × n_mon`).
    #[must_use]
    pub fn stoichiometry(&self) -> ArrayView2<'_, f64> {
        self.inputs.at.view()
    }

    /// Number of monomer species.
    #[must_use]
    pub fn n_monomers(&self) -> usize {
        self.inputs.c0.len()
    }

    /// Total number of species (monomers + complexes).
    #[must_use]
    pub fn n_species(&self) -> usize {
        self.inputs.log_q.len()
    }

    /// Look up a species index by name. Returns `None` if the system
    /// was constructed without names, or the name is unknown.
    #[must_use]
    pub fn species_index(&self, name: &str) -> Option<usize> {
        self.names
            .as_ref()
            .and_then(|n| n.species_index.get(name).copied())
    }

    /// Look up a monomer index by name. Returns `None` if the name
    /// belongs to a complex, the system has no names, or the name is
    /// unknown.
    #[must_use]
    pub fn monomer_index(&self, name: &str) -> Option<usize> {
        let idx = self.species_index(name)?;
        (idx < self.inputs.c0.len()).then_some(idx)
    }

    /// Look up a species name by index. Returns `None` if the system
    /// was constructed without names, or the index is out of range.
    #[must_use]
    pub fn species_name(&self, idx: usize) -> Option<&str> {
        self.names
            .as_ref()
            .and_then(|n| n.species_names.get(idx).map(String::as_str))
    }

    /// Look up a monomer name by index.
    #[must_use]
    pub fn monomer_name(&self, idx: usize) -> Option<&str> {
        if idx >= self.inputs.c0.len() {
            return None;
        }
        self.names
            .as_ref()
            .and_then(|n| n.monomer_names.get(idx).map(String::as_str))
    }

    /// Whether this system has names attached.
    #[must_use]
    pub fn has_names(&self) -> bool {
        self.names.is_some()
    }

    /// Re-run structural invariant checks against the current inputs.
    /// Useful after caller-driven mutation through [`c0_mut`] or
    /// [`log_q_mut`] where the invariants could have been violated.
    ///
    /// [`c0_mut`]: System::c0_mut
    /// [`log_q_mut`]: System::log_q_mut
    pub fn validate(&self) -> Result<(), EquilibriumError> {
        validate_inputs(&self.inputs)
    }
}

// ---------------------------------------------------------------------------
// Equilibrium — borrowed view of a solved System
// ---------------------------------------------------------------------------

/// A borrowed view of the solution stored inside a [`System`].
///
/// The view is tied to the lifetime of the `&System` it borrows from,
/// which prevents any mutating accessor from firing while the view is
/// alive — the compiler rejects such code. To keep a snapshot past the
/// next mutation, copy the data out (e.g. `.concentrations().to_owned()`).
#[derive(Debug, Clone, Copy)]
pub struct Equilibrium<'a> {
    sys: &'a System,
}

impl<'a> Equilibrium<'a> {
    /// All concentrations, monomers first then complexes.
    #[must_use]
    pub fn concentrations(&self) -> ArrayView1<'a, f64> {
        self.sys.solution.concentrations.view()
    }

    /// Free monomer concentrations (first `n_mon` entries).
    #[must_use]
    pub fn free_monomers(&self) -> ArrayView1<'a, f64> {
        let n_mon = self.sys.n_monomers();
        self.sys.solution.concentrations.slice(s![..n_mon])
    }

    /// Complex concentrations (remaining entries).
    #[must_use]
    pub fn complexes(&self) -> ArrayView1<'a, f64> {
        let n_mon = self.sys.n_monomers();
        self.sys.solution.concentrations.slice(s![n_mon..])
    }

    /// Number of solver iterations used to produce this solution.
    ///
    /// When [`System::solve`] is called on an already-fresh system, it
    /// returns the cached result and this value reflects the earlier
    /// compute, not the no-op call.
    #[must_use]
    pub fn iterations(&self) -> usize {
        self.sys.solution.iterations
    }

    /// Whether the solver reached full convergence (RTOL = 1e-7).
    ///
    /// `false` means results were accepted at the relaxed tolerance
    /// (RTOL = 1e-4) after stagnation at f64 precision limits.
    #[must_use]
    pub fn converged_fully(&self) -> bool {
        self.sys.solution.converged_fully
    }

    /// Look up a concentration by name. Returns `None` if the underlying
    /// [`System`] has no names, or the name is unknown.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<f64> {
        self.sys
            .species_index(name)
            .map(|i| self.sys.solution.concentrations[i])
    }

    /// Look up a concentration by index. Panics on out-of-bounds.
    #[must_use]
    pub fn at(&self, idx: usize) -> f64 {
        self.sys.solution.concentrations[idx]
    }
}

impl Index<&str> for Equilibrium<'_> {
    type Output = f64;
    fn index(&self, name: &str) -> &f64 {
        let idx = self
            .sys
            .species_index(name)
            .unwrap_or_else(|| panic!("unknown species: {name}"));
        &self.sys.solution.concentrations[idx]
    }
}

impl Index<usize> for Equilibrium<'_> {
    type Output = f64;
    fn index(&self, idx: usize) -> &f64 {
        &self.sys.solution.concentrations[idx]
    }
}

// ---------------------------------------------------------------------------
// Solve entry point
// ---------------------------------------------------------------------------

fn solve_inputs_into(
    inputs: &ProblemInputs,
    work: &mut WorkBuffers,
    solution: &mut SolutionStorage,
    options: &SolverOptions,
) -> Result<(), EquilibriumError> {
    let n_species = inputs.at.nrows();
    let n_mon = inputs.at.ncols();

    // Short-circuit: no complexes. Primal concentrations are just c0.
    if n_species == n_mon {
        solution.concentrations.assign(&inputs.c0);
        for i in 0..n_mon {
            solution.lambda[i] = inputs.c0[i].ln();
        }
        solution.iterations = 0;
        solution.converged_fully = true;
        return Ok(());
    }

    let convergence = solve_dual_into(
        &inputs.at,
        &inputs.at_nz,
        &inputs.log_q,
        &inputs.c0,
        &mut solution.lambda,
        work,
        options,
    )?;

    // `work.c` holds the concentrations at `solution.lambda` after the
    // solver's final `evaluate_into`. Copy them out.
    solution.concentrations.assign(&work.c);

    match convergence {
        SolverConvergence::Full { iterations } => {
            solution.iterations = iterations;
            solution.converged_fully = true;
        }
        SolverConvergence::Relaxed { iterations, .. } => {
            solution.iterations = iterations;
            solution.converged_fully = false;
        }
    }

    #[cfg(debug_assertions)]
    debug_validate_solution(inputs, &solution.concentrations);

    Ok(())
}

#[cfg(debug_assertions)]
fn debug_validate_solution(inputs: &ProblemInputs, concentrations: &Array1<f64>) {
    let n_species = inputs.at.nrows();
    let n_mon = inputs.at.ncols();

    for i in 0..n_mon {
        let total: f64 = (0..n_species)
            .map(|j| inputs.at[[j, i]] * concentrations[j])
            .sum();
        debug_assert!(
            (total - inputs.c0[i]).abs() < 1e-2 * inputs.c0[i] + 1e-30,
            "mass conservation violated for monomer {i}: {total} != {}",
            inputs.c0[i]
        );
    }

    for k in n_mon..n_species {
        let mut log_expected = inputs.log_q[k];
        for i in 0..n_mon {
            let count = inputs.at[[k, i]];
            if count != 0.0 {
                log_expected += count * concentrations[i].ln();
            }
        }
        let log_actual = concentrations[k].ln();
        let log_err = (log_actual - log_expected).abs();
        debug_assert!(
            log_err < 1e-2 * (1.0 + log_expected.abs()),
            "equilibrium condition violated for species index {k}: \
             log(actual)={log_actual} != log(expected)={log_expected} (log_err={log_err})"
        );
    }
}

// ---------------------------------------------------------------------------
// Linear algebra helpers
// ---------------------------------------------------------------------------

/// L2 norm of a vector.
fn norm(v: &Array1<f64>) -> f64 {
    v.dot(v).sqrt()
}

fn quadratic_form(h: &Array2<f64>, p: &Array1<f64>) -> f64 {
    let n = p.len();
    let mut total = 0.0;
    for i in 0..n {
        let pi = p[i];
        let mut row_dot = 0.0;
        for j in 0..n {
            row_dot += h[[i, j]] * p[j];
        }
        total += pi * row_dot;
    }
    total
}

/// Cholesky factorization of a symmetric positive-definite matrix.
/// Writes the Newton step `p = -H⁻¹g`, returning `false` if `H` is not
/// positive definite.
fn cholesky_solve_into(
    h: &Array2<f64>,
    g: &Array1<f64>,
    l: &mut Array2<f64>,
    z: &mut Array1<f64>,
    p: &mut Array1<f64>,
) -> bool {
    let n = g.len();
    // Compute L (lower triangular, H = LLᵀ)
    l.fill(0.0);
    for i in 0..n {
        for j in 0..=i {
            let mut sum = h[[i, j]];
            for k in 0..j {
                sum -= l[[i, k]] * l[[j, k]];
            }
            if i == j {
                if sum <= 0.0 {
                    return false;
                }
                l[[i, j]] = sum.sqrt();
            } else {
                l[[i, j]] = sum / l[[j, j]];
            }
        }
    }
    // Forward solve: Lz = -g
    for i in 0..n {
        let mut sum = -g[i];
        for k in 0..i {
            sum -= l[[i, k]] * z[k];
        }
        z[i] = sum / l[[i, i]];
    }
    // Back solve: Lᵀp = z
    for i in (0..n).rev() {
        let mut sum = z[i];
        for k in (i + 1)..n {
            sum -= l[[k, i]] * p[k];
        }
        p[i] = sum / l[[i, i]];
    }
    true
}

// ---------------------------------------------------------------------------
// Trust-region Newton solver on the dual
// ---------------------------------------------------------------------------

/// Evaluate the dual objective, gradient, and Hessian at λ, writing into
/// the supplied buffers. Returns f.
///
/// `at` is `n_species × n_mon` (= Aᵀ, row-major so that the `Aᵀλ` multiply
/// reads contiguous rows).
fn evaluate_into(
    at: &Array2<f64>,
    at_nz: &[Vec<(usize, f64)>],
    log_q: &Array1<f64>,
    c0: &Array1<f64>,
    lambda: &Array1<f64>,
    c: &mut Array1<f64>,
    grad: &mut Array1<f64>,
    hessian: &mut Array2<f64>,
    log_c_clamp: f64,
) -> f64 {
    // c = exp(min(log_q + Aᵀλ, log_c_clamp))
    ndarray::linalg::general_mat_vec_mul(1.0, at, lambda, 0.0, c);
    *c += log_q;
    c.mapv_inplace(|lc| lc.min(log_c_clamp).exp());

    // f = -λᵀc⁰ + Σ_j c_j
    let f = -lambda.dot(c0) + c.sum();

    // grad = -c⁰ + Aᵀᵀ·c
    ndarray::linalg::general_mat_vec_mul(1.0, &at.t(), c, 0.0, grad);
    *grad -= c0;

    // H = A diag(c) Aᵀ (sparse loop)
    hessian.fill(0.0);
    for (k, row) in at_nz.iter().enumerate() {
        let ck = c[k];
        for (a_pos, &(i, aik)) in row.iter().enumerate() {
            for &(j, ajk) in &row[a_pos..] {
                let val = aik * ck * ajk;
                hessian[[i, j]] += val;
                if i != j {
                    hessian[[j, i]] += val;
                }
            }
        }
    }

    f
}

/// Evaluate only the dual objective at λ (no gradient or Hessian).
fn evaluate_objective_into(
    at: &Array2<f64>,
    log_q: &Array1<f64>,
    c0: &Array1<f64>,
    lambda: &Array1<f64>,
    c_buf: &mut Array1<f64>,
    log_c_clamp: f64,
) -> f64 {
    ndarray::linalg::general_mat_vec_mul(1.0, at, lambda, 0.0, c_buf);
    *c_buf += log_q;
    let c_sum: f64 = c_buf.iter().map(|&lc| lc.min(log_c_clamp).exp()).sum();

    -lambda.dot(c0) + c_sum
}

/// Compute the dog-leg step within a trust region of radius `delta`, writing
/// into `out` and returning `||out||`.
fn dogleg_step_into(
    grad: &Array1<f64>,
    hessian: &Array2<f64>,
    delta: f64,
    out: &mut Array1<f64>,
    buffers: &mut DoglegBuffers,
) -> f64 {
    if !cholesky_solve_into(
        hessian,
        grad,
        &mut buffers.chol_l,
        &mut buffers.chol_z,
        &mut buffers.p_n,
    ) {
        // Defensive: H = A diag(c) Aᵀ with all c > 0, so H is always
        // positive-definite and Cholesky cannot fail in practice.
        let g_norm = norm(grad);
        if g_norm == 0.0 {
            out.fill(0.0);
            return 0.0;
        }
        out.assign(grad);
        out.mapv_inplace(|v| -(delta / g_norm) * v);
        return delta;
    }

    let p_n_norm = norm(&buffers.p_n);

    if p_n_norm <= delta {
        out.assign(&buffers.p_n);
        return p_n_norm;
    }

    let gtg = grad.dot(grad);
    ndarray::linalg::general_mat_vec_mul(1.0, hessian, grad, 0.0, &mut buffers.hg);
    let gthg = grad.dot(&buffers.hg);

    if gthg < 1e-30 * gtg {
        let g_norm = norm(grad);
        out.assign(grad);
        out.mapv_inplace(|v| -(delta / g_norm) * v);
        return delta;
    }

    buffers.p_c.assign(grad);
    buffers.p_c.mapv_inplace(|v| -(gtg / gthg) * v);
    let p_c_norm = norm(&buffers.p_c);

    if p_c_norm >= delta {
        out.assign(&buffers.p_c);
        out.mapv_inplace(|v| (delta / p_c_norm) * v);
        return delta;
    }

    buffers.d.assign(&buffers.p_n);
    buffers.d -= &buffers.p_c;
    let dd = buffers.d.dot(&buffers.d);
    let pd = buffers.p_c.dot(&buffers.d);
    let pp = buffers.p_c.dot(&buffers.p_c);
    let discriminant = (pd * pd - dd * (pp - delta * delta)).max(0.0);
    let tau = ((-pd + discriminant.sqrt()) / dd).clamp(0.0, 1.0);

    out.assign(&buffers.p_c);
    for (o, &d) in out.iter_mut().zip(buffers.d.iter()) {
        *o += tau * d;
    }
    norm(out)
}

/// Convenience wrapper for tests.
#[cfg(test)]
fn dogleg_step(grad: &Array1<f64>, hessian: &Array2<f64>, delta: f64) -> Array1<f64> {
    let n = grad.len();
    let mut out = Array1::zeros(n);
    let mut buffers = DoglegBuffers {
        chol_l: Array2::zeros((n, n)),
        chol_z: Array1::zeros(n),
        p_n: Array1::zeros(n),
        p_c: Array1::zeros(n),
        d: Array1::zeros(n),
        hg: Array1::zeros(n),
    };
    dogleg_step_into(grad, hessian, delta, &mut out, &mut buffers);
    out
}

/// Convergence quality returned by the dual solver.
#[derive(Debug, Clone, Copy, PartialEq)]
enum SolverConvergence {
    /// Converged to full tolerance (RTOL = 1e-7).
    Full { iterations: usize },
    /// Accepted at relaxed tolerance (RTOL = 1e-4) due to stagnation.
    Relaxed {
        iterations: usize,
        gradient_norm: f64,
    },
}

/// Solve the dual problem. On entry, `lambda` holds the warm start; on
/// exit, it holds the converged optimum. Convergence metadata is
/// returned; primal concentrations at the final `lambda` are left in
/// `work.c`.
fn solve_dual_into(
    at: &Array2<f64>,
    at_nz: &[Vec<(usize, f64)>],
    log_q: &Array1<f64>,
    c0: &Array1<f64>,
    lambda: &mut Array1<f64>,
    work: &mut WorkBuffers,
    opts: &SolverOptions,
) -> Result<SolverConvergence, EquilibriumError> {
    let max_iter = opts.max_iterations;
    let atol = opts.gradient_abs_tol;
    let rtol = opts.gradient_rel_tol;
    let stag_atol = opts.relaxed_gradient_abs_tol;
    let stag_rtol = opts.relaxed_gradient_rel_tol;
    let eta = opts.step_accept_threshold;
    let delta_max = opts.max_trust_region_radius;
    let shrink_rho = opts.trust_region_shrink_rho;
    let grow_rho = opts.trust_region_grow_rho;
    let shrink_scale = opts.trust_region_shrink_scale;
    let grow_scale = opts.trust_region_grow_scale;
    let stag_threshold = opts.stagnation_threshold;
    let log_c_clamp = opts.log_c_clamp;

    // Split-borrow the work buffers so we can pass disjoint &mut
    // references into evaluate_into alongside &lambda_new.
    let WorkBuffers {
        c,
        grad,
        hessian,
        lambda_new,
        step,
        full_step,
        dogleg,
    } = work;

    let mut delta = opts.initial_trust_region_radius;
    let mut stagnation = 0u32;

    // Env-gated per-iteration trace (diagnostic only; zero cost when unset).
    let trace = std::env::var_os("EQUICONC_TRACE").is_some();
    if trace {
        eprintln!("# iter\tf\tgrad_inf_norm\tdelta\tstag");
    }

    for iter in 0..max_iter {
        let f = evaluate_into(at, at_nz, log_q, c0, lambda, c, grad, hessian, log_c_clamp);

        if trace {
            let g_inf = grad.iter().map(|g: &f64| g.abs()).fold(0.0_f64, f64::max);
            eprintln!("{iter}\t{f:.15e}\t{g_inf:.6e}\t{delta:.3e}\t{stagnation}");
        }

        if grad
            .iter()
            .zip(c0.iter())
            .all(|(&g, &cv)| g.abs() < atol + rtol * cv)
        {
            return Ok(SolverConvergence::Full {
                iterations: iter + 1,
            });
        }

        let p_norm = dogleg_step_into(grad, hessian, delta, step, dogleg);

        lambda_new.assign(lambda);
        *lambda_new += &*step;
        let f_new = evaluate_objective_into(at, log_q, c0, lambda_new, c, log_c_clamp);

        let actual_reduction = f - f_new;
        let predicted_reduction =
            -(grad.dot(&*step) + 0.5 * quadratic_form(hessian, &*step));

        if actual_reduction < 4.0 * f64::EPSILON * f.abs().max(1.0) {
            stagnation += 1;
        } else {
            stagnation = 0;
        }

        if stagnation >= stag_threshold {
            dogleg_step_into(grad, hessian, delta_max, full_step, dogleg);
            lambda_new.assign(lambda);
            *lambda_new += &*full_step;
            let f_full =
                evaluate_into(at, at_nz, log_q, c0, lambda_new, c, grad, hessian, log_c_clamp);
            if f_full <= f {
                std::mem::swap(lambda, lambda_new);

                if grad
                    .iter()
                    .zip(c0.iter())
                    .all(|(&g, &cv)| g.abs() < atol + rtol * cv)
                {
                    return Ok(SolverConvergence::Full {
                        iterations: iter + 1,
                    });
                }
                if grad
                    .iter()
                    .zip(c0.iter())
                    .all(|(&g, &cv)| g.abs() < stag_atol + stag_rtol * cv)
                {
                    return Ok(SolverConvergence::Relaxed {
                        iterations: iter + 1,
                        gradient_norm: norm(grad),
                    });
                }

                delta = norm(full_step);
                stagnation = 0;
                continue;
            }
            // Recovery failed: re-evaluate at (unchanged) lambda so that
            // grad/c buffers match.
            evaluate_into(at, at_nz, log_q, c0, lambda, c, grad, hessian, log_c_clamp);
            if grad
                .iter()
                .zip(c0.iter())
                .all(|(&g, &cv)| g.abs() < stag_atol + stag_rtol * cv)
            {
                return Ok(SolverConvergence::Relaxed {
                    iterations: iter + 1,
                    gradient_norm: norm(grad),
                });
            }
            return Err(EquilibriumError::ConvergenceFailure {
                iterations: iter + 1,
                gradient_norm: norm(grad),
            });
        }

        let rho = if predicted_reduction.abs() < 1e-30 {
            if actual_reduction >= 0.0 { 1.0 } else { 0.0 }
        } else {
            actual_reduction / predicted_reduction
        };

        if rho < shrink_rho {
            delta *= shrink_scale;
        } else if rho > grow_rho && (p_norm - delta).abs() < 1e-10 * delta {
            delta = (grow_scale * delta).min(delta_max);
        }

        if rho > eta {
            std::mem::swap(lambda, lambda_new);
        }
    }

    evaluate_into(at, at_nz, log_q, c0, lambda, c, grad, hessian, log_c_clamp);
    Err(EquilibriumError::ConvergenceFailure {
        iterations: max_iter,
        gradient_norm: norm(grad),
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const NM: f64 = 1e-9;

    fn simple_builder() -> SystemBuilder {
        SystemBuilder::new()
            .monomer("A", 100.0 * NM)
            .monomer("B", 100.0 * NM)
            .complex("AB", &[("A", 1), ("B", 1)], -10.0)
    }

    #[test]
    fn no_complexes() {
        let mut sys = SystemBuilder::new()
            .monomer("A", 50e-9)
            .monomer("B", 100e-9)
            .build()
            .unwrap();
        let eq = sys.solve().unwrap();
        assert_eq!(eq.free_monomers()[0], 50e-9);
        assert_eq!(eq.free_monomers()[1], 100e-9);
        assert!(eq.complexes().is_empty());
    }

    #[test]
    fn simple_dimerization() {
        let c0 = 100.0 * NM;
        let dg = -10.0;
        let mut sys = SystemBuilder::new()
            .monomer("A", c0)
            .monomer("B", c0)
            .complex("AB", &[("A", 1), ("B", 1)], dg)
            .build()
            .unwrap();
        let eq = sys.solve().unwrap();

        let rt = R * 298.15;
        let k = (-dg / rt).exp();
        let x = ((2.0 * k * c0 + 1.0) - (4.0 * k * c0 + 1.0).sqrt()) / (2.0 * k);
        let free = c0 - x;

        assert!((eq.get("A").unwrap() - free).abs() < 1e-12);
        assert!((eq.get("B").unwrap() - free).abs() < 1e-12);
        assert!((eq.get("AB").unwrap() - x).abs() < 1e-12);
    }

    #[test]
    fn homotrimerization() {
        let c0 = 1e-6;
        let mut sys = SystemBuilder::new()
            .monomer("A", c0)
            .complex("AAA", &[("A", 3)], -15.0)
            .build()
            .unwrap();
        let eq = sys.solve().unwrap();
        let free_a = eq.get("A").unwrap();
        let aaa = eq.get("AAA").unwrap();
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
        let mut sys = SystemBuilder::new()
            .monomer("a", c0)
            .monomer("b", c0)
            .monomer("c", c0)
            .complex("ab", &[("a", 1), ("b", 1)], -10.0)
            .complex("aaa", &[("a", 3)], -15.0)
            .complex("bc", &[("b", 1), ("c", 1)], -12.0)
            .build()
            .unwrap();
        let eq = sys.solve().unwrap();

        let a = eq.get("a").unwrap();
        let b = eq.get("b").unwrap();
        let c = eq.get("c").unwrap();
        let ab = eq.get("ab").unwrap();
        let aaa = eq.get("aaa").unwrap();
        let bc = eq.get("bc").unwrap();

        assert!((a + ab + 3.0 * aaa - c0).abs() < 1e-5 * c0);
        assert!((b + ab + bc - c0).abs() < 1e-5 * c0);
        assert!((c + bc - c0).abs() < 1e-5 * c0);
    }

    #[test]
    fn strong_binding() {
        let c0 = 100.0 * NM;
        let mut sys = SystemBuilder::new()
            .monomer("A", c0)
            .monomer("B", c0)
            .complex("AB", &[("A", 1), ("B", 1)], -30.0)
            .build()
            .unwrap();
        let eq = sys.solve().unwrap();

        let ab = eq.get("AB").unwrap();
        assert!((ab - c0).abs() < 1e-6 * c0, "[AB] = {ab}, expected ≈ {c0}");
        assert!(eq.get("A").unwrap() < 1e-12);
        assert!(eq.get("B").unwrap() < 1e-12);
    }

    #[test]
    fn asymmetric_concentrations() {
        let mut sys = SystemBuilder::new()
            .monomer("A", 200.0 * NM)
            .monomer("B", 100.0 * NM)
            .complex("AB", &[("A", 1), ("B", 1)], -10.0)
            .build()
            .unwrap();
        let eq = sys.solve().unwrap();

        let a = eq.get("A").unwrap();
        let b = eq.get("B").unwrap();
        let ab = eq.get("AB").unwrap();

        assert!(ab <= 100.0 * NM + 1e-12);
        assert!((a + ab - 200.0 * NM).abs() < 1e-8 * 200.0 * NM);
        assert!((b + ab - 100.0 * NM).abs() < 1e-8 * 100.0 * NM);
    }

    #[test]
    fn negative_concentration() {
        let err = SystemBuilder::new()
            .monomer("A", -1e-9)
            .build()
            .unwrap_err();
        assert!(matches!(err, EquilibriumError::InvalidConcentration(c) if c == -1e-9));
    }

    #[test]
    fn zero_concentration() {
        let err = SystemBuilder::new()
            .monomer("A", 0.0)
            .build()
            .unwrap_err();
        assert!(matches!(err, EquilibriumError::InvalidConcentration(c) if c == 0.0));
    }

    #[test]
    fn nan_concentration() {
        let err = SystemBuilder::new()
            .monomer("A", f64::NAN)
            .build()
            .unwrap_err();
        assert!(matches!(err, EquilibriumError::InvalidConcentration(_)));
    }

    #[test]
    fn inf_concentration() {
        let err = SystemBuilder::new()
            .monomer("A", f64::INFINITY)
            .build()
            .unwrap_err();
        assert!(matches!(err, EquilibriumError::InvalidConcentration(_)));
    }

    #[test]
    fn zero_temperature() {
        let err = SystemBuilder::new()
            .temperature(0.0)
            .monomer("A", 1e-9)
            .build()
            .unwrap_err();
        assert!(matches!(err, EquilibriumError::InvalidTemperature(t) if t == 0.0));
    }

    #[test]
    fn negative_temperature() {
        let err = SystemBuilder::new()
            .temperature(-100.0)
            .monomer("A", 1e-9)
            .build()
            .unwrap_err();
        assert!(matches!(err, EquilibriumError::InvalidTemperature(_)));
    }

    #[test]
    fn duplicate_monomer() {
        let err = SystemBuilder::new()
            .monomer("A", 1e-9)
            .monomer("A", 2e-9)
            .build()
            .unwrap_err();
        assert!(matches!(err, EquilibriumError::DuplicateMonomer(ref n) if n == "A"));
    }

    #[test]
    fn duplicate_complex() {
        let err = SystemBuilder::new()
            .monomer("A", 1e-9)
            .monomer("B", 1e-9)
            .complex("AB", &[("A", 1), ("B", 1)], -10.0)
            .complex("AB", &[("A", 1), ("B", 1)], -12.0)
            .build()
            .unwrap_err();
        assert!(matches!(err, EquilibriumError::DuplicateComplex(ref n) if n == "AB"));
    }

    #[test]
    fn complex_name_collides_with_monomer() {
        let err = SystemBuilder::new()
            .monomer("A", 1e-9)
            .monomer("B", 1e-9)
            .complex("A", &[("A", 1), ("B", 1)], -10.0)
            .build()
            .unwrap_err();
        assert!(matches!(err, EquilibriumError::DuplicateSpeciesName(ref n) if n == "A"));
    }

    #[test]
    fn zero_count_stoichiometry() {
        let err = SystemBuilder::new()
            .monomer("A", 1e-9)
            .monomer("B", 1e-9)
            .complex("AB", &[("A", 0), ("B", 1)], -10.0)
            .build()
            .unwrap_err();
        assert!(matches!(err, EquilibriumError::ZeroCount(ref n) if n == "A"));
    }

    #[test]
    fn empty_composition() {
        let err = SystemBuilder::new()
            .monomer("A", 1e-9)
            .complex("X", &[], -10.0)
            .build()
            .unwrap_err();
        assert!(matches!(err, EquilibriumError::EmptyComposition));
    }

    #[test]
    fn unknown_monomer() {
        let err = SystemBuilder::new()
            .monomer("A", 1e-9)
            .complex("AB", &[("A", 1), ("Z", 1)], -10.0)
            .build()
            .unwrap_err();
        assert!(matches!(err, EquilibriumError::UnknownMonomer(ref n) if n == "Z"));
    }

    #[test]
    fn get_unknown_name() {
        let mut sys = SystemBuilder::new()
            .monomer("A", 1e-9)
            .build()
            .unwrap();
        let eq = sys.solve().unwrap();
        assert!(eq.get("nonexistent").is_none());
    }

    #[test]
    fn error_display() {
        let cases: Vec<(EquilibriumError, &str)> = vec![
            (EquilibriumError::NoMonomers, "no monomers"),
            (EquilibriumError::UnknownMonomer("X".into()), "unknown monomer"),
            (EquilibriumError::EmptyComposition, "empty composition"),
            (
                EquilibriumError::InvalidConcentration(-1.0),
                "invalid concentration",
            ),
            (
                EquilibriumError::InvalidTemperature(-1.0),
                "invalid temperature",
            ),
            (
                EquilibriumError::DuplicateMonomer("A".into()),
                "duplicate monomer",
            ),
            (
                EquilibriumError::DuplicateComplex("AB".into()),
                "duplicate complex",
            ),
            (
                EquilibriumError::ZeroCount("A".into()),
                "zero stoichiometric count",
            ),
            (EquilibriumError::InvalidDeltaG(f64::NAN), "invalid delta_g"),
            (EquilibriumError::EmptyName, "must not be empty"),
            (
                EquilibriumError::DuplicateSpeciesName("A".into()),
                "species name already in use",
            ),
            (
                EquilibriumError::InvalidInputs("mismatch".into()),
                "invalid inputs",
            ),
            (
                EquilibriumError::InvalidOptions("shrink_rho >= grow_rho".into()),
                "invalid solver options",
            ),
            (
                EquilibriumError::ConvergenceFailure {
                    iterations: 100,
                    gradient_norm: 1.0,
                },
                "did not converge",
            ),
        ];
        for (err, expected_substr) in &cases {
            let msg = err.to_string();
            assert!(
                msg.contains(expected_substr),
                "expected {:?} to contain {:?}, got {:?}",
                err,
                expected_substr,
                msg,
            );
        }
    }

    #[test]
    fn duplicate_monomer_in_composition_sums() {
        let mut sys_dup = SystemBuilder::new()
            .monomer("A", 1e-6)
            .complex("A3", &[("A", 1), ("A", 2)], -15.0)
            .build()
            .unwrap();
        let eq_dup = sys_dup.solve().unwrap();
        let a3_dup = eq_dup.get("A3").unwrap();
        let a_dup = eq_dup.get("A").unwrap();

        let mut sys_merged = SystemBuilder::new()
            .monomer("A", 1e-6)
            .complex("A3", &[("A", 3)], -15.0)
            .build()
            .unwrap();
        let eq_merged = sys_merged.solve().unwrap();
        assert!((a3_dup - eq_merged.get("A3").unwrap()).abs() < 1e-20);
        assert!((a_dup - eq_merged.get("A").unwrap()).abs() < 1e-20);
    }

    #[test]
    fn nan_delta_g() {
        let err = SystemBuilder::new()
            .monomer("A", 1e-9)
            .monomer("B", 1e-9)
            .complex("AB", &[("A", 1), ("B", 1)], f64::NAN)
            .build()
            .unwrap_err();
        assert!(matches!(err, EquilibriumError::InvalidDeltaG(_)));
    }

    #[test]
    fn inf_delta_g() {
        let err = SystemBuilder::new()
            .monomer("A", 1e-9)
            .monomer("B", 1e-9)
            .complex("AB", &[("A", 1), ("B", 1)], f64::INFINITY)
            .build()
            .unwrap_err();
        assert!(matches!(err, EquilibriumError::InvalidDeltaG(_)));
    }

    #[test]
    fn neg_inf_delta_g() {
        let err = SystemBuilder::new()
            .monomer("A", 1e-9)
            .monomer("B", 1e-9)
            .complex("AB", &[("A", 1), ("B", 1)], f64::NEG_INFINITY)
            .build()
            .unwrap_err();
        assert!(matches!(err, EquilibriumError::InvalidDeltaG(_)));
    }

    #[test]
    fn empty_monomer_name() {
        let err = SystemBuilder::new()
            .monomer("", 1e-9)
            .build()
            .unwrap_err();
        assert!(matches!(err, EquilibriumError::EmptyName));
    }

    #[test]
    fn empty_complex_name() {
        let err = SystemBuilder::new()
            .monomer("A", 1e-9)
            .complex("", &[("A", 1)], -10.0)
            .build()
            .unwrap_err();
        assert!(matches!(err, EquilibriumError::EmptyName));
    }

    #[test]
    fn system_accessors() {
        let sys = simple_builder().build().unwrap();
        assert_eq!(sys.n_monomers(), 2);
        assert_eq!(sys.n_species(), 3);
        assert_eq!(sys.monomer_name(0), Some("A"));
        assert_eq!(sys.monomer_name(1), Some("B"));
        assert_eq!(sys.monomer_name(2), None);
        assert_eq!(sys.species_name(2), Some("AB"));
        assert_eq!(sys.species_index("A"), Some(0));
        assert_eq!(sys.species_index("AB"), Some(2));
        assert_eq!(sys.monomer_index("AB"), None);
        assert!(sys.has_names());
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
            "step norm {} should equal delta {}",
            p_norm,
            delta,
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
            "step norm {} should equal delta {}",
            p_norm,
            delta,
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
            "step norm {} should equal delta 1.0",
            p_norm,
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
            "step norm {} should equal delta {}",
            p_norm,
            delta,
        );
    }

    #[test]
    fn trust_region_adjustment() {
        let mut sys = SystemBuilder::new()
            .monomer("A", 1e-6)
            .monomer("B", 1e-8)
            .complex("AB", &[("A", 1), ("B", 1)], -15.0)
            .complex("A2B", &[("A", 2), ("B", 1)], -20.0)
            .complex("AB2", &[("A", 1), ("B", 2)], -18.0)
            .build()
            .unwrap();
        let eq = sys.solve().unwrap();

        let a = eq.get("A").unwrap();
        let b = eq.get("B").unwrap();
        let ab = eq.get("AB").unwrap();
        let a2b = eq.get("A2B").unwrap();
        let ab2 = eq.get("AB2").unwrap();
        assert!((a + ab + 2.0 * a2b + ab2 - 1e-6).abs() < 1e-5 * 1e-6);
        assert!((b + ab + a2b + 2.0 * ab2 - 1e-8).abs() < 1e-5 * 1e-8);
    }

    // ---------------------------------------------------------------------
    // Low-level API tests
    // ---------------------------------------------------------------------

    fn build_arrays_for_dimerization(
        c0: f64,
        dg: f64,
        temp_k: f64,
    ) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
        let mut at = Array2::zeros((3, 2));
        at[[0, 0]] = 1.0;
        at[[1, 1]] = 1.0;
        at[[2, 0]] = 1.0;
        at[[2, 1]] = 1.0;
        let rt = R * temp_k;
        let mut log_q = Array1::zeros(3);
        log_q[2] = -dg / rt;
        let c0_arr = Array1::from_vec(vec![c0, c0]);
        (at, log_q, c0_arr)
    }

    #[test]
    fn from_arrays_matches_builder() {
        let c0 = 100.0 * NM;
        let dg = -10.0;
        let temp_k = 298.15;

        let mut sys_builder = SystemBuilder::new()
            .temperature(temp_k)
            .monomer("A", c0)
            .monomer("B", c0)
            .complex("AB", &[("A", 1), ("B", 1)], dg)
            .build()
            .unwrap();
        let eq_builder = sys_builder.solve().unwrap();
        let ab_builder = eq_builder.get("AB").unwrap();
        let a_builder = eq_builder.get("A").unwrap();

        let (at, log_q, c0_arr) = build_arrays_for_dimerization(c0, dg, temp_k);
        let mut sys_arr = System::from_arrays(at, log_q, c0_arr).unwrap();
        let eq_arr = sys_arr.solve().unwrap();
        assert!((eq_arr.at(2) - ab_builder).abs() < 1e-15);
        assert!((eq_arr.at(0) - a_builder).abs() < 1e-15);
    }

    #[test]
    fn solve_is_idempotent_when_fresh() {
        let mut sys = simple_builder().build().unwrap();
        let iters1 = sys.solve().unwrap().iterations();
        let iters2 = sys.solve().unwrap().iterations();
        assert_eq!(iters1, iters2);
    }

    #[test]
    fn mutation_invalidates_fresh() {
        let mut sys = simple_builder().build().unwrap();
        sys.solve().unwrap();
        assert!(sys.last_solution().is_some());
        sys.set_c0(0, 2e-7);
        assert!(sys.last_solution().is_none());
    }

    #[test]
    fn c0_mut_invalidates_fresh() {
        let mut sys = simple_builder().build().unwrap();
        sys.solve().unwrap();
        assert!(sys.last_solution().is_some());
        {
            let _ = sys.c0_mut();
        }
        assert!(sys.last_solution().is_none());
    }

    #[test]
    fn sweep_matches_per_step_rebuild() {
        let c_b = 1e-7;
        let dg = -10.0;
        let temp_k = 298.15;

        // Tight sweep over c_a, reusing one System.
        let mut sys = SystemBuilder::new()
            .temperature(temp_k)
            .monomer("A", 1e-9)
            .monomer("B", c_b)
            .complex("AB", &[("A", 1), ("B", 1)], dg)
            .build()
            .unwrap();
        let a_idx = sys.monomer_index("A").unwrap();
        let ab_idx = sys.species_index("AB").unwrap();

        for i in 1..=5 {
            let c_a = 1e-7 * (i as f64) / 5.0;
            sys.set_c0(a_idx, c_a);
            let reused = sys.solve().unwrap().at(ab_idx);

            // Reference: rebuild the entire system for this c_a.
            let mut ref_sys = SystemBuilder::new()
                .temperature(temp_k)
                .monomer("A", c_a)
                .monomer("B", c_b)
                .complex("AB", &[("A", 1), ("B", 1)], dg)
                .build()
                .unwrap();
            let reference = ref_sys.solve().unwrap().get("AB").unwrap();

            // Solver converges to RTOL 1e-7 from each starting λ; the
            // warm-start and cold-start paths need not end at the same
            // f64 value. Allow relative tolerance comparable to RTOL.
            assert!(
                (reused - reference).abs() < 1e-14 + 1e-6 * reference.abs(),
                "sweep diverged at i={i}: reused={reused}, reference={reference}"
            );
        }
    }

    #[test]
    fn anonymous_system_no_names() {
        let (at, log_q, c0) = build_arrays_for_dimerization(100.0 * NM, -10.0, 298.15);
        let mut sys = System::from_arrays(at, log_q, c0).unwrap();
        assert!(!sys.has_names());
        sys.solve().unwrap();
        assert!(sys.species_index("AB").is_none());
        assert!(sys.last_solution().unwrap().get("AB").is_none());
    }

    #[test]
    fn from_arrays_with_names_round_trip() {
        let (at, log_q, c0) = build_arrays_for_dimerization(100.0 * NM, -10.0, 298.15);
        let mut sys = System::from_arrays_with_names(
            at,
            log_q,
            c0,
            vec!["A".into(), "B".into()],
            vec!["A".into(), "B".into(), "AB".into()],
        )
        .unwrap();
        let eq = sys.solve().unwrap();
        assert!(eq.get("AB").is_some());
        assert_eq!(eq["AB"], eq.at(2));
    }

    #[test]
    fn from_arrays_rejects_shape_mismatch() {
        let at = Array2::zeros((3, 2));
        let log_q = Array1::zeros(4);
        let c0 = Array1::from_vec(vec![1e-7, 1e-7]);
        let err = System::from_arrays(at, log_q, c0).unwrap_err();
        assert!(matches!(err, EquilibriumError::InvalidInputs(_)));
    }

    #[test]
    fn from_arrays_rejects_non_identity_monomer_block() {
        let mut at = Array2::zeros((3, 2));
        at[[0, 0]] = 1.0;
        at[[0, 1]] = 0.5; // spurious off-diagonal in monomer row
        at[[1, 1]] = 1.0;
        at[[2, 0]] = 1.0;
        at[[2, 1]] = 1.0;
        let log_q = Array1::zeros(3);
        let c0 = Array1::from_vec(vec![1e-7, 1e-7]);
        let err = System::from_arrays(at, log_q, c0).unwrap_err();
        assert!(matches!(err, EquilibriumError::InvalidInputs(_)));
    }

    #[test]
    fn from_arrays_rejects_nonzero_monomer_log_q() {
        let (at, _log_q, c0) = build_arrays_for_dimerization(100.0 * NM, -10.0, 298.15);
        let mut bad_log_q = Array1::zeros(3);
        bad_log_q[0] = 1.0;
        bad_log_q[2] = 16.0;
        let err = System::from_arrays(at, bad_log_q, c0).unwrap_err();
        assert!(matches!(err, EquilibriumError::InvalidInputs(_)));
    }

    #[test]
    fn from_arrays_rejects_non_positive_c0() {
        let (at, log_q, _) = build_arrays_for_dimerization(100.0 * NM, -10.0, 298.15);
        let c0 = Array1::from_vec(vec![0.0, 1e-7]);
        let err = System::from_arrays(at, log_q, c0).unwrap_err();
        assert!(matches!(err, EquilibriumError::InvalidConcentration(_)));
    }

    #[test]
    fn from_arrays_with_names_rejects_name_mismatch() {
        let (at, log_q, c0) = build_arrays_for_dimerization(100.0 * NM, -10.0, 298.15);
        let err = System::from_arrays_with_names(
            at,
            log_q,
            c0,
            vec!["A".into(), "B".into()],
            vec!["A".into(), "X".into(), "AB".into()], // second entry doesn't match
        )
        .unwrap_err();
        assert!(matches!(err, EquilibriumError::InvalidInputs(_)));
    }

    #[test]
    fn validate_after_mutation() {
        let mut sys = simple_builder().build().unwrap();
        assert!(sys.validate().is_ok());
        sys.c0_mut()[0] = -1.0; // break the invariant
        assert!(sys.validate().is_err());
    }

    #[test]
    fn solve_rejects_invalid_mutated_c0() {
        let mut sys = simple_builder().build().unwrap();
        sys.set_c0(0, 0.0);
        let err = sys.solve().unwrap_err();
        assert!(matches!(err, EquilibriumError::InvalidConcentration(0.0)));
    }

    #[test]
    fn solve_rejects_invalid_mutated_log_q() {
        let mut sys = simple_builder().build().unwrap();
        sys.set_log_q(0, 1.0);
        let err = sys.solve().unwrap_err();
        assert!(matches!(err, EquilibriumError::InvalidInputs(_)));
    }

    #[test]
    fn solve_rejects_invalid_mutated_options() {
        let mut sys = simple_builder().build().unwrap();
        sys.options_mut().max_iterations = 0;
        let err = sys.solve().unwrap_err();
        assert!(matches!(err, EquilibriumError::InvalidOptions(_)));
    }

    #[test]
    fn index_by_name_and_usize() {
        let mut sys = simple_builder().build().unwrap();
        let eq = sys.solve().unwrap();
        let by_name = eq["AB"];
        let by_idx = eq[2];
        assert_eq!(by_name, by_idx);
    }

    // ---------------------------------------------------------------------
    // SolverOptions tests
    // ---------------------------------------------------------------------

    #[test]
    fn default_options_roundtrip() {
        let opts = SolverOptions::default();
        assert!(opts.validate().is_ok());
        assert_eq!(opts.max_iterations, 1000);
        assert_eq!(opts.gradient_rel_tol, 1e-7);
        assert_eq!(opts.log_q_clamp, None);
    }

    #[test]
    fn validate_rejects_bad_tolerances() {
        let mut opts = SolverOptions::default();
        opts.gradient_rel_tol = -1.0;
        assert!(matches!(
            opts.validate(),
            Err(EquilibriumError::InvalidOptions(_))
        ));

        let mut opts = SolverOptions::default();
        opts.gradient_abs_tol = f64::NAN;
        assert!(matches!(
            opts.validate(),
            Err(EquilibriumError::InvalidOptions(_))
        ));
    }

    #[test]
    fn validate_rejects_bad_rho_thresholds() {
        let mut opts = SolverOptions::default();
        opts.trust_region_shrink_rho = 0.8;
        opts.trust_region_grow_rho = 0.75;
        assert!(matches!(
            opts.validate(),
            Err(EquilibriumError::InvalidOptions(_))
        ));
    }

    #[test]
    fn validate_rejects_bad_scale_factors() {
        let mut opts = SolverOptions::default();
        opts.trust_region_shrink_scale = 1.5;
        assert!(matches!(
            opts.validate(),
            Err(EquilibriumError::InvalidOptions(_))
        ));

        let mut opts = SolverOptions::default();
        opts.trust_region_grow_scale = 0.9;
        assert!(matches!(
            opts.validate(),
            Err(EquilibriumError::InvalidOptions(_))
        ));
    }

    #[test]
    fn validate_rejects_zero_max_iterations() {
        let mut opts = SolverOptions::default();
        opts.max_iterations = 0;
        assert!(matches!(
            opts.validate(),
            Err(EquilibriumError::InvalidOptions(_))
        ));
    }

    #[test]
    fn validate_rejects_delta_ordering() {
        let mut opts = SolverOptions::default();
        opts.initial_trust_region_radius = 100.0;
        opts.max_trust_region_radius = 10.0;
        assert!(matches!(
            opts.validate(),
            Err(EquilibriumError::InvalidOptions(_))
        ));
    }

    #[test]
    fn builder_options_surfaces_invalid_at_build() {
        let opts = SolverOptions {
            max_iterations: 0,
            ..Default::default()
        };
        let err = SystemBuilder::new()
            .monomer("A", 1e-7)
            .options(opts)
            .build()
            .unwrap_err();
        assert!(matches!(err, EquilibriumError::InvalidOptions(_)));
    }

    #[test]
    fn builder_applies_log_q_clamp() {
        // Artificial complex with huge -ΔG/RT. At the default (no clamp)
        // log_q would be ~5000; with clamp 100 it should be capped.
        let dg = -5000.0 * R * 298.15; // so -dg/RT = 5000
        let builder = SystemBuilder::new()
            .monomer("A", 1e-7)
            .monomer("B", 1e-7)
            .complex("AB", &[("A", 1), ("B", 1)], dg);
        let opts = SolverOptions {
            log_q_clamp: Some(100.0),
            ..Default::default()
        };
        let sys_unclamped = builder.clone().build().unwrap();
        let sys_clamped = builder.options(opts).build().unwrap();

        let unclamped_logq = sys_unclamped.log_q()[2];
        let clamped_logq = sys_clamped.log_q()[2];
        assert!(unclamped_logq > 1000.0, "expected huge log_q, got {unclamped_logq}");
        assert!((clamped_logq - 100.0).abs() < 1e-12, "clamp should cap at 100, got {clamped_logq}");
    }

    #[test]
    fn tighter_tolerance_requires_more_iterations() {
        // Pick a problem where the default 1e-7 tolerance completes in
        // ~10 iterations; then tightening by 5 orders of magnitude
        // should require at least a few more.
        let c0 = 100e-9;
        let dg = -10.0;
        let builder = SystemBuilder::new()
            .monomer("A", c0)
            .monomer("B", c0)
            .complex("AB", &[("A", 1), ("B", 1)], dg);

        let mut default = builder.clone().build().unwrap();
        let default_iters = default.solve().unwrap().iterations();

        let tight_opts = SolverOptions {
            gradient_rel_tol: 1e-12,
            gradient_abs_tol: 0.0,
            ..Default::default()
        };
        let mut tight = builder.options(tight_opts).build().unwrap();
        let tight_iters = tight.solve().unwrap().iterations();

        assert!(
            tight_iters >= default_iters,
            "tight tolerance ({tight_iters}) should take at least as many iters as default ({default_iters})"
        );
    }

    #[test]
    fn max_iterations_cap_fails_fast() {
        let opts = SolverOptions {
            max_iterations: 1,
            ..Default::default()
        };
        let mut sys = SystemBuilder::new()
            .monomer("A", 100e-9)
            .monomer("B", 100e-9)
            .complex("AB", &[("A", 1), ("B", 1)], -20.0)
            .options(opts)
            .build()
            .unwrap();
        let err = sys.solve().unwrap_err();
        assert!(matches!(
            err,
            EquilibriumError::ConvergenceFailure { iterations: 1, .. }
        ));
    }

    #[test]
    fn options_mut_invalidates_fresh() {
        let mut sys = simple_builder().build().unwrap();
        sys.solve().unwrap();
        assert!(sys.last_solution().is_some());
        let _ = sys.options_mut(); // just accessing flips freshness
        assert!(sys.last_solution().is_none());
    }

    #[test]
    fn set_options_reapplies_clamp() {
        let c0 = 1e-7;
        let dg = -5000.0 * R * 298.15; // huge log_q again
        let mut sys = SystemBuilder::new()
            .monomer("A", c0)
            .monomer("B", c0)
            .complex("AB", &[("A", 1), ("B", 1)], dg)
            .build()
            .unwrap();
        // No clamp yet: log_q is huge.
        assert!(sys.log_q()[2] > 1000.0);
        sys.set_options(SolverOptions {
            log_q_clamp: Some(50.0),
            ..Default::default()
        })
        .unwrap();
        assert!((sys.log_q()[2] - 50.0).abs() < 1e-12);
        assert!(sys.last_solution().is_none());
    }

    #[test]
    fn from_arrays_with_options_applies_clamp() {
        let (at, mut log_q, c0) = build_arrays_for_dimerization(100.0 * NM, -10.0, 298.15);
        // Bump log_q[2] to something large, then clamp.
        log_q[2] = 500.0;
        let opts = SolverOptions {
            log_q_clamp: Some(50.0),
            ..Default::default()
        };
        let sys = System::from_arrays_with_options(at, log_q, c0, opts).unwrap();
        assert!((sys.log_q()[2] - 50.0).abs() < 1e-12);
    }

    #[test]
    fn clone_gives_independent_solver() {
        let mut sys = simple_builder().build().unwrap();
        sys.solve().unwrap();
        let mut clone = sys.clone();
        // Mutating the clone must not affect the original's freshness.
        clone.set_c0(0, 200.0 * NM);
        assert!(sys.last_solution().is_some());
        assert!(clone.last_solution().is_none());
    }
}

// ---------------------------------------------------------------------------
// Python bindings (behind "python" feature)
// ---------------------------------------------------------------------------

#[cfg(feature = "python")]
mod python;
