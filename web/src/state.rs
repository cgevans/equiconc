//! UI-side state types. The solver runs entirely in-process via the
//! `equiconc` crate; nothing here is serialized over a network boundary.

use equiconc::{SolverObjective, SolverOptions};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Whether the temperature input is read in Celsius or Kelvin.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TempUnit {
    Celsius,
    Kelvin,
}

impl TempUnit {
    pub fn label(self) -> &'static str {
        match self {
            TempUnit::Celsius => "°C",
            TempUnit::Kelvin => "K",
        }
    }
}

/// Units of the per-species ΔG values in the composition input.
///
/// `KcalPerMol` (default) treats inputs as kcal/mol and converts via
/// `log_q = −ΔG/(RT)` using the user's temperature. `RT` treats inputs
/// as already-dimensionless `ΔG/RT` and uses them as `log_q = −ΔG`
/// directly — no temperature divide. With `RT`, temperature only
/// matters when scalarity is on (water density for the c₀ rescale).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EnergyUnit {
    KcalPerMol,
    RT,
}

/// User-side options surface: solver knobs plus the wrapper-level knobs
/// (temperature, energy units, mole-fraction scaling, ΔG clamp).
#[derive(Clone, Debug, PartialEq)]
pub struct UiOptions {
    pub temperature_value: f64,
    pub temperature_unit: TempUnit,
    pub energy_unit: EnergyUnit,
    pub scalarity: bool,
    pub dg_clamp_on: bool,
    pub dg_clamp_kcal: f64,
    pub solver: SolverOptions,
}

impl UiOptions {
    /// equiconc defaults: 25 °C, kcal/mol input, no mole-fraction
    /// scaling, no ΔG clamp, linear objective, default solver knobs.
    pub fn equiconc_default() -> Self {
        Self {
            temperature_value: 25.0,
            temperature_unit: TempUnit::Celsius,
            energy_unit: EnergyUnit::KcalPerMol,
            scalarity: false,
            dg_clamp_on: false,
            dg_clamp_kcal: 230.0,
            solver: SolverOptions::default(),
        }
    }

    /// COFFEE-compatible: 37 °C, kcal/mol input, mole-fraction scaling
    /// on, ΔG clamp at 230 kcal/mol, linear objective,
    /// max_iterations = 5000.
    pub fn coffee_compatible() -> Self {
        Self {
            temperature_value: 37.0,
            temperature_unit: TempUnit::Celsius,
            energy_unit: EnergyUnit::KcalPerMol,
            scalarity: true,
            dg_clamp_on: true,
            dg_clamp_kcal: 230.0,
            solver: SolverOptions {
                max_iterations: 5000,
                objective: SolverObjective::Linear,
                ..SolverOptions::default()
            },
        }
    }

    pub fn temperature_kelvin(&self) -> f64 {
        match self.temperature_unit {
            TempUnit::Kelvin => self.temperature_value,
            TempUnit::Celsius => self.temperature_value + 273.15,
        }
    }

    pub fn temperature_celsius(&self) -> f64 {
        match self.temperature_unit {
            TempUnit::Kelvin => self.temperature_value - 273.15,
            TempUnit::Celsius => self.temperature_value,
        }
    }
}

impl Default for UiOptions {
    fn default() -> Self {
        Self::equiconc_default()
    }
}

#[derive(Clone, Debug)]
pub struct SolveResult {
    /// Per-species concentration in molar units (after un-scaling out of
    /// mole fractions, if scalarity was on).
    pub concentrations: Array1<f64>,
    /// Original `c0` (in molar units).
    pub c0: Array1<f64>,
    /// `n_species × n_mon` stoichiometry matrix.
    pub stoich: Array2<f64>,
    /// Reference free energies (kcal/mol) actually used for the solve,
    /// after ΔG clamping. Kept for the JSON reproducibility export.
    pub dg_kcal_used: Array1<f64>,
    /// Number of monomers (= first n_mon rows of stoich are identity).
    pub n_mon: usize,
    /// Solver iterations used.
    pub iterations: usize,
    /// Whether full-precision convergence was reached.
    pub converged_fully: bool,
    /// Largest per-monomer mass-balance residual, in molar units.
    pub residual: f64,
    /// Wall-clock time spent inside `System::solve()`, in milliseconds.
    pub elapsed_ms: f64,
    /// Snapshot of the options used. Embedded in the JSON export.
    pub options: UiOptions,
}

impl SolveResult {
    /// Total mass per monomer at equilibrium: `Σ_j A_{ji} · c_j`.
    pub fn mass_per_monomer(&self) -> Array1<f64> {
        let mut totals = Array1::<f64>::zeros(self.n_mon);
        let n_species = self.stoich.nrows();
        for i in 0..self.n_mon {
            let mut s = 0.0;
            for j in 0..n_species {
                s += self.stoich[[j, i]] * self.concentrations[j];
            }
            totals[i] = s;
        }
        totals
    }

    /// Per-species `share-of-mass` for a given monomer index `i`:
    /// `A_{ji} · c_j / Σ_k A_{ki} · c_k`.
    pub fn share_of_monomer(&self, i: usize) -> Array1<f64> {
        let n_species = self.stoich.nrows();
        let total: f64 = (0..n_species)
            .map(|j| self.stoich[[j, i]] * self.concentrations[j])
            .sum();
        let mut out = Array1::<f64>::zeros(n_species);
        if total <= 0.0 {
            return out;
        }
        for j in 0..n_species {
            out[j] = self.stoich[[j, i]] * self.concentrations[j] / total;
        }
        out
    }
}

#[derive(Clone, Debug)]
pub struct SolveError {
    pub message: String,
    pub source: ErrSource,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrSource {
    Composition,
    Concentrations,
    Solver,
    Options,
}

impl SolveError {
    pub fn from_composition(msg: impl Into<String>) -> Self {
        Self {
            message: msg.into(),
            source: ErrSource::Composition,
        }
    }
    pub fn from_concentrations(msg: impl Into<String>) -> Self {
        Self {
            message: msg.into(),
            source: ErrSource::Concentrations,
        }
    }
    pub fn from_options(msg: impl Into<String>) -> Self {
        Self {
            message: msg.into(),
            source: ErrSource::Options,
        }
    }
}
