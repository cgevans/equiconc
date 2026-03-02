use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

use crate::{Equilibrium, EquilibriumError};

// ---------------------------------------------------------------------------
// Error mapping
// ---------------------------------------------------------------------------

fn map_err(e: EquilibriumError) -> PyErr {
    match e {
        // Defensive: the convex dual always converges for valid inputs.
        EquilibriumError::ConvergenceFailure { .. } => PyRuntimeError::new_err(e.to_string()),
        _ => PyValueError::new_err(e.to_string()),
    }
}

// ---------------------------------------------------------------------------
// PySystem — deferred-construction wrapper
// ---------------------------------------------------------------------------

/// Equilibrium concentration solver.
///
/// Build a system by chaining `.monomer()` and `.complex()` calls, then
/// call `.equilibrium()` to solve.
#[pyclass(name = "System", module = "equiconc")]
struct PySystem {
    inner: crate::System,
}

#[pymethods]
impl PySystem {
    #[new]
    #[pyo3(signature = (temperature = 310.15))]
    fn new(temperature: f64) -> Self {
        PySystem {
            inner: crate::System::new().temperature(temperature),
        }
    }

    /// Add a monomer species with a given total concentration (molar).
    fn monomer(slf: Py<Self>, py: Python<'_>, name: &str, total_concentration: f64) -> Py<Self> {
        let mut inner = slf.borrow_mut(py);
        inner.inner = inner.inner.clone().monomer(name, total_concentration);
        drop(inner);
        slf
    }

    /// Add a complex with composition `[(monomer_name, count), ...]` and ΔG° (kcal/mol).
    fn complex(
        slf: Py<Self>,
        py: Python<'_>,
        name: &str,
        composition: Vec<(String, usize)>,
        delta_g: f64,
    ) -> Py<Self> {
        let comp_refs: Vec<(&str, usize)> =
            composition.iter().map(|(n, c)| (n.as_str(), *c)).collect();
        let mut inner = slf.borrow_mut(py);
        inner.inner = inner.inner.clone().complex(name, &comp_refs, delta_g);
        drop(inner);
        slf
    }

    /// Solve for equilibrium concentrations.
    fn equilibrium(&self) -> PyResult<PyEquilibrium> {
        let eq = self.inner.equilibrium().map_err(map_err)?;
        Ok(PyEquilibrium::from_equilibrium(eq))
    }

    fn __repr__(&self) -> String {
        format!(
            "System(temperature={}, monomers={}, complexes={})",
            self.inner.get_temperature(),
            self.inner.monomer_count(),
            self.inner.complex_count()
        )
    }
}

// ---------------------------------------------------------------------------
// PyEquilibrium — wraps the result with Pythonic access
// ---------------------------------------------------------------------------

/// Result of an equilibrium calculation.
///
/// Supports dict-like access: `eq["A"]`, `"A" in eq`, `len(eq)`.
#[pyclass(name = "Equilibrium", module = "equiconc")]
struct PyEquilibrium {
    concentrations: HashMap<String, f64>,
    monomer_names: Vec<String>,
    complex_names: Vec<String>,
    free_monomer_concentrations: Vec<f64>,
    complex_concentrations: Vec<f64>,
    converged_fully: bool,
}

impl PyEquilibrium {
    fn from_equilibrium(eq: Equilibrium) -> Self {
        let mut concentrations = HashMap::new();
        for (name, &conc) in eq.monomer_names().iter().zip(eq.free_monomer_concentrations()) {
            concentrations.insert(name.clone(), conc);
        }
        for (name, &conc) in eq.complex_names().iter().zip(eq.complex_concentrations()) {
            concentrations.insert(name.clone(), conc);
        }
        let converged_fully = eq.converged_fully();
        PyEquilibrium {
            concentrations,
            monomer_names: eq.monomer_names().to_vec(),
            complex_names: eq.complex_names().to_vec(),
            free_monomer_concentrations: eq.free_monomer_concentrations().to_vec(),
            complex_concentrations: eq.complex_concentrations().to_vec(),
            converged_fully,
        }
    }

    /// All species names in deterministic order (monomers first, then complexes).
    fn ordered_names(&self) -> impl Iterator<Item = &String> {
        self.monomer_names.iter().chain(self.complex_names.iter())
    }
}

#[pymethods]
impl PyEquilibrium {
    /// Look up a concentration by species name. Returns `None` if unknown.
    fn concentration(&self, name: &str) -> Option<f64> {
        self.concentrations.get(name).copied()
    }

    /// Dict-like access: `eq["AB"]`. Raises `KeyError` if unknown.
    fn __getitem__(&self, name: &str) -> PyResult<f64> {
        self.concentrations
            .get(name)
            .copied()
            .ok_or_else(|| PyKeyError::new_err(name.to_string()))
    }

    /// Supports `"AB" in eq`.
    fn __contains__(&self, name: &str) -> bool {
        self.concentrations.contains_key(name)
    }

    /// Number of species.
    fn __len__(&self) -> usize {
        self.concentrations.len()
    }

    /// All species as a Python dict `{name: concentration}` in deterministic
    /// order (monomers first, then complexes, in addition order).
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for name in self.ordered_names() {
            dict.set_item(name, self.concentrations[name])?;
        }
        Ok(dict)
    }

    /// Species names in deterministic order (monomers first, then complexes).
    fn keys(&self) -> Vec<String> {
        self.ordered_names().cloned().collect()
    }

    /// Concentrations in deterministic order (monomers first, then complexes).
    fn values(&self) -> Vec<f64> {
        self.ordered_names().map(|n| self.concentrations[n]).collect()
    }

    /// `(name, concentration)` pairs in deterministic order.
    fn items(&self) -> Vec<(String, f64)> {
        self.ordered_names()
            .map(|n| (n.clone(), self.concentrations[n]))
            .collect()
    }

    /// Supports `for name in eq:`.
    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<EquilibriumKeyIter>> {
        let py = slf.py();
        Py::new(py, EquilibriumKeyIter {
            keys: slf.keys(),
            index: 0,
        })
    }

    /// Free monomer concentrations as a list.
    #[getter]
    fn free_monomer_concentrations(&self) -> Vec<f64> {
        self.free_monomer_concentrations.clone()
    }

    /// Complex concentrations as a list.
    #[getter]
    fn complex_concentrations(&self) -> Vec<f64> {
        self.complex_concentrations.clone()
    }

    /// Monomer names as a list.
    #[getter]
    fn monomer_names(&self) -> Vec<String> {
        self.monomer_names.clone()
    }

    /// Complex names as a list.
    #[getter]
    fn complex_names(&self) -> Vec<String> {
        self.complex_names.clone()
    }

    /// Whether the solver achieved full convergence.
    ///
    /// `False` if the solver accepted the result at a relaxed tolerance
    /// due to stagnation at f64 precision limits.
    #[getter]
    fn converged_fully(&self) -> bool {
        self.converged_fully
    }

    fn __repr__(&self) -> String {
        let entries: Vec<String> = self
            .ordered_names()
            .map(|name| {
                let conc = self.concentrations[name];
                format!("  {name}: {conc:.6e} M")
            })
            .collect();
        format!("Equilibrium(\n{}\n)", entries.join("\n"))
    }
}

// ---------------------------------------------------------------------------
// Iterator for `for name in eq:`
// ---------------------------------------------------------------------------

#[pyclass]
struct EquilibriumKeyIter {
    keys: Vec<String>,
    index: usize,
}

#[pymethods]
impl EquilibriumKeyIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self) -> Option<String> {
        if self.index < self.keys.len() {
            let key = self.keys[self.index].clone();
            self.index += 1;
            Some(key)
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

#[pymodule]
fn equiconc(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySystem>()?;
    m.add_class::<PyEquilibrium>()?;
    Ok(())
}
