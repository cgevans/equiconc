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
#[pyclass(name = "System")]
struct PySystem {
    temperature: f64,
    monomers: Vec<(String, f64)>,
    complexes: Vec<(String, Vec<(String, usize)>, f64)>,
}

#[pymethods]
impl PySystem {
    #[new]
    #[pyo3(signature = (temperature = 310.15))]
    fn new(temperature: f64) -> Self {
        PySystem {
            temperature,
            monomers: Vec::new(),
            complexes: Vec::new(),
        }
    }

    /// Add a monomer species with a given total concentration (molar).
    fn monomer(slf: Py<Self>, py: Python<'_>, name: &str, total_concentration: f64) -> Py<Self> {
        slf.borrow_mut(py)
            .monomers
            .push((name.to_string(), total_concentration));
        slf
    }

    /// Add a complex with composition `[(monomer_name, count), ...]` and ΔG° (kcal/mol).
    fn complex(
        slf: Py<Self>,
        py: Python<'_>,
        name: &str,
        composition: Vec<(String, usize)>,
        delta_g: f64,
    ) -> PyResult<Py<Self>> {
        if composition.is_empty() {
            return Err(PyValueError::new_err("complex has empty composition"));
        }
        {
            let mut inner = slf.borrow_mut(py);
            // Eagerly validate monomer names
            let known: Vec<&str> = inner.monomers.iter().map(|(n, _)| n.as_str()).collect();
            for (monomer_name, _) in &composition {
                if !known.contains(&monomer_name.as_str()) {
                    return Err(PyValueError::new_err(format!(
                        "unknown monomer: {monomer_name}"
                    )));
                }
            }
            inner.complexes.push((name.to_string(), composition, delta_g));
        }
        Ok(slf)
    }

    /// Solve for equilibrium concentrations.
    fn equilibrium(&self) -> PyResult<PyEquilibrium> {
        // Build the Rust System from stored inputs
        let mut sys = crate::System::new().temperature(self.temperature);
        for (name, conc) in &self.monomers {
            sys = sys.monomer(name, *conc);
        }
        for (name, comp, dg) in &self.complexes {
            let comp_refs: Vec<(&str, usize)> =
                comp.iter().map(|(n, c)| (n.as_str(), *c)).collect();
            sys = sys.complex(name, &comp_refs, *dg).map_err(map_err)?;
        }
        let eq = sys.equilibrium().map_err(map_err)?;
        Ok(PyEquilibrium::from_equilibrium(eq))
    }

    fn __repr__(&self) -> String {
        format!(
            "System(temperature={}, monomers={}, complexes={})",
            self.temperature,
            self.monomers.len(),
            self.complexes.len()
        )
    }
}

// ---------------------------------------------------------------------------
// PyEquilibrium — wraps the result with Pythonic access
// ---------------------------------------------------------------------------

/// Result of an equilibrium calculation.
///
/// Supports dict-like access: `eq["A"]`, `"A" in eq`, `len(eq)`.
#[pyclass(name = "Equilibrium")]
struct PyEquilibrium {
    concentrations: HashMap<String, f64>,
    monomer_names: Vec<String>,
    complex_names: Vec<String>,
    free_monomer_concentrations: Vec<f64>,
    complex_concentrations: Vec<f64>,
}

impl PyEquilibrium {
    fn from_equilibrium(eq: Equilibrium) -> Self {
        let mut concentrations = HashMap::new();
        for (name, &conc) in eq.monomer_names.iter().zip(&eq.free_monomer_concentrations) {
            concentrations.insert(name.clone(), conc);
        }
        for (name, &conc) in eq.complex_names.iter().zip(&eq.complex_concentrations) {
            concentrations.insert(name.clone(), conc);
        }
        PyEquilibrium {
            concentrations,
            monomer_names: eq.monomer_names,
            complex_names: eq.complex_names,
            free_monomer_concentrations: eq.free_monomer_concentrations,
            complex_concentrations: eq.complex_concentrations,
        }
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

    /// All species as a Python dict `{name: concentration}`.
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (name, &conc) in &self.concentrations {
            dict.set_item(name, conc)?;
        }
        Ok(dict)
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

    fn __repr__(&self) -> String {
        let entries: Vec<String> = self
            .monomer_names
            .iter()
            .chain(self.complex_names.iter())
            .map(|name| {
                let conc = self.concentrations[name];
                format!("  {name}: {conc:.6e} M")
            })
            .collect();
        format!("Equilibrium(\n{}\n)", entries.join("\n"))
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
