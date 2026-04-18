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
// Energy specification for complexes
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum EnergySpec {
    /// ΔG in kcal/mol
    DeltaG(f64),
    /// ΔG/RT (unitless)
    DeltaGOverRT(f64),
    /// ΔH (kcal/mol) and ΔS (kcal/(mol·K))
    DeltaHS { delta_h: f64, delta_s: f64 },
}

/// Accepts either a plain float or a (value, temperature_C) tuple from Python.
#[derive(Debug, Clone, FromPyObject)]
enum DeltaGInput {
    Scalar(f64),
    AtTemp((f64, f64)),
}

fn resolve_energy_spec(
    dg_st: Option<DeltaGInput>,
    delta_g_over_rt: Option<f64>,
    dh_st: Option<f64>,
    ds_st: Option<f64>,
) -> PyResult<EnergySpec> {
    // Split dg_st into its two forms for cleaner matching.
    let (dg_stcalar, dg_at_temp) = match dg_st {
        Some(DeltaGInput::Scalar(v)) => (Some(v), None),
        Some(DeltaGInput::AtTemp(pair)) => (None, Some(pair)),
        None => (None, None),
    };

    match (dg_stcalar, dg_at_temp, delta_g_over_rt, dh_st, ds_st) {
        // dg_st=<float>
        (Some(dg), None, None, None, None) => Ok(EnergySpec::DeltaG(dg)),
        // delta_g_over_rt=<float>
        (None, None, Some(dgrt), None, None) => Ok(EnergySpec::DeltaGOverRT(dgrt)),
        // dh_st=<float>, ds_st=<float>
        (None, None, None, Some(dh), Some(ds)) => Ok(EnergySpec::DeltaHS {
            delta_h: dh,
            delta_s: ds,
        }),
        // dg_st=(<float>, <temp_C>), ds_st=<float>  →  derive ΔH
        (None, Some((dg, temp_c)), None, None, Some(ds)) => {
            let temp_k = temp_c + 273.15;
            let dh = dg + temp_k * ds;
            Ok(EnergySpec::DeltaHS {
                delta_h: dh,
                delta_s: ds,
            })
        }
        // dg_st=(<float>, <temp_C>) without ds_st
        (None, Some(_), None, None, None) => Err(PyValueError::new_err(
            "dg_st as (value, temperature_C) requires ds_st",
        )),
        // dg_st=<float> with ds_st but no dh_st
        (Some(_), None, None, None, Some(_)) => Err(PyValueError::new_err(
            "dg_st as a scalar cannot be combined with ds_st; \
             use dg_st=(value, temperature_C) tuple form, or dh_st + ds_st",
        )),
        // dh_st without ds_st, or vice versa
        (None, None, None, Some(_), None) | (None, None, None, None, Some(_)) => Err(
            PyValueError::new_err("dh_st and ds_st must both be specified"),
        ),
        // Nothing specified
        (None, None, None, None, None) => Err(PyValueError::new_err(
            "must specify energy: dg_st, delta_g_over_rt, or (dh_st and ds_st)",
        )),
        // Conflicting specifications
        _ => Err(PyValueError::new_err(
            "specify only one of: dg_st, delta_g_over_rt, or (dh_st and ds_st)",
        )),
    }
}

// ---------------------------------------------------------------------------
// PySystem — deferred-construction wrapper
// ---------------------------------------------------------------------------

/// Equilibrium concentration solver for nucleic acid strand systems.
///
/// Build a system by chaining ``monomer()`` and ``complex()`` calls,
/// then call ``equilibrium()`` to solve for equilibrium concentrations.
///
/// Parameters
/// ----------
/// temperature_C : float, optional
///     Temperature in degrees Celsius (default: 25.0).
/// temperature_K : float, optional
///     Temperature in kelvin. Cannot be combined with ``temperature_C``.
///
/// Examples
/// --------
/// >>> import equiconc
/// >>> eq = (equiconc.System()
/// ...     .monomer("A", 100e-9)
/// ...     .monomer("B", 100e-9)
/// ...     .complex("AB", [("A", 1), ("B", 1)], dg_st=-10.0)
/// ...     .equilibrium())
/// >>> eq["AB"] > 0
/// True
#[pyclass(name = "System", module = "equiconc")]
struct PySystem {
    temperature_k: Option<f64>,
    monomers: Vec<(String, f64)>,
    complexes: Vec<(String, Vec<(String, usize)>, EnergySpec)>,
}

#[pymethods]
#[allow(non_snake_case)]
impl PySystem {
    #[new]
    #[pyo3(signature = (*, temperature_C=None, temperature_K=None))]
    fn new(temperature_C: Option<f64>, temperature_K: Option<f64>) -> PyResult<Self> {
        let temp_k = match (temperature_C, temperature_K) {
            (None, None) => None,
            (Some(c), None) => Some(c + 273.15),
            (None, Some(k)) => Some(k),
            (Some(_), Some(_)) => {
                return Err(PyValueError::new_err(
                    "cannot specify both temperature_C and temperature_K",
                ))
            }
        };
        Ok(PySystem {
            temperature_k: temp_k,
            monomers: Vec::new(),
            complexes: Vec::new(),
        })
    }

    /// Add a monomer species with a given total concentration.
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///     Name of the monomer species. Must be unique and non-empty.
    /// total_concentration : float
    ///     Total concentration in molar (mol/L). Must be finite and
    ///     positive.
    ///
    /// Returns
    /// -------
    /// System
    ///     The same system instance, for method chaining.
    fn monomer(slf: Py<Self>, py: Python<'_>, name: &str, total_concentration: f64) -> Py<Self> {
        let mut inner = slf.borrow_mut(py);
        inner.monomers.push((name.to_string(), total_concentration));
        drop(inner);
        slf
    }

    /// Add a complex species with a given stoichiometry and energy.
    ///
    /// Exactly one energy specification must be provided:
    ///
    /// - ``dg_st``: standard free energy of formation in kcal/mol
    /// - ``dg_st=(value, temperature_C)`` + ``ds_st``: ΔG at a
    ///   known temperature plus ΔS; ΔH is derived as
    ///   ΔH = ΔG + T·ΔS and ΔG at the system temperature is
    ///   computed as ΔH − T_sys·ΔS
    /// - ``delta_g_over_rt``: dimensionless ΔG/RT (no temperature needed)
    /// - ``dh_st`` + ``ds_st``: enthalpy (kcal/mol) and entropy
    ///   (kcal/(mol·K)); ΔG is computed as ΔH − TΔS at solve time
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///     Name of the complex. Must be unique across all species.
    /// composition : list of (str, int)
    ///     Monomer composition as ``[(monomer_name, count), ...]``.
    ///     Each monomer must have been previously added and count
    ///     must be >= 1.
    /// dg_st : float or (float, float), optional
    ///     Standard free energy of formation in kcal/mol at 1 M
    ///     standard state. Either a scalar (must be finite), or a
    ///     tuple ``(dg_st, temperature_C)`` giving ΔG at a known
    ///     temperature in °C; the latter form requires ``ds_st``.
    /// delta_g_over_rt : float, optional
    ///     Dimensionless free energy ΔG/(RT). When all complexes use
    ///     this form, temperature is not required.
    /// dh_st : float, optional
    ///     Enthalpy of formation in kcal/mol. Must be paired with
    ///     ``ds_st``.
    /// ds_st : float, optional
    ///     Entropy of formation in kcal/(mol·K). Must be paired with
    ///     ``dh_st``, or with the tuple form of ``dg_st``.
    ///
    /// Returns
    /// -------
    /// System
    ///     The same system instance, for method chaining.
    #[pyo3(signature = (name, composition, *, dg_st=None, delta_g_over_rt=None, dh_st=None, ds_st=None))]
    fn complex(
        slf: Py<Self>,
        py: Python<'_>,
        name: &str,
        composition: Vec<(String, usize)>,
        dg_st: Option<DeltaGInput>,
        delta_g_over_rt: Option<f64>,
        dh_st: Option<f64>,
        ds_st: Option<f64>,
    ) -> PyResult<Py<Self>> {
        let energy = resolve_energy_spec(dg_st, delta_g_over_rt, dh_st, ds_st)?;
        let mut inner = slf.borrow_mut(py);
        inner.complexes.push((name.to_string(), composition, energy));
        drop(inner);
        Ok(slf)
    }

    /// Solve for equilibrium concentrations.
    ///
    /// Returns
    /// -------
    /// Equilibrium
    ///     The result containing concentrations of all species.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the system specification is invalid (no monomers, unknown
    ///     monomers in complexes, invalid concentrations, etc.).
    /// RuntimeError
    ///     If the solver fails to converge.
    fn equilibrium(&self) -> PyResult<PyEquilibrium> {
        // Default to 25 °C if no temperature was given.
        let temp_k = self.temperature_k.unwrap_or(298.15);

        let mut builder = crate::SystemBuilder::new().temperature(temp_k);

        for (name, conc) in &self.monomers {
            builder = builder.monomer(name, *conc);
        }

        let rt = crate::R * temp_k;
        for (name, comp, energy) in &self.complexes {
            let dg = match energy {
                EnergySpec::DeltaG(dg) => *dg,
                EnergySpec::DeltaGOverRT(dgrt) => *dgrt * rt,
                EnergySpec::DeltaHS { delta_h, delta_s } => *delta_h - temp_k * *delta_s,
            };
            let comp_refs: Vec<(&str, usize)> =
                comp.iter().map(|(n, c)| (n.as_str(), *c)).collect();
            builder = builder.complex(name, &comp_refs, dg);
        }

        let mut sys = builder.build().map_err(map_err)?;
        let n_mon = sys.n_monomers();
        let n_species = sys.n_species();
        let monomer_names: Vec<String> = (0..n_mon)
            .map(|i| sys.monomer_name(i).unwrap_or_default().to_string())
            .collect();
        let complex_names: Vec<String> = (n_mon..n_species)
            .map(|i| sys.species_name(i).unwrap_or_default().to_string())
            .collect();

        let eq = sys.solve().map_err(map_err)?;
        Ok(PyEquilibrium::from_equilibrium(
            monomer_names,
            complex_names,
            &eq,
        ))
    }

    fn __repr__(&self) -> String {
        let temp_c = self.temperature_k.unwrap_or(298.15) - 273.15;
        format!(
            "System(temperature={temp_c:.2}°C, monomers={}, complexes={})",
            self.monomers.len(),
            self.complexes.len()
        )
    }
}

// ---------------------------------------------------------------------------
// PyEquilibrium — wraps the result with Pythonic access
// ---------------------------------------------------------------------------

/// Result of an equilibrium concentration calculation.
///
/// Supports dict-like access: ``eq["A"]``, ``"A" in eq``, ``len(eq)``,
/// and iteration over species names with ``for name in eq``.
///
/// Attributes
/// ----------
/// monomer_names : list of str
///     Monomer species names, in addition order.
/// complex_names : list of str
///     Complex species names, in addition order.
/// free_monomer_concentrations : list of float
///     Free monomer concentrations in molar, in addition order.
/// complex_concentrations : list of float
///     Complex concentrations in molar, in addition order.
/// converged_fully : bool
///     Whether the solver achieved full convergence. ``False`` if the
///     solver accepted results at a relaxed tolerance.
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
    fn from_equilibrium(
        monomer_names: Vec<String>,
        complex_names: Vec<String>,
        eq: &Equilibrium<'_>,
    ) -> Self {
        let free_monomer_concentrations: Vec<f64> = eq.free_monomers().to_vec();
        let complex_concentrations: Vec<f64> = eq.complexes().to_vec();

        let mut concentrations =
            HashMap::with_capacity(monomer_names.len() + complex_names.len());
        for (name, &conc) in monomer_names.iter().zip(&free_monomer_concentrations) {
            concentrations.insert(name.clone(), conc);
        }
        for (name, &conc) in complex_names.iter().zip(&complex_concentrations) {
            concentrations.insert(name.clone(), conc);
        }

        PyEquilibrium {
            concentrations,
            monomer_names,
            complex_names,
            free_monomer_concentrations,
            complex_concentrations,
            converged_fully: eq.converged_fully(),
        }
    }

    /// All species names in deterministic order (monomers first, then complexes).
    fn ordered_names(&self) -> impl Iterator<Item = &String> {
        self.monomer_names.iter().chain(self.complex_names.iter())
    }
}

#[pymethods]
impl PyEquilibrium {
    /// Look up a concentration by species name.
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///     Species name (monomer or complex).
    ///
    /// Returns
    /// -------
    /// float
    ///     Concentration in molar.
    ///
    /// Raises
    /// ------
    /// KeyError
    ///     If the species name is not found.
    fn concentration(&self, name: &str) -> PyResult<f64> {
        self.concentrations
            .get(name)
            .copied()
            .ok_or_else(|| PyKeyError::new_err(name.to_string()))
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

    /// Convert to a dict mapping species names to concentrations.
    ///
    /// Returns
    /// -------
    /// dict
    ///     ``{name: concentration}`` in deterministic order (monomers
    ///     first, then complexes, in addition order).
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for name in self.ordered_names() {
            dict.set_item(name, self.concentrations[name])?;
        }
        Ok(dict)
    }

    /// Species names in deterministic order.
    ///
    /// Returns
    /// -------
    /// list of str
    ///     Names in order: monomers first, then complexes.
    fn keys(&self) -> Vec<String> {
        self.ordered_names().cloned().collect()
    }

    /// Concentrations in deterministic order.
    ///
    /// Returns
    /// -------
    /// list of float
    ///     Concentrations in molar, monomers first, then complexes.
    fn values(&self) -> Vec<f64> {
        self.ordered_names().map(|n| self.concentrations[n]).collect()
    }

    /// ``(name, concentration)`` pairs in deterministic order.
    ///
    /// Returns
    /// -------
    /// list of (str, float)
    ///     Pairs in order: monomers first, then complexes.
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
