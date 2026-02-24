use nalgebra::{DMatrix, DVector};

/// Gas constant in kcal/(mol·K).
const R: f64 = 1.987204e-3;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum EquilibriumError {
    NoMonomers,
    UnknownMonomer(String),
    EmptyComposition,
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
// Internal types
// ---------------------------------------------------------------------------

struct Monomer {
    name: String,
    total_concentration: f64,
}

struct Complex {
    name: String,
    composition: Vec<(usize, usize)>, // (monomer_index, count)
    delta_g: f64,                      // kcal/mol
}

// ---------------------------------------------------------------------------
// Public builder / system
// ---------------------------------------------------------------------------

pub struct System {
    monomers: Vec<Monomer>,
    complexes: Vec<Complex>,
    temperature: f64, // Kelvin
}

impl System {
    /// Create an empty system at 37 °C (310.15 K).
    pub fn new() -> Self {
        System {
            monomers: Vec::new(),
            complexes: Vec::new(),
            temperature: 310.15,
        }
    }

    /// Set the temperature in Kelvin.
    pub fn temperature(mut self, t: f64) -> Self {
        self.temperature = t;
        self
    }

    /// Add a monomer species with a given total concentration (molar).
    pub fn monomer(mut self, name: &str, total_concentration: f64) -> Self {
        self.monomers.push(Monomer {
            name: name.to_string(),
            total_concentration,
        });
        self
    }

    /// Add a complex with the given composition and standard free energy of
    /// formation (ΔG° in kcal/mol). Composition entries are `(monomer_name, count)`.
    pub fn complex(
        mut self,
        name: &str,
        composition: &[(&str, usize)],
        delta_g: f64,
    ) -> Result<Self, EquilibriumError> {
        if composition.is_empty() {
            return Err(EquilibriumError::EmptyComposition);
        }
        let mut comp = Vec::new();
        for &(monomer_name, count) in composition {
            let idx = self
                .monomers
                .iter()
                .position(|m| m.name == monomer_name)
                .ok_or_else(|| EquilibriumError::UnknownMonomer(monomer_name.to_string()))?;
            comp.push((idx, count));
        }
        self.complexes.push(Complex {
            name: name.to_string(),
            composition: comp,
            delta_g,
        });
        Ok(self)
    }

    /// Build the stoichiometry matrix **A**, the log-partition-function vector
    /// **log_q**, and the total-concentration vector **c⁰**.
    fn build_problem(&self) -> (DMatrix<f64>, DVector<f64>, DVector<f64>) {
        let n_mon = self.monomers.len();
        let n_cplx = self.complexes.len();
        let n_species = n_mon + n_cplx;

        // A: n_monomers × n_species
        let mut a = DMatrix::zeros(n_mon, n_species);
        // Identity block for free monomers
        for i in 0..n_mon {
            a[(i, i)] = 1.0;
        }
        // Complex columns
        for (k, cplx) in self.complexes.iter().enumerate() {
            for &(mi, count) in &cplx.composition {
                a[(mi, n_mon + k)] = count as f64;
            }
        }

        // log_q: 0 for free monomers, -ΔG°/(RT) for complexes
        let mut log_q = DVector::zeros(n_species);
        let rt = R * self.temperature;
        for (k, cplx) in self.complexes.iter().enumerate() {
            log_q[n_mon + k] = -cplx.delta_g / rt;
        }

        // c⁰
        let c0 = DVector::from_iterator(
            n_mon,
            self.monomers.iter().map(|m| m.total_concentration),
        );

        (a, log_q, c0)
    }

    /// Solve for equilibrium concentrations.
    pub fn equilibrium(&self) -> Result<Equilibrium, EquilibriumError> {
        if self.monomers.is_empty() {
            return Err(EquilibriumError::NoMonomers);
        }

        let n_mon = self.monomers.len();

        // Short-circuit: no complexes
        if self.complexes.is_empty() {
            return Ok(Equilibrium {
                monomer_names: self.monomers.iter().map(|m| m.name.clone()).collect(),
                complex_names: Vec::new(),
                free_monomer_concentrations: self
                    .monomers
                    .iter()
                    .map(|m| m.total_concentration)
                    .collect(),
                complex_concentrations: Vec::new(),
            });
        }

        let (a, log_q, c0) = self.build_problem();
        let lambda = solve_dual(&a, &log_q, &c0)?;

        // Recover concentrations: c_j = exp(log_q_j + (Aᵀλ)_j)
        let log_c = &log_q + a.transpose() * &lambda;
        let concentrations: Vec<f64> =
            log_c.iter().map(|&lc| lc.min(700.0).exp()).collect();

        // Debug-mode validation
        #[cfg(debug_assertions)]
        {
            let rt = R * self.temperature;
            // Mass conservation
            for i in 0..n_mon {
                let total: f64 =
                    (0..concentrations.len()).map(|j| a[(i, j)] * concentrations[j]).sum();
                debug_assert!(
                    (total - c0[i]).abs() < 1e-6 * c0[i] + 1e-30,
                    "mass conservation violated for monomer {}: {total} != {}",
                    self.monomers[i].name,
                    c0[i]
                );
            }
            // Equilibrium condition
            for (k, cplx) in self.complexes.iter().enumerate() {
                let k_eq = (-cplx.delta_g / rt).exp();
                let mut product = 1.0;
                for &(mi, count) in &cplx.composition {
                    product *= concentrations[mi].powi(count as i32);
                }
                let expected = k_eq * product;
                let actual = concentrations[n_mon + k];
                let rel_err = (actual - expected).abs() / (expected + 1e-300);
                debug_assert!(
                    rel_err < 1e-6,
                    "equilibrium condition violated for {}: {actual} != {expected} (rel err: {rel_err})",
                    cplx.name
                );
            }
        }

        Ok(Equilibrium {
            monomer_names: self.monomers.iter().map(|m| m.name.clone()).collect(),
            complex_names: self.complexes.iter().map(|c| c.name.clone()).collect(),
            free_monomer_concentrations: concentrations[..n_mon].to_vec(),
            complex_concentrations: concentrations[n_mon..].to_vec(),
        })
    }
}

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

pub struct Equilibrium {
    monomer_names: Vec<String>,
    complex_names: Vec<String>,
    pub free_monomer_concentrations: Vec<f64>,
    pub complex_concentrations: Vec<f64>,
}

impl Equilibrium {
    /// Look up a concentration by species name (monomer or complex).
    pub fn concentration(&self, name: &str) -> Option<f64> {
        if let Some(i) = self.monomer_names.iter().position(|n| n == name) {
            return Some(self.free_monomer_concentrations[i]);
        }
        if let Some(i) = self.complex_names.iter().position(|n| n == name) {
            return Some(self.complex_concentrations[i]);
        }
        None
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
        log_c.iter().map(|&lc| lc.min(700.0).exp()),
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
            // Fallback to scaled steepest-descent
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

/// Solve the dual problem, returning the optimal λ*.
fn solve_dual(
    a: &DMatrix<f64>,
    log_q: &DVector<f64>,
    c0: &DVector<f64>,
) -> Result<DVector<f64>, EquilibriumError> {
    const MAX_ITER: usize = 1000;
    const TOL: f64 = 1e-12;
    const ETA: f64 = 1e-4;
    const DELTA_MAX: f64 = 1e10;

    // Initial guess: λ_i = ln(c⁰_i)
    let mut lambda = DVector::from_iterator(c0.len(), c0.iter().map(|&c| c.ln()));
    let mut delta = 1.0;

    for _ in 0..MAX_ITER {
        let eval = evaluate(a, log_q, c0, &lambda);
        let grad_norm = eval.grad.norm();

        if grad_norm < TOL {
            return Ok(lambda);
        }

        let p = dogleg_step(&eval.grad, &eval.hessian, delta);
        let p_norm = p.norm();

        // Evaluate at candidate point
        let lambda_new = &lambda + &p;
        let eval_new = evaluate(a, log_q, c0, &lambda_new);

        let actual_reduction = eval.f - eval_new.f;
        let predicted_reduction = -(eval.grad.dot(&p) + 0.5 * p.dot(&(&eval.hessian * &p)));

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
        let sys = System::new().monomer("A", 50e-9).monomer("B", 100e-9);
        let eq = sys.equilibrium().unwrap();
        assert_eq!(eq.free_monomer_concentrations[0], 50e-9);
        assert_eq!(eq.free_monomer_concentrations[1], 100e-9);
        assert!(eq.complex_concentrations.is_empty());
    }

    #[test]
    fn simple_dimerization() {
        let c0 = 100.0 * NM;
        let dg = -10.0;
        let sys = System::new()
            .monomer("A", c0)
            .monomer("B", c0)
            .complex("AB", &[("A", 1), ("B", 1)], dg)
            .unwrap();
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
            .complex("AAA", &[("A", 3)], -15.0)
            .unwrap();
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
            .unwrap()
            .complex("aaa", &[("a", 3)], -15.0)
            .unwrap()
            .complex("bc", &[("b", 1), ("c", 1)], -12.0)
            .unwrap();
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
            .complex("AB", &[("A", 1), ("B", 1)], -30.0)
            .unwrap();
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
            .complex("AB", &[("A", 1), ("B", 1)], -10.0)
            .unwrap();
        let eq = sys.equilibrium().unwrap();

        let a = eq.concentration("A").unwrap();
        let b = eq.concentration("B").unwrap();
        let ab = eq.concentration("AB").unwrap();

        assert!(ab <= 100.0 * NM + 1e-12);
        // Mass conservation
        assert!((a + ab - 200.0 * NM).abs() < 1e-8 * 200.0 * NM);
        assert!((b + ab - 100.0 * NM).abs() < 1e-8 * 100.0 * NM);
    }
}
