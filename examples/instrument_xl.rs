//! Diagnostic on a synthetic rgrow-scale system (m≈1000).
//!
//! Builds a reproducible system with `m` monomers and `n_cplx` complexes,
//! each complex a random small combination of 2–6 monomers with ΔG drawn
//! from a realistic range (−30 … −5 kcal/mol at 298 K). Reports iteration
//! counts and wall time for both solvers.

use coffee::{extras::OptimizerArgs, optimize::Optimizer};
use equiconc::{R, System};
use ndarray::{Array1, Array2};
use std::time::Instant;

const TEMP_K: f64 = 298.15;
const RT: f64 = R * TEMP_K;

/// Lightweight deterministic PRNG (splitmix64) so this example is
/// reproducible without pulling in `rand` as a dev-dep.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn uniform_range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.uniform()
    }
    fn usize_range(&mut self, lo: usize, hi: usize) -> usize {
        lo + (self.next() as usize) % (hi - lo)
    }
}

struct Synth {
    at: Array2<f64>,
    log_q: Array1<f64>,
    c0: Array1<f64>,
    coffee_at: Array2<f64>,
    coffee_c0: Array1<f64>,
    coffee_q_nonexp: Array1<f64>,
}

fn build_synth(m: usize, n_cplx: usize, seed: u64) -> Synth {
    let n_species = m + n_cplx;
    let mut at = Array2::<f64>::zeros((n_species, m));
    let mut log_q = Array1::<f64>::zeros(n_species);
    let mut coffee_q_nonexp = Array1::<f64>::zeros(n_species);

    for i in 0..m {
        at[[i, i]] = 1.0;
    }

    let mut rng = Rng(seed);

    // Log-uniform monomer concentrations in [1e-9, 1e-7].
    let mut c0 = Array1::<f64>::zeros(m);
    for i in 0..m {
        let log_c = rng.uniform_range(-9.0f64.ln() / std::f64::consts::LOG10_E, -7.0);
        c0[i] = 10f64.powf(log_c);
    }
    for i in 0..m {
        c0[i] = 10f64.powf(rng.uniform_range(-9.0, -7.0));
    }

    for k in 0..n_cplx {
        let size = rng.usize_range(2, 7); // 2..=6 strands
        let row = m + k;
        let mut total_strands = 0;
        for _ in 0..size {
            let j = rng.usize_range(0, m);
            at[[row, j]] += 1.0;
            total_strands += 1;
        }
        // ΔG ≈ −(3 … 8) kcal/mol per strand interaction
        let dg = -(rng.uniform_range(3.0, 8.0) * total_strands as f64);
        log_q[row] = -dg / RT;
        coffee_q_nonexp[row] = dg / RT;
    }

    let coffee_at = Array2::from_shape_vec((n_species, m), at.iter().copied().collect())
        .expect("coffee_at shape");
    let coffee_c0 = Array1::from_vec(c0.to_vec());

    Synth {
        at,
        log_q,
        c0,
        coffee_at,
        coffee_c0,
        coffee_q_nonexp,
    }
}

fn time_equiconc(s: &Synth) -> (usize, f64, bool) {
    let t = Instant::now();
    let mut sys = System::from_arrays(s.at.clone(), s.log_q.clone(), s.c0.clone()).unwrap();
    match sys.solve() {
        Ok(eq) => (eq.iterations(), t.elapsed().as_secs_f64(), true),
        Err(e) => {
            eprintln!("  equiconc failed: {e}");
            (0, t.elapsed().as_secs_f64(), false)
        }
    }
}

// Upstream COFFEE doesn't expose an iteration count; only wall time.
fn time_coffee(s: &Synth) -> (f64, bool) {
    let args = OptimizerArgs {
        scalarity: false,
        use_terminal: false,
        verbose: false,
        ..OptimizerArgs::default()
    };
    let t = Instant::now();
    let mut opt =
        Optimizer::new(&s.coffee_c0, &s.coffee_at, &s.coffee_q_nonexp, &args).unwrap();
    match opt.optimize(1.0) {
        Ok(_) => (t.elapsed().as_secs_f64(), true),
        Err(e) => {
            eprintln!("  coffee failed: {e}");
            (t.elapsed().as_secs_f64(), false)
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let default_cases: &[(usize, usize)] = &[
        (100, 200),
        (250, 500),
        (500, 1000),
        (1000, 2000),
        (1000, 5000),
        (2000, 4000),
    ];
    let cases: Vec<(usize, usize)> = if args.len() >= 3 {
        let m: usize = args[1].parse().unwrap();
        let n: usize = args[2].parse().unwrap();
        vec![(m, n)]
    } else {
        default_cases.to_vec()
    };

    // `co_iters` is omitted because upstream COFFEE doesn't expose the
    // iteration count; coffee is reported by total wall time only.
    println!(
        "{:>6} {:>6} {:>16} {:>10} {:>10}",
        "m", "n_cplx", "equiconc (s/it)", "coffee (s)", "eq_iters"
    );
    println!("{}", "-".repeat(70));

    for &(m, n_cplx) in &cases {
        let s = build_synth(m, n_cplx, 0xCAFE_F00D);
        let (eq_iters, eq_secs, eq_ok) = time_equiconc(&s);
        let (co_secs, co_ok) = time_coffee(&s);
        let eq_str = if eq_ok {
            format!(
                "{:>6.3} ({:>4.1})",
                eq_secs,
                eq_secs / eq_iters.max(1) as f64 * 1e3
            )
        } else {
            "FAIL".to_string()
        };
        let co_str = if co_ok {
            format!("{co_secs:>6.3}")
        } else {
            "FAIL".to_string()
        };
        println!(
            "{:>6} {:>6} {:>16} {:>10} {:>10}",
            m, n_cplx, eq_str, co_str, eq_iters
        );
    }
}
