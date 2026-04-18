//! Diagnostic instrumentation for the large coffeesolverdev testcases.
//!
//! Prints iteration counts for both solvers and a per-phase breakdown
//! of equiconc's inner loop (mat-vec forward, exp sweep, mat-vec back,
//! Hessian assembly) on the loaded `ProblemInputs`, so we can tell
//! whether equiconc's deficit at m=15, n≈50k is iteration-count-driven
//! or per-iteration-work-driven.
//!
//! Usage:
//!   COFFEE_TESTCASES=/tmp/coffeesolverdev/testcases \
//!     cargo run --release --example instrument_large -- <case>
//!
//! `<case>` is the testcase directory name (0, 1, 2). Defaults to `0`.

#[path = "../tests/coffee_vendor/mod.rs"]
mod coffee_vendor;

use coffee_vendor::{Optimizer, OptimizerArgs};
use equiconc::System;
use ndarray::{Array1, Array2};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

struct Testcase {
    at: Array2<f64>,           // n_species × n_mon
    log_q: Array1<f64>,        // n_species (clamped, -ΔG/RT ≤ 230)
    coffee_q_nonexp: Array1<f64>,
    c0: Array1<f64>,
    n_species: usize,
}

const LOG_Q_MAX: f64 = 230.0;

fn load(name: &str, dir: &Path) -> Testcase {
    let con = fs::read_to_string(dir.join("input.con")).expect("input.con");
    let ocx = fs::read_to_string(dir.join("input.ocx")).expect("input.ocx");

    let c0_vec: Vec<f64> = con
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.trim().parse().expect("parse c0"))
        .collect();
    let n_mon = c0_vec.len();

    let rows: Vec<(Vec<f64>, f64)> = ocx
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            assert_eq!(f.len(), n_mon + 3, "ocx width");
            let counts: Vec<f64> =
                f[2..2 + n_mon].iter().map(|s| s.parse().unwrap()).collect();
            let energy: f64 = f[2 + n_mon].parse().unwrap();
            (counts, energy)
        })
        .collect();
    let n_species = rows.len();

    let mut at = Array2::<f64>::zeros((n_species, n_mon));
    let mut log_q = Array1::<f64>::zeros(n_species);
    let mut q_nx = Array1::<f64>::zeros(n_species);
    for (i, (counts, energy)) in rows.iter().enumerate() {
        for (j, &c) in counts.iter().enumerate() {
            at[[i, j]] = c;
        }
        log_q[i] = (-energy).min(LOG_Q_MAX);
        q_nx[i] = *energy;
    }

    eprintln!(
        "Loaded testcase {name}: m={n_mon}, n_species={n_species}"
    );
    Testcase {
        at,
        log_q,
        coffee_q_nonexp: q_nx,
        c0: Array1::from_vec(c0_vec),
        n_species,
    }
}

fn time_equiconc(tc: &Testcase) -> (usize, f64) {
    let t0 = Instant::now();
    let mut sys =
        System::from_arrays(tc.at.clone(), tc.log_q.clone(), tc.c0.clone()).expect("build");
    let eq = sys.solve().expect("solve");
    let elapsed = t0.elapsed().as_secs_f64();
    let iters = eq.iterations();
    (iters, elapsed)
}

fn time_coffee(tc: &Testcase) -> (usize, f64) {
    let args = OptimizerArgs {
        scalarity: false,
        use_terminal: false,
        verbose: false,
        ..OptimizerArgs::default()
    };
    let t0 = Instant::now();
    let mut opt =
        Optimizer::new(&tc.c0, &tc.at, &tc.coffee_q_nonexp, &args).expect("Optimizer::new");
    opt.optimize(1.0).expect("optimize");
    let elapsed = t0.elapsed().as_secs_f64();
    (opt.iterations(), elapsed)
}

/// Replicates equiconc's evaluate_into inner loop for measurement.
/// Runs `reps` passes over the given data and returns per-phase totals.
fn time_phases(at: &Array2<f64>, log_q: &Array1<f64>, lambda: &Array1<f64>, reps: usize) -> [f64; 4] {
    let n_species = at.nrows();
    let n_mon = at.ncols();
    let mut c = Array1::<f64>::zeros(n_species);
    let mut grad = Array1::<f64>::zeros(n_mon);
    let mut hessian = Array2::<f64>::zeros((n_mon, n_mon));

    // Phase A: Aᵀλ into c (= A·λ in storage terms — at is n_species × n_mon)
    let mut t_matvec_fwd = 0.0;
    // Phase B: c += log_q; exp in place
    let mut t_exp = 0.0;
    // Phase C: grad = at.t() · c
    let mut t_matvec_back = 0.0;
    // Phase D: Hessian assembly
    let mut t_hessian = 0.0;

    for _ in 0..reps {
        let t = Instant::now();
        ndarray::linalg::general_mat_vec_mul(1.0, at, lambda, 0.0, &mut c);
        t_matvec_fwd += t.elapsed().as_secs_f64();

        let t = Instant::now();
        c += log_q;
        c.mapv_inplace(|lc| lc.min(700.0).exp());
        t_exp += t.elapsed().as_secs_f64();

        let t = Instant::now();
        ndarray::linalg::general_mat_vec_mul(1.0, &at.t(), &c, 0.0, &mut grad);
        t_matvec_back += t.elapsed().as_secs_f64();

        let t = Instant::now();
        hessian.fill(0.0);
        for k in 0..n_species {
            let ck = c[k];
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
                    hessian[[i, j]] += val;
                    if i != j {
                        hessian[[j, i]] += val;
                    }
                }
            }
        }
        t_hessian += t.elapsed().as_secs_f64();
    }

    [
        t_matvec_fwd / reps as f64,
        t_exp / reps as f64,
        t_matvec_back / reps as f64,
        t_hessian / reps as f64,
    ]
}

fn count_nonzeros_per_row(at: &Array2<f64>) -> (f64, usize) {
    let n = at.nrows();
    let mut total = 0usize;
    let mut max_row = 0usize;
    for i in 0..n {
        let mut k = 0;
        for j in 0..at.ncols() {
            if at[[i, j]] != 0.0 {
                k += 1;
            }
        }
        total += k;
        if k > max_row {
            max_row = k;
        }
    }
    (total as f64 / n as f64, max_row)
}

fn main() {
    let case = std::env::args().nth(1).unwrap_or_else(|| "0".to_string());
    let root: PathBuf = std::env::var("COFFEE_TESTCASES")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp/coffeesolverdev/testcases"));

    let dir = root.join(&case);
    let tc = load(&case, &dir);

    // Sparsity / composition density.
    let (avg_nnz, max_nnz) = count_nonzeros_per_row(&tc.at);
    println!(
        "\nStoichiometry density: avg {avg_nnz:.2} non-zeros per species row, max {max_nnz}"
    );

    // End-to-end solve timings + iteration counts.
    println!("\n== Full solve ==");
    let (eq_iters, eq_secs) = time_equiconc(&tc);
    let eq_per_iter = eq_secs / eq_iters as f64;
    println!(
        "  equiconc: {eq_iters} iterations, {eq_secs:.3} s total ({:.1} ms/iter)",
        eq_per_iter * 1e3
    );

    let (co_iters, co_secs) = time_coffee(&tc);
    let co_per_iter = co_secs / co_iters as f64;
    println!(
        "  coffee:   {co_iters} iterations, {co_secs:.3} s total ({:.1} ms/iter)",
        co_per_iter * 1e3
    );

    // Per-phase breakdown of equiconc's evaluate_into at initial λ.
    // Using a moderate number of reps to amortize Instant overhead.
    let lambda_init = tc.c0.mapv(|c| c.ln());
    let reps = if tc.n_species > 10_000 { 10 } else { 200 };
    println!(
        "\n== equiconc evaluate_into phase breakdown (avg of {reps} reps at λ=ln(c0)) =="
    );
    let [t_fwd, t_exp, t_back, t_hess] = time_phases(&tc.at, &tc.log_q, &lambda_init, reps);
    let total_inner = t_fwd + t_exp + t_back + t_hess;
    let pct = |x: f64| format!("{:>5.1}%", 100.0 * x / total_inner);
    println!(
        "  Aᵀλ matvec : {:>8.3} ms  ({})",
        t_fwd * 1e3,
        pct(t_fwd)
    );
    println!("  exp+add    : {:>8.3} ms  ({})", t_exp * 1e3, pct(t_exp));
    println!(
        "  Aᵀᵀc matvec: {:>8.3} ms  ({})",
        t_back * 1e3,
        pct(t_back)
    );
    println!(
        "  Hessian    : {:>8.3} ms  ({})",
        t_hess * 1e3,
        pct(t_hess)
    );
    println!(
        "  ------------------------------\n  total inner: {:>8.3} ms",
        total_inner * 1e3
    );

    // How much of equiconc's per-iter cost is Hessian assembly vs the rest?
    let hessian_fraction_of_iter = t_hess / eq_per_iter;
    let hv_equivalent_per_iter = t_fwd + t_back;
    println!(
        "\n== What switching to matrix-free (Steihaug-CG) would save ==\n\
         Assumption: outer iteration count stays the same, Hessian assembly goes away,\n\
         each CG iteration costs ~(Aᵀλ + Aᵀᵀc) = 2 mat-vecs.\n\
         Projected per-iter cost without Hessian = per-iter − Hessian + k × 2·matvec.\n\
         Current:  per-iter = {:.1} ms (Hessian = {:.1} ms, {:.0}% of per-iter)\n\
         Break-even k (CG iters) so matrix-free ties current cost:\n    k = Hessian / (2·matvec) = {:.1}",
        eq_per_iter * 1e3,
        t_hess * 1e3,
        100.0 * hessian_fraction_of_iter,
        t_hess / (2.0 * (t_fwd + t_back) / 2.0)
    );
    // Note: 2·matvec ≈ t_fwd + t_back. The formula simplifies:
    println!(
        "    (= {:.1} ms / {:.3} ms = {:.1})",
        t_hess * 1e3,
        hv_equivalent_per_iter * 1e3,
        t_hess / hv_equivalent_per_iter
    );
}
