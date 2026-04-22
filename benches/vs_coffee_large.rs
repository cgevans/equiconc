//! Comparative benchmarks on the large testcases shipped with the
//! upstream COFFEE repository (`coffeesolverdev/coffee/testcases`).
//!
//! The testcases are not vendored in this repo — set `COFFEE_TESTCASES`
//! to the directory containing subdirectories `0/`, `1/`, `2/` if
//! they live somewhere other than `/tmp/coffeesolverdev/testcases`.
//!
//! File format:
//! - `input.con`: one monomer concentration per line (`n_mon` lines)
//! - `input.ocx`: tab-separated, one row per species. NUPACK layout:
//!     `id \t size \t count_1 \t ... \t count_m \t energy`
//!   where `energy` is ΔG/RT (0 for free monomers). The first `n_mon`
//!   rows are the identity block.

use coffee::{extras::OptimizerArgs, optimize::Optimizer};
use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use equiconc::{SolverOptions, System};
use ndarray::{Array1, Array2};
use std::fs;
use std::path::{Path, PathBuf};

/// Parsed numerical inputs shared by both solvers.
struct TestcaseInputs {
    name: String,
    n_mon: usize,
    n_species: usize,
    /// `n_species × n_mon`. Identity block in the first `n_mon` rows.
    at: Array2<f64>,
    /// `n_species`. `-ΔG/RT` per species (0 for monomers).
    log_q: Array1<f64>,
    /// `n_mon`. Monomer concentrations.
    c0: Array1<f64>,
    /// Same as `at`, kept as a separate clone for COFFEE's API.
    coffee_at: Array2<f64>,
    /// Same as `c0`, kept as a separate clone for COFFEE's API.
    coffee_c0: Array1<f64>,
    /// `n_species`. `ΔG/RT` per species (COFFEE's `q_nonexp`, 0 for monomers).
    coffee_q_nonexp: Array1<f64>,
}

fn testcase_root() -> PathBuf {
    let default = "/tmp/coffeesolverdev/testcases";
    std::env::var("COFFEE_TESTCASES")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(default))
}

fn load_testcase(name: &str, dir: &Path) -> TestcaseInputs {
    let con_text = fs::read_to_string(dir.join("input.con"))
        .unwrap_or_else(|e| panic!("read input.con for {name}: {e}"));
    let ocx_text = fs::read_to_string(dir.join("input.ocx"))
        .unwrap_or_else(|e| panic!("read input.ocx for {name}: {e}"));

    let c0_vec: Vec<f64> = con_text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            l.trim()
                .parse::<f64>()
                .unwrap_or_else(|e| panic!("parse c0 line {l:?}: {e}"))
        })
        .collect();
    let n_mon = c0_vec.len();

    // Collect ocx rows: each gives (counts[m], energy).
    let rows: Vec<(Vec<f64>, f64)> = ocx_text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let fields: Vec<&str> = l.split('\t').collect();
            // NUPACK: id, size, counts (m), energy  → width == m + 3.
            assert_eq!(
                fields.len(),
                n_mon + 3,
                "unexpected ocx row width {} (expected {}): {}",
                fields.len(),
                n_mon + 3,
                l
            );
            let counts: Vec<f64> = fields[2..2 + n_mon]
                .iter()
                .map(|f| {
                    f.parse::<f64>()
                        .unwrap_or_else(|e| panic!("count parse {f:?}: {e}"))
                })
                .collect();
            let energy: f64 = fields[2 + n_mon]
                .parse()
                .unwrap_or_else(|e| panic!("energy parse {:?}: {e}", fields[2 + n_mon]));
            (counts, energy)
        })
        .collect();
    let n_species = rows.len();

    // Sanity: first n_mon rows are the identity block.
    for i in 0..n_mon {
        for j in 0..n_mon {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_eq!(
                rows[i].0[j], expected,
                "non-identity in monomer row {i}, col {j}"
            );
        }
        assert_eq!(rows[i].1, 0.0, "non-zero energy on monomer row {i}");
    }

    let mut at = Array2::zeros((n_species, n_mon));
    let mut log_q = Array1::zeros(n_species);
    let mut coffee_q_nonexp = Array1::zeros(n_species);
    // COFFEE internally clamps q_nonexp at SMALLEST_EXP_VALUE = -230.
    // We set the same cap on equiconc via `SolverOptions::log_q_clamp`
    // below (see `bench_equiconc`). Pass raw unclamped log_q; the
    // library applies the clamp at construction time.
    for (i, (counts, energy)) in rows.iter().enumerate() {
        for (j, &c) in counts.iter().enumerate() {
            at[[i, j]] = c;
        }
        // ocx energy column = ΔG/RT (0 for monomers).
        // equiconc log_q = -ΔG/RT; COFFEE q_nonexp = ΔG/RT (it does exp(-)).
        log_q[i] = -energy;
        coffee_q_nonexp[i] = *energy;
    }

    let c0 = Array1::from_vec(c0_vec);
    let coffee_at = Array2::from_shape_vec((n_species, n_mon), at.iter().copied().collect())
        .expect("coffee_at shape");
    let coffee_c0 = Array1::from_vec(c0.to_vec());

    TestcaseInputs {
        name: name.to_string(),
        n_mon,
        n_species,
        at,
        log_q,
        c0,
        coffee_at,
        coffee_c0,
        coffee_q_nonexp,
    }
}

fn coffee_args() -> OptimizerArgs {
    OptimizerArgs {
        scalarity: false,
        use_terminal: false,
        verbose: false,
        ..OptimizerArgs::default()
    }
}

fn bench_equiconc(bencher: &mut criterion::Bencher<'_>, tc: &TestcaseInputs) {
    // Build a fresh System per batch (allocates work buffers and seeds λ
    // = ln(c0)); the timed body is just solve(). This mirrors what
    // bench_coffee does with Optimizer::new in its setup.
    let opts = SolverOptions {
        log_q_clamp: Some(230.0),
        ..Default::default()
    };
    bencher.iter_batched(
        || {
            System::from_arrays_with_options(
                tc.at.clone(),
                tc.log_q.clone(),
                tc.c0.clone(),
                opts.clone(),
            )
            .expect("from_arrays")
        },
        |mut sys| {
            sys.solve().unwrap();
        },
        BatchSize::SmallInput,
    );
}

fn bench_coffee(bencher: &mut criterion::Bencher<'_>, tc: &TestcaseInputs) {
    let args = coffee_args();
    bencher.iter_batched(
        || Optimizer::new(&tc.coffee_c0, &tc.coffee_at, &tc.coffee_q_nonexp, &args)
            .expect("Optimizer::new"),
        |mut opt| {
            opt.optimize(1.0).unwrap();
        },
        BatchSize::SmallInput,
    );
}

fn bench_large_testcases(c: &mut Criterion) {
    let root = testcase_root();
    if !root.exists() {
        eprintln!(
            "testcase root {root:?} not found; set COFFEE_TESTCASES to override. Skipping."
        );
        return;
    }

    let mut group = c.benchmark_group("large_testcases");
    // Large systems: one sample is hundreds of ms; use fewer samples and
    // a short measurement window so the whole suite finishes in minutes.
    group.sample_size(20);

    for name in ["0", "1", "2"] {
        let dir = root.join(name);
        if !dir.exists() {
            eprintln!("skip {dir:?} (missing)");
            continue;
        }
        let tc = load_testcase(name, &dir);
        let label = format!("{}_m{}_n{}", tc.name, tc.n_mon, tc.n_species - tc.n_mon);

        group.bench_with_input(BenchmarkId::new("equiconc", &label), &tc, |b, tc| {
            bench_equiconc(b, tc);
        });
        group.bench_with_input(BenchmarkId::new("coffee", &label), &tc, |b, tc| {
            bench_coffee(b, tc);
        });
    }

    group.finish();
}

criterion_group!(benches, bench_large_testcases);
criterion_main!(benches);
