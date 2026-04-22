//! Comparative benchmarks: equiconc vs vendored COFFEE optimizer.
//!
//! Varies both the number of monomers (m) and the number of complexes (n)
//! to identify where each solver is faster and how they scale.

use coffee::{extras::OptimizerArgs, optimize::Optimizer};
use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use equiconc::{SystemBuilder, R};
use ndarray::{Array1, Array2};

const TEMP_K: f64 = 298.15;
const RT: f64 = R * TEMP_K;
const C0: f64 = 100e-9; // 100 nM

/// A pre-built system in both equiconc and COFFEE formats.
struct DualSystem {
    equiconc: SystemBuilder,
    // COFFEE inputs (scalarity=false)
    coffee_monomers: Array1<f64>,
    coffee_polymers: Array2<f64>,
    coffee_q_nonexp: Array1<f64>,
}

/// Generate a system with `n_mon` monomers and `n_cplx` complexes.
///
/// Complexes are generated deterministically: complex k uses monomers
/// (k % n_mon) and ((k + 1) % n_mon) with a ΔG that varies slightly.
fn build_system(n_mon: usize, n_cplx: usize) -> DualSystem {
    let names: Vec<String> = (0..n_mon).map(|i| format!("S{i}")).collect();
    let n_species = n_mon + n_cplx;

    let mut builder = SystemBuilder::new().temperature(TEMP_K);
    for name in &names {
        builder = builder.monomer(name, C0);
    }

    let mut coffee_polymers = Array2::zeros((n_species, n_mon));
    let mut coffee_q_nonexp = Array1::zeros(n_species);

    // Identity block for free monomers
    for i in 0..n_mon {
        coffee_polymers[[i, i]] = 1.0;
    }

    for k in 0..n_cplx {
        // Deterministic complex: two monomers per complex
        let i = k % n_mon;
        let j = (k + 1 + k / n_mon) % n_mon;
        let j = if j == i { (i + 1) % n_mon } else { j };

        // Vary ΔG slightly per complex
        let dg = -10.0 - 0.001 * (k as f64);

        let comp = if i == j {
            vec![(&names[i] as &str, 2)]
        } else {
            vec![(&names[i] as &str, 1), (&names[j] as &str, 1)]
        };
        builder = builder.complex(&format!("c{k}"), &comp, dg);

        // COFFEE matrix row
        if i == j {
            coffee_polymers[[n_mon + k, i]] = 2.0;
        } else {
            coffee_polymers[[n_mon + k, i]] = 1.0;
            coffee_polymers[[n_mon + k, j]] = 1.0;
        }
        coffee_q_nonexp[n_mon + k] = dg / RT;
    }

    let coffee_monomers = Array1::from_vec(vec![C0; n_mon]);

    DualSystem {
        equiconc: builder,
        coffee_monomers,
        coffee_polymers,
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

/// Bench the equiconc solver alone: setup (clone the builder and validate
/// + compile it into a `System`) is hoisted into `iter_batched`'s setup
/// closure and is excluded from the measurement. Only `solve()` is timed.
fn bench_equiconc(
    bencher: &mut criterion::Bencher<'_>,
    builder: &SystemBuilder,
) {
    bencher.iter_batched(
        || builder.clone().build().unwrap(),
        |mut sys| {
            sys.solve().unwrap();
        },
        BatchSize::SmallInput,
    );
}

/// Bench the COFFEE solver alone: `Optimizer::new` is in setup; only
/// `optimize` is timed.
fn bench_coffee(
    bencher: &mut criterion::Bencher<'_>,
    monomers: &Array1<f64>,
    polymers: &Array2<f64>,
    q_nonexp: &Array1<f64>,
) {
    let args = coffee_args();
    bencher.iter_batched(
        || Optimizer::new(monomers, polymers, q_nonexp, &args).unwrap(),
        |mut opt| opt.optimize(1.0).unwrap(),
        BatchSize::SmallInput,
    );
}

/// Benchmark scaling with number of monomers (all-pairs dimers).
fn bench_monomer_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("monomer_scaling");

    for n_mon in [2, 4, 6, 8, 10] {
        let n_cplx = n_mon * (n_mon - 1) / 2; // all pairs
        let ds = build_system(n_mon, n_cplx);

        let label = format!("m{n_mon}_n{n_cplx}");
        group.bench_with_input(
            BenchmarkId::new("equiconc", &label), &ds,
            |b, ds| bench_equiconc(b, &ds.equiconc),
        );
        group.bench_with_input(
            BenchmarkId::new("coffee", &label), &ds,
            |b, ds| bench_coffee(b, &ds.coffee_monomers, &ds.coffee_polymers, &ds.coffee_q_nonexp),
        );
    }
    group.finish();
}

/// Benchmark scaling with number of complexes (fixed monomers).
fn bench_complex_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("complex_scaling_m4");
    let n_mon = 4;

    for n_cplx in [6, 50, 100, 500, 1000] {
        let ds = build_system(n_mon, n_cplx);
        let label = format!("n{n_cplx}");
        group.bench_with_input(
            BenchmarkId::new("equiconc", &label), &ds,
            |b, ds| bench_equiconc(b, &ds.equiconc),
        );
        group.bench_with_input(
            BenchmarkId::new("coffee", &label), &ds,
            |b, ds| bench_coffee(b, &ds.coffee_monomers, &ds.coffee_polymers, &ds.coffee_q_nonexp),
        );
    }
    group.finish();
}

/// Benchmark large-scale systems (more monomers, many complexes).
fn bench_large_scale(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_scale_m10");
    let n_mon = 10;

    for n_cplx in [45, 100, 500, 1000, 5000] {
        let ds = build_system(n_mon, n_cplx);
        let label = format!("n{n_cplx}");
        group.bench_with_input(
            BenchmarkId::new("equiconc", &label), &ds,
            |b, ds| bench_equiconc(b, &ds.equiconc),
        );
        group.bench_with_input(
            BenchmarkId::new("coffee", &label), &ds,
            |b, ds| bench_coffee(b, &ds.coffee_monomers, &ds.coffee_polymers, &ds.coffee_q_nonexp),
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_monomer_scaling,
    bench_complex_scaling,
    bench_large_scale,
);
criterion_main!(benches);
