use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use equiconc::SystemBuilder;

const NM: f64 = 1e-9;

/// Build a builder with `n` monomers and all pairwise dimers (n choose 2).
fn all_pairs_builder(n: usize) -> SystemBuilder {
    let mut b = SystemBuilder::new();
    let names: Vec<String> = (0..n).map(|i| format!("S{i}")).collect();

    for name in &names {
        b = b.monomer(name, 100.0 * NM);
    }

    let mut dg = -9.0;
    for i in 0..n {
        for j in (i + 1)..n {
            let cname = format!("{}{}", names[i], names[j]);
            b = b.complex(&cname, &[(&names[i], 1), (&names[j], 1)], dg);
            dg -= 0.1; // vary ΔG slightly so each dimer is distinct
        }
    }
    b
}

fn bench_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling");
    for (n_mon, n_cplx) in [(2, 1), (4, 6), (6, 15), (8, 28)] {
        let builder = all_pairs_builder(n_mon);
        group.bench_with_input(
            BenchmarkId::new("monomers_complexes", format!("{n_mon}_{n_cplx}")),
            &builder,
            |b, builder| {
                b.iter(|| {
                    let mut sys = builder.clone().build().unwrap();
                    sys.solve().unwrap();
                })
            },
        );
    }
    group.finish();
}

fn bench_binding_strength(c: &mut Criterion) {
    let mut group = c.benchmark_group("binding_strength");
    for dg in [-1.0, -10.0, -25.0] {
        let builder = SystemBuilder::new()
            .monomer("A", 100.0 * NM)
            .monomer("B", 100.0 * NM)
            .complex("AB", &[("A", 1), ("B", 1)], dg);
        group.bench_with_input(
            BenchmarkId::new("dG", format!("{dg:.0}")),
            &builder,
            |b, builder| {
                b.iter(|| {
                    let mut sys = builder.clone().build().unwrap();
                    sys.solve().unwrap();
                })
            },
        );
    }
    group.finish();
}

fn bench_stoichiometry(c: &mut Criterion) {
    let mut group = c.benchmark_group("stoichiometry");

    let ab = SystemBuilder::new()
        .monomer("A", 100.0 * NM)
        .monomer("B", 100.0 * NM)
        .complex("AB", &[("A", 1), ("B", 1)], -10.0);
    group.bench_with_input(BenchmarkId::new("complex", "1:1"), &ab, |b, builder| {
        b.iter(|| {
            let mut sys = builder.clone().build().unwrap();
            sys.solve().unwrap();
        })
    });

    let a2b3 = SystemBuilder::new()
        .monomer("A", 500.0 * NM)
        .monomer("B", 500.0 * NM)
        .complex("A2B3", &[("A", 2), ("B", 3)], -20.0);
    group.bench_with_input(BenchmarkId::new("complex", "2:3"), &a2b3, |b, builder| {
        b.iter(|| {
            let mut sys = builder.clone().build().unwrap();
            sys.solve().unwrap();
        })
    });

    group.finish();
}

fn bench_no_complexes(c: &mut Criterion) {
    let builder = SystemBuilder::new()
        .monomer("A", 100.0 * NM)
        .monomer("B", 100.0 * NM);
    c.bench_function("no_complexes", |b| {
        b.iter(|| {
            let mut sys = builder.clone().build().unwrap();
            sys.solve().unwrap();
        })
    });
}

/// Warm-path bench: build the System once, then measure just the solver
/// on repeated calls where a single c0 entry is bumped each iteration
/// (otherwise `solve()` is a no-op on a fresh system).
fn bench_warm_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("warm_sweep");
    for n_mon in [4, 8] {
        let n_cplx = n_mon * (n_mon - 1) / 2;
        let mut sys = all_pairs_builder(n_mon).build().unwrap();
        // Prime the warm-start λ with one solve so the steady state is
        // measured, not the cold start.
        sys.solve().unwrap();

        let base_c0 = sys.c0()[0];
        let mut tick: f64 = 0.0;
        group.bench_function(
            BenchmarkId::new("solve_after_bump", format!("m{n_mon}_n{n_cplx}")),
            |b| {
                b.iter(|| {
                    // A tiny perturbation forces a non-trivial re-solve
                    // while remaining close to the warm λ.
                    tick = (tick + 1.0) % 16.0;
                    sys.set_c0(0, base_c0 * (1.0 + 1e-6 * tick));
                    sys.solve().unwrap();
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_scaling,
    bench_binding_strength,
    bench_stoichiometry,
    bench_no_complexes,
    bench_warm_sweep,
);
criterion_main!(benches);
