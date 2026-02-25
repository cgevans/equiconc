use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use equiconc::System;

const NM: f64 = 1e-9;

/// Build a system with `n` monomers and all pairwise dimers (n choose 2).
fn all_pairs_system(n: usize) -> System {
    let mut sys = System::new();
    let names: Vec<String> = (0..n).map(|i| format!("S{i}")).collect();

    for name in &names {
        sys = sys.monomer(name, 100.0 * NM).unwrap();
    }

    let mut dg = -9.0;
    for i in 0..n {
        for j in (i + 1)..n {
            let cname = format!("{}{}", names[i], names[j]);
            sys = sys
                .complex(&cname, &[(&names[i], 1), (&names[j], 1)], dg)
                .unwrap();
            dg -= 0.1; // vary ΔG slightly so each dimer is distinct
        }
    }
    sys
}

fn bench_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling");
    for (n_mon, n_cplx) in [(2, 1), (4, 6), (6, 15), (8, 28)] {
        let sys = all_pairs_system(n_mon);
        group.bench_with_input(
            BenchmarkId::new("monomers_complexes", format!("{n_mon}_{n_cplx}")),
            &sys,
            |b, sys| b.iter(|| sys.equilibrium().unwrap()),
        );
    }
    group.finish();
}

fn bench_binding_strength(c: &mut Criterion) {
    let mut group = c.benchmark_group("binding_strength");
    for dg in [-1.0, -10.0, -25.0] {
        let sys = System::new()
            .monomer("A", 100.0 * NM).unwrap()
            .monomer("B", 100.0 * NM).unwrap()
            .complex("AB", &[("A", 1), ("B", 1)], dg)
            .unwrap();
        group.bench_with_input(
            BenchmarkId::new("dG", format!("{dg:.0}")),
            &sys,
            |b, sys| b.iter(|| sys.equilibrium().unwrap()),
        );
    }
    group.finish();
}

fn bench_stoichiometry(c: &mut Criterion) {
    let mut group = c.benchmark_group("stoichiometry");

    let ab = System::new()
        .monomer("A", 100.0 * NM).unwrap()
        .monomer("B", 100.0 * NM).unwrap()
        .complex("AB", &[("A", 1), ("B", 1)], -10.0)
        .unwrap();
    group.bench_with_input(BenchmarkId::new("complex", "1:1"), &ab, |b, sys| {
        b.iter(|| sys.equilibrium().unwrap())
    });

    let a2b3 = System::new()
        .monomer("A", 500.0 * NM).unwrap()
        .monomer("B", 500.0 * NM).unwrap()
        .complex("A2B3", &[("A", 2), ("B", 3)], -20.0)
        .unwrap();
    group.bench_with_input(BenchmarkId::new("complex", "2:3"), &a2b3, |b, sys| {
        b.iter(|| sys.equilibrium().unwrap())
    });

    group.finish();
}

fn bench_no_complexes(c: &mut Criterion) {
    let sys = System::new()
        .monomer("A", 100.0 * NM).unwrap()
        .monomer("B", 100.0 * NM).unwrap();
    c.bench_function("no_complexes", |b| {
        b.iter(|| sys.equilibrium().unwrap())
    });
}

criterion_group!(
    benches,
    bench_scaling,
    bench_binding_strength,
    bench_stoichiometry,
    bench_no_complexes,
);
criterion_main!(benches);
