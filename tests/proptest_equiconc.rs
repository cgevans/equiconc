use equiconc::System;
use proptest::prelude::*;

const R: f64 = 1.987204e-3;
const MONOMER_NAMES: [&str; 4] = ["A", "B", "C", "D"];
const REL_TOL: f64 = 1e-4;

/// Metadata about a generated system for verifying physical invariants.
#[derive(Debug, Clone)]
struct SystemSpec {
    temperature: f64,
    monomers: Vec<(&'static str, f64)>,
    complexes: Vec<(String, Vec<(&'static str, usize)>, f64)>,
}

impl SystemSpec {
    fn solve(&self) -> Result<equiconc::Equilibrium, equiconc::EquilibriumError> {
        let mut sys = System::new().temperature(self.temperature).unwrap();
        for &(name, conc) in &self.monomers {
            sys = sys.monomer(name, conc).unwrap();
        }
        for (name, comp, dg) in &self.complexes {
            let comp_refs: Vec<(&str, usize)> = comp.iter().copied().collect();
            sys = sys.complex(name, &comp_refs, *dg).unwrap();
        }
        sys.equilibrium()
    }
}

fn log_uniform_concentration() -> impl Strategy<Value = f64> {
    (-6.0f64..=-3.0).prop_map(|e| 10f64.powf(e))
}

fn arb_system() -> impl Strategy<Value = SystemSpec> {
    (293.15..=373.15f64, 1..=4usize)
        .prop_flat_map(|(temp, n_mon)| {
            let concs = prop::collection::vec(log_uniform_concentration(), n_mon);
            (Just(temp), Just(n_mon), concs, 0..=6usize)
        })
        .prop_flat_map(|(temp, n_mon, concs, n_cplx)| {
            let complexes = prop::collection::vec(
                (
                    // For each monomer, a count of 0 (absent) to 3
                    prop::collection::vec(0..=3usize, n_mon),
                    -15.0..=5.0f64,
                ),
                n_cplx,
            );
            (Just(temp), Just(n_mon), Just(concs), complexes)
        })
        .prop_map(|(temp, n_mon, concs, raw_complexes)| {
            let monomers: Vec<_> = (0..n_mon)
                .map(|i| (MONOMER_NAMES[i], concs[i]))
                .collect();
            let complexes: Vec<_> = raw_complexes
                .into_iter()
                .enumerate()
                .map(|(k, (mut counts, dg))| {
                    // Ensure at least one monomer participates
                    if counts.iter().all(|&c| c == 0) {
                        counts[0] = 1;
                    }
                    let comp: Vec<(&'static str, usize)> = counts
                        .into_iter()
                        .enumerate()
                        .filter(|&(_, count)| count > 0)
                        .map(|(i, count)| (MONOMER_NAMES[i], count))
                        .collect();
                    (format!("c{k}"), comp, dg)
                })
                .collect();

            SystemSpec {
                temperature: temp,
                monomers,
                complexes,
            }
        })
}

fn arb_monomer_only() -> impl Strategy<Value = SystemSpec> {
    (293.15..=373.15f64, 1..=4usize)
        .prop_flat_map(|(temp, n_mon)| {
            let concs = prop::collection::vec(log_uniform_concentration(), n_mon);
            (Just(temp), Just(n_mon), concs)
        })
        .prop_map(|(temp, n_mon, concs)| {
            let monomers = (0..n_mon)
                .map(|i| (MONOMER_NAMES[i], concs[i]))
                .collect();
            SystemSpec {
                temperature: temp,
                monomers,
                complexes: Vec::new(),
            }
        })
}

proptest! {
    #[test]
    fn prop_convergence(spec in arb_system()) {
        prop_assert!(spec.solve().is_ok(), "solver failed for {:?}", spec);
    }

    #[test]
    fn prop_mass_conservation(spec in arb_system()) {
        let eq = spec.solve().unwrap();
        for &(mon_name, c0) in &spec.monomers {
            let mut total = eq.concentration(mon_name).unwrap();
            for (cplx_name, comp, _) in &spec.complexes {
                for &(n, count) in comp {
                    if n == mon_name {
                        total += count as f64 * eq.concentration(cplx_name).unwrap();
                    }
                }
            }
            let rel_err = (total - c0).abs() / c0;
            prop_assert!(
                rel_err < REL_TOL,
                "mass conservation violated for {}: total={}, c0={}, rel_err={}",
                mon_name, total, c0, rel_err
            );
        }
    }

    #[test]
    fn prop_equilibrium_condition(spec in arb_system()) {
        let eq = spec.solve().unwrap();
        let rt = R * spec.temperature;
        for (cplx_name, comp, dg) in &spec.complexes {
            let k_eq = (-dg / rt).exp();
            let mut product = 1.0;
            for &(mon_name, count) in comp {
                product *= eq.concentration(mon_name).unwrap().powi(count as i32);
            }
            let expected = k_eq * product;
            let actual = eq.concentration(cplx_name).unwrap();
            let rel_err = (actual - expected).abs() / (expected + 1e-300);
            prop_assert!(
                rel_err < REL_TOL,
                "equilibrium violated for {}: actual={}, expected={}, rel_err={}",
                cplx_name, actual, expected, rel_err
            );
        }
    }

    #[test]
    fn prop_concentrations_non_negative(spec in arb_system()) {
        let eq = spec.solve().unwrap();
        for &(name, _) in &spec.monomers {
            let c = eq.concentration(name).unwrap();
            prop_assert!(c >= 0.0, "negative concentration for {}: {}", name, c);
        }
        for (name, _, _) in &spec.complexes {
            let c = eq.concentration(name).unwrap();
            prop_assert!(c >= 0.0, "negative concentration for {}: {}", name, c);
        }
    }

    #[test]
    fn prop_monomer_only_identity(spec in arb_monomer_only()) {
        let eq = spec.solve().unwrap();
        for &(name, c0) in &spec.monomers {
            let c = eq.concentration(name).unwrap();
            prop_assert!(
                (c - c0).abs() < 1e-15 * c0,
                "monomer-only: {} = {} (expected {})", name, c, c0
            );
        }
    }

    #[test]
    fn prop_dimerization_analytical(
        (c0, dg, temp) in (
            log_uniform_concentration(),
            -15.0..=5.0f64,
            293.15..=373.15f64,
        )
    ) {
        let sys = System::new()
            .temperature(temp).unwrap()
            .monomer("A", c0).unwrap()
            .monomer("B", c0).unwrap()
            .complex("AB", &[("A", 1), ("B", 1)], dg).unwrap();
        let eq = sys.equilibrium().unwrap();

        let rt = R * temp;
        let k = (-dg / rt).exp();
        // Numerically stable closed-form: avoid catastrophic cancellation
        let disc = (4.0 * k * c0 + 1.0).sqrt();
        let free = 2.0 * c0 / (disc + 1.0);
        let x = k * free * free;

        let tol = 1e-6;
        let a_conc = eq.concentration("A").unwrap();
        let b_conc = eq.concentration("B").unwrap();
        let ab_conc = eq.concentration("AB").unwrap();

        prop_assert!(
            (a_conc - free).abs() / (free + 1e-300) < tol,
            "[A]={} != expected {}", a_conc, free
        );
        prop_assert!(
            (b_conc - free).abs() / (free + 1e-300) < tol,
            "[B]={} != expected {}", b_conc, free
        );
        prop_assert!(
            (ab_conc - x).abs() / (x + 1e-300) < tol,
            "[AB]={} != expected {}", ab_conc, x
        );
    }
}
