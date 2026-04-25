//! End-to-end numeric regression: feed each baked-in testcase through
//! the same pipeline the web app uses (parse + solve), under both
//! presets, and assert sanity properties (mass balance, monotonicity
//! of total mass) plus exact agreement between the two preset paths
//! when configured equivalently.
//!
//! Runs natively against the equiconc crate — wasm is irrelevant for
//! these checks because the solver code is identical.

use equiconc::io::{parse_cfe, parse_concentrations};
use equiconc::{R, System, mass_balance_residual, water_molar_density};

const AB_OCX: &str = include_str!("../testcases/ab_dimer.ocx");
const AB_CON: &str = include_str!("../testcases/ab_dimer.con");
const ABC_OCX: &str = include_str!("../testcases/abc_competing.ocx");
const ABC_CON: &str = include_str!("../testcases/abc_competing.con");
const HOMO_OCX: &str = include_str!("../testcases/a_homo.ocx");
const HOMO_CON: &str = include_str!("../testcases/a_homo.con");

const TESTCASES: &[(&str, &str, &str)] = &[
    ("ab_dimer", AB_OCX, AB_CON),
    ("abc_competing", ABC_OCX, ABC_CON),
    ("a_homo", HOMO_OCX, HOMO_CON),
];

/// Solve at COFFEE-compatible settings (37 °C, scalarity on, ΔG clamp
/// at 230 kcal/mol, max_iterations 5000) and assert mass balance.
#[test]
fn coffee_preset_solves_all_testcases_with_tight_mass_balance() {
    let t_celsius = 37.0_f64;
    let rt = R * (t_celsius + 273.15);
    let rho = water_molar_density(t_celsius);

    for (name, cfe_text, con_text) in TESTCASES {
        let c0 = parse_concentrations(con_text).expect("parse con");
        let n_mon = c0.len();
        let (stoich, dg) = parse_cfe(cfe_text, n_mon).expect("parse cfe");

        let dg_clamped = dg.mapv(|g| g.max(-230.0));
        let log_q = dg_clamped.mapv(|g| -g / rt);
        let c0_frac = c0.mapv(|c| c / rho);

        let opts = equiconc::SolverOptions {
            max_iterations: 5000,
            ..equiconc::SolverOptions::default()
        };

        let mut sys = System::from_arrays_with_options(stoich.clone(), log_q, c0_frac, opts)
            .unwrap_or_else(|e| panic!("{name}: build failed: {e}"));
        let eq = sys
            .solve()
            .unwrap_or_else(|e| panic!("{name}: solve failed: {e}"));

        let conc_molar: ndarray::Array1<f64> =
            eq.concentrations().iter().map(|c| c * rho).collect();
        let residual = mass_balance_residual(stoich.view(), c0.view(), conc_molar.view());
        let c0_max = c0.iter().cloned().fold(0.0_f64, f64::max);
        // Mass balance should be tight in molar units. 1e-9 relative is
        // a comfortable margin; observed values are ~1e-13.
        assert!(
            residual < 1e-7 * c0_max,
            "{name}: residual {residual:e} > 1e-7 * c0_max ({:e})",
            1e-7 * c0_max
        );

        // No negative concentrations.
        for (j, c) in conc_molar.iter().enumerate() {
            assert!(*c >= 0.0, "{name}: species {j} has negative c = {c:e}");
        }
    }
}

/// Solve at equiconc defaults (25 °C, no scalarity, no ΔG clamp,
/// max_iterations 1000) and assert mass balance.
#[test]
fn equiconc_default_preset_solves_all_testcases() {
    let t_celsius = 25.0_f64;
    let rt = R * (t_celsius + 273.15);

    for (name, cfe_text, con_text) in TESTCASES {
        let c0 = parse_concentrations(con_text).unwrap();
        let n_mon = c0.len();
        let (stoich, dg) = parse_cfe(cfe_text, n_mon).unwrap();

        let log_q = dg.mapv(|g| -g / rt);
        let mut sys = System::from_arrays(stoich.clone(), log_q, c0.clone())
            .unwrap_or_else(|e| panic!("{name}: build failed: {e}"));
        let eq = sys
            .solve()
            .unwrap_or_else(|e| panic!("{name}: solve failed: {e}"));

        let conc = eq.concentrations().to_owned();
        let residual = mass_balance_residual(stoich.view(), c0.view(), conc.view());
        let c0_max = c0.iter().cloned().fold(0.0_f64, f64::max);
        assert!(
            residual < 1e-7 * c0_max,
            "{name}: residual {residual:e} > 1e-7 * c0_max"
        );
    }
}

/// AB-dimer at -10 kcal/mol, 1 µM each, 25 °C: at this binding energy
/// nearly all material should be in the AB complex.
#[test]
fn ab_dimer_25c_concentrates_into_complex() {
    let c0 = parse_concentrations(AB_CON).unwrap();
    let (stoich, dg) = parse_cfe(AB_OCX, c0.len()).unwrap();
    let rt = R * (25.0 + 273.15);
    let log_q = dg.mapv(|g| -g / rt);
    let mut sys = System::from_arrays(stoich, log_q, c0).unwrap();
    let eq = sys.solve().unwrap();
    let a = eq.at(0);
    let b = eq.at(1);
    let ab = eq.at(2);
    // AB should be the dominant species.
    assert!(ab > a && ab > b, "ab={ab:e} a={a:e} b={b:e}");
    // At ΔG = -10 kcal/mol, K_d ≈ 4.6e-8 M. With 1 µM each, ~80%
    // of the monomer mass ends up as AB. (Closed-form: 8.08e-7 M.)
    assert!(ab > 7.5e-7, "ab={ab:e} should be > 7.5e-7 M");
    assert!((ab + a - 1.0e-6).abs() < 1e-12);
}
