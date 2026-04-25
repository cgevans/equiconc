//! Liquid-water properties used to convert between molarity and mole
//! fraction in the solver wrapper layers.

/// Molar density of liquid water at `t_c` degrees Celsius, returned in
/// mol/L (= mol/dm³).
///
/// Implemented from the empirical mass-density formula of Tanaka et al.
/// 2001, *Metrologia* 38, 301–309 (VSMOW water, 0–40 °C), converted to
/// molar units via the IUPAC-recommended molar mass of H₂O,
/// `M = 18.015 28 g/mol` (Meija et al. 2016, *Pure Appl. Chem.* 88, 265).
#[must_use]
pub fn water_molar_density(t_c: f64) -> f64 {
    // Tanaka et al. 2001, Table 1 (constants for VSMOW).
    const A1: f64 = -3.983_035_f64;
    const A2: f64 = 301.797_f64;
    const A3: f64 = 522_528.9_f64;
    const A4: f64 = 69.348_81_f64;
    const RHO_MAX_KG_PER_M3: f64 = 999.974_950_f64;

    // Molar mass of water (g/mol).
    const M_WATER_G_PER_MOL: f64 = 18.015_28_f64;

    let offset = t_c + A1;
    let mass_density = RHO_MAX_KG_PER_M3 * (1.0 - offset * offset * (t_c + A2) / (A3 * (t_c + A4)));
    mass_density / M_WATER_G_PER_MOL
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn density_water_matches_reference_values() {
        let cases = [
            (0.0_f64, 55.498_f64),
            (25.0_f64, 55.343_f64),
            (37.0_f64, 55.137_f64),
        ];
        for (t, expected) in cases {
            let got = water_molar_density(t);
            assert!(
                (got - expected).abs() / expected < 1e-3,
                "ρ(T={t} °C) = {got}, expected ≈ {expected}"
            );
        }
    }
}
