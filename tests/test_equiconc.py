import math

import pytest

import equiconc

R = 1.987204e-3  # kcal/(mol·K)
NM = 1e-9


def test_no_complexes():
    sys = equiconc.System().monomer("A", 50e-9).monomer("B", 100e-9)
    eq = sys.equilibrium()
    assert eq["A"] == pytest.approx(50e-9)
    assert eq["B"] == pytest.approx(100e-9)
    assert eq.complex_concentrations == []
    assert len(eq) == 2


def test_simple_dimerization():
    c0 = 100.0 * NM
    dg = -10.0
    sys = (
        equiconc.System()
        .monomer("A", c0)
        .monomer("B", c0)
        .complex("AB", [("A", 1), ("B", 1)], dg_st=dg)
    )
    eq = sys.equilibrium()

    # Analytical solution (default temperature is 25 °C = 298.15 K)
    rt = R * 298.15
    k = math.exp(-dg / rt)
    x = ((2 * k * c0 + 1) - math.sqrt(4 * k * c0 + 1)) / (2 * k)
    free = c0 - x

    assert eq["A"] == pytest.approx(free, abs=1e-12)
    assert eq["B"] == pytest.approx(free, abs=1e-12)
    assert eq["AB"] == pytest.approx(x, abs=1e-12)


def test_concentration_method():
    sys = equiconc.System().monomer("A", 50e-9)
    eq = sys.equilibrium()
    assert eq.concentration("A") == pytest.approx(50e-9)
    with pytest.raises(KeyError):
        eq.concentration("nonexistent")


def test_contains():
    sys = equiconc.System().monomer("A", 50e-9)
    eq = sys.equilibrium()
    assert "A" in eq
    assert "B" not in eq


def test_getitem_keyerror():
    sys = equiconc.System().monomer("A", 50e-9)
    eq = sys.equilibrium()
    with pytest.raises(KeyError):
        eq["nonexistent"]


def test_to_dict():
    c0 = 100.0 * NM
    sys = (
        equiconc.System()
        .monomer("A", c0)
        .monomer("B", c0)
        .complex("AB", [("A", 1), ("B", 1)], dg_st=-10.0)
    )
    eq = sys.equilibrium()
    d = eq.to_dict()
    assert isinstance(d, dict)
    assert set(d.keys()) == {"A", "B", "AB"}
    assert all(isinstance(v, float) for v in d.values())


def test_getters():
    c0 = 100.0 * NM
    sys = (
        equiconc.System()
        .monomer("A", c0)
        .monomer("B", c0)
        .complex("AB", [("A", 1), ("B", 1)], dg_st=-10.0)
    )
    eq = sys.equilibrium()
    assert eq.monomer_names == ["A", "B"]
    assert eq.complex_names == ["AB"]
    assert len(eq.free_monomer_concentrations) == 2
    assert len(eq.complex_concentrations) == 1


def test_no_monomers_error():
    sys = equiconc.System()
    with pytest.raises(ValueError, match="no monomers"):
        sys.equilibrium()


def test_unknown_monomer_error():
    sys = equiconc.System().monomer("A", 50e-9).complex("AB", [("A", 1), ("B", 1)], dg_st=-10.0)
    with pytest.raises(ValueError, match="unknown monomer"):
        sys.equilibrium()


def test_empty_composition_error():
    sys = equiconc.System().monomer("A", 50e-9).complex("X", [], dg_st=-10.0)
    with pytest.raises(ValueError, match="empty composition"):
        sys.equilibrium()


def test_custom_temperature_K():
    sys = equiconc.System(temperature_K=300.0).monomer("A", 50e-9)
    eq = sys.equilibrium()
    assert eq["A"] == pytest.approx(50e-9)


def test_custom_temperature_C():
    sys = equiconc.System(temperature_C=25.0).monomer("A", 50e-9)
    eq = sys.equilibrium()
    assert eq["A"] == pytest.approx(50e-9)


def test_repr():
    sys = equiconc.System()
    assert "System" in repr(sys)
    sys2 = sys.monomer("A", 50e-9)
    eq = sys2.equilibrium()
    assert "Equilibrium" in repr(eq)


def test_negative_concentration_error():
    with pytest.raises(ValueError, match="invalid concentration"):
        equiconc.System().monomer("A", -1e-9).equilibrium()


def test_zero_concentration_error():
    with pytest.raises(ValueError, match="invalid concentration"):
        equiconc.System().monomer("A", 0.0).equilibrium()


def test_zero_temperature_error():
    with pytest.raises(ValueError, match="invalid temperature"):
        equiconc.System(temperature_K=0.0).monomer("A", 1e-9).equilibrium()


def test_negative_temperature_error():
    with pytest.raises(ValueError, match="invalid temperature"):
        equiconc.System(temperature_K=-100.0).monomer("A", 1e-9).equilibrium()


def test_duplicate_monomer_error():
    with pytest.raises(ValueError, match="duplicate monomer"):
        equiconc.System().monomer("A", 1e-9).monomer("A", 2e-9).equilibrium()


def test_duplicate_complex_error():
    with pytest.raises(ValueError, match="duplicate complex"):
        (
            equiconc.System()
            .monomer("A", 1e-9)
            .monomer("B", 1e-9)
            .complex("AB", [("A", 1), ("B", 1)], dg_st=-10.0)
            .complex("AB", [("A", 1), ("B", 1)], dg_st=-12.0)
            .equilibrium()
        )


def test_zero_count_error():
    with pytest.raises(ValueError, match="zero stoichiometric count"):
        (
            equiconc.System()
            .monomer("A", 1e-9)
            .monomer("B", 1e-9)
            .complex("AB", [("A", 0), ("B", 1)], dg_st=-10.0)
            .equilibrium()
        )


def test_duplicate_monomer_in_composition_sums():
    """Duplicate monomers in a composition should have their counts summed."""
    eq_dup = (
        equiconc.System()
        .monomer("A", 1e-6)
        .complex("A3", [("A", 1), ("A", 2)], dg_st=-15.0)
        .equilibrium()
    )
    eq_merged = (
        equiconc.System()
        .monomer("A", 1e-6)
        .complex("A3", [("A", 3)], dg_st=-15.0)
        .equilibrium()
    )
    assert eq_dup["A3"] == pytest.approx(eq_merged["A3"])
    assert eq_dup["A"] == pytest.approx(eq_merged["A"])


def test_nan_delta_g_error():
    with pytest.raises(ValueError, match="invalid delta_g"):
        (
            equiconc.System()
            .monomer("A", 1e-9)
            .monomer("B", 1e-9)
            .complex("AB", [("A", 1), ("B", 1)], dg_st=float("nan"))
            .equilibrium()
        )


def test_inf_delta_g_error():
    with pytest.raises(ValueError, match="invalid delta_g"):
        (
            equiconc.System()
            .monomer("A", 1e-9)
            .monomer("B", 1e-9)
            .complex("AB", [("A", 1), ("B", 1)], dg_st=float("inf"))
            .equilibrium()
        )


def test_iter():
    sys = (
        equiconc.System()
        .monomer("A", 50e-9)
        .monomer("B", 100e-9)
        .complex("AB", [("A", 1), ("B", 1)], dg_st=-10.0)
    )
    eq = sys.equilibrium()
    names = list(eq)
    assert names == ["A", "B", "AB"]


def test_keys_values_items():
    sys = (
        equiconc.System()
        .monomer("A", 50e-9)
        .monomer("B", 100e-9)
        .complex("AB", [("A", 1), ("B", 1)], dg_st=-10.0)
    )
    eq = sys.equilibrium()
    assert eq.keys() == ["A", "B", "AB"]
    assert len(eq.values()) == 3
    assert all(isinstance(v, float) for v in eq.values())
    items = eq.items()
    assert len(items) == 3
    assert items[0][0] == "A"
    assert items[1][0] == "B"
    assert items[2][0] == "AB"


def test_to_dict_order():
    sys = (
        equiconc.System()
        .monomer("A", 50e-9)
        .monomer("B", 100e-9)
        .complex("AB", [("A", 1), ("B", 1)], dg_st=-10.0)
    )
    eq = sys.equilibrium()
    d = eq.to_dict()
    assert list(d.keys()) == ["A", "B", "AB"]


def test_converged_fully():
    sys = equiconc.System().monomer("A", 50e-9)
    eq = sys.equilibrium()
    assert eq.converged_fully is True


def test_empty_monomer_name_error():
    with pytest.raises(ValueError, match="must not be empty"):
        equiconc.System().monomer("", 1e-9).equilibrium()


def test_empty_complex_name_error():
    with pytest.raises(ValueError, match="must not be empty"):
        equiconc.System().monomer("A", 1e-9).complex("", [("A", 1)], dg_st=-10.0).equilibrium()


def test_complex_name_collides_with_monomer():
    with pytest.raises(ValueError, match="species name already in use"):
        (
            equiconc.System()
            .monomer("A", 1e-9)
            .monomer("B", 1e-9)
            .complex("A", [("A", 1), ("B", 1)], dg_st=-10.0)
            .equilibrium()
        )


# ---------------------------------------------------------------------------
# Temperature unit tests
# ---------------------------------------------------------------------------


def test_default_temperature_is_25C():
    """System() defaults to 25 °C."""
    sys = equiconc.System().monomer("A", 100e-9).monomer("B", 100e-9)
    # Using explicit 25 °C should give identical results
    sys_explicit = equiconc.System(temperature_C=25.0).monomer("A", 100e-9).monomer("B", 100e-9)
    eq = sys.complex("AB", [("A", 1), ("B", 1)], dg_st=-10.0).equilibrium()
    eq_explicit = sys_explicit.complex("AB", [("A", 1), ("B", 1)], dg_st=-10.0).equilibrium()
    assert eq["AB"] == pytest.approx(eq_explicit["AB"])


def test_temperature_C_converts_correctly():
    """temperature_C=25 should equal temperature_K=298.15."""
    c0 = 100e-9
    dg = -10.0
    eq_c = (
        equiconc.System(temperature_C=25.0)
        .monomer("A", c0)
        .monomer("B", c0)
        .complex("AB", [("A", 1), ("B", 1)], dg_st=dg)
        .equilibrium()
    )
    eq_k = (
        equiconc.System(temperature_K=298.15)
        .monomer("A", c0)
        .monomer("B", c0)
        .complex("AB", [("A", 1), ("B", 1)], dg_st=dg)
        .equilibrium()
    )
    assert eq_c["AB"] == pytest.approx(eq_k["AB"])
    assert eq_c["A"] == pytest.approx(eq_k["A"])


def test_both_temperatures_error():
    with pytest.raises(ValueError, match="cannot specify both"):
        equiconc.System(temperature_C=37.0, temperature_K=310.15)


# ---------------------------------------------------------------------------
# delta_g_over_rt tests
# ---------------------------------------------------------------------------


def test_delta_g_over_rt_matches_dg_st():
    """delta_g_over_rt should give same result as equivalent dg_st."""
    c0 = 100e-9
    dg = -10.0
    temp_k = 310.15
    dgrt = dg / (R * temp_k)

    eq_dg = (
        equiconc.System(temperature_K=temp_k)
        .monomer("A", c0)
        .monomer("B", c0)
        .complex("AB", [("A", 1), ("B", 1)], dg_st=dg)
        .equilibrium()
    )
    eq_dgrt = (
        equiconc.System(temperature_K=temp_k)
        .monomer("A", c0)
        .monomer("B", c0)
        .complex("AB", [("A", 1), ("B", 1)], delta_g_over_rt=dgrt)
        .equilibrium()
    )
    assert eq_dg["AB"] == pytest.approx(eq_dgrt["AB"], rel=1e-10)
    assert eq_dg["A"] == pytest.approx(eq_dgrt["A"], rel=1e-10)


def test_delta_g_over_rt_no_temperature():
    """When using delta_g_over_rt, temperature need not be specified."""
    c0 = 100e-9
    dgrt = -16.0

    eq = (
        equiconc.System()  # no temperature specified
        .monomer("A", c0)
        .monomer("B", c0)
        .complex("AB", [("A", 1), ("B", 1)], delta_g_over_rt=dgrt)
        .equilibrium()
    )
    assert eq["AB"] > 0
    # Result should be the same regardless of what temperature is set,
    # since delta_g_over_rt * R * T / (R * T) cancels.
    eq2 = (
        equiconc.System(temperature_K=500.0)
        .monomer("A", c0)
        .monomer("B", c0)
        .complex("AB", [("A", 1), ("B", 1)], delta_g_over_rt=dgrt)
        .equilibrium()
    )
    assert eq["AB"] == pytest.approx(eq2["AB"], rel=1e-10)


# ---------------------------------------------------------------------------
# dh_st / ds_st tests
# ---------------------------------------------------------------------------


def test_dh_st_ds_st_matches_dg_st():
    """ΔG = ΔH − TΔS should give same result as explicit dg_st."""
    c0 = 100e-9
    temp_k = 310.15
    delta_h = -50.0  # kcal/mol
    delta_s = -0.13  # kcal/(mol·K)
    dg = delta_h - temp_k * delta_s

    eq_dg = (
        equiconc.System(temperature_K=temp_k)
        .monomer("A", c0)
        .monomer("B", c0)
        .complex("AB", [("A", 1), ("B", 1)], dg_st=dg)
        .equilibrium()
    )
    eq_hs = (
        equiconc.System(temperature_K=temp_k)
        .monomer("A", c0)
        .monomer("B", c0)
        .complex("AB", [("A", 1), ("B", 1)], dh_st=delta_h, ds_st=delta_s)
        .equilibrium()
    )
    assert eq_dg["AB"] == pytest.approx(eq_hs["AB"], rel=1e-10)
    assert eq_dg["A"] == pytest.approx(eq_hs["A"], rel=1e-10)


def test_dh_st_ds_st_temperature_dependence():
    """Changing temperature should change equilibrium when using ΔH/ΔS."""
    c0 = 100e-9
    delta_h = -50.0
    delta_s = -0.13

    eq_25 = (
        equiconc.System(temperature_C=25.0)
        .monomer("A", c0)
        .monomer("B", c0)
        .complex("AB", [("A", 1), ("B", 1)], dh_st=delta_h, ds_st=delta_s)
        .equilibrium()
    )
    eq_50 = (
        equiconc.System(temperature_C=50.0)
        .monomer("A", c0)
        .monomer("B", c0)
        .complex("AB", [("A", 1), ("B", 1)], dh_st=delta_h, ds_st=delta_s)
        .equilibrium()
    )
    # Higher temperature should shift equilibrium (with these values,
    # ΔG becomes less negative → less complex formed)
    assert eq_25["AB"] != pytest.approx(eq_50["AB"], rel=0.01)


# ---------------------------------------------------------------------------
# Energy specification error tests
# ---------------------------------------------------------------------------


def test_no_energy_spec_error():
    with pytest.raises(ValueError, match="must specify energy"):
        equiconc.System().monomer("A", 1e-9).complex("AB", [("A", 1)])


def test_dh_st_without_ds_st_error():
    with pytest.raises(ValueError, match="dh_st and ds_st must both"):
        equiconc.System().monomer("A", 1e-9).complex(
            "AB", [("A", 1)], dh_st=-50.0
        )


def test_ds_st_without_dh_st_error():
    with pytest.raises(ValueError, match="dh_st and ds_st must both"):
        equiconc.System().monomer("A", 1e-9).complex(
            "AB", [("A", 1)], ds_st=-0.13
        )


def test_multiple_energy_specs_error():
    with pytest.raises(ValueError, match="specify only one"):
        equiconc.System().monomer("A", 1e-9).complex(
            "AB", [("A", 1)], dg_st=-10.0, delta_g_over_rt=-16.0
        )


# ---------------------------------------------------------------------------
# dg_st tuple form tests
# ---------------------------------------------------------------------------


def test_dg_st_tuple_matches_dh_st_ds_st():
    """dg_st=(dg, temp_C) + ds_st should give same result as dh_st + ds_st."""
    c0 = 100e-9
    temp_k = 310.15  # system temperature
    delta_s = -0.13  # kcal/(mol·K)

    # Known ΔG at 25 °C
    ref_temp_c = 25.0
    ref_temp_k = ref_temp_c + 273.15
    delta_h = -50.0
    dg_at_ref = delta_h - ref_temp_k * delta_s

    eq_tuple = (
        equiconc.System(temperature_K=temp_k)
        .monomer("A", c0)
        .monomer("B", c0)
        .complex(
            "AB",
            [("A", 1), ("B", 1)],
            dg_st=(dg_at_ref, ref_temp_c),
            ds_st=delta_s,
        )
        .equilibrium()
    )
    eq_hs = (
        equiconc.System(temperature_K=temp_k)
        .monomer("A", c0)
        .monomer("B", c0)
        .complex(
            "AB",
            [("A", 1), ("B", 1)],
            dh_st=delta_h,
            ds_st=delta_s,
        )
        .equilibrium()
    )
    assert eq_tuple["AB"] == pytest.approx(eq_hs["AB"], rel=1e-10)
    assert eq_tuple["A"] == pytest.approx(eq_hs["A"], rel=1e-10)


def test_dg_st_tuple_same_temp_matches_scalar():
    """dg_st=(dg, T) at system temperature should match dg_st=dg."""
    c0 = 100e-9
    dg = -10.0
    temp_c = 37.0

    eq_scalar = (
        equiconc.System(temperature_C=temp_c)
        .monomer("A", c0)
        .monomer("B", c0)
        .complex("AB", [("A", 1), ("B", 1)], dg_st=dg)
        .equilibrium()
    )
    # With ds_st=0 and same reference temperature, result should match scalar dg_st.
    eq_tuple = (
        equiconc.System(temperature_C=temp_c)
        .monomer("A", c0)
        .monomer("B", c0)
        .complex(
            "AB",
            [("A", 1), ("B", 1)],
            dg_st=(dg, temp_c),
            ds_st=0.0,
        )
        .equilibrium()
    )
    assert eq_scalar["AB"] == pytest.approx(eq_tuple["AB"], rel=1e-10)


def test_dg_st_tuple_without_ds_st_error():
    with pytest.raises(ValueError, match="requires ds_st"):
        equiconc.System().monomer("A", 1e-9).complex(
            "AB", [("A", 1)], dg_st=(-10.0, 25.0)
        )


def test_dg_st_scalar_with_ds_st_error():
    with pytest.raises(ValueError, match="cannot be combined with ds_st"):
        equiconc.System().monomer("A", 1e-9).complex(
            "AB", [("A", 1)], dg_st=-10.0, ds_st=-0.13
        )


# ---------------------------------------------------------------------------
# SolverOptions tests
# ---------------------------------------------------------------------------


def test_default_options():
    opts = equiconc.SolverOptions()
    assert opts.max_iterations == 1000
    assert opts.gradient_rel_tol == 1e-7
    assert opts.log_q_clamp is None


def test_options_with_kwargs():
    opts = equiconc.SolverOptions(max_iterations=50, gradient_rel_tol=1e-9)
    assert opts.max_iterations == 50
    assert opts.gradient_rel_tol == 1e-9


def test_invalid_options_rejected():
    with pytest.raises(ValueError, match="max_iterations"):
        equiconc.SolverOptions(max_iterations=0)
    with pytest.raises(ValueError, match="invalid solver options"):
        equiconc.SolverOptions(
            trust_region_shrink_rho=0.9, trust_region_grow_rho=0.5
        )
    with pytest.raises(ValueError, match="invalid solver options"):
        equiconc.SolverOptions(gradient_rel_tol=-1.0)


def test_options_passed_through_to_solver():
    c0 = 100e-9
    opts = equiconc.SolverOptions(max_iterations=1)
    with pytest.raises(RuntimeError, match="did not converge"):
        (
            equiconc.System(options=opts)
            .monomer("A", c0)
            .monomer("B", c0)
            .complex("AB", [("A", 1), ("B", 1)], dg_st=-20.0)
            .equilibrium()
        )


def test_log_q_clamp_bounds_extreme_energy():
    """With log_q_clamp set, even pathologically strong binding stays finite."""
    # dg = -5000 * R * T gives log_q ≈ 5000 (way into overflow territory).
    # Without clamp, the solver diverges; with clamp it converges.
    R_gas = 1.987204e-3
    dg = -5000 * R_gas * 298.15
    opts = equiconc.SolverOptions(log_q_clamp=100.0)
    eq = (
        equiconc.System(options=opts)
        .monomer("A", 100e-9)
        .monomer("B", 100e-9)
        .complex("AB", [("A", 1), ("B", 1)], dg_st=dg)
        .equilibrium()
    )
    # Effectively all the material should end up in the complex.
    assert eq["AB"] > 99e-9


def test_options_are_kwargs_only():
    # options must be keyword, per the signature.
    with pytest.raises(TypeError):
        equiconc.System(equiconc.SolverOptions())  # type: ignore


def test_solver_options_repr():
    opts = equiconc.SolverOptions(max_iterations=42)
    r = repr(opts)
    assert "SolverOptions" in r
    assert "42" in r


# ---------------------------------------------------------------------------
# Log-objective tests
# ---------------------------------------------------------------------------


def test_objective_default_is_linear():
    opts = equiconc.SolverOptions()
    assert opts.objective == "linear"


def test_objective_log_kwarg_round_trip():
    opts = equiconc.SolverOptions(objective="log")
    assert opts.objective == "log"


def test_objective_invalid_raises_with_helpful_message():
    with pytest.raises(ValueError, match='"linear" or "log"'):
        equiconc.SolverOptions(objective="bogus")


def test_objective_log_matches_linear_on_simple_dimer():
    c0 = 100e-9
    dg = -10.0
    lin = (
        equiconc.System()
        .monomer("A", c0)
        .monomer("B", c0)
        .complex("AB", [("A", 1), ("B", 1)], dg_st=dg)
        .equilibrium()
    )
    log_eq = (
        equiconc.System(options=equiconc.SolverOptions(objective="log"))
        .monomer("A", c0)
        .monomer("B", c0)
        .complex("AB", [("A", 1), ("B", 1)], dg_st=dg)
        .equilibrium()
    )
    for name in ("A", "B", "AB"):
        assert log_eq[name] == pytest.approx(lin[name], rel=1e-6)


def test_objective_log_handles_coffee_bug1():
    # Single monomer with a positive-ΔG conformer at 20 °C — the case
    # that produces NaN in coffee. Log path must return finite, correct
    # concentrations.
    c0 = 1e-3
    eq = (
        equiconc.System(
            temperature_C=20.0,
            options=equiconc.SolverOptions(objective="log"),
        )
        .monomer("A", c0)
        .complex("Astar", [("A", 1)], dg_st=3.9)
        .equilibrium()
    )
    a = eq["A"]
    a_star = eq["Astar"]
    # Mass conservation.
    assert (a + a_star) == pytest.approx(c0, rel=1e-9)
    # Both finite.
    assert a > 0 and a_star > 0
    # Astar/A = exp(-3.9 / RT) ≈ 0.0012 at 20 °C.
    R_gas = 1.987204e-3
    expected_ratio = math.exp(-3.9 / (R_gas * 293.15))
    assert (a_star / a) == pytest.approx(expected_ratio, rel=1e-7)


def test_objective_log_handles_coffee_bug2_strong_binding():
    # A + 2B ⇌ AB2 with extreme ΔG and asymmetric c0 — coffee fails
    # mass conservation here. Log path must respect it.
    eq = (
        equiconc.System(
            temperature_K=349.7,
            options=equiconc.SolverOptions(objective="log"),
        )
        .monomer("A", 1e-3)
        .monomer("B", 162.4e-6)
        .complex("AB2", [("A", 1), ("B", 2)], dg_st=-39.47)
        .equilibrium()
    )
    a, b, ab2 = eq["A"], eq["B"], eq["AB2"]
    # Mass conservation tight.
    assert (a + ab2) == pytest.approx(1e-3, rel=1e-9)
    assert (b + 2 * ab2) == pytest.approx(162.4e-6, rel=1e-9)
    # And specifically: free A must not exceed initial A (coffee bug).
    assert a <= 1e-3
