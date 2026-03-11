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
        .complex("AB", [("A", 1), ("B", 1)], dg_s=dg)
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
        .complex("AB", [("A", 1), ("B", 1)], dg_s=-10.0)
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
        .complex("AB", [("A", 1), ("B", 1)], dg_s=-10.0)
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
    sys = equiconc.System().monomer("A", 50e-9).complex("AB", [("A", 1), ("B", 1)], dg_s=-10.0)
    with pytest.raises(ValueError, match="unknown monomer"):
        sys.equilibrium()


def test_empty_composition_error():
    sys = equiconc.System().monomer("A", 50e-9).complex("X", [], dg_s=-10.0)
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
            .complex("AB", [("A", 1), ("B", 1)], dg_s=-10.0)
            .complex("AB", [("A", 1), ("B", 1)], dg_s=-12.0)
            .equilibrium()
        )


def test_zero_count_error():
    with pytest.raises(ValueError, match="zero stoichiometric count"):
        (
            equiconc.System()
            .monomer("A", 1e-9)
            .monomer("B", 1e-9)
            .complex("AB", [("A", 0), ("B", 1)], dg_s=-10.0)
            .equilibrium()
        )


def test_duplicate_monomer_in_composition_error():
    with pytest.raises(ValueError, match="duplicate monomer in composition"):
        (
            equiconc.System()
            .monomer("A", 1e-9)
            .monomer("B", 1e-9)
            .complex("AB", [("A", 1), ("A", 2)], dg_s=-10.0)
            .equilibrium()
        )


def test_nan_delta_g_error():
    with pytest.raises(ValueError, match="invalid delta_g"):
        (
            equiconc.System()
            .monomer("A", 1e-9)
            .monomer("B", 1e-9)
            .complex("AB", [("A", 1), ("B", 1)], dg_s=float("nan"))
            .equilibrium()
        )


def test_inf_delta_g_error():
    with pytest.raises(ValueError, match="invalid delta_g"):
        (
            equiconc.System()
            .monomer("A", 1e-9)
            .monomer("B", 1e-9)
            .complex("AB", [("A", 1), ("B", 1)], dg_s=float("inf"))
            .equilibrium()
        )


def test_iter():
    sys = (
        equiconc.System()
        .monomer("A", 50e-9)
        .monomer("B", 100e-9)
        .complex("AB", [("A", 1), ("B", 1)], dg_s=-10.0)
    )
    eq = sys.equilibrium()
    names = list(eq)
    assert names == ["A", "B", "AB"]


def test_keys_values_items():
    sys = (
        equiconc.System()
        .monomer("A", 50e-9)
        .monomer("B", 100e-9)
        .complex("AB", [("A", 1), ("B", 1)], dg_s=-10.0)
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
        .complex("AB", [("A", 1), ("B", 1)], dg_s=-10.0)
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
        equiconc.System().monomer("A", 1e-9).complex("", [("A", 1)], dg_s=-10.0).equilibrium()


def test_complex_name_collides_with_monomer():
    with pytest.raises(ValueError, match="species name already in use"):
        (
            equiconc.System()
            .monomer("A", 1e-9)
            .monomer("B", 1e-9)
            .complex("A", [("A", 1), ("B", 1)], dg_s=-10.0)
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
    eq = sys.complex("AB", [("A", 1), ("B", 1)], dg_s=-10.0).equilibrium()
    eq_explicit = sys_explicit.complex("AB", [("A", 1), ("B", 1)], dg_s=-10.0).equilibrium()
    assert eq["AB"] == pytest.approx(eq_explicit["AB"])


def test_temperature_C_converts_correctly():
    """temperature_C=25 should equal temperature_K=298.15."""
    c0 = 100e-9
    dg = -10.0
    eq_c = (
        equiconc.System(temperature_C=25.0)
        .monomer("A", c0)
        .monomer("B", c0)
        .complex("AB", [("A", 1), ("B", 1)], dg_s=dg)
        .equilibrium()
    )
    eq_k = (
        equiconc.System(temperature_K=298.15)
        .monomer("A", c0)
        .monomer("B", c0)
        .complex("AB", [("A", 1), ("B", 1)], dg_s=dg)
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


def test_delta_g_over_rt_matches_dg_s():
    """delta_g_over_rt should give same result as equivalent dg_s."""
    c0 = 100e-9
    dg = -10.0
    temp_k = 310.15
    dgrt = dg / (R * temp_k)

    eq_dg = (
        equiconc.System(temperature_K=temp_k)
        .monomer("A", c0)
        .monomer("B", c0)
        .complex("AB", [("A", 1), ("B", 1)], dg_s=dg)
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
# dh_s / ds_s tests
# ---------------------------------------------------------------------------


def test_dh_s_ds_s_matches_dg_s():
    """ΔG = ΔH − TΔS should give same result as explicit dg_s."""
    c0 = 100e-9
    temp_k = 310.15
    delta_h = -50.0  # kcal/mol
    delta_s = -0.13  # kcal/(mol·K)
    dg = delta_h - temp_k * delta_s

    eq_dg = (
        equiconc.System(temperature_K=temp_k)
        .monomer("A", c0)
        .monomer("B", c0)
        .complex("AB", [("A", 1), ("B", 1)], dg_s=dg)
        .equilibrium()
    )
    eq_hs = (
        equiconc.System(temperature_K=temp_k)
        .monomer("A", c0)
        .monomer("B", c0)
        .complex("AB", [("A", 1), ("B", 1)], dh_s=delta_h, ds_s=delta_s)
        .equilibrium()
    )
    assert eq_dg["AB"] == pytest.approx(eq_hs["AB"], rel=1e-10)
    assert eq_dg["A"] == pytest.approx(eq_hs["A"], rel=1e-10)


def test_dh_s_ds_s_temperature_dependence():
    """Changing temperature should change equilibrium when using ΔH/ΔS."""
    c0 = 100e-9
    delta_h = -50.0
    delta_s = -0.13

    eq_25 = (
        equiconc.System(temperature_C=25.0)
        .monomer("A", c0)
        .monomer("B", c0)
        .complex("AB", [("A", 1), ("B", 1)], dh_s=delta_h, ds_s=delta_s)
        .equilibrium()
    )
    eq_50 = (
        equiconc.System(temperature_C=50.0)
        .monomer("A", c0)
        .monomer("B", c0)
        .complex("AB", [("A", 1), ("B", 1)], dh_s=delta_h, ds_s=delta_s)
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


def test_dh_s_without_ds_s_error():
    with pytest.raises(ValueError, match="dh_s and ds_s must both"):
        equiconc.System().monomer("A", 1e-9).complex(
            "AB", [("A", 1)], dh_s=-50.0
        )


def test_ds_s_without_dh_s_error():
    with pytest.raises(ValueError, match="dh_s and ds_s must both"):
        equiconc.System().monomer("A", 1e-9).complex(
            "AB", [("A", 1)], ds_s=-0.13
        )


def test_multiple_energy_specs_error():
    with pytest.raises(ValueError, match="specify only one"):
        equiconc.System().monomer("A", 1e-9).complex(
            "AB", [("A", 1)], dg_s=-10.0, delta_g_over_rt=-16.0
        )


# ---------------------------------------------------------------------------
# dg_s tuple form tests
# ---------------------------------------------------------------------------


def test_dg_s_tuple_matches_dh_s_ds_s():
    """dg_s=(dg, temp_C) + ds_s should give same result as dh_s + ds_s."""
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
            dg_s=(dg_at_ref, ref_temp_c),
            ds_s=delta_s,
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
            dh_s=delta_h,
            ds_s=delta_s,
        )
        .equilibrium()
    )
    assert eq_tuple["AB"] == pytest.approx(eq_hs["AB"], rel=1e-10)
    assert eq_tuple["A"] == pytest.approx(eq_hs["A"], rel=1e-10)


def test_dg_s_tuple_same_temp_matches_scalar():
    """dg_s=(dg, T) at system temperature should match dg_s=dg."""
    c0 = 100e-9
    dg = -10.0
    temp_c = 37.0

    eq_scalar = (
        equiconc.System(temperature_C=temp_c)
        .monomer("A", c0)
        .monomer("B", c0)
        .complex("AB", [("A", 1), ("B", 1)], dg_s=dg)
        .equilibrium()
    )
    # With ds_s=0 and same reference temperature, result should match scalar dg_s.
    eq_tuple = (
        equiconc.System(temperature_C=temp_c)
        .monomer("A", c0)
        .monomer("B", c0)
        .complex(
            "AB",
            [("A", 1), ("B", 1)],
            dg_s=(dg, temp_c),
            ds_s=0.0,
        )
        .equilibrium()
    )
    assert eq_scalar["AB"] == pytest.approx(eq_tuple["AB"], rel=1e-10)


def test_dg_s_tuple_without_ds_s_error():
    with pytest.raises(ValueError, match="requires ds_s"):
        equiconc.System().monomer("A", 1e-9).complex(
            "AB", [("A", 1)], dg_s=(-10.0, 25.0)
        )


def test_dg_s_scalar_with_ds_s_error():
    with pytest.raises(ValueError, match="cannot be combined with ds_s"):
        equiconc.System().monomer("A", 1e-9).complex(
            "AB", [("A", 1)], dg_s=-10.0, ds_s=-0.13
        )
