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
        .complex("AB", [("A", 1), ("B", 1)], delta_g=dg)
    )
    eq = sys.equilibrium()

    # Analytical solution
    rt = R * 310.15
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
    assert eq.concentration("nonexistent") is None


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
        .complex("AB", [("A", 1), ("B", 1)], delta_g=-10.0)
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
        .complex("AB", [("A", 1), ("B", 1)], delta_g=-10.0)
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
    sys = equiconc.System().monomer("A", 50e-9)
    with pytest.raises(ValueError, match="unknown monomer"):
        sys.complex("AB", [("A", 1), ("B", 1)], delta_g=-10.0)


def test_empty_composition_error():
    sys = equiconc.System().monomer("A", 50e-9)
    with pytest.raises(ValueError, match="empty composition"):
        sys.complex("X", [], delta_g=-10.0)


def test_custom_temperature():
    sys = equiconc.System(temperature=300.0).monomer("A", 50e-9)
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
        equiconc.System().monomer("A", -1e-9)


def test_zero_concentration_error():
    with pytest.raises(ValueError, match="invalid concentration"):
        equiconc.System().monomer("A", 0.0)


def test_zero_temperature_error():
    with pytest.raises(ValueError, match="invalid temperature"):
        equiconc.System(temperature=0.0)


def test_negative_temperature_error():
    with pytest.raises(ValueError, match="invalid temperature"):
        equiconc.System(temperature=-100.0)


def test_duplicate_monomer_error():
    with pytest.raises(ValueError, match="duplicate monomer"):
        equiconc.System().monomer("A", 1e-9).monomer("A", 2e-9)


def test_duplicate_complex_error():
    with pytest.raises(ValueError, match="duplicate complex"):
        (
            equiconc.System()
            .monomer("A", 1e-9)
            .monomer("B", 1e-9)
            .complex("AB", [("A", 1), ("B", 1)], delta_g=-10.0)
            .complex("AB", [("A", 1), ("B", 1)], delta_g=-12.0)
        )


def test_zero_count_error():
    with pytest.raises(ValueError, match="zero stoichiometric count"):
        (
            equiconc.System()
            .monomer("A", 1e-9)
            .monomer("B", 1e-9)
            .complex("AB", [("A", 0), ("B", 1)], delta_g=-10.0)
        )


def test_duplicate_monomer_in_composition_error():
    with pytest.raises(ValueError, match="duplicate monomer in composition"):
        (
            equiconc.System()
            .monomer("A", 1e-9)
            .monomer("B", 1e-9)
            .complex("AB", [("A", 1), ("A", 2)], delta_g=-10.0)
        )


def test_nan_delta_g_error():
    with pytest.raises(ValueError, match="invalid delta_g"):
        (
            equiconc.System()
            .monomer("A", 1e-9)
            .monomer("B", 1e-9)
            .complex("AB", [("A", 1), ("B", 1)], delta_g=float("nan"))
        )


def test_inf_delta_g_error():
    with pytest.raises(ValueError, match="invalid delta_g"):
        (
            equiconc.System()
            .monomer("A", 1e-9)
            .monomer("B", 1e-9)
            .complex("AB", [("A", 1), ("B", 1)], delta_g=float("inf"))
        )
