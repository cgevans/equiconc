import math

from hypothesis import given
from hypothesis import strategies as st

import equiconc

R = 1.987204e-3  # kcal/(mol·K)
MONOMER_NAMES = ["A", "B", "C", "D"]
REL_TOL = 1e-4


def _log_uniform_concentration():
    """Log-uniform concentration in [1e-9, 1e-3]."""
    return st.floats(min_value=-9.0, max_value=-3.0).map(lambda e: 10.0**e)


@st.composite
def system_strategy(draw):
    """Generate a random valid system with metadata for verification."""
    temperature = draw(st.floats(min_value=293.15, max_value=373.15))
    n_mon = draw(st.integers(min_value=1, max_value=4))

    monomers = {}
    sys = equiconc.System(temperature_K=temperature)
    for i in range(n_mon):
        name = MONOMER_NAMES[i]
        conc = draw(_log_uniform_concentration())
        monomers[name] = conc
        sys = sys.monomer(name, conc)

    n_cplx = draw(st.integers(min_value=0, max_value=6))
    complexes = {}
    for k in range(n_cplx):
        cplx_name = f"c{k}"
        # For each monomer, draw a count of 0 (absent) to 3
        counts = {
            MONOMER_NAMES[i]: draw(st.integers(min_value=0, max_value=3))
            for i in range(n_mon)
        }
        comp = {name: count for name, count in counts.items() if count > 0}
        if not comp:
            comp = {MONOMER_NAMES[0]: 1}
        delta_g = draw(st.floats(min_value=-30.0, max_value=5.0))
        sys = sys.complex(cplx_name, list(comp.items()), dg_st=delta_g)
        complexes[cplx_name] = (comp, delta_g)

    return sys, temperature, monomers, complexes


@st.composite
def monomer_only_strategy(draw):
    """Generate a system with no complexes."""
    temperature = draw(st.floats(min_value=293.15, max_value=373.15))
    n_mon = draw(st.integers(min_value=1, max_value=4))

    monomers = {}
    sys = equiconc.System(temperature_K=temperature)
    for i in range(n_mon):
        name = MONOMER_NAMES[i]
        conc = draw(_log_uniform_concentration())
        monomers[name] = conc
        sys = sys.monomer(name, conc)

    return sys, monomers


@st.composite
def dimerization_strategy(draw):
    """Generate parameters for A + B -> AB dimerization."""
    c0 = draw(_log_uniform_concentration())
    dg = draw(st.floats(min_value=-30.0, max_value=5.0))
    temp = draw(st.floats(min_value=293.15, max_value=373.15))
    return c0, dg, temp


@given(data=system_strategy())
def test_prop_mass_conservation(data):
    sys, temperature, monomers, complexes = data
    eq = sys.equilibrium()
    for mon_name, c0 in monomers.items():
        total = eq[mon_name]
        for cplx_name, (comp, _) in complexes.items():
            if mon_name in comp:
                total += comp[mon_name] * eq[cplx_name]
        rel_err = abs(total - c0) / c0
        assert rel_err < REL_TOL, (
            f"mass conservation violated for {mon_name}: "
            f"total={total}, c0={c0}, rel_err={rel_err}"
        )


@given(data=system_strategy())
def test_prop_equilibrium_condition(data):
    sys, temperature, monomers, complexes = data
    eq = sys.equilibrium()
    rt = R * temperature
    for cplx_name, (comp, dg) in complexes.items():
        k_eq = math.exp(-dg / rt)
        product = 1.0
        for mon_name, count in comp.items():
            product *= eq[mon_name] ** count
        expected = k_eq * product
        actual = eq[cplx_name]
        rel_err = abs(actual - expected) / (expected + 1e-300)
        assert rel_err < REL_TOL, (
            f"equilibrium violated for {cplx_name}: "
            f"actual={actual}, expected={expected}, rel_err={rel_err}"
        )


@given(data=system_strategy())
def test_prop_concentrations_non_negative(data):
    sys, temperature, monomers, complexes = data
    eq = sys.equilibrium()
    d = eq.to_dict()
    for name, conc in d.items():
        assert conc >= 0.0, f"negative concentration for {name}: {conc}"


@given(data=system_strategy())
def test_prop_api_consistency(data):
    sys, temperature, monomers, complexes = data
    eq = sys.equilibrium()
    d = eq.to_dict()
    all_names = list(monomers.keys()) + list(complexes.keys())
    assert len(eq) == len(all_names)
    for name in all_names:
        assert eq[name] == eq.concentration(name)
        assert name in eq
        assert name in d


@given(data=monomer_only_strategy())
def test_prop_monomer_only_identity(data):
    sys, monomers = data
    eq = sys.equilibrium()
    for name, c0 in monomers.items():
        c = eq[name]
        assert c == c0, f"{name}: {c} != {c0}"


@given(data=dimerization_strategy())
def test_prop_dimerization_analytical(data):
    c0, dg, temp = data
    sys = (
        equiconc.System(temperature_K=temp)
        .monomer("A", c0)
        .monomer("B", c0)
        .complex("AB", [("A", 1), ("B", 1)], dg_st=dg)
    )
    eq = sys.equilibrium()

    rt = R * temp
    k = math.exp(-dg / rt)
    # Numerically stable closed-form: avoid catastrophic cancellation
    disc = math.sqrt(4.0 * k * c0 + 1.0)
    free = 2.0 * c0 / (disc + 1.0)
    x = k * free * free

    tol = REL_TOL
    assert abs(eq["A"] - free) / (free + 1e-300) < tol, f'[A]={eq["A"]} != expected {free}'
    assert abs(eq["B"] - free) / (free + 1e-300) < tol, f'[B]={eq["B"]} != expected {free}'
    assert abs(eq["AB"] - x) / (x + 1e-300) < tol, f'[AB]={eq["AB"]} != expected {x}'
