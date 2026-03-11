"""Cross-validate equiconc against NUPACK's concentration solver.

Both implement the Dirks et al. (2007) trust-region Newton method on the
convex dual problem, so their results should agree to high precision.

NUPACK's solver operates in non-dimensionalized units (concentrations divided
by water molarity), so its log partition functions require a correction:
    logq_nupack = -ΔG/RT + (n - 1) * ln(c_water)
where n is the number of strands in the complex.
"""

import numpy as np
import pytest

nupack = pytest.importorskip("nupack")
from nupack.concentration import solve_complex_concentrations
from nupack.constants import water_molarity

import equiconc

R = 1.987204e-3  # kcal/(mol·K)
NM = 1e-9


def _nupack_conc(strand_concs, complexes, T=310.15):
    """Solve equilibrium concentrations with NUPACK.

    Args:
        strand_concs: dict mapping strand name → total concentration (M).
        complexes: list of (name, composition, delta_g) tuples where
            composition is a list of (strand_name, count) pairs.
        T: temperature in Kelvin.

    Returns:
        dict mapping species name → equilibrium concentration.
    """
    strand_names = list(strand_concs.keys())
    strand_idx = {name: i for i, name in enumerate(strand_names)}
    x0 = [strand_concs[s] for s in strand_names]
    cw = water_molarity(T)
    ln_cw = np.log(cw)

    # Build indices and logq for all species (monomers + complexes)
    indices = []
    logq = []
    species_names = []

    # Monomers as single-strand complexes
    for i, name in enumerate(strand_names):
        indices.append([i])
        logq.append(0.0)
        species_names.append(name)

    # Multi-strand complexes
    for name, composition, dg in complexes:
        idx_list = []
        for strand_name, count in composition:
            idx_list.extend([strand_idx[strand_name]] * count)
        n_strands = len(idx_list)
        indices.append(idx_list)
        logq.append(-dg / (R * T) + (n_strands - 1) * ln_cw)
        species_names.append(name)

    conc = solve_complex_concentrations(
        indices, logq, x0, T, rotational_correction=False,
    )

    return {name: float(conc[i]) for i, name in enumerate(species_names)}


def _equiconc(strand_concs, complexes, T=310.15):
    """Solve equilibrium concentrations with equiconc."""
    sys = equiconc.System(temperature=T)
    for name, c0 in strand_concs.items():
        sys = sys.monomer(name, c0)
    for name, composition, dg in complexes:
        sys = sys.complex(name, composition, dg_s=dg)
    eq = sys.equilibrium()
    return eq.to_dict()


def _assert_match(eq, nupack, rel=1e-8):
    """Assert all species concentrations match between solvers."""
    for name in eq:
        assert eq[name] == pytest.approx(nupack[name], rel=rel), \
            f"{name}: equiconc={eq[name]:.6e} vs nupack={nupack[name]:.6e}"


# --------------------------------------------------------------------------- #
# Test cases
# --------------------------------------------------------------------------- #


class TestSimpleDimerization:
    """A + B ⇌ AB"""

    def test_vs_nupack(self):
        strands = {"A": 100 * NM, "B": 100 * NM}
        complexes = [("AB", [("A", 1), ("B", 1)], -10.0)]

        eq = _equiconc(strands, complexes)
        nu = _nupack_conc(strands, complexes)
        _assert_match(eq, nu)


class TestAsymmetricConcentrations:
    """A + B ⇌ AB with [A]₀ ≫ [B]₀"""

    def test_vs_nupack(self):
        strands = {"A": 1e-6, "B": 10 * NM}
        complexes = [("AB", [("A", 1), ("B", 1)], -12.0)]

        eq = _equiconc(strands, complexes)
        nu = _nupack_conc(strands, complexes)
        _assert_match(eq, nu)


class TestHomodimer:
    """A + A ⇌ AA (stoichiometry coefficient = 2)"""

    def test_vs_nupack(self):
        strands = {"A": 200 * NM}
        complexes = [("AA", [("A", 2)], -8.0)]

        eq = _equiconc(strands, complexes)
        nu = _nupack_conc(strands, complexes)
        _assert_match(eq, nu)


class TestCompetingComplexes:
    """A + B ⇌ AB, A + C ⇌ AC — two complexes competing for A"""

    def test_vs_nupack(self):
        strands = {"A": 100 * NM, "B": 100 * NM, "C": 100 * NM}
        complexes = [
            ("AB", [("A", 1), ("B", 1)], -10.0),
            ("AC", [("A", 1), ("C", 1)], -9.0),
        ]

        eq = _equiconc(strands, complexes)
        nu = _nupack_conc(strands, complexes)
        _assert_match(eq, nu)


class TestTrimer:
    """A + B + C ⇌ ABC (three-body complex)"""

    def test_vs_nupack(self):
        strands = {"A": 50 * NM, "B": 80 * NM, "C": 120 * NM}
        complexes = [("ABC", [("A", 1), ("B", 1), ("C", 1)], -15.0)]

        eq = _equiconc(strands, complexes)
        nu = _nupack_conc(strands, complexes)
        _assert_match(eq, nu)


class TestHigherOrderComplex:
    """A₂B₃ — complex with stoichiometry (2, 3)"""

    def test_vs_nupack(self):
        strands = {"A": 500 * NM, "B": 500 * NM}
        complexes = [("A2B3", [("A", 2), ("B", 3)], -20.0)]

        eq = _equiconc(strands, complexes)
        nu = _nupack_conc(strands, complexes)
        _assert_match(eq, nu)


class TestWeakBinding:
    """Very weak binding (ΔG ≈ 0) — almost no complex formation."""

    def test_vs_nupack(self):
        strands = {"A": 100 * NM, "B": 100 * NM}
        complexes = [("AB", [("A", 1), ("B", 1)], -1.0)]

        eq = _equiconc(strands, complexes)
        nu = _nupack_conc(strands, complexes)
        _assert_match(eq, nu)


class TestStrongBinding:
    """Very strong binding — almost complete complex formation."""

    def test_vs_nupack(self):
        strands = {"A": 100 * NM, "B": 100 * NM}
        complexes = [("AB", [("A", 1), ("B", 1)], -25.0)]

        eq = _equiconc(strands, complexes)
        nu = _nupack_conc(strands, complexes)
        _assert_match(eq, nu)


class TestCustomTemperature:
    """Explicit temperature matching the default (25 °C = 298.15 K)."""

    def test_vs_nupack(self):
        strands = {"A": 100 * NM, "B": 100 * NM}
        complexes = [("AB", [("A", 1), ("B", 1)], -10.0)]
        T = 298.15

        eq = _equiconc(strands, complexes, T)
        nu = _nupack_conc(strands, complexes, T)
        _assert_match(eq, nu)


class TestManyComplexes:
    """Four monomers forming six pairwise dimers."""

    def test_vs_nupack(self):
        strands = {"A": 50 * NM, "B": 100 * NM, "C": 150 * NM, "D": 200 * NM}
        complexes = [
            ("AB", [("A", 1), ("B", 1)], -10.0),
            ("AC", [("A", 1), ("C", 1)], -9.5),
            ("AD", [("A", 1), ("D", 1)], -8.0),
            ("BC", [("B", 1), ("C", 1)], -11.0),
            ("BD", [("B", 1), ("D", 1)], -7.5),
            ("CD", [("C", 1), ("D", 1)], -10.5),
        ]

        eq = _equiconc(strands, complexes)
        nu = _nupack_conc(strands, complexes)
        _assert_match(eq, nu)


class TestDimersAndTrimer:
    """A + B ⇌ AB, A + C ⇌ AC, A + B + C ⇌ ABC — mixed 2- and 3-body."""

    def test_vs_nupack(self):
        strands = {"A": 100 * NM, "B": 100 * NM, "C": 100 * NM}
        complexes = [
            ("AB", [("A", 1), ("B", 1)], -10.0),
            ("AC", [("A", 1), ("C", 1)], -9.0),
            ("ABC", [("A", 1), ("B", 1), ("C", 1)], -18.0),
        ]

        eq = _equiconc(strands, complexes)
        nu = _nupack_conc(strands, complexes)
        _assert_match(eq, nu)


class TestTitration:
    """Sweep [A]₀ over several orders of magnitude."""

    def test_vs_nupack(self):
        complexes = [("AB", [("A", 1), ("B", 1)], -10.0)]

        for c0_a in np.geomspace(1 * NM, 10e-6, 20):
            strands = {"A": c0_a, "B": 100 * NM}
            eq = _equiconc(strands, complexes)
            nu = _nupack_conc(strands, complexes)
            _assert_match(eq, nu)


class TestDeltaGSweep:
    """Sweep ΔG from weakly repulsive to strongly attractive."""

    def test_vs_nupack(self):
        strands = {"A": 100 * NM, "B": 100 * NM}

        for dg in np.linspace(2.0, -25.0, 30):
            complexes = [("AB", [("A", 1), ("B", 1)], dg)]
            eq = _equiconc(strands, complexes)
            nu = _nupack_conc(strands, complexes)
            _assert_match(eq, nu)


class TestTemperatureSweep:
    """Sweep temperature from 10 °C to 90 °C."""

    def test_vs_nupack(self):
        strands = {"A": 100 * NM, "B": 100 * NM}
        complexes = [("AB", [("A", 1), ("B", 1)], -10.0)]

        for T in np.linspace(283.15, 363.15, 20):
            eq = _equiconc(strands, complexes, T)
            nu = _nupack_conc(strands, complexes, T)
            _assert_match(eq, nu)
