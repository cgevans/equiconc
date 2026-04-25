from typing import Iterator, overload

class SolverOptions:
    """Solver configuration: tolerances, iteration caps, trust-region
    parameters, and numerical clamps.
    """

    def __init__(
        self,
        *,
        max_iterations: int | None = None,
        gradient_abs_tol: float | None = None,
        gradient_rel_tol: float | None = None,
        relaxed_gradient_abs_tol: float | None = None,
        relaxed_gradient_rel_tol: float | None = None,
        stagnation_threshold: int | None = None,
        initial_trust_region_radius: float | None = None,
        max_trust_region_radius: float | None = None,
        step_accept_threshold: float | None = None,
        trust_region_shrink_rho: float | None = None,
        trust_region_grow_rho: float | None = None,
        trust_region_shrink_scale: float | None = None,
        trust_region_grow_scale: float | None = None,
        log_c_clamp: float | None = None,
        log_q_clamp: float | None = None,
    ) -> None: ...
    @property
    def max_iterations(self) -> int: ...
    @property
    def gradient_abs_tol(self) -> float: ...
    @property
    def gradient_rel_tol(self) -> float: ...
    @property
    def relaxed_gradient_abs_tol(self) -> float: ...
    @property
    def relaxed_gradient_rel_tol(self) -> float: ...
    @property
    def stagnation_threshold(self) -> int: ...
    @property
    def initial_trust_region_radius(self) -> float: ...
    @property
    def max_trust_region_radius(self) -> float: ...
    @property
    def step_accept_threshold(self) -> float: ...
    @property
    def trust_region_shrink_rho(self) -> float: ...
    @property
    def trust_region_grow_rho(self) -> float: ...
    @property
    def trust_region_shrink_scale(self) -> float: ...
    @property
    def trust_region_grow_scale(self) -> float: ...
    @property
    def log_c_clamp(self) -> float: ...
    @property
    def log_q_clamp(self) -> float | None: ...
    def __repr__(self) -> str: ...

class System:
    """Equilibrium concentration solver for nucleic acid strand systems.

    Build a system by chaining ``monomer()`` and ``complex()`` calls,
    then call ``equilibrium()`` to solve for equilibrium concentrations.

    Parameters
    ----------
    temperature_C : float, optional
        Temperature in degrees Celsius (default: 25.0).
    temperature_K : float, optional
        Temperature in kelvin. Cannot be combined with ``temperature_C``.

    Examples
    --------
    >>> import equiconc
    >>> eq = (equiconc.System()
    ...     .monomer("A", 100e-9)
    ...     .monomer("B", 100e-9)
    ...     .complex("AB", [("A", 1), ("B", 1)], dg_st=-10.0)
    ...     .equilibrium())
    >>> eq["AB"] > 0
    True
    """

    def __init__(
        self,
        *,
        temperature_C: float | None = None,
        temperature_K: float | None = None,
        options: SolverOptions | None = None,
    ) -> None: ...
    def monomer(self, name: str, total_concentration: float) -> System:
        """Add a monomer species with a given total concentration.

        Parameters
        ----------
        name : str
            Name of the monomer species. Must be unique and non-empty.
        total_concentration : float
            Total concentration in molar (mol/L). Must be finite and positive.

        Returns
        -------
        System
            The same system instance, for method chaining.
        """
        ...

    @overload
    def complex(
        self,
        name: str,
        composition: list[tuple[str, int]],
        *,
        dg_st: float,
    ) -> System: ...
    @overload
    def complex(
        self,
        name: str,
        composition: list[tuple[str, int]],
        *,
        dg_st: tuple[float, float],
        ds_st: float,
    ) -> System: ...
    @overload
    def complex(
        self,
        name: str,
        composition: list[tuple[str, int]],
        *,
        delta_g_over_rt: float,
    ) -> System: ...
    @overload
    def complex(
        self,
        name: str,
        composition: list[tuple[str, int]],
        *,
        dh_st: float,
        ds_st: float,
    ) -> System: ...
    def complex(
        self,
        name: str,
        composition: list[tuple[str, int]],
        *,
        dg_st: float | tuple[float, float] | None = None,
        delta_g_over_rt: float | None = None,
        dh_st: float | None = None,
        ds_st: float | None = None,
    ) -> System:
        """Add a complex species with a given stoichiometry and energy.

        Exactly one energy specification must be provided:

        - ``dg_st``: standard free energy of formation in kcal/mol
        - ``dg_st=(value, temperature_C)`` + ``ds_st``: |DeltaG| at a
          known temperature plus |DeltaS|; |DeltaH| is derived as
          |DeltaH| = |DeltaG| + T |cdot| |DeltaS|
        - ``delta_g_over_rt``: dimensionless |DeltaG|/RT
        - ``dh_st`` + ``ds_st``: enthalpy (kcal/mol) and entropy
          (kcal/(mol |cdot| K))

        Parameters
        ----------
        name : str
            Name of the complex. Must be unique across all species.
        composition : list of (str, int)
            Monomer composition as ``[(monomer_name, count), ...]``.
        dg_st : float or (float, float), optional
            Standard free energy of formation in kcal/mol, or a
            ``(dg_st, temperature_C)`` tuple requiring ``ds_st``.
        delta_g_over_rt : float, optional
            Dimensionless free energy |DeltaG|/(RT).
        dh_st : float, optional
            Enthalpy of formation in kcal/mol. Must be paired with ``ds_st``.
        ds_st : float, optional
            Entropy of formation in kcal/(mol |cdot| K).

        Returns
        -------
        System
            The same system instance, for method chaining.
        """
        ...

    def equilibrium(self) -> Equilibrium:
        """Solve for equilibrium concentrations.

        Returns
        -------
        Equilibrium
            The result containing concentrations of all species.

        Raises
        ------
        ValueError
            If the system specification is invalid.
        RuntimeError
            If the solver fails to converge.
        """
        ...

class Equilibrium:
    """Result of an equilibrium concentration calculation.

    Supports dict-like access: ``eq["A"]``, ``"A" in eq``, ``len(eq)``,
    and iteration over species names with ``for name in eq``.
    """

    @property
    def monomer_names(self) -> list[str]:
        """Monomer species names, in addition order."""
        ...

    @property
    def complex_names(self) -> list[str]:
        """Complex species names, in addition order."""
        ...

    @property
    def free_monomer_concentrations(self) -> list[float]:
        """Free monomer concentrations in molar, in addition order."""
        ...

    @property
    def complex_concentrations(self) -> list[float]:
        """Complex concentrations in molar, in addition order."""
        ...

    @property
    def converged_fully(self) -> bool:
        """Whether the solver achieved full convergence."""
        ...

    def concentration(self, name: str) -> float:
        """Look up a concentration by species name.

        Parameters
        ----------
        name : str
            Species name (monomer or complex).

        Returns
        -------
        float
            Concentration in molar.

        Raises
        ------
        KeyError
            If the species name is not found.
        """
        ...

    def to_dict(self) -> dict[str, float]:
        """Convert to a dict mapping species names to concentrations.

        Returns
        -------
        dict
            ``{name: concentration}`` in deterministic order (monomers
            first, then complexes, in addition order).
        """
        ...

    def keys(self) -> list[str]:
        """Species names in deterministic order.

        Returns
        -------
        list of str
            Names in order: monomers first, then complexes.
        """
        ...

    def values(self) -> list[float]:
        """Concentrations in deterministic order.

        Returns
        -------
        list of float
            Concentrations in molar, monomers first, then complexes.
        """
        ...

    def items(self) -> list[tuple[str, float]]:
        """``(name, concentration)`` pairs in deterministic order.

        Returns
        -------
        list of (str, float)
            Pairs in order: monomers first, then complexes.
        """
        ...

    def __getitem__(self, name: str) -> float: ...
    def __contains__(self, name: object) -> bool: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[str]: ...
    def __repr__(self) -> str: ...
