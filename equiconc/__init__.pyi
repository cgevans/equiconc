from typing import Iterator, overload

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
    ...     .complex("AB", [("A", 1), ("B", 1)], dg_s=-10.0)
    ...     .equilibrium())
    >>> eq["AB"] > 0
    True
    """

    def __init__(
        self,
        *,
        temperature_C: float | None = None,
        temperature_K: float | None = None,
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
        dg_s: float,
    ) -> System: ...
    @overload
    def complex(
        self,
        name: str,
        composition: list[tuple[str, int]],
        *,
        dg_s: tuple[float, float],
        ds_s: float,
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
        dh_s: float,
        ds_s: float,
    ) -> System: ...
    def complex(
        self,
        name: str,
        composition: list[tuple[str, int]],
        *,
        dg_s: float | tuple[float, float] | None = None,
        delta_g_over_rt: float | None = None,
        dh_s: float | None = None,
        ds_s: float | None = None,
    ) -> System:
        """Add a complex species with a given stoichiometry and energy.

        Exactly one energy specification must be provided:

        - ``dg_s``: standard free energy of formation in kcal/mol
        - ``dg_s=(value, temperature_C)`` + ``ds_s``: |DeltaG| at a
          known temperature plus |DeltaS|; |DeltaH| is derived as
          |DeltaH| = |DeltaG| + T |cdot| |DeltaS|
        - ``delta_g_over_rt``: dimensionless |DeltaG|/RT
        - ``dh_s`` + ``ds_s``: enthalpy (kcal/mol) and entropy
          (kcal/(mol |cdot| K))

        Parameters
        ----------
        name : str
            Name of the complex. Must be unique across all species.
        composition : list of (str, int)
            Monomer composition as ``[(monomer_name, count), ...]``.
        dg_s : float or (float, float), optional
            Standard free energy of formation in kcal/mol, or a
            ``(dg_s, temperature_C)`` tuple requiring ``ds_s``.
        delta_g_over_rt : float, optional
            Dimensionless free energy |DeltaG|/(RT).
        dh_s : float, optional
            Enthalpy of formation in kcal/mol. Must be paired with ``ds_s``.
        ds_s : float, optional
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
