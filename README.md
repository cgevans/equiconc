[![Documentation](https://img.shields.io/badge/docs-docs.costi.net%2Fequiconc-blue)](https://docs.costi.net/equiconc/)

Equiconc is an equilibrium concentration solver for monomer/complex systems (like networks of interactions of DNA/RNA strands),
_when the complexes are already enumerated and standard free energies of binding are known_.  It implements the convex optimization
method of [Dirks et al. (2007)](https://doi.org/10.1137/060651100), without any other portions of the paper; as such, it can be
applied generally, for example, to find equilibrium concentrations in tile assembly systems.

The library is written in Rust, with a Python interface that is intended to be easily usable.

There is also an [in-browser web interface](https://docs.costi.net/equiconc/app/), which runs the solver in your browser directly (not a server).  This interface was inspired by [COFFEE](https://coffeesolver.dev/); Equiconc also supports a modified version of their log-based optimization ([Yu et al, 2025](https://doi.org/10.1109/SIEDS65500.2025.11021092)).

## Quick example

```python
import equiconc

# A + B <=> AB with DG = -10 kcal/mol at 25 C (default)
eq = (
    equiconc.System()
    .monomer("A", 100e-9)       # 100 nM
    .monomer("B", 100e-9)
    .complex("AB", [("A", 1), ("B", 1)], dg_st=-10.0)
    .equilibrium()
)

print(f"Free [A] = {eq['A']:.2e} M")
print(f"Free [B] = {eq['B']:.2e} M")
print(f"[AB]     = {eq['AB']:.2e} M")
```

## Note regarding Codeberg 

I am aware that Codeberg is banning majority-LLM projects; while I do fully understand how this code works, and have reviewed the code, the direct code was LLM-generated, and makes no attempt to hide that fact.  I am also now aware that "side projects and experiments" are not welcome on Codeberg, and that it might not be the right place for academic code projects.  I am in the process of moving my projects off of Codeberg, but ask for some patience and understanding while I do so.
