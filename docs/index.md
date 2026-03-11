# equiconc

Equiconc is an equilibrium concentration solver for monomer/complex systems (like networks of interactions of DNA/RNA strands),
_when the complexes are already enumerated and standard free energies of binding are known_.  It implements the convex optimization
method of [Dirks et al. (2007)](https://doi.org/10.1137/060651100), without any other portions of the paper; as such, it can be
applied generally, for example, to find equilibrium concentrations in tile assembly systems.

The library is written in Rust, with a Python interface that is intended to be easily usable.

## Documentation

- [Getting Started](getting-started.md) — installation and basic usage
- [API Reference](api.md) — full API docs for `System` and `Equilibrium`
- [Theory](theory.md) — mathematical background and algorithm details
- Examples
    - [Quickstart](notebooks/quickstart.md) — simple walkthrough
    - [Competitive Binding](notebooks/competitive_binding.md) — multi-complex systems
- [Changelog](changelog.md)

## Quick example

```python
import equiconc

# A + B <=> AB with DG = -10 kcal/mol at 25 C (default)
eq = (
    equiconc.System()
    .monomer("A", 100e-9)       # 100 nM
    .monomer("B", 100e-9)
    .complex("AB", [("A", 1), ("B", 1)], dg_s=-10.0)
    .equilibrium()
)

print(f"Free [A] = {eq['A']:.2e} M")
print(f"Free [B] = {eq['B']:.2e} M")
print(f"[AB]     = {eq['AB']:.2e} M")
```
