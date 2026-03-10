# Getting Started

## Prerequisites

- Python >= 3.10
- Rust toolchain (for building from source)
- [maturin](https://www.maturin.rs/) (`uv tool install maturin`)

## Installation

equiconc is a Rust library with Python bindings built via [maturin](https://www.maturin.rs/). Install from source:

```bash
git clone <repo-url>
cd equiconc
python -m venv .venv
source .venv/bin/activate
maturin develop --release
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run maturin develop --release
```

## Quick start

### Simple dimerization

Compute equilibrium concentrations for A + B &#x21CC; AB:

```python
import equiconc

eq = (
    equiconc.System()
    .monomer("A", 100e-9)       # 100 nM total
    .monomer("B", 100e-9)       # 100 nM total
    .complex("AB", [("A", 1), ("B", 1)], delta_g=-10.0)
    .equilibrium()
)

for name, conc in eq.items():
    print(f"{name}: {conc:.3e} M")
```

### Temperature

The default temperature is 25 &deg;C. Set it explicitly in Celsius or
Kelvin:

```python
sys = equiconc.System()                     # 25 C (default)
sys = equiconc.System(temperature_C=37)     # 37 C
sys = equiconc.System(temperature_K=310.15) # 310.15 K = 37 C
```

### Energy specifications

You can specify complex energies in three ways:

**Standard free energy** (\(\Delta G^\circ\) in kcal/mol):

```python
sys.complex("AB", [("A", 1), ("B", 1)], delta_g=-10.0)
```

**Dimensionless free energy** (\(\Delta G / RT\), unitless). When all
complexes use this form, temperature is not required:

```python
sys.complex("AB", [("A", 1), ("B", 1)], delta_g_over_rt=-16.2)
```

**Enthalpy and entropy** (\(\Delta H\) in kcal/mol, \(\Delta S\) in
kcal/(mol&middot;K)). The free energy \(\Delta G = \Delta H - T \Delta S\)
is computed at solve time, so changing temperature shifts the equilibrium:

```python
sys.complex("AB", [("A", 1), ("B", 1)], delta_h=-50.0, delta_s=-0.13)
```

### Builder pattern

The API uses a fluent builder pattern. Chain calls to `monomer()` and
`complex()`, then call `equilibrium()` to solve:

```python
sys = equiconc.System(temperature_C=37)
sys = sys.monomer("A", 1e-6)
sys = sys.monomer("B", 1e-6)
sys = sys.complex("AB", [("A", 1), ("B", 1)], delta_g=-10.0)
eq = sys.equilibrium()
```

### Working with results

The `Equilibrium` object supports dict-like access:

```python
# Bracket access
conc_ab = eq["AB"]

# Membership test
assert "AB" in eq

# Iteration
for name in eq:
    print(name, eq[name])

# Convert to dict
d = eq.to_dict()

# Properties
eq.monomer_names           # ["A", "B"]
eq.complex_names           # ["AB"]
eq.free_monomer_concentrations  # [float, float]
eq.complex_concentrations       # [float]
```

### Multi-complex systems

Define as many complexes as needed. The solver scales with the number of
**monomer** species (not complexes):

```python
eq = (
    equiconc.System()
    .monomer("A", 1e-6)
    .monomer("B", 1e-6)
    .monomer("C", 1e-6)
    .complex("AB", [("A", 1), ("B", 1)], delta_g=-10.0)
    .complex("AC", [("A", 1), ("C", 1)], delta_g=-8.0)
    .complex("ABC", [("A", 1), ("B", 1), ("C", 1)], delta_g=-15.0)
    .equilibrium()
)
```

## Conventions

- **Concentrations** are in **molar** (mol/L).
- **Temperature** defaults to **25 &deg;C** (298.15 K). Specify via
  `temperature_C` (Celsius) or `temperature_K` (Kelvin).
- **Free energies** (`delta_g`) are in **kcal/mol** with a **1 M standard
  state** (\(u_0 = 1 \text{M}\)).
- **Enthalpy** (`delta_h`) is in **kcal/mol**;
  **entropy** (`delta_s`) is in **kcal/(mol&middot;K)**.
- **Dimensionless energies** (`delta_g_over_rt`) are unitless
  \(\Delta G / RT\) values.

!!! note "Symmetry corrections"
    Homodimer and higher homo-oligomer symmetry corrections are **not** applied
    automatically: equiconc has no way of knowing whether the complex is
    actually symmetric. If your complex contains identical strands, include the
    symmetry correction in the \(\Delta G^\circ\) value you provide
    (e.g., add \(+RT \ln \sigma\) where \(\sigma\) is the symmetry number).

## Error handling

Invalid inputs raise `ValueError`:

```python
import equiconc

# No monomers
try:
    equiconc.System().equilibrium()
except ValueError as e:
    print(e)  # "system has no monomers"

# Negative concentration
try:
    equiconc.System().monomer("A", -1e-9).equilibrium()
except ValueError as e:
    print(e)  # "invalid concentration: -0.000000001 ..."

# Missing energy specification
try:
    equiconc.System().monomer("A", 1e-9).complex("AB", [("A", 1)])
except ValueError as e:
    print(e)  # "must specify energy: delta_g, delta_g_over_rt, ..."
```
