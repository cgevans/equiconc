# Changelog

## [Unreleased]

### Changed

- Reverted the crate `license` field from `"BSD-3-Clause AND Apache-2.0"` back
  to `"BSD-3-Clause"`. The dual declaration existed only because of the
  vendored COFFEE sources; with vendoring removed, the published crate
  contains no Apache-2.0-licensed code.
- Comparative benchmarks, the `proptest_vs_coffee` cross-validation test, and
  the `instrument_large` / `instrument_xl` diagnostic examples now depend on
  the `coffee` crate as a pinned git dev-dependency (via the `cgevans/coffee`
  fork, which gates the polars-backed file-input path behind a feature) with
  `default-features = false`, instead of carrying a vendored copy under
  `tests/coffee_vendor/`. These are all dev-only consumers, so the git source
  doesn't affect the published crate. Because COFFEE still pins
  `ndarray = 0.16` while equiconc is on 0.17, an aliased
  `ndarray_coffee = { package = "ndarray", version = "0.16" }` dev-dep
  supplies the array types COFFEE's API requires.
- `instrument_large` / `instrument_xl` no longer report a COFFEE iteration
  count (upstream's `Optimizer` doesn't expose one); wall time only.

## [0.2.0] - 2026-04-18

### Changed

- **BREAKING**: Renamed the Rust `System` builder type to `SystemBuilder`. The
  new `System` is a stateful solver handle that owns numerical inputs, work
  buffers, and the most recent solution; it supports in-place mutation for
  titration / parameter sweeps and re-solves with warm-started λ. The one-shot
  pattern is now `SystemBuilder::new()…build()?.solve()?` instead of
  `System::new()…equilibrium()?`.
- **BREAKING**: `Equilibrium` is now a borrowed view (`Equilibrium<'a>`) into
  the owning `System` rather than an owned struct. Use `eq.get(name)`,
  `eq.at(idx)`, or indexing (`eq["AB"]`, `eq[idx]`) for lookups. To keep
  results past a `System` mutation, copy the data out
  (e.g. `eq.concentrations().to_owned()`). The borrow checker now enforces
  "no stale reads": mutating accessors on `System` cannot fire while an
  `Equilibrium` view is alive.
- Duplicate monomers in a complex composition now have their counts summed
  instead of raising an error.
- **BREAKING** (Rust): bumped `ndarray` from 0.16 to 0.17. Downstream crates
  consuming equiconc's `ArrayView{1,2}` / `Array{1,2}`-valued API must upgrade
  ndarray in lockstep.

### Added

- `SolverOptions` struct exposing previously-hard-coded solver knobs:
  `max_iterations`, gradient tolerances (full + relaxed), trust-region
  parameters (initial / max δ, ρ thresholds, shrink / grow scale
  factors), stagnation threshold, and two numerical clamps
  (`log_c_clamp`, optional `log_q_clamp`). Every field has a default
  matching the previous constant, so `SolverOptions::default()`
  reproduces pre-configuration behavior bit-for-bit.
- `SystemBuilder::options` / `options_ref`, `System::options` /
  `options_mut` / `set_options`, plus
  `System::from_arrays_with_options` and
  `System::from_arrays_with_names_and_options` for passing options
  directly alongside numerical inputs.
- `EquilibriumError::InvalidOptions` variant, raised by
  `SolverOptions::validate()` on inconsistent combinations
  (non-positive tolerances, `shrink_rho >= grow_rho`, etc.) and
  surfaced by every constructor that accepts options.
- Python `equiconc.SolverOptions` class with keyword-only constructor
  mirroring the Rust fields. Pass to `System(options=opts)`.
- `System::from_arrays` and `System::from_arrays_with_names` for constructing
  a solver directly from numerical inputs without going through the
  string-keyed builder. Temperature is not stored at this level — callers
  bake it into `log_q`.
- `System::c0_mut`, `System::log_q_mut`, `System::set_c0`, `System::set_log_q`
  for in-place mutation in titration / parameter sweep workflows.
- `System::last_solution` returning `Option<Equilibrium<'_>>`, `None` if any
  input has been modified since the last successful solve.
- `System::validate` to re-run structural invariant checks after caller-driven
  mutation.
- `EquilibriumError::InvalidInputs` variant, raised by `from_arrays` /
  `from_arrays_with_names` when shapes, the identity block, monomer `log_q`,
  or name tables are inconsistent.

Python bindings are unchanged.

## [0.1.0] - 2026-03-10

- Initial release

