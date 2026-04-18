# Changelog

## [Unreleased]

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

### Added

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

