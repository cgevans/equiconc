# Changelog

## [Unreleased]

## [0.4.0] - 2026-04-25

### Added

- `System::solve_with_progress(on_iter)` / `IterationStatus` /
  `SolveControl` / `EquilibriumError::Aborted`. The new method invokes
  a callback once per outer trust-region iteration (linear and log
  paths both); the callback may return `SolveControl::Abort` to
  short-circuit with `EquilibriumError::Aborted`. The existing
  `System::solve()` is now a thin shim that supplies a no-op callback,
  so its behavior and performance are unchanged. Intended as the hook
  for live progress reporting from the web UI's worker, but useful
  anywhere a long sweep wants to surface convergence telemetry.

- Browser-side equilibrium-concentration solver as a separate
  `web/` crate (Leptos + Trunk). Builds to a single static-site
  bundle (`trunk build --release` → `web/dist/`); no backend, no
  JS code, all compute runs in WebAssembly via the unmodified
  `equiconc` solver. The page exposes the full `SolverOptions`
  surface tiered Basic / Advanced / Expert, includes
  `equiconc-defaults` and `COFFEE-compatible` preset buttons, a
  kcal/mol vs RT energy-units toggle (with the temperature input
  hiding itself when not consumed), drag-and-drop file loading for
  `.cfe` / `.ocx` / `.con`, baked-in testcases, sortable concentrations
  table with per-monomer share-of-mass and 100-row pagination for
  COFFEE-scale 50k-species systems, a horizontal bar chart of top
  species by absolute concentration, a per-monomer share-of-mass
  stacked bar chart, a compact live convergence chart
  (log₁₀ ‖∇‖ vs. iteration), and TSV / CSV / JSON-report exports.
  The solve runs in a dedicated Web Worker so the UI thread stays
  responsive; a Cancel button terminates the worker mid-iteration.
  New `just web` and `just web-dev` recipes; CI builds the dist
  artifact and on `main` / tag pushes deploys it to GitHub Pages
  alongside the docs (docs at the project URL root, web app at
  `/app/`).
- `pub mod equiconc::io` with `parse_cfe(text, n_mon)` and
  `parse_concentrations(text)` parsers for NUPACK-style complex
  tables (`.cfe` / `.ocx`, with NUPACK header auto-detection) and
  one-value-per-line concentration files (`.con`). Accepts
  whitespace, `,`, `;`, and `|` as delimiters.
- `pub fn equiconc::water_molar_density(t_c)` — molar density of
  liquid water from the Tanaka 2001 mass-density formula. Useful
  for converting between molarity and mole fraction in callers
  that mirror COFFEE's "scalarity" wrapper.
- `equiconc::Equilibrium::mass_balance_residual` /
  `mass_balance_residual_self` and a free-function
  `equiconc::mass_balance_residual` for `max_i |c0_i − Σ_j A_{ji} c_j|`.

### Changed

- `equiconc-coffee-cli` now consumes `equiconc::io::parse_cfe`,
  `equiconc::io::parse_concentrations`, `equiconc::water_molar_density`,
  and `equiconc::mass_balance_residual` instead of carrying its own
  copies. No behavior change.
- The repository is now a Cargo workspace; the published `equiconc`
  package is unchanged but the workspace also contains the
  `equiconc-web` crate (`publish = false`).

- New `simd` Cargo feature (opt-in, off by default) that vectorizes
  the per-species element-wise hot loops in `evaluate_into` and
  `evaluate_log_into` via `pulp` runtime ISA dispatch (SSE2 / AVX2 /
  AVX-512 on x86, NEON on aarch64, scalar on wasm). Enable with
  `cargo build --features simd` (or `--features python,simd` for the
  Python wheel). Translates to ~9% end-to-end speedup on the
  COFFEE-large `equiconc_linear` testcases; preserves scalar
  performance on `equiconc_log`.

  Numerical contract: linear-path kernels use a degree-12 Taylor
  polynomial after Cody-Waite range reduction (≤2 ulps vs libm);
  log-path kernels keep parallelism for everything around the `exp`
  call but route the exp itself through scalar libm per lane, because
  the trust-region step acceptance on `g = ln f` requires
  per-iteration progress measurable above `4·eps·|g|` and the
  polynomial residual stalls the iteration on extremely stiff systems
  (COFFEE testcase 0 with sub-nanomolar `c0` and ~54k species was the
  canonical failure). The full and candidate evaluators always agree
  on rounding so the trust-region ρ check stays consistent.

## [0.3.0] - 2026-04-25

### Added

- Optional log-objective trust-region path, selected via
  `SolverOptions::objective = SolverObjective::Log` (Rust) or
  `SolverOptions(objective="log")` (Python). The default remains
  `SolverObjective::Linear`, so existing callers see no behavior change.
  The log path minimizes `g(λ) = ln f(λ)` rather than the linear dual
  `f(λ)`; on stiff systems (very strong binding, asymmetric `c⁰`) it
  often converges in many fewer iterations because `g` compresses the
  exponential dynamic range of `f`. The log objective is non-convex
  (`H_g` can be indefinite away from the optimum); equiconc compensates
  with on-the-fly modified-Cholesky regularization of the model
  Hessian, so the dog-leg step always sees a PD matrix and
  `predicted_reduction > 0` by construction. The implementation
  structurally avoids the three documented failure modes of COFFEE's
  log-Lagrangian solver (see `coffee-bugs.md` and
  `coffee/docs/issue2-analysis.md`):

  - Bug 1 (NaN from `∞ − ∞`): `evaluate_log_into` computes the
    objective via log-sum-exp on the un-clamped `log_q + Aᵀλ`, never
    forming `f` and then taking its log. Steps that would push `f ≤ 0`
    are rejected and the trust region shrinks.
  - Bug 2 (premature convergence at `λ ≈ 0` under strong binding): the
    convergence test is on the primal mass-conservation residual
    `|Ac − c⁰|_i < atol + rtol · c0_i` for both objectives — never on
    the log-rescaled gradient `∇g = ∇f / f`, which is the term COFFEE
    suppresses to floating-point underflow.
  - Bug 3 / coffee issue #2 (trust-region oscillation on indefinite
    Hessians): the model Hessian is regularized to PD before dog-leg
    sees it, and a defensive `pred_reduction ≤ 0 → ρ = -1` sentinel
    catches any residual case where regularization saturates.

  Validated against COFFEE on the existing
  `tests/proptest_vs_coffee.rs` cross-check (new
  `prop_equiconc_log_matches_coffee`) and against the linear path on
  `tests/proptest_equiconc.rs` (`prop_log_matches_linear`). Explicit
  reproducers for the three documented coffee failure cases now live in
  `src/lib.rs` (`coffee_bug1_positive_dg_conformer_log`,
  `coffee_bug2_strong_binding_log`, `coffee_issue2_strong_dimer_log`).
- New optional binary `equiconc-coffee-cli` (gated behind the `coffee-cli`
  Cargo feature) that accepts the same NUPACK-style `.ocx`/`.cfe` +
  `.con` input files as COFFEE's `coffee_cli` and produces the same
  space-separated 2-decimal-scientific results payload. Hard-codes
  `T = 37 °C`, mole-fraction scaling, and the `ΔG ≥ -230 kcal/mol`
  clamp to match COFFEE's non-configurable defaults — producing
  byte-for-byte agreement with `coffee_cli` on the monomer free
  concentrations of `../coffee/testcases/{0,1,2}` at the `{:.2e}`
  output precision, and on the full 8-species payload of testcase 2.
  Integration tests in `tests/coffee_cli_compat.rs` verify per-species
  agreement on a synthetic 2-monomer/1-dimer system and on all three
  COFFEE testcases (skipped if `../coffee/testcases/` is absent).
  Build with `cargo build --release --features coffee-cli`.
- `cargo-deny` configuration (`deny.toml`) and a CI job that runs three
  checks against the runtime dependency tree:
  - **licenses**: fails on any SPDX expression outside the allow-list
    (`MIT`, `Apache-2.0`, `Apache-2.0 WITH LLVM-exception`, `BSD-3-Clause`,
    `Unicode-3.0`).
  - **advisories**: fails on any RustSec vulnerability, unmaintained
    crate, unsound advisory, or yanked version.
  - **sources**: fails if any runtime crate comes from anywhere other
    than the default crates.io registry.

  Dev-dependencies (criterion, proptest, and the `cgevans/coffee` git
  dep used for cross-checks) are excluded via `[graph] exclude-dev`
  since they ship in neither the crate tarball nor the wheel.
- CI lint job: `cargo fmt --all --check`,
  `cargo clippy --all-features --all-targets -- -D warnings`, and
  `cargo doc --no-deps --all-features` with `RUSTDOCFLAGS=-D warnings`.
  Catches formatting drift, clippy regressions, and broken intra-doc
  links ahead of a docs.rs publish.
- CI cross-platform / cross-Python smoke tests (`tests-matrix` job):
  the existing `tests` job runs only on Linux + Python 3.12 because of
  the cargo-llvm-cov coverage instrumentation; the new matrix exercises
  the corners that ship in the PyPI wheel but were never otherwise
  tested — minimum supported Python (3.10), free-threaded Python
  (3.13t), macOS, and Windows.
- Dependabot configuration (`.github/dependabot.yml`) for weekly Cargo,
  GitHub Actions, and uv (Python) dependency updates, with non-major
  bumps grouped per ecosystem to reduce PR churn.

### Changed

- `cargo fmt --all` sweep across `src/`, `tests/`, `benches/`, and
  `examples/`. No behavior change; lands ahead of the new
  `cargo fmt --check` CI gate so the gate starts green.
- Clippy cleanup so `cargo clippy --all-features --all-targets -- -D warnings`
  passes. Mechanical fixes: derive `Default` on `SolverObjective`;
  collapse `field_reassign_with_default` patterns in tests into struct
  literals; rewrite `!(f > 0.0) || !f.is_finite()` as
  `f <= 0.0 || !f.is_finite()` (NaN-equivalent, clippy-clean);
  `iter().copied().collect()` → `to_vec()`; minor doc-comment
  re-indentation in benches; `for i in 0..n` index loop → `enumerate`;
  collapse nested `if let` into a Rust 2024 let-chain. Type aliases
  `ComplexSpec` / `PyComplexSpec` introduced for the `Vec<(String,
  Vec<(String, usize)>, _)>` builder fields. `#[allow]` annotations
  with one-line comments where the lint can't tell the code is
  intentional: NaN-safe `!(a < b)` rho ordering check in
  `SolverOptions::validate`; `too_many_arguments` on the
  `evaluate_into` / `evaluate_log_into` hot-path inner functions and
  on the pyo3-bound `PySystem::complex` method.
- `builds.yml` now uses `concurrency: cancel-in-progress` keyed on
  `${{ github.ref }}` for `pull_request` events, so superseded PR runs
  are cancelled. Tag and main pushes are unaffected.
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
- Bumped `criterion` dev-dependency from 0.5 to 0.8.

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

