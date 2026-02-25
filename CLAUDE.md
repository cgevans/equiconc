# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

equiconc is a Rust library (with Python bindings via PyO3) that computes equilibrium concentrations for systems of interacting nucleic acid strands that form complexes. It implements the trust-region Newton method on the convex dual problem from Dirks et al. (2007), SIAM Review 49(1), 65-88.

## Build & Test Commands

```bash
# Rust
cargo build --release
cargo test
cargo check

# Python
maturin develop --release              # build Python wheel into .venv
.venv/bin/pytest tests/test_equiconc.py -v   # all Python tests
.venv/bin/pytest tests/test_equiconc.py -v -k test_name   # single test
```

## Coverage

Requires: `cargo-llvm-cov` (`cargo install cargo-llvm-cov`) and `just`.

```bash
just coverage          # text summary, Rust + Python merged
just coverage-rust     # Rust tests only
just coverage-python   # Rust-from-Python only
just coverage-html     # HTML report in target/coverage/
just coverage-open     # HTML report, opened in browser
```

## Architecture

### Core library (`src/lib.rs`)

Public API uses a builder pattern: `System::new()` → `.temperature()` → `.monomer()` → `.complex()` → `.equilibrium()` → `Result<Equilibrium, EquilibriumError>`.

Internally, the solver:
1. `build_problem()` constructs the stoichiometry matrix **A** (monomers × all species), reference free energies **log_q**, and initial concentrations **c⁰**
2. `solve_dual()` minimizes the dual objective `f(λ) = -λᵀc⁰ + Σⱼ Q̃ⱼ exp(Aᵀλ)_j` using trust-region Newton with dog-leg steps (Cholesky on the Hessian, which is guaranteed positive definite)
3. `evaluate()` computes f, gradient, and Hessian in log-space to prevent overflow
4. Primal concentrations recovered via `c_j = exp(log_q_j + (Aᵀλ)_j)`

Key insight: the dual problem dimension equals the number of monomer species (typically 2-10), making the optimization fast regardless of how many complexes exist.

Debug-mode assertions verify mass conservation and equilibrium conditions post-solve.

### Python bindings (`src/python.rs`)

Gated behind the `python` Cargo feature. `PySystem` mirrors the Rust builder; `PyEquilibrium` provides dict-like access (`eq["AB"]`, `"AB" in eq`) plus `to_dict()` and property getters.

### Dependencies

- **nalgebra** — matrix/vector ops and Cholesky decomposition
- **pyo3** (optional, feature = "python") — Python FFI
- **maturin** — builds the Rust cdylib as a Python wheel
