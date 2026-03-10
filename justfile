# Default: list available recipes
default:
    @just --list

venv := justfile_directory() / ".venv"

# Rust test coverage only
coverage-rust:
    cargo llvm-cov --summary-only

# Rust-from-Python coverage only (instruments cdylib, runs pytest)
coverage-python:
    #!/usr/bin/env bash
    set -euo pipefail
    export VIRTUAL_ENV="{{ venv }}"
    source <(cargo llvm-cov show-env --export-prefix)
    cargo llvm-cov clean --workspace
    maturin develop
    {{ venv }}/bin/pytest tests/test_equiconc.py -v
    cargo llvm-cov report

# Combined Rust + Python coverage (merged report)
coverage:
    #!/usr/bin/env bash
    set -euo pipefail
    export VIRTUAL_ENV="{{ venv }}"
    source <(cargo llvm-cov show-env --export-prefix)
    cargo llvm-cov clean --workspace
    cargo test
    maturin develop
    {{ venv }}/bin/pytest tests/test_equiconc.py -v
    cargo llvm-cov report

# Combined coverage with HTML report in target/coverage/
coverage-html:
    #!/usr/bin/env bash
    set -euo pipefail
    export VIRTUAL_ENV="{{ venv }}"
    source <(cargo llvm-cov show-env --export-prefix)
    cargo llvm-cov clean --workspace
    cargo test
    maturin develop
    {{ venv }}/bin/pytest tests/test_equiconc.py -v
    cargo llvm-cov report --html --output-dir target/coverage

# Combined coverage with HTML report, opened in browser
coverage-open:
    #!/usr/bin/env bash
    set -euo pipefail
    export VIRTUAL_ENV="{{ venv }}"
    source <(cargo llvm-cov show-env --export-prefix)
    cargo llvm-cov clean --workspace
    cargo test
    maturin develop
    {{ venv }}/bin/pytest tests/test_equiconc.py -v
    cargo llvm-cov report --html --output-dir target/coverage --open

# Serve the HTML coverage report on localhost:8000
coverage-serve: coverage-html
    python3 -m http.server 8000 -d target/coverage/html

# Build Python extension into venv
develop:
    uv run maturin develop --release

# Pre-execute notebooks and convert to markdown
docs-notebooks: develop
    {{ venv }}/bin/jupyter nbconvert --to markdown --execute \
        --output-dir docs/notebooks/ \
        docs/notebooks/quickstart.ipynb \
        docs/notebooks/competitive_binding.ipynb

# Build documentation
docs: docs-notebooks
    {{ venv }}/bin/zensical build

# Serve documentation with live reload
docs-serve: docs-notebooks
    {{ venv }}/bin/zensical serve
