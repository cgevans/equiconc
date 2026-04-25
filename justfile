# Default: list available recipes
default:
    @just --list

venv := justfile_directory() / ".venv"

# License check on the runtime dependency tree (cargo-deny)
deny:
    cargo deny check licenses

# Rust coverage from Rust tests
coverage-rust:
    #!/usr/bin/env bash
    set -euo pipefail
    source <(cargo llvm-cov show-env --export-prefix --no-cfg-coverage)
    export CARGO_TARGET_DIR=$CARGO_LLVM_COV_TARGET_DIR
    export CARGO_INCREMENTAL=1
    cargo llvm-cov clean --workspace
    cargo test
    cargo llvm-cov --no-run --lcov --output-path coverage-rust.lcov

# Python + Rust coverage from Python tests (instrumented build, both outputs)
coverage-python-full:
    #!/usr/bin/env bash
    set -euo pipefail
    source <(cargo llvm-cov show-env --export-prefix --no-cfg-coverage)
    export CARGO_TARGET_DIR=$CARGO_LLVM_COV_TARGET_DIR
    export CARGO_INCREMENTAL=1
    cargo llvm-cov clean --workspace
    if [ -f .venv/bin/activate ]; then
        source .venv/bin/activate
    else
        source .venv/Scripts/activate
    fi
    maturin develop --uv --profile dev
    pytest --cov equiconc --cov-report term-missing --cov-report xml:coverage-python.xml
    cargo llvm-cov report --lcov --output-path coverage-rust-from-python.lcov

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

# Pre-execute notebooks and convert to markdown
docs-notebooks: 
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
