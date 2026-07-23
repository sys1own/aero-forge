# Aero-Forge

Aero-Forge is an LLM-integrated build tool that automatically generates, compiles, tests, and fixes numeric Python functions by transpiling them to Rust/PyO3 extensions.

## Goals

- Accept a Python function (or `.aero` blueprint) and transpile it to a native Rust extension.
- Run tests in an isolated sandbox.
- If compilation or tests fail, attribute the error and prompt an LLM for a fix.
- Iterate until the function passes, then merge the result back.

## Setup

1. Install Rust and cargo: https://rustup.rs/
2. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```
3. Install the package in editable mode:
   ```bash
   pip install -e ".[dev]"
   ```

## Usage

Fix and compile a function:

```bash
aero-forge fix tests/fixtures/fibonacci.py --function fibonacci
```

Use the accelerator without LLM generation:

```bash
aero-forge fix tests/fixtures/fibonacci.py --function fibonacci --no-llm
```

Run the test suite:

```bash
pytest
```

## How it works

1. Parses the target Python function and lowers it to a UAST/HIN graph.
2. The precision shield selects Rust types and traits.
3. The scaffold engine emits a temporary PyO3 crate.
4. `cargo build --release` produces a shared library.
5. The sandbox runs `pytest` against the compiled extension.
6. On failure, the orchestrator parses errors, routes through a small self-healing layer, and then asks an LLM for a corrected function.
7. When tests pass, the native extension and wrapper are copied back to the source directory.

## Notes

- The first build may take a while as PyO3 is compiled.
- Without `OPENAI_API_KEY`, `--no-llm` mode still compiles valid functions but cannot repair broken ones.
