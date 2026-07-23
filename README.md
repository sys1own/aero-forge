# Aero-Forge

Aero-Forge is an LLM-integrated build tool that automatically generates, compiles, tests, and fixes numeric Python functions by transpiling them to Rust/PyO3 extensions.

## Goals

- Accept a Python function (or `.aero` blueprint) and transpile it to a native Rust extension.
- Run tests in an isolated sandbox.
- If compilation or tests fail, attribute the error and prompt an LLM for a fix.
- Iterate until the function passes, then merge the result back.

## Setup

1. Install Rust and cargo: https://rustup.rs/
2. Set one or more API keys:
   - OpenAI: `export OPENAI_API_KEY="sk-..."`
   - OpenRouter: `export OPENROUTER_API_KEY="sk-or-..."`
   - Google Gemini: `export GEMINI_API_KEY="..."` or `export GOOGLE_API_KEY="..."`
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

Use a custom model priority list and retry settings:

```bash
aero-forge fix tests/fixtures/fibonacci.py --function fibonacci \
  --model-priority openrouter/free,gpt-4 \
  --max-retries 3
```

Run the test suite:

```bash
pytest
```

## Configuration

Aero-Forge merges configuration from (lowest to highest precedence):

1. Built-in defaults
2. `accelerate.toml` in the project or parent directories
3. Environment variables
4. CLI flags

### Environment variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AERO_FORGE_MODEL_PRIORITY` | Comma-separated LLM priority list | `openrouter/free,gpt-4` |
| `AERO_FORGE_FALLBACK_MODEL` | Final fallback model | `openrouter/free` |
| `AERO_FORGE_MAX_RETRIES` | Retries per model | `3` |
| `AERO_FORGE_MAX_ITERATIONS` | Maximum forge iterations | `5` |
| `AERO_FORGE_CACHE_ENABLED` | Enable fix cache | `true` |
| `AERO_FORGE_USE_LLM` | Allow LLM calls | `true` |

### CLI flags

- `--no-llm` – skip the LLM entirely (router-only mode).
- `--no-cache` – disable the fix cache.
- `--model-priority MODEL1,MODEL2` – override model priority.
- `--fallback-model MODEL` – set final fallback model.
- `--max-retries N` – retries per model.
- `--max-iterations N` – maximum iterations.
- `--verbose` – show debug logs and full output.

## How it works

1. Parses the target Python function and lowers it to a UAST/HIN graph.
2. The precision shield selects Rust types and traits.
3. The scaffold engine emits a temporary PyO3 crate.
4. `cargo build --release` produces a shared library.
5. The sandbox runs `pytest` against the compiled extension.
6. On failure, the orchestrator:
   - Classifies the error (transient / recoverable / fatal).
   - First tries the self-healing router (zero API calls).
   - Falls back to a cached fix if one exists.
   - Finally calls an LLM with retry, model fallback, and exponential backoff.
7. When tests pass, the native extension and wrapper are copied back to the source directory.

## Free-tier / low-quota usage

- `openrouter/free` is included as the first default model. Set `OPENROUTER_API_KEY` to use it.
- If a model rate-limits or errors, Aero-Forge waits with exponential backoff and tries the next model in the list.
- Disable the LLM entirely with `--no-llm` to rely on the router and cache.

## Notes

- The first build may take a while as PyO3 is compiled.
- Without any API key, `--no-llm` mode still compiles valid functions but cannot repair broken ones.
- Fix cache is stored in `~/.cache/aero-forge/fix_cache.json`.
