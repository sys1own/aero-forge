# Aero-Forge

Aero-Forge is an LLM-integrated build tool that automatically generates, compiles, tests, and fixes numeric Python functions by transpiling them to Rust/PyO3 extensions.

## Goals

- Accept a Python function (or `.aero` blueprint) and transpile it to a native Rust extension.
- Run tests in an isolated sandbox.
- If compilation or tests fail, attribute the error and prompt an LLM for a fix.
- Iterate until the function passes, then merge the result back.

## Setup

1. Install Rust and cargo: https://rustup.rs/
2. Install the package in editable mode:
   ```bash
   pip install -e ".[dev]"
   ```
3. Choose an LLM provider and set the corresponding environment variable.

### Provider configuration

| Provider | `AERO_FORGE_LLM_PROVIDER` | Required API key | Optional model override |
|----------|-----------------------------|------------------|-------------------------|
| OpenAI | `openai` | `OPENAI_API_KEY` | `AERO_FORGE_MODEL` (default `gpt-4`) |
| OpenRouter | `openrouter` | `OPENROUTER_API_KEY` | `AERO_FORGE_MODEL` (default `openrouter/free`) |
| Gemini | `gemini` | `GEMINI_API_KEY` + `pip install google-generativeai` | `AERO_FORGE_MODEL` (default `gemini-2.0-flash`) |
| Router only | `none` | none | none |

All providers also fall back to `AERO_FORGE_API_KEY` if their provider-specific key is not set.

### Examples

```bash
# OpenAI
export AERO_FORGE_LLM_PROVIDER=openai
export OPENAI_API_KEY="sk-..."

# OpenRouter
export AERO_FORGE_LLM_PROVIDER=openrouter
export OPENROUTER_API_KEY="sk-or-..."

# Gemini
export AERO_FORGE_LLM_PROVIDER=gemini
export GEMINI_API_KEY="..."
pip install google-generativeai

# Generic key fallback
export AERO_FORGE_LLM_PROVIDER=openai
export AERO_FORGE_API_KEY="sk-..."
```

## Usage

Fix and compile a function (uses the configured LLM provider, or router-only if none is set):

```bash
aero-forge fix tests/fixtures/fibonacci.py --function fibonacci
```

Use the accelerator without LLM generation:

```bash
aero-forge fix tests/fixtures/fibonacci.py --function fibonacci --no-llm
```

Select a provider or model on the command line:

```bash
aero-forge fix broken.py --function bad_syntax --llm-provider gemini
aero-forge fix broken.py --function bad_syntax --llm-provider openrouter --model openai/gpt-4
aero-forge fix broken.py --function bad_syntax --llm-provider openai --model gpt-4o
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
| `AERO_FORGE_LLM_PROVIDER` | LLM provider: `openai`, `openrouter`, `gemini`, or `none` | `none` |
| `AERO_FORGE_MODEL` | Model name to use | provider default |
| `OPENAI_API_KEY` | OpenAI API key | (unset) |
| `OPENROUTER_API_KEY` | OpenRouter API key | (unset) |
| `GEMINI_API_KEY` | Google Gemini API key | (unset) |
| `AERO_FORGE_API_KEY` | Generic fallback API key | (unset) |
| `AERO_FORGE_BASE_URL` | Override the provider base URL | (provider default) |
| `AERO_FORGE_MAX_RETRIES` | Retries per LLM call | `3` |
| `AERO_FORGE_MAX_ITERATIONS` | Maximum forge iterations | `5` |
| `AERO_FORGE_CACHE_ENABLED` | Enable fix cache | `true` |

### CLI flags

- `--llm-provider {openai,openrouter,gemini,none}` – select the LLM provider.
- `--model MODEL` – override the model name.
- `--no-llm` – skip the LLM entirely (router-only mode).
- `--no-cache` – disable the fix cache.
- `--max-retries N` – retries per LLM call.
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
   - Finally calls the configured LLM provider with retry and exponential backoff.
7. When tests pass, the native extension and wrapper are copied back to the source directory.

## Free-tier / low-quota usage

- Use `aero-forge fix ... --llm-provider openrouter --model openrouter/free` and set `OPENROUTER_API_KEY` to leverage OpenRouter's free models.
- Use `aero-forge fix ... --llm-provider gemini` with `GEMINI_API_KEY` and `pip install google-generativeai` for Google Gemini.
- If the provider is not configured or the key is missing, Aero-Forge logs a clear message and falls back to router-only mode.
- Disable the LLM entirely with `--no-llm` or `AERO_FORGE_LLM_PROVIDER=none` to rely on the router and cache.

## Notes

- The first build may take a while as PyO3 is compiled.
- `AERO_FORGE_LLM_PROVIDER` defaults to `none`, so no API calls are attempted unless explicitly configured.
- Fix cache is stored in `~/.cache/aero-forge/fix_cache.json`.
