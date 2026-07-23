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

## Multi-function builds with `aero-forge build`

Create a project skeleton:

```bash
aero-forge init my_project
cd my_project
```

`init` creates:

```
my_project/
├── blueprint.aero
├── src/
│   └── example.py
├── tests/
│   └── test_example.py
└── dist/
```

Build all functions listed in the blueprint:

```bash
aero-forge build
```

Build from a YAML blueprint:

```bash
aero-forge build blueprint.yaml
```

### Blueprint file format

`blueprint.aero` (YAML-compatible):

```aero
project: "my_project"

functions:
  - file: "src/compute.py"
    name: "mandelbrot"
    tests: ["tests/test_compute.py"]
  - file: "src/transform.py"
    name: "fft"
    tests: ["tests/test_transform.py"]

compiler_flags:
  - "-C target-cpu=native"
  - "-C opt-level=3"

output_dir: "./dist"

llm:
  provider: "gemini"
  model: "gemini-2.0-flash"
```

Or `blueprint.yaml`:

```yaml
project: my_project
functions:
  - file: src/compute.py
    name: mandelbrot
    tests:
      - tests/test_compute.py
compiler_flags:
  - -C target-cpu=native
output_dir: ./dist
llm:
  provider: gemini
  model: gemini-2.0-flash
```

### Auto-discovery

`--auto` scans a Python file and compiles all public top-level functions:

```bash
aero-forge build --auto src/my_module.py --no-llm
```

### Compile all public functions in a file

Use `name: "*"` or `compile_all: true` to compile every public top-level function in a source file:

```aero
functions:
  - file: src/utils.py
    name: "*"
    tests: [tests/test_utils.py]
  - file: src/transform.py
    compile_all: true
```

Functions from the same file are compiled into one shared library.

### Preview a build

```bash
aero-forge build --dry-run
```

`--dry-run` parses the blueprint, expands wildcards, and lists what would be compiled without invoking `cargo`.

### Build CLI flags

- `--auto FILE` – auto-discover functions.
- `--llm-provider {openai,openrouter,gemini,none}` – override provider.
- `--model MODEL` – override model.
- `--output-dir PATH` / `-o PATH` – override `output_dir`.
- `--jobs N` / `-j N` – parallel build jobs (default `min(4, functions)`).
- `--no-llm` – skip LLM-based healing.
- `--no-cache` – disable the build cache.
- `--write-blueprint` – when using `--auto`, write a generated `blueprint.aero`.
- `--dry-run` – preview what would be built.
- `--verbose` – show debug logs and per-function results.

See `BLUEPRINT.md` for a complete field reference and a fully commented example.

### Build caching and parallelism

- Each source file's compilation result is cached under `~/.cache/aero-forge/build_cache/`.
- Re-running `aero-forge build` skips unchanged functions.
- Functions from different source files are compiled in parallel (configurable with `--jobs`).

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

## Supported Python constructs (stress-tested)

Aero-Forge is designed for numeric/scalar Python functions. The `stress_tests/`
directory covers real-world patterns:

**Supported:**
- Scalar arithmetic, comparisons, and boolean returns.
- `if`/`elif`/`else`, nested conditions.
- `for`/`while` loops over `range(...)`.
- `break` and `continue` inside loops.
- Recursion.
- Tuple unpacking assignments (`a, b = b, a`).
- `min`/`max` with multiple scalar arguments.
- `math.sqrt`, `math.sin`, `math.cos`, `math.tan`, `math.exp`, `math.log`,
  `math.log10`, `math.ceil`, `math.floor`, `math.trunc`, `math.pow`,
  `math.radians`, `math.degrees`, and `math.pi`/`e`/`tau`.
- Multi-source builds via `compile_all` and per-function `tests`.
- Parallel builds (`--jobs`) and incremental build caching.
- Simple Python classes with `__init__`, instance methods, `@staticmethod`, and `@classmethod`.

**Currently unsupported (clear error messages):**
- Lists, list comprehensions, slicing, `append`, `len`, `sum`, `enumerate`, `zip`.
- Dictionaries and sets.
- Class inheritance, properties, and dataclasses.
- `try`/`except`, `with`, `yield`, `async`/`await`.
- `random`, `datetime`, `re`, `json`, and other non-math stdlib modules.
- I/O (`print`, file access, network, `os`, `subprocess`, etc.).

See `stress_tests/README.md` for the full campaign report.

## LLM healing

When compilation or tests fail, Aero-Forge can prompt a configured LLM to fix the source code:

```bash
export AERO_FORGE_LLM_PROVIDER=openrouter
export OPENROUTER_API_KEY="sk-or-..."
export AERO_FORGE_MODEL=openrouter/free
aero-forge fix broken.py --function bad_syntax
```

Supported providers: `openai`, `openrouter`, `gemini`, `none`.
Missing or invalid keys always fall back to router-only mode (or `--no-llm`) without crashing.

### What LLM healing can fix

- Simple syntax errors such as a missing colon (`if x > 0\n    ...`).
- Type mismatches such as `return a / b` in an `int`-declared function.
- Single broken functions inside a multi-function source file (the other functions are preserved).
- Missing imports when the router has a matching rule.

### Retry and rate-limit behavior

- Every LLM call retries up to `AERO_FORGE_MAX_RETRIES` (default 3) with exponential backoff.
- Backoff honors server-provided retry hints (`Retry-After` headers for OpenAI/OpenRouter, `google.rpc.retry_delay` blocks for Gemini) up to the `backoff_max` cap.
- After exhausting retries, the forge loop returns a partial result and the original source is left untouched.

### Limitations

- The LLM only edits the target function; it does not add new files or change the project structure.
- Healing is limited to the supported Python constructs listed above (numeric/scalar code).
- Free-tier API keys can be rate-limited or quota-exhausted; if you hit limits, the tool will retry and then fall back to router-only mode.
- Results depend on the model. `openrouter/free` and `gemini-2.0-flash` work for the bundled stress tests, but more complex broken code may need a stronger model.

## Notes

- The first build may take a while as PyO3 is compiled.
- `AERO_FORGE_LLM_PROVIDER` defaults to `none`, so no API calls are attempted unless explicitly configured.
- Fix cache is stored in `~/.cache/aero-forge/fix_cache.json`.
- Build cache is stored in `~/.cache/aero-forge/build_cache/`.
