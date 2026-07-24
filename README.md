# Aero-Forge: Prompt-Driven Python-to-Rust Build System

Aero-Forge turns a natural language prompt (or a Python function) into a fast, compiled Rust extension that you can import from Python. It handles code generation, transpilation, native compilation, testing, and optional LLM-driven healing and optimization.

## What is Aero-Forge?

Aero-Forge is a prompt-driven build system for Python. You describe the function you want in plain English, or point it at an existing `.py` file, and it produces a native Rust/PyO3 extension that is typically **10-100x faster** than the equivalent pure-Python implementation.

It is designed for numerical, algorithmic, and performance-critical code. The toolchain is opinionated: it focuses on scalar arithmetic, loops, conditionals, lists, simple classes, and NumPy-style 1D/2D array operations. When it cannot compile something, it produces a clear error message instead of silently generating broken Rust.

## Key Features

- **Natural Language Prompts** - Describe what you want and Aero-Forge generates Python code, tests, and a build blueprint.
- **Zero Manual Rust Boilerplate** - No Cargo.toml, `#[pyfunction]` annotations, or linker flags are required from the user.
- **LLM Self-Healing** - If `cargo build` or tests fail, Aero-Forge can ask an LLM to fix the code, retry, and continue iterating.
- **Algorithm Library** - Pick from a curated library of reference implementations (sorting, matrix, FFT, math) or let the LLM select one automatically.
- **Multi-Variant Testing** - Generate several implementations, compile them in parallel, benchmark each, and select the fastest variant that passes.
- **Explainable Builds** - Add `--explain` to get the LLM to describe the algorithm choice, complexity, and tradeoffs.
- **Blueprint Support** - Declarative `.aero` files for multi-function projects with tests, compiler flags, and LLM configuration.
- **Auto-Discovery** - `aero-forge build --auto-detect` discovers `src/` and `tests/` and compiles everything it understands.
- **Project Builds & Zip Bundles** - `aero-forge build --project <dir>` compiles every public function in a project directory and produces a downloadable zip with source, compiled libraries, a Python package, and a build manifest.
- **Zip Uploads** - `aero-forge build --upload project.zip` extracts, builds, and re-bundles an uploaded project.
- **Project-Aware Generation** - `aero-forge generate --prompt "..." --project <dir>` adds a new function to an existing project and rebuilds the bundle.
- **Interactive Chat** - Refine prompts conversationally with `aero-forge chat`.
- **Examples Gallery** - Try pre-built examples and build them with one command.
- **Multiple LLM Providers** - OpenAI, Gemini, OpenRouter, and DeepSeek are supported.
- **Cross-Compilation and WASM** - Build for other Rust target triples or `wasm32-unknown-unknown`.

## Installation

Requires Python 3.10+ and a working Rust toolchain with `cargo` and `rustup`.

```bash
pip install aero-forge
```

Or from source:

```bash
git clone https://github.com/sys1own/aero-forge.git
cd aero-forge
pip install -e ".[dev]"
```

## Quick Start

### 1. Generate from a prompt

```bash
export AERO_FORGE_LLM_PROVIDER=openrouter
export OPENROUTER_API_KEY="sk-or-v1-..."

aero-forge generate --prompt "Build a fast iterative Fibonacci function" --build
```

Aero-Forge will:

1. Send the prompt to the configured LLM.
2. Parse the generated Python code and tests.
3. Write `src/generated.py` and `tests/test_generated.py`.
4. Build a native Rust extension and run the tests.

Example output:

```
Generated: .../src/generated.py
Tests:     .../tests/test_generated.py
Blueprint: .../blueprint.aero
Build: 1/1 succeeded (.../dist)
```

### 2. Build an existing Python file

```bash
aero-forge fix src/my_function.py --function my_function
```

`fix` transpiles the function, compiles it, and runs any matching tests. If compilation fails and an LLM provider is configured, it attempts to heal the code.

```bash
aero-forge fix src/my_function.py --function my_function --json
```

With `--json`, `fix` prints a structured result with `status`, `rust_extensions`, `execution_time_ms`, and `error`.

### 3. Build from a blueprint

Create `blueprint.aero`:

```aero
project: my_project
functions:
  - file: src/math_ops.py
    compile_all: true
    tests:
      - tests/test_math_ops.py
output_dir: dist
llm:
  provider: openrouter
  model: openrouter/free
```

Then run:

```bash
aero-forge build
```

### 4. Build without the LLM

For existing, well-typed code you can skip LLM calls entirely:

```bash
aero-forge fix src/my_function.py --function my_function --no-llm
aero-forge build blueprint.aero --no-llm
```

### 5. Interactive Chat Mode

Start a conversational session. Aero-Forge remembers your previous prompts, generated code, and build results, so you can iterate naturally:

```bash
aero-forge chat
```

Example session:

```text
$ aero-forge chat
Aero-Forge chat is ready. What would you like to build?
> Build a fast Fibonacci function.
[Generating code from your prompt...]
[Compiling to Rust...]
[Running tests...]
[Build passed.]
Done! I generated `fibonacci`, compiled it to a Rust extension, and it passed all tests. The compiled library is in `dist/`.

> Make it faster.
[Alright, optimizing...]
Done! The optimized version is even faster. The build completed in 0.8s.

> Show me the code.
Here's the code:

def fibonacci(n: int) -> int:
    ...

> exit
Goodbye!
```

You can resume a previous session with `--session-id`:

```bash
aero-forge chat --session-id abc123
```

For machine integration, use `--json` to emit NDJSON events:

```bash
aero-forge chat --json
aero-forge generate --prompt "..." --build --json --stream
```

Useful chat phrases:

- `Build a <function>` - generate and compile code
- `Make it faster` / `Use less memory` - optimize the current code
- `Benchmark it` - build and time the project
- `Show me the code` - display the generated source
- `Explain the algorithm` - get a plain-English explanation
- `Explain` - explain the last build error
- `help` - list available commands

### 6. Post-Build Summaries

After every successful `aero-forge generate --build` or `aero-forge build`, Aero-Forge prints a short, friendly summary of what was built, whether tests passed, and where the compiled library is. In chat mode the summary is part of the assistant's reply.

## Commands Reference

| Command | Description |
|---------|-------------|
| `aero-forge fix <file> --function <name>` | Transpile and compile a single function. |
| `aero-forge build [blueprint]` | Build all functions in a blueprint. |
| `aero-forge build --project <dir>` | Build every public function in a project directory and bundle it as a zip. |
| `aero-forge build --upload <zip>` | Extract an uploaded project zip, build it, and produce a result zip. |
| `aero-forge build --output-zip <path>` | Path for the bundled output zip. |
| `aero-forge generate --prompt "..."` | Generate code from a natural language prompt. |
| `aero-forge generate --prompt "..." --project <dir>` | Generate a new function into an existing project and rebuild it. |
| `aero-forge chat` | Start an interactive chat session (`--session-id` to resume). |
| `aero-forge examples list` | List available example projects. |
| `aero-forge examples run <name>` | Build an example. |
| `aero-forge examples create <name> --prompt "..."` | Create a new example from a prompt. |
| `aero-forge init <project>` | Create a new project skeleton with a blueprint. |

## Advanced Generation Flags

| Flag | Description |
|------|-------------|
| `--algorithm-library` | Pick an algorithm from the built-in library and adapt it. |
| `--selected-algorithm <name>` | Force a specific library algorithm. |
| `--variants N` | Generate and benchmark N implementations, then select the best. |
| `--explain` | Request and display an explanation of the algorithm choice. |
| `--discover` | Allow the LLM to design a new algorithm when no library entry matches. |
| `--review` | Run an LLM self-review step before compilation. |
| `--optimize` | Run an iterative LLM optimization loop. |
| `--prompt-template <name>` | Choose one of `v1_minimal`, `v2_structured`, `v3_algorithm`, `v4_performance`, `v5_balanced` (default), `v6_creative`, `v7_conservative`, `v8_iterative`, `v9_transpiler_friendly`, `v10_correctness_focused`. |
| `--build` | Run `aero-forge build` immediately after generation. |
| `--json` | Output the final result as structured JSON for frontend integration. |
| `--stream` | Emit NDJSON progress events during generation/build. |

## Blueprint Reference

### Top-Level Fields

| Field | Type | Description |
|-------|------|-------------|
| `project` | string | Project name. |
| `functions` | list | Function specifications (see below). |
| `prompt` | string | Natural language prompt to generate code. |
| `constraints` | string or object | Constraints passed to the LLM. |
| `output_dir` | path | Output directory (default `dist`). |
| `llm` | object | LLM provider and model configuration. |
| `compiler_flags` | list | Global Rust compiler flags. |

### Function Specification

| Field | Type | Description |
|-------|------|-------------|
| `file` | path | Source Python file (required). |
| `name` | string | Function name, or `*` / `compile_all: true` for all public functions. |
| `compile_all` | boolean | Compile every public function in the file. |
| `tests` | list | Test files to run after compilation. |
| `output_name` | string | Custom output module name. |
| `compiler_flags` | list | Per-function Rust flags. |

### Example Blueprint

```aero
project: my_optimized_project

functions:
  - file: src/math_ops.py
    compile_all: true
    tests:
      - tests/test_math_ops.py
  - file: src/heavy.py
    name: simulation_step
    compiler_flags:
      - "-C target-cpu=native"

compiler_flags:
  - "-C opt-level=3"

output_dir: ./dist

llm:
  provider: openrouter
  model: openrouter/free
```

## LLM Configuration

### Supported Providers

| Provider | Environment Variable | Default Model |
|----------|---------------------|---------------|
| OpenAI | `OPENAI_API_KEY` | `gpt-4` |
| OpenRouter | `OPENROUTER_API_KEY` | `openrouter/free` |
| DeepSeek | `DEEPSEEK_API_KEY` | `deepseek-chat` |
| Gemini | `GEMINI_API_KEY` | `gemini-2.0-flash` |
| Router only | none | `none` |

`AERO_FORGE_API_KEY` works as a generic fallback for any provider.

### Configuration Precedence

1. CLI flags
2. Environment variables
3. `llm` block in `blueprint.aero`
4. Built-in defaults

Example environment setup:

```bash
export AERO_FORGE_LLM_PROVIDER=openrouter
export AERO_FORGE_MODEL=openrouter/free
export OPENROUTER_API_KEY="sk-or-v1-..."
```

## Algorithm Library

Aero-Forge ships with reference Python implementations in `aero_forge/algorithms/`:

- Sorting: `quicksort`, `mergesort`, `timsort`, `insertion_sort`, `selection_sort`, `heap_sort`
- Matrix: `matrix_multiply`, `naive_multiply`, `blocked_multiply`, `strassen`
- FFT: `cooley_tukey`
- Searching: `binary_search`
- Math: `fibonacci`, `gcd`, `is_prime`

Each file includes a `METADATA` dict describing complexity, use cases, and constraints. Use `--algorithm-library` to let the LLM select and adapt the right one.

## Prompt Templates

Nine templates are included for different generation styles:

| Template | Description |
|----------|-------------|
| `v1_minimal` | Minimal instruction, maximum creativity. |
| `v2_structured` | Structured output with constraints. |
| `v3_algorithm` | Algorithm-focused. |
| `v4_performance` | Performance-focused (SIMD, caching, parallelism). |
| `v5_balanced` | Balanced algorithm/performance (default). |
| `v6_creative` | Encourages novel algorithms. |
| `v7_conservative` | Uses only well-known algorithms. |
| `v8_iterative` | Includes feedback from previous runs. |
| `v9_transpiler_friendly` | Explicitly forbids edge-case constructs for maximum first-pass success. |

Use `--prompt-template v5_balanced` to select one. `v5_balanced` is the default and was the most reliable in the prompt-engineering campaign.

## Advanced LLM Intelligence (D-series)

These flags turn Aero-Forge into a senior-engineer-style assistant:

- `--algorithm-library` selects a reference implementation from the built-in `aero_forge/algorithms/` library and asks the LLM to adapt it.
- `--selected-algorithm <name>` forces a specific library entry.
- `--variants N` generates N implementations, compiles each, and selects the fastest variant that passes all tests using a Pareto frontier over accuracy and build time.
- `--explain` requests an `## Explanation` section covering algorithm choice, complexity, and tradeoffs.
- `--discover` lets the LLM design a new algorithm when the library has no match.
- `--review` runs a second LLM pass that checks the generated code for correctness, performance, security, and style.

## Performance

Aero-Forge targets 10-100x speedups for hot numerical loops. Actual speedup depends on the function and the quality of the generated Rust. The benchmark loop in `aero-forge generate --optimize` compares the native extension against the original Python and reports the relative improvement.

## Supported Python Constructs

The transpiler handles common numerical and algorithmic Python patterns:

- Primitive numeric types (`int`, `float`, `bool`) and `list`/`List[T]` annotations, plus `numpy.ndarray` which maps to `Vec<f64>`.
- Nested `for`/`while` loops, `if`/`elif`/`else`, `break`, `continue`, and early `return`.
- `range(...)` loops with one, two, or three arguments (step is supported).
- List comprehensions (e.g., `[x * x for x in range(10)]` or `[x * 2 for x in arr]`) and nested list comprehensions.
- Tuple unpacking assignments (`a, b = b, a + b`) and chain assignments (`i = j = 0`).
- `enumerate()` and `zip()` in `for` loop iteration.
- List slicing for reads (`a[:]`, `a[1:3]`) and slice assignment (`a[1:3] = b`).
- `len()` on lists and nested list rows.
- `append()`, `extend()`, and `pop()` on lists.
- `not list` emptiness tests (e.g. `if not a: return []`).
- Negative literal subscripts (`arr[-1]`).
- Generic `list`/`List[T]` annotations where the element type is inferred from usage.
- Basic `list[list[T]]` matrices and indexing (`m[i][j]`), including row caching (`row = m[i]`) and direct nested subscript assignment (`m[i][j] = value`).
- Tuple unpacking on name and subscript targets (`a, b = b, a` and `a[i], a[j] = a[j], a[i]`).
- `min()` and `max()` on two scalar values.
- `sorted(values)` with no key.
- `int()` and `float()` casts.
- Mixed `int`/`float` arithmetic and `math` functions (`math.cos`, `math.sin`, `math.sqrt`, etc.), including bare math names and constants when `import math` is used.
- Bitwise operators (`&`, `|`, `^`, `<<`, `>>`) on integer-typed values.
- Automatic empty-list guard for scalar-returning functions that index into a list.
- List replication (`[0] * n`) with safe ordering relative to input guards.

## Known Limitations

The transpiler is intentionally narrow. It works well for numerical/algorithmic code and produces clear errors for unsupported constructs.

Currently not supported:

- `insert`, `remove`, and most other list methods (only `append`, `extend`, `pop`, and indexing/slicing are supported).
- Nested function, class, or method definitions (refactor to top-level functions).
- Dictionaries and sets.
- Complex class inheritance, properties, and dataclasses.
- `try`/`except`, `with`, `yield`, `async`/`await`.
- `eval`/`exec` and dynamic imports.
- `random`, `datetime`, `re`, `json`, and other non-math stdlib modules.
- I/O, networking, and `os`/`subprocess`.
- Full `ndarray` broadcasting and n-dimensional operations.

See `BLUEPRINT.md` and `stress_tests/README.md` for the full supported-construct list and the stress-test campaign results.

## How It Works

1. **Parse** - The Python source is parsed and, for prompt-driven builds, generated by the LLM.
2. **Transpile** - A Python-to-Rust transpiler emits PyO3 `#[pyfunction]`/`#[pyclass]` code.
3. **Scaffold** - A temporary Cargo crate is generated automatically.
4. **Compile** - `cargo build --release` produces a shared library.
5. **Test** - `pytest` runs against the compiled extension in an isolated sandbox.
6. **Heal** - On failure, the orchestrator classifies the error, tries router-only fixes, then falls back to the configured LLM with retry and exponential backoff.
7. **Cache** - Compilation and fix results are cached so unchanged files rebuild instantly.

## Running Tests

```bash
python -m pytest
```

The repository includes unit tests plus a `stress_tests/` campaign that exercises supported constructs end-to-end.

## Building from Source

```bash
git clone https://github.com/sys1own/aero-forge.git
cd aero-forge
pip install -e ".[dev]"
rustup target add wasm32-unknown-unknown  # optional, for WASM builds
```

The first build may take a few minutes while PyO3 compiles.

## Further Reading

- [`BLUEPRINT.md`](BLUEPRINT.md) - Complete blueprint reference and commented examples.
- [`PROMPT_ENGINEERING.md`](PROMPT_ENGINEERING.md) - Prompt template guide and campaign results.
- [`stress_tests/README.md`](stress_tests/README.md) - Stress-test campaign report.

## License

MIT
