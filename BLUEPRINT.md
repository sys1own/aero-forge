# Aero-Forge Blueprint Reference

Aero-Forge builds are driven by a declarative **blueprint** file.
Blueprints can be written as `.aero` (YAML-compatible) or `.yaml` files.

## File locations

- Default file for `aero-forge build`: `blueprint.aero`
- You can pass any file explicitly: `aero-forge build my_blueprint.yaml`

## Top-level fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `project` | string | No | Human-readable project name. Default: `aero_forge_project`. |
| `functions` | list | Yes | List of `FunctionSpec` entries to compile. See below. |
| `compiler_flags` | list of strings | No | Extra `RUSTFLAGS` applied to every function. Default: `[]`. |
| `output_dir` | string/path | No | Directory for compiled `.so` and loader `.py` files. Default: `./dist`. |
| `llm` | object | No | LLM provider/model configuration. See below. |
| `prompt` | string | No | Natural language prompt used by `aero-forge generate` to produce `src/generated.py` and `tests/test_generated.py`. |
| `constraints` | string | No | Optional constraints for generated code (e.g. "iterative only", "O(n) time"). |

## `functions` entries

Each entry describes one or more Python functions or classes to compile.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | string/path | Yes | Python source file containing the function(s). Relative to the blueprint file. |
| `name` | string | Yes* | Name of the function or class to compile. Use `"*"` or set `compile_all: true` to compile every public top-level function in `file` (classes must be listed explicitly). |
| `compile_all` | boolean | No | If `true`, compile every public function in `file` (those not starting with `_`). Default: `false`. |
| `tests` | list of strings | No | Test file(s) to run against the compiled module. Relative to the blueprint file. |
| `output_name` | string | No | Reserved for future per-function output naming. |
| `compiler_flags` | list of strings | No | Extra `RUSTFLAGS` for this entry only. |

`*` `name` is required unless `compile_all: true`.

## `llm` configuration

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `provider` | string | No | One of `openai`, `openrouter`, `gemini`, `none`. Default: `none`. |
| `model` | string | No | Model name. Defaults are provider-specific (`gpt-4`, `openrouter/free`, `gemini-2.0-flash`). |

## Special values

- `name: "*"` and `compile_all: true` are equivalent.
- If both are present, `compile_all` wins.

## Fully commented example

```aero
# Project name used in logs and output metadata.
project: "my_physics_pipeline"

# Functions to compile. Each entry becomes one compiled native module.
functions:
    # Compile a single named function from a source file.
  - file: "src/compute.py"
    name: "mandelbrot"
    # Test files are copied into the sandbox and run with pytest.
    tests: ["tests/test_compute.py"]

    # Compile every public top-level function in utils.py.
    # The *wildcard* form is useful when a module contains many helpers.
  - file: "src/utils.py"
    name: "*"
    tests: ["tests/test_utils.py"]

    # Same as above, using the explicit compile_all flag.
  - file: "src/transform.py"
    compile_all: true
    tests:
      - "tests/test_transform.py"

    # You can also add per-function compiler flags.
  - file: "src/heavy.py"
    name: "simulation_step"
    compiler_flags:
      - "-C target-cpu=native"

# Global compiler flags applied to every function.
compiler_flags:
  - "-C opt-level=3"

# Where compiled libraries and loaders are written.
output_dir: "./dist"

# Optional LLM healing configuration.
llm:
  provider: "gemini"
  model: "gemini-2.0-flash"
```

## CLI overrides

Any field can be overridden from the command line:

```bash
# Override output directory
aero-forge build -o ./build_output

# Disable LLM and preview what would be built
aero-forge build --no-llm --dry-run

# Use auto-discovery instead of a blueprint
aero-forge build --auto src/my_module.py --no-llm

# Auto-detect a standard project structure (src/ + tests/)
aero-forge build --auto-detect

# Show build progress with a progress bar
aero-forge build blueprint.aero --progress

# Explain a compilation error
aero-forge explain src/broken.py --error-file error.log

# List and run curated examples
aero-forge examples list
aero-forge examples run fibonacci

# Create a new example from a prompt
aero-forge examples create primes --prompt "fast prime sieve"

# Generate from the algorithm library
aero-forge generate --algorithm-library --prompt "sort a list quickly"

# Generate and benchmark multiple variants
aero-forge generate --variants 3 --prompt "fast prime check" --build

# Generate with an explanation and self-review
aero-forge generate --explain --review --prompt "fast gcd"
```

## Precedence

Settings are merged from lowest to highest priority:

1. Blueprint file values
2. Environment variables (`AERO_FORGE_LLM_PROVIDER`, `AERO_FORGE_MODEL`, etc.)
3. CLI flags (`--llm-provider`, `--model`, `--output-dir`, `--no-llm`, etc.)

## Supported Python Constructs

Aero-Forge's transpiler targets numerical and algorithmic Python. The following patterns are supported:

- Scalar numeric types (`int`, `float`, `bool`) and `list`/`List[T]` type annotations, plus `numpy.ndarray` which maps to `Vec<f64>`.
- `for` and `while` loops, `if`/`elif`/`else`, `break`, `continue`, and early `return`.
- `range(...)` with one, two, or three arguments (step is supported).
- List comprehensions, including nested comprehensions like `[[0 for _ in cols] for _ in rows]`.
- Tuple unpacking assignments (`a, b = b, a + b`) inside loops and chain assignments (`i = j = 0`).
- `enumerate()` and `zip()` in `for` loop iteration.
- List slicing for reads (`a[:]`, `a[1:3]`, `a[2:]`) and assignment (`a[1:3] = b`).
- `len()` on lists and nested list rows (`len(a)`, `len(a[0])`).
- `append()` and `extend()` on list variables.
- Nested `list[list[T]]` matrices with row/column indexing (`m[i][j]`), including caching a row (`row = m[i]`) via `.clone()`.
- `min()` and `max()` on two scalar values.

The following are intentionally not supported and produce clear errors:

- Nested function, class, or method definitions (refactor to top-level functions).
- `try`/`except`, `with`, generators, `eval`/`exec`, dynamic imports, and complex class inheritance.

## Interactive Chat

`aero-forge chat` starts a stateful, conversational session:

- Chat history, the last prompt, generated source, and last build result are kept in memory.
- The session can be persisted and resumed with `--session-id`. Session files are stored in `~/.cache/aero-forge/sessions/<id>.json`.
- Natural commands are recognized: `Build a ...`, `Make it faster`, `Use less memory`, `Benchmark it`, `Show me the code`, `Explain the algorithm`, `Explain`, `help`, `exit`/`quit`.
- Unrecognized input falls back to the LLM for a conversational reply. Typos are matched against the command list and a suggestion is offered.
- Progress messages are printed during generation, compilation, testing, and optimization.

## Post-Build Summaries

After a successful `aero-forge generate --build` or `aero-forge build`, Aero-Forge prints a short, friendly summary (2-4 sentences) of what was built, whether tests passed, build timing when available, and where the output is. The summary is generated by the LLM when a provider is configured; otherwise a deterministic fallback is used.

## Notes

- Functions from the same source file are compiled into a single shared library.
- Functions from different source files are built in parallel (controlled by `--jobs`).
- Unchanged source files are skipped when a cached artifact exists in `~/.cache/aero-forge/build_cache/`.
- See `stress_tests/` for a broad set of real-world Python patterns and the
  current limits of what Aero-Forge can compile.
