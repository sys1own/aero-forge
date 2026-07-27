# Aero-Forge: Natively Accelerated Polyglot Build Engine & Web Workspace

Aero-Forge turns a plain-English prompt or an existing source file into a complete, tested, **natively accelerated** software project. It is a universal build orchestrator for **natively accelerated Python**, **pure Rust**, **native C++**, **Rust/C++ systems**, and **Python/Rust/C++ tri-polyglot** applications, with automatic PyO3/Maturin extension generation, `extern "C"` C-ABI dynamic libraries, in-memory HIN JIT compilation, a zero-copy native bridge, and an embedded **web-first workspace**.

> **Web-first by design:** The fastest way to use Aero-Forge is the embedded web dashboard (`aero-forge web` or `python3 -m aero_forge.server`). It provides a full workspace environment — interactive Co-Pilot chat with Action Cards, a multi-tab file explorer & editor, real-time build/accelerator log streaming, drag-and-drop `blueprint.aero` importing, and one-click workspace regeneration. The CLI remains fully functional for scripting and automation.

## What is Aero-Forge?

Aero-Forge is a prompt-driven build system for high-performance software. You describe what you want, point it at a `.py` file, or upload a ZIP, and it produces working source, native extensions, packaging manifests, tests, and a downloadable project bundle.

The engine is built around a declarative contract called `blueprint.aero`. For every request, Aero-Forge first classifies the prompt to infer the target architecture (`pure_python`, `pure_rust`, `hybrid_rust_python`, `hybrid_cpp_python`, `hybrid_cpp_rust`, `multi_crate_rust`, `tri_polyglot_rust_cpp_python`, `wasm`), the required toolchains (`python`, `cargo`, `maturin`, `cmake`, `cpp`, `gcc`, `clang`), the file manifest, and the exported contracts. It then materializes every declared file and invokes the appropriate native toolchains. When compilation or tests fail, it applies deterministic AST and pattern-based repairs first, escalates to full-workspace LLM healing when needed, and surfaces precise diagnostics. LLM calls are strictly confined to intent interpretation, high-level strategy selection, and human-facing summaries.

Core value propositions:

- **Zero-boilerplate native acceleration** - No `Cargo.toml`, `#[pyfunction]`, `build.rs`, or linker flags required.
- **C-ABI Zero-Copy Dynamic Bridge** - Accelerated numerical functions compile to `.so`/`.dylib`/`.dll` via `clang++`/`g++` with native FFI bindings emitted by `cpp_emitter.py`.
- **Multi-Language Build Matrix** - Native support for six build targets: `pure_python`, `pure_rust`, `hybrid_rust_python`, `hybrid_cpp_python`, `hybrid_cpp_rust`, and `tri_polyglot_rust_cpp_python`.
- **Selective Acceleration Heuristics** - AST node evaluation routes heavy vector loops to C++ `extern "C"` shared libraries, concurrent/memory-safe work to Rust PyO3, and light or incompatible workloads back to CPython.
- **Sub-millisecond execution pathways** - Numeric Python functions compile to native code and are cached at the AST node level for instant re-execution.
- **Wavefront Parallel Acceleration Engine** - Dependency-graph wavefront analysis batches independent functions and UAST nodes across multi-crate and polyglot targets for parallel compilation and execution, cutting build matrix times and hot-loop overhead.
- **Drop-In Blueprint Portability** - `blueprint.aero` is a self-contained project contract. Drag or copy it into any Aero-Forge workspace to scaffold and compile the complete project.
- **Self-Healing Workspace Regeneration** - The "Regenerate Workspace from Blueprint" action cleanly purges a broken `src/` tree and re-scaffolds the full polyglot codebase from `blueprint.aero`.
- **Interactive Co-Pilot & Action Cards** - The web Co-Pilot chat is workspace-aware (via `bundle_repo.py`), proposes optimized target-aware build prompts, and renders `PROPOSE_BUILD` Action Cards with one-click `[ 🚀 Send to Builder & Run ]` triggers.
- **Fall-forward safety** - Unsupported Python constructs gracefully fall back to CPython without panics.

## Core Supported Build Targets

Aero-Forge natively supports eight primary build targets, each with deterministic materialization and native toolchain invocation:

### 1. Natively Accelerated Python

Python functions are transpiled and compiled into either a PyO3 extension or a C-ABI shared library, then exposed through a Pythonic wrapper. The `@accelerate` decorator lets you mark any numeric function for native compilation:

```python
from aero_forge import accelerate

@accelerate(target="rust_hin")
def weighted_sum(scores: list[float], weights: list[float]) -> float:
    total = 0.0
    for s, w in zip(scores, weights):
        total += s * w
    return total
```

The first call compiles; subsequent calls reuse the cached UAST node hash and execute in the native HIN VM or the compiled `.so` with sub-millisecond latency.

### 2. Pure Rust Crates & Cargo Workspaces

Aero-Forge generates standalone Rust binaries, library crates, and multi-crate Cargo workspaces from prompts or blueprints:

```bash
aero-forge generate --prompt "Build a pure Rust command-line prime sieve" --build
```

### 3. Python / Rust Polyglot Extensions (`hybrid_rust_python`)

High-performance PyO3/Maturin hybrid extension modules with zero-copy buffer handoffs for numeric matrices and arrays:

```bash
aero-forge generate --prompt "Scaffold a PyO3 extension for batch matrix multiplication with Python bindings" --build
```

### 4. Python / C++ Dynamic Extensions (`hybrid_cpp_python`)

C++ numerical kernels compiled to `extern "C"` dynamic shared libraries (`-fPIC -shared`) and loaded by Python through `native_bridge.py` using `ctypes`:

```bash
aero-forge generate --prompt "Build Python calling a C++ vector transform via ctypes" --build
```

### 5. Native C++ / Rust Systems (`hybrid_cpp_rust`)

Pure native binaries where Rust handles concurrency, CLI, and memory safety while C++ provides accelerated numerical kernels. The C++ source under `src/cpp_core/` is compiled into a static archive by a generated `build.rs` and linked directly into the Rust binary — no Python runtime:

```bash
aero-forge generate --prompt "Build a native Rust CLI that links an extern C C++ math module via build.rs" --build
```

### 6. Tri-Polyglot Orchestration (`tri_polyglot_rust_cpp_python`)

A Python CLI/REPL orchestrates input, a Rust PyO3 crate manages concurrent state, and a C++ `extern "C"` shared library executes heavy numeric array transformations. All three languages are materialized and compiled in a single multi-stage pipeline:

```bash
aero-forge generate --prompt "Build a Python CLI that uses Rust PyO3 for token validation and C++ for matrix transforms" --build
```

### 7. Multi-Crate Rust Workspaces (`multi_crate_rust`)

Generate Cargo workspaces with multiple interdependent crates, shared library members, and a top-level binary or workspace root:

```bash
aero-forge generate --prompt "Build a multi-crate Rust workspace with a shared math crate and a CLI crate" --build
```

### 8. WebAssembly (`wasm`)

Compile Rust targets to `wasm32-unknown-unknown` for browser or Wasmtime deployment:

```bash
aero-forge generate --prompt "Compile a Rust numeric kernel to wasm32" --target wasm --build
```

Supported outputs also include:

- Pure Python packages and libraries.
- Pure Rust crates and Cargo workspaces.
- Hybrid Rust/Python extensions via PyO3/Maturin.
- Hybrid C++/Python extensions via `extern "C"` shared libraries and `ctypes`.
- Native C++/Rust binaries linked via `build.rs`.
- Tri-polyglot Python/Rust/C++ orchestration projects.
- Multi-crate Rust monorepos.
- Cross-compiled Rust artifacts and `wasm32-unknown-unknown` binaries.

## Key Features

- **Native HIN Acceleration Core** - A Rust-based execution backend (`hin_vm`) lowers numeric Python into a High-level Interaction Net (HIN) for fast, safe reduction.
- **Fine-Grained AST Node Hash Caching** - Compilation artifacts are keyed by the hash of individual UAST nodes. Identical functions bypass `cargo build` entirely and load the cached native snippet instantly.
- **Fall-Forward Precision Shield** - If the transpiler encounters an unsupported Python construct, the function is transparently routed back to native CPython execution. No panics, no crashes.
- **Universal Intent Detection** - Parses any high-level prompt to infer languages, build tools, module boundaries, and concurrency patterns. Hybrid stacks are never silently downgraded to a fallback.
- **Declarative Blueprinting (`blueprint.aero`)** - Every build starts with a generated contract that declares `architecture`, `toolchains`, `manifest`, `contracts`, `functions`, and verification steps.
- **Strict Blueprint Materialization** - Every manifest file, source module, native backend library, compiler config, and target binding declared in `blueprint.aero` is physically emitted and built.
- **Natural Language Prompts** - Describe what you want and Aero-Forge generates Python code, tests, and a build blueprint.
- **Zero Manual Rust Boilerplate** - No `Cargo.toml`, `#[pyfunction]` annotations, or linker flags are required from the user for standard Python/Rust hybrid builds.
- **Interactive Web Dashboard & Terminal REPL** - Start a local web server (`aero-forge web`) to prompt, build, test, monitor real-time build logs, browse generated files in a multi-tab editor, and download compiled ZIP artifacts. An embedded xterm.js terminal supports copy/paste and live command execution.
- **Workspace Co-Pilot & Action Cards** - The Co-Pilot chat is workspace-aware via `bundle_repo.py`, understands all build targets, proposes optimized build prompts, and renders `PROPOSE_BUILD` Action Cards with `[ 🚀 Send to Builder & Run ]` and `[ 📝 Edit in Build Tab ]` buttons.
- **Wavefront Parallel Acceleration Engine** - Dependency-graph wavefront analysis schedules independent functions and UAST nodes for parallel compilation and execution, reducing build matrix times for multi-crate and polyglot targets.
- **Drop-In Blueprint Import & Workspace Regeneration** - Drag `blueprint.aero` into the web explorer to scaffold a full project, or click "Regenerate Workspace from Blueprint" to purge a broken workspace and rebuild strictly from the blueprint contract.
- **Symbolic & AST Static Healing Core** - If `cargo build` or tests fail, Aero-Forge applies deterministic AST/pattern-based repairs first and escalates to full-workspace LLM healing when a static patch is insufficient. Failures surface precise exception type, file, and line diagnostics.
- **Algorithm Library** - Pick from a curated library of reference implementations (sorting, matrix, FFT, math) or let the LLM select one automatically.
- **Multi-Variant Testing** - Generate several implementations, compile them in parallel, benchmark each, and select the fastest variant that passes.
- **Explainable Builds** - Add `--explain` to get the LLM to describe the algorithm choice, complexity, and tradeoffs.
- **Auto-Discovery** - `aero-forge build --auto-detect` discovers `src/` and `tests/` and compiles everything it understands.
- **Project Builds & Zip Bundles** - `aero-forge build --project <dir>` compiles every public function in a project directory and produces a downloadable zip with source, compiled libraries, a Python package, and a build manifest.
- **Zip Uploads** - `aero-forge build --upload project.zip` extracts, builds, and re-bundles an uploaded project.
- **Project-Aware Generation** - `aero-forge generate --prompt "..." --project <dir>` adds a new function to an existing project and rebuilds the bundle.
- **Interactive Chat** - Refine prompts conversationally with `aero-forge chat`.
- **Examples Gallery** - Try pre-built examples and build them with one command.
- **Multiple LLM Providers** - OpenAI, Gemini, OpenRouter, and DeepSeek are supported.
- **Cross-Compilation and WASM** - Build for other Rust target triples or `wasm32-unknown-unknown`.

## Installation

Requires Python 3.10+ and a working Rust toolchain with `cargo` and `rustup`. Aero-Forge will auto-bootstrap a minimal Rust toolchain if one is missing.

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

### 1. Launch the Web Dashboard

The web dashboard is the fastest way to build, iterate, and inspect projects:

```bash
export AERO_FORGE_LLM_PROVIDER=openrouter
export OPENROUTER_API_KEY="sk-or-v1-..."

aero-forge web
# or
python3 -m aero_forge.server --port 8080
```

Open `http://localhost:8080` in your browser.

The dashboard gives you:

- **Co-Pilot Chat** — ask questions, get build suggestions, and trigger `PROPOSE_BUILD` Action Cards.
- **Build Tab** — generate and compile polyglot projects with target and acceleration selectors.
- **Multi-Tab File Explorer & Editor** — browse, edit, and download generated files.
- **Real-Time BUILD LOG / ACCELERATOR LOG** — streaming `cargo`, `g++`, `maturin`, and heuristic telemetry.
- **Drag-and-Drop Blueprint Import** — drop a `blueprint.aero` file into the explorer to scaffold the entire workspace.

### 2. Generate from a prompt

If you prefer the CLI, generate and build in one command:

```bash
export AERO_FORGE_LLM_PROVIDER=openrouter
export OPENROUTER_API_KEY="sk-or-v1-..."

aero-forge generate --prompt "Build a fast iterative Fibonacci function" --build
```

Aero-Forge will:

1. Parse the prompt and classify the required architecture and toolchains.
2. Generate `blueprint.aero` describing the workspace.
3. Generate Python source and tests.
4. Materialize any declared native crate, build the Rust extension, and run the tests.

Example output:

```
Generated: .../src/generated.py
Tests:     .../tests/test_generated.py
Blueprint: .../blueprint.aero
Build: 1/1 succeeded (.../dist)
```

### 4. Accelerate a Python function with `@accelerate`

```python
from aero_forge import accelerate

@accelerate(target="rust_hin")
def dot_product(a: list[float], b: list[float]) -> float:
    total = 0.0
    for x, y in zip(a, b):
        total += x * y
    return total

print(dot_product([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]))
```

The first call compiles the function to a native Rust `.so`. The second call reuses the cached node key and executes in under a millisecond.

### 5. Build an existing Python file

```bash
aero-forge fix src/my_function.py --function my_function
```

`fix` transpiles the function, compiles it, and runs any matching tests. If compilation fails, it attempts deterministic AST/pattern-based repairs and returns a clear diagnostic if the error cannot be fixed.

```bash
aero-forge fix src/my_function.py --function my_function --json
```

With `--json`, `fix` prints a structured result with `status`, `rust_extensions`, `execution_time_ms`, and `error`.

### 6. Build from a blueprint

Create `blueprint.aero`:

```aero
project: my_project
architecture: hybrid_rust_python
toolchains:
  - python
  - cargo
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

### 7. Build Deterministically

The transpile/build/test path is always deterministic. For existing, well-typed code you can skip LLM-driven code generation entirely:

```bash
aero-forge fix src/my_function.py --function my_function --no-llm
aero-forge build blueprint.aero --no-llm
```

### 8. Interactive Chat Mode

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

### 7. Post-Build Summaries

After every successful `aero-forge generate --build` or `aero-forge build`, Aero-Forge prints a short, friendly summary of what was built, whether tests passed, and where the compiled library is. In chat mode the summary is part of the assistant's reply.

## Web Interface Setup

Aero-Forge is designed to be operated primarily through its embedded web dashboard. The dashboard runs a single HTTP + WebSocket server (`aero-forge web` or `python3 -m aero_forge.server`) and provides a full workspace environment in any modern browser.

**Tagline:** `Universal AST & Polyglot Accelerator (Python · Rust · C++)`

Start the web server on the default port:

```bash
python3 -m aero_forge.server
```

The dashboard is then available at `http://localhost:8080`.

To bind a custom host or port:

```bash
python3 -m aero_forge.server --host 0.0.0.0 --port 8889
```

Add `--no-browser` to prevent a browser tab from opening automatically:

```bash
python3 -m aero_forge.server --port 8889 --no-browser
```

Alternatively, use the CLI wrapper:

```bash
aero-forge web --port 8080
```

The web interface supports the same providers and environment variables as the CLI. Set `AERO_FORGE_LLM_PROVIDER` and the appropriate API key before starting the server.

### Dashboard Layout

The workspace is organized around two primary tabs plus a shared file explorer:

- **Build Tab** — One-shot polyglot generation. Enter a prompt, choose a target language and acceleration policy, and click **Build**. The engine generates `blueprint.aero`, materializes source/manifests, compiles native artifacts, and streams `BUILD LOG` / `ACCELERATOR LOG` output in real time.
- **Co-Pilot Chat Tab** — A workspace-aware assistant powered by `bundle_repo.py`. Ask questions about the current project, request optimizations, or debug build failures. When the Co-Pilot proposes a build, it returns a structured `PROPOSE_BUILD` Action Card showing the optimized prompt, selected `target`, and `acceleration` mode. Click `[ 🚀 Send to Builder & Run ]` to populate the Build tab and start compilation immediately, or `[ 📝 Edit in Build Tab ]` to review the prompt first.
- **File Explorer (left sidebar)** — Browse, open, and edit generated files. Drag-and-drop any `blueprint.aero` file into the explorer to scaffold the complete workspace. When a `blueprint.aero` is detected, an **"Initialize Project from Blueprint"** banner appears.
- **Embedded Terminal (bottom panel)** — An `xterm.js` terminal with copy/paste and live command execution.
- **Real-Time Logs**:
  - **BUILD LOG:** Output from `cargo`, `g++`, `clang++`, `maturin`, and test runners.
  - **ACCELERATOR LOG:** Heuristic routing telemetry, AST hash evaluation, and bridge binding verdicts (e.g., `ACCELERATED: C++ selected for extern "C" dynamic shared library`).

### Dashboard Controls

- **Target Language selector:** `Auto-Detect / Polyglot`, `Python`, `Rust`, `C++`, `Hybrid C++ / Rust`, `Multi-Crate Rust`, `Tri-Polyglot (Python + Rust + C++)`, `WebAssembly (wasm32)`.
- **Acceleration Policy selector:** `Selective Acceleration (Auto-Detect Heavy Compute)`, `Force Native Bridge`, `Standard Runtime (Bypass Bridge)`.
- **Blueprint actions**:
  - **Explorer header** — click the regenerate icon to open the *Regenerate Workspace from Blueprint* confirmation modal.
  - **Right-click `blueprint.aero`** in the explorer — choose *Rebuild Workspace from Blueprint*.
  - **Error Recovery Panel** — when a build fails, the terminal banner offers `[ 🔄 Hard Reset & Rebuild Workspace from Blueprint ]` alongside AST and LLM healing options.

### Workspace Regeneration

If source files become corrupted or you want a clean slate, use **Regenerate Workspace from Blueprint** (`POST /api/workspace/regenerate_blueprint`). The engine:

1. Backs up the current workspace to `.aero_backup/`.
2. Purges generated directories (`src/`, `tests/`, `rust_core/`, `cpp_core/`, `target/`, `dist/`, etc.) and native manifests (`Cargo.toml`, `pyproject.toml`, `build.rs`).
3. Re-scaffolds directories and writes syntactically valid stubs from the `blueprint.aero` manifest and contracts.
4. Optionally runs a full build and removes the backup on success.

## Commands Reference

| Command | Description |
|---------|-------------|
| `aero-forge web` | Start the embedded web dashboard. |
| `python3 -m aero_forge.server` | Start the web server directly (`--host`, `--port`, `--no-browser`). |
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

`blueprint.aero` is the authoritative declarative contract for every build. The engine writes it first, validates it against the detected intent, and then materializes every declared file before invoking toolchains.

### Top-Level Fields

| Field | Type | Description |
|-------|------|-------------|
| `project` | string | Project name. |
| `architecture` | string | Build strategy: `pure_python`, `pure_rust`, `hybrid_rust_python`, `hybrid_cpp_python`, `hybrid_cpp_rust`, `multi_crate_rust`, `tri_polyglot_rust_cpp_python`, `wasm`. |
| `toolchains` | list | Required tools: `python`, `pip`, `cargo`, `maturin`, `cmake`, `cpp`, `gcc`, `clang`, `npm`, `go`, etc. |
| `manifest` | list | Files to create (see Manifest Entry). |
| `contracts` | list | Exported symbols, FFI bindings, or shared data structures (see Contract Entry). |
| `functions` | list | Function specifications (see Function Specification). |
| `prompt` | string | Natural language prompt used to generate the blueprint. |
| `constraints` | string or object | Constraints passed to the LLM or materializer. |
| `output_dir` | path | Output directory (default `dist`). |
| `llm` | object | LLM provider and model configuration. |
| `compiler_flags` | list | Global Rust compiler flags. |
| `languages` | list | Detected languages (e.g. `python`, `rust`, `cpp`). |
| `features` | list | Detected features (e.g. `web`, `async`, `gpu`, `wasm`). |

### Function Specification

| Field | Type | Description |
|-------|------|-------------|
| `file` | path | Source Python file (required). |
| `name` | string | Function name, or `*` / `compile_all: true` for all public functions. |
| `compile_all` | boolean | Compile every public function in the file. |
| `tests` | list | Test files to run after compilation. |
| `output_name` | string | Custom output module name. |
| `compiler_flags` | list | Per-function Rust flags. |

### Manifest Entry

| Field | Type | Description |
|-------|------|-------------|
| `path` | path | Relative path of the file to materialize. |
| `lang` | string | Language or format: `python`, `rust`, `cpp`, `toml`, `markdown`, `cmake`. |
| `purpose` | string | Human-readable description of the file's role. |

### Contract Entry

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Exported symbol name. |
| `signature` | string | Function signature, quoted to avoid YAML parsing issues. |
| `language` | string | Language boundary: `python`, `rust`, `cpp`, `python/rust`, `python/cpp`, `rust/cpp`, `python/rust/cpp`, etc. |
| `python_name` | string | Python import path for the binding. |
| `purpose` | string | Description of the contract. |

### Example Blueprint

```aero
project: my_optimized_project
architecture: hybrid_rust_python
toolchains:
  - python
  - cargo
  - maturin

manifest:
  - path: Cargo.toml
    lang: toml
    purpose: Workspace Cargo manifest
  - path: rust_core/Cargo.toml
    lang: toml
    purpose: PyO3 crate manifest
  - path: rust_core/src/lib.rs
    lang: rust
    purpose: Native core exposing the primary function
  - path: pyproject.toml
    lang: toml
    purpose: Python package configuration
  - path: my_engine/__init__.py
    lang: python
    purpose: Python driver package exports
  - path: my_engine/core.py
    lang: python
    purpose: Python wrapper importing rust_core
  - path: tests/test_core.py
    lang: python
    purpose: pytest tests

contracts:
  - name: compute_batch
    signature: "def compute_batch(data: list[float]) -> list[float]"
    language: python/rust
    python_name: my_engine.core.compute_batch
    purpose: Native/PyO3 exported core function

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

### Tri-Polyglot Example Blueprint

```aero
project: tri_optimized_project
architecture: tri_polyglot_rust_cpp_python
toolchains:
  - python
  - rust
  - cpp
  - cargo

manifest:
  - path: Cargo.toml
    lang: toml
    purpose: Rust workspace manifest
  - path: rust_core/Cargo.toml
    lang: toml
    purpose: PyO3 crate manifest
  - path: rust_core/src/lib.rs
    lang: rust
    purpose: Rust concurrent state/token validation core
  - path: cpp_core/native.cpp
    lang: cpp
    purpose: C-ABI dynamic shared library source
  - path: pyproject.toml
    lang: toml
    purpose: Python package configuration
  - path: tri_optimized_project/__init__.py
    lang: python
    purpose: Python driver package exports
  - path: tri_optimized_project/main.py
    lang: python
    purpose: Python CLI / REPL entrypoint
  - path: run_shell.py
    lang: python
    purpose: Headless launcher for automated commands
  - path: tests/test_tri.py
    lang: python
    purpose: pytest tests

contracts:
  - name: validate_token
    signature: "def validate_token(token: str) -> bool"
    language: python/rust
    python_name: tri_optimized_project.main.validate_token
    purpose: Safe token validation exported via PyO3
  - name: transform_numeric_array
    signature: "def transform_numeric_array(arr: list[float]) -> list[float]"
    language: python/cpp
    python_name: tri_optimized_project.main.transform_numeric_array
    purpose: Numeric array transformation via ctypes loaded from C++ shared library

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
| DeepSeek | `DEEPSEEK_API_KEY` | `deepseek-v4-pro` |
| Gemini | `GEMINI_API_KEY` | `gemini-2.0-flash` |
| Router only | none | `none` |

`AERO_FORGE_API_KEY` works as a generic fallback for any provider.

### Configuration Precedence

1. `ConfigOverride` passed to a build or generation task (request-scoped)
2. CLI flags
3. Environment variables
4. `llm` block in `blueprint.aero`
5. Built-in defaults

Example environment setup:

```bash
export AERO_FORGE_LLM_PROVIDER=openrouter
export AERO_FORGE_MODEL=openrouter/free
export OPENROUTER_API_KEY="sk-or-v1-..."
```

For web or programmatic use, pass a request-scoped ``ConfigOverride`` instead of mutating the environment:

```python
from aero_forge.config import ConfigOverride, override
from aero_forge.generate import generate_and_build

with override(ConfigOverride(llm_provider="deepseek", model="deepseek-v4-flash", api_key="...")):
    result = generate_and_build("Build a fast fibonacci function")
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
| `v10_correctness_focused` | Prioritizes correct, maintainable code. |

Use `--prompt-template v5_balanced` to select one. `v5_balanced` is the default and was the most reliable in the prompt-engineering campaign.

## Advanced LLM Intelligence (D-series)

These flags turn Aero-Forge into a senior-engineer-style assistant:

- `--algorithm-library` selects a reference implementation from the built-in `aero_forge/algorithms/` library and asks the LLM to adapt it.
- `--selected-algorithm <name>` forces a specific library entry.
- `--variants N` generates N implementations, compiles each, and selects the fastest variant that passes all tests using a Pareto frontier over accuracy and build time.
- `--explain` requests an `## Explanation` section covering algorithm choice, complexity, and tradeoffs.
- `--discover` lets the LLM design a new algorithm when the library has no match.
- `--review` runs a second LLM pass that checks the generated code for correctness, performance, security, and style.

## Engine Philosophy & System Boundaries

Aero-Forge separates **deterministic execution** from **LLM-assisted intent interpretation**:

- **Deterministic Core (no LLM calls):** prompt classification, blueprint validation, AST/UAST/HIN lowering, type inference, symbolic constraint verification, vectorization/SIMD planning, Fiedler graph partitioning, Rust/Python/C++ code generation, C-ABI header generation, `extern "C"` dynamic/shared library compilation (`-fPIC -shared`/static archive), Cargo/pip/maturin invocation, pytest/cargo test execution, and all healing attempts are 100% deterministic. The build loop never calls an LLM.
- **LLM Boundary (intent & diagnostics only):** LLMs are invoked only for initial natural language prompt interpretation, blueprint generation, high-level algorithm selection, `--explain` summaries, `--review` feedback, chat responses, and `aero-forge explain` human-facing diagnostics. No generated text enters the transpiler or build loop without being written to disk by an explicit generation step.

This boundary makes builds reproducible, auditable, and safe to run unattended or inside a web backend.

## Performance

Aero-Forge targets 10-100x speedups for hot numerical loops. Actual speedup depends on the function and the quality of the generated Rust. The benchmark loop in `aero-forge generate --optimize` compares the native extension against the original Python and reports the relative improvement.

Once a function is compiled, the AST node cache lets subsequent invocations skip `cargo build` entirely and load the cached native artifact in under a millisecond.

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
- `min()` and `max()` on two scalar values or a single iterable.
- `sum()` over a single iterable.
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

1. **Intent & Classification** - The user provides a natural language prompt (LLM), an existing `.py` file, or a `blueprint.aero` drag-and-drop. The prompt is classified into an `architecture` (e.g. `pure_python`, `pure_rust`, `hybrid_rust_python`, `hybrid_cpp_python`, `hybrid_cpp_rust`, `multi_crate_rust`, `tri_polyglot_rust_cpp_python`, `wasm`) and `toolchains`.
2. **Blueprint** - A `blueprint.aero` file is generated describing the workspace, manifest, contracts, and verification steps.
3. **Materialize** - Every file declared in the blueprint is physically emitted, including `Cargo.toml`, `pyproject.toml`, `build.rs`, `src/cpp_core/native.cpp`, `src/lib.rs`, `src/main.rs`, Python wrappers, and tests.
4. **Parse** - The Python source is parsed into an AST.
5. **Transpile** - A deterministic Python-to-Rust transpiler lowers the AST through a UAST/HIN intermediate and emits PyO3 `#[pyfunction]`/`#[pyclass]` code. C-ABI `extern "C"` C++ wrappers are emitted by `cpp_emitter.py` for heavy numeric loops.
6. **Scaffold** - A temporary Cargo crate, full workspace, or polyglot package is generated automatically, with `.cargo/config.toml` network resilience settings.
7. **Compile** - `cargo build --release`, `g++/clang++ -fPIC -shared`, or `maturin build` produces the native artifact, depending on the selected architecture.
8. **Cache** - The compiled native artifact is keyed by the hash of the UAST node so identical functions re-execute without recompiling.
9. **Wavefront Schedule** - Independent UAST nodes and functions are batched into wavefronts and compiled/executed in parallel across crates and languages, reducing build matrix times.
10. **Test** - `pytest` and `cargo test` run against the generated code in an isolated sandbox.
11. **Heal** - On failure, the orchestrator applies deterministic AST/pattern-based repairs, escalates to full-workspace LLM healing when static repairs are insufficient, and offers one-click "Regenerate Workspace from Blueprint" recovery.
12. **Explain** - Optional LLM-generated summaries are produced for human viewing after the build.

## Web Integration and Session Isolation

For embedded or web backends, use ``SandboxManager`` to create UUID-isolated sandbox directories:

```python
from aero_forge.sandbox.manager import SandboxManager

manager = SandboxManager()
sandbox_dir = manager.create_session_sandbox("550e8400-e29b-41d4-a716-446655440000")
# ... write or run files ...
zip_bytes = manager.archive_session_sandbox(session_id)
manager.clean_session_sandbox(session_id)
```

``ConfigOverride`` and ``override()`` let you pass per-request LLM settings without touching global environment variables. ``BuildRunner``, ``Orchestrator``, ``generate_and_build``, and ``ProjectBuilder`` all accept a ``config_override`` argument.

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

The first build may take a few minutes while PyO3 compiles. If `cargo` is not present, Aero-Forge will attempt to auto-bootstrap a minimal stable toolchain.

## Further Reading

- [`BLUEPRINT.md`](BLUEPRINT.md) - Complete blueprint reference and commented examples.
- [`PROMPT_ENGINEERING.md`](PROMPT_ENGINEERING.md) - Prompt template guide and campaign results.
- [`stress_tests/README.md`](stress_tests/README.md) - Stress-test campaign report.

## License

