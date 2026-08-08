# Aero-Forge: Natively Accelerated Polyglot Build Engine & Web Workspace

Aero-Forge turns a plain-English prompt or an existing source file into a complete, tested, **natively accelerated** software project. It is a universal build orchestrator for **natively accelerated Python**, **pure Rust**, **native C++**, **Rust/C++ systems**, and **Python/Rust/C++ tri-polyglot** applications, with automatic PyO3/Maturin extension generation, `extern "C"` C-ABI dynamic libraries, in-memory Holographic Interaction Net (HIN) JIT compilation, Geometry-of-Interaction (GoI) matrix scheduling, a zero-copy native bridge, and an embedded **web-first workspace**.

> **Web-first by design:** The fastest way to use Aero-Forge is the embedded web dashboard (`aero-forge web` or `python3 -m aero_forge.server`). It provides a full workspace environment — interactive Co-Pilot chat with Action Cards, a multi-tab file explorer & editor, real-time build/accelerator log streaming, drag-and-drop `workspace.aeroc` importing, and one-click workspace regeneration. The CLI remains fully functional for scripting and automation.

## What is Aero-Forge?

Aero-Forge is a prompt-driven build system for high-performance software. You describe what you want, point it at a `.py` file, or upload a ZIP, and it produces working source, native extensions, packaging manifests, tests, and a downloadable project bundle.

The engine is built around a declarative contract called `blueprint.aero`. For every request, Aero-Forge first classifies the prompt to infer the target architecture (`pure_python`, `pure_rust`, `hybrid_rust_python`, `hybrid_cpp_python`, `hybrid_cpp_rust`, `multi_crate_rust`, `tri_polyglot_rust_cpp_python`, `wasm`), the required toolchains (`python`, `cargo`, `maturin`, `cmake`, `cpp`, `gcc`, `clang`), the file manifest, and the exported contracts. It then materializes every declared file and invokes the appropriate native toolchains. When compilation or tests fail, a **100% deterministic, proof-theoretic self-healing core** repairs the workspace at native Rust speed. The build/repair loop never calls an LLM; LLMs are confined to intent interpretation, high-level strategy selection, and human-facing summaries.

Core value propositions:

- **Zero-boilerplate native acceleration** - No `Cargo.toml`, `#[pyfunction]`, `build.rs`, or linker flags required.
- **C-ABI Zero-Copy Dynamic Bridge** - Accelerated numerical functions compile to `.so`/`.dylib`/`.dll` via `clang++`/`g++` with native FFI bindings emitted by `cpp_emitter.py`.
- **Universal Multi-Language Build Matrix** - Native support for core targets (`pure_python`, `pure_rust`, `hybrid_rust_python`, `hybrid_cpp_python`, `hybrid_cpp_rust`, `multi_crate_rust`, `tri_polyglot_rust_cpp_python`, `wasm`) plus on-demand JIT emitter synthesis for any language/toolchain the LLM proposes (Go, C#, Java, Zig, Mojo, D, Nim, Fortran, etc.).
- **Selective Acceleration Heuristics** - AST node evaluation routes heavy vector loops to C++ `extern "C"` shared libraries, concurrent/memory-safe work to Rust PyO3, and light or incompatible workloads back to CPython.
- **Sub-millisecond execution pathways** - Numeric Python functions compile to native code and are cached at the AST node level for instant re-execution.
- **Wavefront Parallel Acceleration Engine** - Dependency-graph wavefront analysis batches independent functions and UAST nodes across multi-crate and polyglot targets for parallel compilation and execution, cutting build matrix times and hot-loop overhead.
- **Drop-In Workspace Portability** - `workspace.aeroc` is a packed binary workspace archive containing the `blueprint.aero` contract, source tree, and build metadata. Drag or copy it into an empty Aero-Forge workspace to scaffold, compile, and run the complete project.
- **100% Deterministic Proof-Theoretic Self-Healing** - Build failures are repaired at native Rust speed (<1 ms per rewrite) by MELL energy invariants, E-Graph equality saturation (`egg`), category-theoretic FFI contract synthesis, and GoI `ΔM` perturbation bounding. The repair loop never calls an LLM.
- **Interactive Co-Pilot & Action Cards** - The web Co-Pilot chat is workspace-aware (via `bundle_repo.py`), separates conversational advice from the executable build prompt using isolated ` ```build_prompt ` blocks, and renders `SUGGEST_BUILD_PROMPT` Action Cards with an editable prompt box and a one-click `[ Send to Builder & Run ]` trigger.
- **Proactive Synthesis Healing** - Unsupported Python constructs are healed into HIN-native forms by E-Graph equality saturation and SMT constraint solving before code is emitted. The legacy fall-forward-to-CPython path remains available as an explicit opt-in for cases that cannot be statically healed.

## Proactive Formal Synthesis Engine (Zero-Failure Architecture)

Aero-Forge has evolved from a reactive build-and-repair engine into a **proactive, proof-theoretic synthesis system**. It no longer relies on post-facto compiler error logs; instead, it mathematically guarantees buildability and cross-language semantic consistency *before* any file is written or a native toolchain (`rustc`, `g++`, `pytest`) is invoked.

### From Reactive Repair to Pre-Materialization Verification

- Build failures are prevented rather than observed after emission.
- Every multi-language boundary (Rust ↔ C++ ↔ Python) is verified for type, ownership, FFI layout, and concurrency safety before source files are materialized.
- Native toolchains are only invoked once the design has been proven sound.

### Core Verification Pipeline

- **HIN AST Normalization** — Python, Rust, and C++ AST fragments are unified into a single `networkx.MultiDiGraph` **Heterogeneous Information Network** $G_{\text{HIN}} = (V, E, \mathcal{T}, \mathcal{R})$. Double-Pushout (DPO) graph rewrites inject `FFIBoundary` nodes around raw string FFI calls, and an affine ownership lattice (`1`, `&`, `&mut`, `!`, $\bot$) propagates constraints across `TransfersOwnershipTo` edges.
- **Neuro-Symbolic SMT Solving (Z3)** — Unresolved typed holes ($\square_i$) and cross-language FFI constraints (`OffsetOf`, `AlignOf`, import visibility) are solved by `SMTASTEngine`. For every dynamic variable in a Python UAST, the engine creates a Z3 native-type variable and adds constraints from literals, arithmetic, subscripts, comparisons, calls (`range`, `len`, `list`, `dict`, `set`, `sorted`), assignments, and loop-carried dependencies. The most specific native type (`i64`, `f64`, `usize`, `bool`, `String`, `Vec<T>`, `BTreeMap<K, V>`, `HashSet<T>`) is injected into the generated Rust/C++/bridge code, allowing the native compiler to optimize the function as if it were statically typed. UNSAT cores are captured for in-memory healing rather than allowed to reach disk.
- **GoI Proof Net Verification** — Girard's *Geometry of Interaction* solver computes the wavefront execution matrix $EX(M, U) = (I - U \cdot M)^{-1} \cdot U$ and verifies nilpotency $(\sigma M)^N = 0$ on the loop-carried dependency matrix (with self-loops removed). This proves that even dynamic Python `for`/`while` loops are mathematically incapable of deadlocking across the HIN, because all cross-iteration dependencies eventually vanish. Singularity of $(I - U \cdot M)$ also detects cyclic build graphs before any disk write.
- **Pre-Materialization In-Memory Healing** — `FallbackManager` applies structural AST repairs in memory using SMT UNSAT cores and GoI path cuts. Dict/set idioms (`d.get(k)`, `dict(...)`, `{...}`) are rewritten into HIN-native `DictConstructor`/`SetConstructor`/`KeyLookup` agents with MELL linear typing. Only after all verification gates pass does `ProactivePolyglotBuilder` call the materialization step.

### Pipeline Workflow

```mermaid
flowchart LR
    A[Blueprint / Sketch AST] --> B[HIN Normalization]
    B --> C[SMT Z3 Constraint Solving]
    C --> D[GoI Proof Net Deadlock Check]
    D --> E[Pre-Write AST Healing]
    E --> F[Zero-Failure Emission & Build]
```

## Architecture

### A. Holographic Interaction Net (HIN) Engine & MELL Linear Typing

Aero-Forge's Python AST is lowered into a **Holographic Interaction Net (HIN)** — a compact graph of principal/auxiliary ports where computation happens by *localized active-pair reduction* rather than heap-allocated expression trees.

- **Homomorphic UAST lowering**: Python AST nodes (`BinOp`, `IfExp`, function calls, `return`, etc.) are translated into interaction-net agents: `Constructor`, `Destructor`, `Switch`, `Duplicator`, `Eraser`, `Value`, and `CausalProjection`.
- **Active-pair reduction**: Computation advances by repeatedly collapsing connected principal-port pairs. Supported rules include annihilation, duplication, erasure, conditional switching, and causal projection. Each rule rewires only the immediately adjacent ports, so the cost of one step is `O(1)`.
- **MELL linear typing**: Every wire carries a **Multiplicative-Exponential Linear Logic (MELL)** type (`I`, `Tensor`, `Implication`, `Bang`). These types replace dynamic symbol tables — variables are bound directly to physical topological edges. When a value is consumed, its wire is discharged, giving exact, zero-dynamic-heap memory accounting.
- **Native Rust arena**: The HIN kernel lives in `_native/src/hin_engine.rs`. Nodes and ports are stored in flat `Vec`s of `u32`-indexed slots. Reduction runs with the Python GIL released, and the resulting live graph is serialized back to JSON only once at the end.

### B. Execution Matrix Core & Geometry of Interaction (GoI)

For DAG-structured build tasks, Aero-Forge encodes the dependency graph as an execution matrix `M` and a routing rule matrix `U`.

```text
EX(M, U) = (I - U · M)^(-1) · U
```

- `M` is the dependency adjacency matrix (`M[i, j] = 1` if task `j` depends on task `i`).
- `U` is the routing/execution rule matrix that propagates completed work to dependent tasks.
- `EX(M, U)` returns the transitive execution wave matrix: each row gives the total influence (precedence ordering) of every other task.

**Incremental Schedule Repair (`ΔM`)**: When a local patch, LLM edit, or build failure changes only a few edges, Aero-Forge applies a `ΔM` update and recomputes only the affected wavefronts instead of rebuilding the full DAG. This keeps multi-round generation and healing responsive on 500+ node project graphs.

### C. Dual-Acceleration Paradigm

Aero-Forge accelerates *both* the artifacts it produces and the engine itself.

- **Target Acceleration**: User functions are routed to the fastest backend for their shape — Rust/PyO3 for memory-safe numeric kernels, C++ `extern "C"` shared libraries for vectorized loops, WASM for portable numeric kernels, and CUDA C for `# @accelerate gpu` pointwise array kernels.
- **Engine Self-Acceleration**: Internal hot paths use the same HIN/GoI machinery. AST lowering runs through the zero-heap Rust HIN arena, GoI matrix wavefronts schedule build tasks, and deterministic `ΔM` influence-zone computation restricts any downstream diagnostics to the minimal affected subgraph.

### D. Multi-Tier Execution Fallback Matrix

| Tier | HIN Kernel | GoI Solver | Use Case |
|------|-----------|-----------|----------|
| **Tier 1** | Native Rust HIN arena (GIL released) | JAX/XLA GPU GoI solver + CUDA kernel dispatch | GPU/TPU-backed builds and pointwise numeric kernels |
| **Tier 2** | Native Rust HIN arena | NumPy CPU GoI solver (`goi_solver.py`) | Standard workstations and CI |
| **Tier 3** | Python fallback HIN VM | Classic shell wavefront scheduler (`wavefront.py`) | Environments without the Rust extension or NumPy |

Tier 3 is fully backward-compatible; tiers 1 and 2 activate automatically when the Rust extension and optional JAX/NumPy dependencies are available.

## Graph-Driven Polyglot Materializer

Aero-Forge also supports a `graph_polyglot` architecture that models the whole build as a Heterogeneous Information Network (HIN). The `GraphPolyglotMaterializer` consumes a graph of language nodes and cross-language edges, schedules parallel build wavefronts with the GoI solver, synthesizes FFI bridge contracts, and writes all source files and toolchain manifests atomically.

### Architecture

The graph blueprint is a HIN:

```text
G_HIN = (V, E, T, R)
```

- $V$: language nodes (`rust_core`, `cpp_engine`, `py_client`, ...).
- $E$: directed FFI-boundary edges between nodes.
- $\mathcal{T}$: node target runtime / language tag.
- $\mathcal{R}$: edge boundary contract type (`c_abi`, `pyo3_maturin`, `cgo`, `pinvoke`, `jni`, ...).

`PolyglotGraphBlueprint` in `aero_forge/blueprint/schema.py` validates this graph: every edge endpoint must exist and the graph must be acyclic before materialization begins.

### GoI Wavefront Scheduling

Given a dependency matrix $M$ where $M_{ij} = 1$ when node $i$ depends on node $j$ (edge $j \to i$), and a routing matrix $U$, the execution wave operator is:

$$EX(M, U) = (I - U \cdot M)^{-1} \cdot U$$

If $\det(I - U \cdot M) = 0$, the operator is singular and the graph contains a cycle. `GoIWavefrontSolver` raises a cyclic-dependency exception before any file is written to disk, so circular builds are caught during pre-materialization verification.

### Supported Runtimes and FFI Boundaries

| Runtime | Boundaries | Toolchains |
|---------|------------|------------|
| **Python** | C-ABI (`ctypes`), PyO3/Maturin | `python3`, `maturin` |
| **Rust** | C-ABI (`extern "C"`), PyO3/Maturin | `rustc`, `cargo`, `maturin` |
| **C/C++** | C-ABI shared libraries, CUDA-HIP-C | `gcc`, `clang`, `clang++`, `nvcc` |
| **Go** | CGO `//export` c-shared | `go` |
| **C#** | .NET NativeAOT `[UnmanagedCallersOnly]`, `[LibraryImport]` | `dotnet` |
| **Java** | JNI native method signatures | `javac`, `gcc`, `clang++` |

### `blueprint.aero` Example

```yaml
metadata:
  project_name: graph_demo
  schema_version: "3.0.0"
  architecture: graph_polyglot

nodes:
  - node_id: rust_core
    lang: rust
    toolchain: cargo
  - node_id: cpp_engine
    lang: cpp
    toolchain: clang++
  - node_id: py_client
    lang: python
    toolchain: python

edges:
  - source: rust_core
    target: cpp_engine
    boundary_type: c_abi
    symbol: rust_compute
    args: [int64, int64]
    return_type: int64
  - source: cpp_engine
    target: py_client
    boundary_type: c_abi
    symbol: cpp_compute
    args: [int64, int64]
    return_type: int64
```

The `GraphPolyglotMaterializer` turns this into three ordered wavefront stages (`rust_core`, then `cpp_engine`, then `py_client`), synthesizes the C-ABI headers and `ctypes` loaders for each edge, and emits `Cargo.toml`, `CMakeLists.txt`, and `pyproject.toml` manifests.

### Extending the Engine with Custom Plugins

New language targets implement `PolyglotEmitterPlugin` and register with `EmitterRegistry`:

```python
from aero_forge.builder.emitters.base import (
    BoundaryContract,
    CapabilityDescriptor,
    CodeArtifact,
    EmitterRegistry,
    PolyglotEmitterPlugin,
)

class ZigEmitterPlugin(PolyglotEmitterPlugin):
    @property
    def descriptor(self) -> CapabilityDescriptor:
        return CapabilityDescriptor(
            language_id="zig",
            supported_boundaries={BoundaryContract.C_ABI},
            toolchains=["zig"],
            file_extensions=[".zig"],
            supports_zero_copy=False,
            supports_async_ffi=False,
        )

    def emit_source_files(self, node_id, node_spec, boundary_contracts):
        return [
            CodeArtifact(
                file_path=f"{node_id}.zig",
                content="// generated",
                language="zig",
            )
        ]

    def emit_build_manifest(self, node_id, dependencies, compiler_flags):
        return CodeArtifact(
            file_path="build.zig",
            content="// build manifest",
            language="zig",
        )

EmitterRegistry.get_instance().register(ZigEmitterPlugin())
```

## Universal Plugin Synthesis

Aero-Forge is a **Universal Build System**: it is not constrained to the languages that ship with built-in emitters. When `blueprint.aero` requests a language or toolchain that has no hardcoded `PolyglotEmitterPlugin` (e.g., `zig`, `mojo`, `d`, `nim`, or `fortran`), the engine JIT-synthesizes a plugin on demand.

The synthesis flow is deterministic and auditable:

1. **Blueprint request** — the LLM planner emits a `graph_polyglot` node with `lang` and `toolchain` set to the desired value.
2. **Registry miss** — `EmitterRegistry.get_plugin(..., synthesize=True)` notices the language is not registered and loads the `EMITTER_PLUGIN_SYNTHESIS_SYSTEM_PROMPT`.
3. **LLM emitter generation** — an LLM writes a self-contained `PolyglotEmitterPlugin` subclass, including a `CapabilityDescriptor` with the supported `BoundaryContract` values and `emit_source_files` / `emit_build_manifest` implementations.
4. **Deterministic validation** — the engine checks that the synthesized plugin defines a concrete class, exposes the requested language id and boundary type, returns `List[CodeArtifact]`, and implements a real exported function matching the first boundary contract.
5. **Materialization** — the validated plugin is registered temporarily and used to emit source and manifest files; the native toolchain is then invoked through `SystemToolchainRouter`.

This lets the LLM architect propose the right tool for the job (Go for a web server, C# NativeAOT for a P/Invoke kernel, Java JNI for an enterprise integration, Zig for a fast math library) without requiring hand-written plugins in the repo.

## Core Supported Build Targets

Aero-Forge natively supports eight primary build targets, each with deterministic materialization and native toolchain invocation. Beyond these, the **Universal Plugin Synthesis** pipeline can materialize any language the LLM proposes.

### 1. Natively Accelerated Python

Python functions are transpiled and compiled into either a PyO3 extension or a C-ABI shared library, then exposed through a Pythonic wrapper. The `@accelerate` decorator lets you mark any numeric function for native compilation:

```python
from aero_forge import accelerate

@accelerate(target="rust_hin")
def weighted_sum(scores: list[float], weights: list[float]) -> float:
    total = 0.0
    for i in range(len(scores)):
        total += scores[i] * weights[i]
    return total
```

The first call compiles; subsequent calls reuse the cached UAST node hash and execute in the **Holographic Interaction Net (HIN)** kernel with MELL linear typing and zero dynamic heap allocations, or in the compiled `.so`, with sub-millisecond latency.

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
- **JIT-synthesized language targets** (Go, C#, Java, Zig, Mojo, D, Nim, etc.) produced by the Universal Plugin Synthesis pipeline.

## Key Features

- **Native HIN Acceleration Core** - A Rust-based execution backend (`_native/src/hin_engine.rs`) lowers numeric Python into a **Holographic Interaction Net (HIN)** for fast, safe reduction with MELL linear typing and zero dynamic heap allocations.
- **Fine-Grained AST Node Hash Caching** - Compilation artifacts are keyed by the hash of individual UAST nodes. Identical functions bypass `cargo build` entirely and load the cached native snippet instantly.
- **Proactive Synthesis Healing** - Unsupported Python constructs are rewritten into HIN-native equivalents (`d.get(k)` → `d[k]`, `dict(...)` → `{...}`, `set.add(...)` → set constructors) by E-Graph equality saturation and SMT-driven type inference before a single file is written. The legacy fall-forward-to-CPython path remains as an explicit opt-in (`--precision-shield-mode permissive`).
- **Universal Intent Detection** - Parses any high-level prompt to infer languages, build tools, module boundaries, and concurrency patterns. Hybrid stacks are never silently downgraded to a fallback.
- **Declarative Blueprinting (`blueprint.aero`)** - Every build starts with a generated contract that declares `architecture`, `toolchains`, `manifest`, `contracts`, `functions`, and verification steps.
- **Strict Blueprint Materialization** - Every manifest file, source module, native backend library, compiler config, and target binding declared in `blueprint.aero` is physically emitted and built.
- **Natural Language Prompts** - Describe what you want and Aero-Forge generates Python code, tests, and a build blueprint.
- **Zero Manual Rust Boilerplate** - No `Cargo.toml`, `#[pyfunction]` annotations, or linker flags are required from the user for standard Python/Rust hybrid builds.
- **Interactive Web Dashboard & Terminal REPL** - Start a local web server (`aero-forge web`) to prompt, build, test, monitor real-time build logs, browse generated files in a multi-tab editor, and download compiled ZIP artifacts. An embedded xterm.js terminal supports copy/paste and live command execution.
- **Workspace Co-Pilot & Action Cards** - The Co-Pilot chat is workspace-aware via `bundle_repo.py`, understands all build targets, proposes optimized build prompts inside isolated ` ```build_prompt ` blocks, and renders `SUGGEST_BUILD_PROMPT` Action Cards with `[ Send to Builder & Run ]` and `[ Edit in Build Tab ]` buttons.
- **Geometry-of-Interaction Wavefront Engine** - DAG dependency graphs are encoded into matrices `M` and `U` and scheduled via the GoI formula `EX(M, U) = (I - U·M)⁻¹·U`. Incremental `ΔM` repairs avoid full recomputation during multi-round generation and healing.
- **Drop-In Workspace Import & Workspace Regeneration** - Drag `workspace.aeroc` into the web explorer to scaffold a full project, or click "Regenerate Workspace from Blueprint" to purge a broken workspace and rebuild strictly from the `blueprint.aero` contract.
- **100% Deterministic Proof-Theoretic Self-Healing Core** - Build failures are repaired by native Rust primitives: MELL `E(G)` energy invariants localize faults, `egg` equality saturation rewrites UAST expressions, category-theoretic contract synthesis emits missing FFI wrappers, and GoI `ΔM` perturbation bounding guarantees repair isolation. The repair loop never calls an LLM.
- **Algorithm Library** - Pick from a curated library of reference implementations (sorting, matrix, FFT, math) or let the LLM select one automatically.
- **Multi-Variant Testing** - Generate several implementations, compile them in parallel, benchmark each, and select the fastest variant that passes.
- **Explainable Builds** - Add `--explain` to get the LLM to describe the algorithm choice, complexity, and tradeoffs.
- **Auto-Discovery** - `aero-forge build --auto-detect` discovers `src/` and `tests/` and compiles everything it understands.
- **Project Builds & Bundles** - `aero-forge build --project <dir>` compiles every public function in a project directory and produces a downloadable zip with source, compiled libraries, a Python package, and a build manifest. The web dashboard also exports `workspace.aeroc` binary IR via `/api/workspace/download-aeroc` and Wavefront scaffold ZIPs via `/api/workspace/export-scaffold`.
- **Workspace Uploads** - `aero-forge build --upload project.zip` extracts, builds, and re-bundles an uploaded project. The web dashboard additionally accepts `workspace.aeroc` archives and auto-materializes the workspace through `/api/upload-aeroc`.
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

- **Co-Pilot Chat** — ask questions, get build suggestions, and trigger `SUGGEST_BUILD_PROMPT` Action Cards with editable `build_prompt` boxes.
- **Build Tab** — generate and compile polyglot projects with target and acceleration selectors.
- **Multi-Tab File Explorer & Editor** — browse, edit, and download generated files.
- **Real-Time BUILD LOG / ACCELERATOR LOG** — streaming `cargo`, `g++`, `maturin`, and heuristic telemetry.
- **Drag-and-Drop Workspace Import** — drop a `workspace.aeroc` archive into the explorer to scaffold the entire workspace.

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

The first call compiles the function to a native Rust `.so` or lowers it into the zero-heap HIN arena. The second call reuses the cached node key and executes in under a millisecond.

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
- **Co-Pilot Chat Tab** — A workspace-aware assistant powered by `bundle_repo.py`. Ask questions about the current project, request optimizations, or debug build failures. When the Co-Pilot proposes a build, it separates the conversational explanation from an executable `build_prompt` wrapped in an isolated code block and renders a `SUGGEST_BUILD_PROMPT` Action Card with an editable prompt box. Click `[ Send to Builder & Run ]` to dispatch the exact prompt to the Build tab and start compilation immediately, or `[ Edit in Build Tab ]` to review the prompt first.
- **File Explorer (left sidebar)** — Browse, open, and edit generated files. Drag-and-drop any `workspace.aeroc` archive into the explorer to scaffold the complete workspace. When a `workspace.aeroc` archive is detected, an **"Initialize Project from Archive"** banner appears.
- **Embedded Terminal (bottom panel)** — An `xterm.js` terminal with copy/paste and live command execution.
- **Real-Time Logs**:
  - **BUILD LOG:** Output from `cargo`, `g++`, `clang++`, `maturin`, and test runners.
  - **ACCELERATOR LOG:** Heuristic routing telemetry, AST hash evaluation, and bridge binding verdicts (e.g., `ACCELERATED: C++ selected for extern "C" dynamic shared library`).
  - **GoI/HIN STREAM:** Streaming `goi_wave_state` and `hin_reduction_steps` NDJSON events from the wavefront scheduler and HIN arena, surfaced in the dashboard for live execution tracking.

### Dashboard Controls

- **Target Language selector:** `Auto-Detect / Polyglot`, `Python`, `Rust`, `C++`, `Hybrid C++ / Rust`, `Multi-Crate Rust`, `Tri-Polyglot (Python + Rust + C++)`, `WebAssembly (wasm32)`.
- **Acceleration Policy selector:** `Selective Acceleration (Auto-Detect Heavy Compute)`, `Force Native Bridge`, `Standard Runtime (Bypass Bridge)`.
- **Blueprint actions**:
  - **Explorer header** — click the regenerate icon to open the *Regenerate Workspace from Blueprint* confirmation modal.
  - **Right-click `blueprint.aero`** in the explorer — choose *Rebuild Workspace from Blueprint*.
  - **Right-click `workspace.aeroc`** in the explorer — choose *Build Workspace* to run the packed binary archive directly.
  - **Error Recovery Panel** — when a build fails, the terminal banner offers `[ Hard Reset & Rebuild Workspace from Blueprint ]` alongside AST and LLM healing options.

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
| `aero-forge aeroc compile --input <dir|file> --output workspace.aeroc` | Compile a workspace or blueprint into the binary `workspace.aeroc` container. |
| `aero-forge aeroc exec --file workspace.aeroc --jobs 4` | Execute a packed `.aeroc` workspace with the native wavefront scheduler. |
| `aero-forge aeroc unpack --file workspace.aeroc --target-dir <dir>` | Extract a `.aeroc` archive back into source files. |
| `aero-forge aeroc export --file workspace.aeroc --output workspace.aeroc.bin` | Bundle a `.aeroc` archive with the native runner into a standalone executable. |

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
| `--prompt-template <name>` | Choose one of `v1_minimal`, `v2_structured`, `v3_algorithm`, `v4_performance`, `v5_balanced` (default), `v6_creative`, `v7_conservative`, `v8_iterative`, `v9_transpiler_friendly`, `v10_correctness_focused`, `v11_universal_architect`. |
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
| DeepSeek | `DEEPSEEK_API_KEY` | `deepseek-chat` |
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
| `v11_universal_architect` | Universal polyglot design: any toolchain, SMT/GoI-backed safety, multi-language boundary planning. |

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

- **Deterministic Core (no LLM calls):** prompt classification, blueprint validation, AST/UAST/HIN lowering, type inference, symbolic constraint verification, vectorization/SIMD planning, Fiedler graph partitioning, Rust/Python/C++ code generation, C-ABI header generation, `extern "C"` dynamic/shared library compilation (`-fPIC -shared`/static archive), Cargo/pip/maturin invocation, pytest/cargo test execution, and **all self-healing attempts** are 100% deterministic. The build/repair loop never calls an LLM.

- **Deterministic Proof-Theoretic Self-Healing Core:** When compilation or tests fail, repair is performed at native Rust speed by four proof-theoretic pillars:
  1. **MELL Linear Logic `E(G)` Invariants** (`aero_forge/_native/src/hin_engine.rs`) localize faults by quantifying stalled active pairs, dangling wires, and broken edges in the Holographic Interaction Net.
  2. **E-Graph Equality Saturation (`egg`)** (`aero_forge/_native/src/deterministic_healer.rs`) rewrites algebraic UAST expressions to their cheapest normal form.
  3. **Category-Theoretic Contract Synthesis** (`aero_forge/scaffold/contract_synth.py`) emits canonical PyO3, C-ABI, and zero-copy Rust morphism wrappers for missing cross-language symbols.
  4. **Geometry of Interaction `ΔM` Perturbation Bounding** (`aero_forge/_native/src/goi_solver.rs`) verifies that any repair perturbation keeps the spectral radius `ρ(U·(M + ΔM))` strictly below the unit boundary, guaranteeing zero upstream or parallel wavefront regressions.

- **LLM Boundary (intent & diagnostics only):** LLMs are invoked only for initial natural language prompt interpretation, blueprint generation, high-level algorithm selection, `--explain` summaries, `--review` feedback, chat responses, and `aero-forge explain` human-facing diagnostics. No generated text enters the transpiler or build loop without being written to disk by an explicit generation step.

This boundary makes builds reproducible, auditable, and safe to run unattended or inside a web backend.

## Performance

Aero-Forge targets 10-100x speedups for hot numerical loops and sub-second build matrix scheduling for large multi-crate projects. Actual speedup depends on the function and the selected backend.

- **HIN target speedups**: Numeric Python functions lowered to the Holographic Interaction Net run with zero dynamic heap allocations and the GIL released during reduction, avoiding CPython refcount churn.
- **GoI wavefront speedups**: Build-task DAGs scheduled via the Geometry-of-Interaction matrix are up to 5× faster to recompute incrementally than full DAG rebuilds on 500+ node graphs.
- **AST node cache**: Once a function is compiled, the AST node cache lets subsequent invocations skip `cargo build` entirely and load the cached native artifact in under a millisecond.

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

The transpiler targets numerical and algorithmic code. Most constructs are proactively healed into HIN-native equivalents before materialization.

Currently not natively supported (the engine will attempt AST healing; unhealed cases can opt-in to legacy CPython fallback):

- `insert`, `remove`, and most other list methods (only `append`, `extend`, `pop`, and indexing/slicing are supported; `dict`/`set` are now HIN-native).
- Nested function, class, or method definitions (refactor to top-level functions).
- Complex class inheritance, properties, and dataclasses.
- `try`/`except`, `with`, `yield`, `async`/`await`.
- `eval`/`exec` and dynamic imports.
- `random`, `datetime`, `re`, `json`, and other non-math stdlib modules.
- I/O, networking, and `os`/`subprocess`.
- Full `ndarray` broadcasting and n-dimensional operations.

See `BLUEPRINT.md` and `stress_tests/README.md` for the full supported-construct list and the stress-test campaign results.

## How It Works

1. **Intent & Classification** - The user provides a natural language prompt (LLM), an existing `.py` file, or drags-and-drops a `workspace.aeroc` archive. The prompt is classified into an `architecture` (e.g. `pure_python`, `pure_rust`, `hybrid_rust_python`, `hybrid_cpp_python`, `hybrid_cpp_rust`, `multi_crate_rust`, `tri_polyglot_rust_cpp_python`, `wasm`) and `toolchains`. The LLM may propose any language or toolchain (Go, C#, Java, Zig, Mojo, etc.); the engine will synthesize and validate the required emitter plugin.
2. **Blueprint** - A `blueprint.aero` file is generated describing the workspace, manifest, contracts, and verification steps.
3. **Materialize** - Every file declared in the blueprint is physically emitted, including `Cargo.toml`, `pyproject.toml`, `build.rs`, `src/cpp_core/native.cpp`, `src/lib.rs`, `src/main.rs`, Python wrappers, and tests.
4. **Parse** - The Python source is parsed into an AST.
5. **SMT Type Inference** - `SMTASTEngine` resolves every dynamic typed hole ($\square_i$) from usage, producing a concrete native type for each variable and injecting it into the generated bridge code.
6. **GoI Deadlock Proving** - The Geometry-of-Interaction solver builds the loop-carried dependency matrix, removes self-loops, and verifies $(\sigma M)^N = 0$ for pure-Python `for`/`while` loops, certifying that no dynamic HIN deadlock can occur.
7. **Transpile** - A deterministic Python-to-native transpiler lowers the AST through a UAST/HIN intermediate. The Holographic Interaction Net (HIN) is reduced in the zero-heap Rust arena (`_native/src/hin_engine.rs`) with MELL-typed wires, and code generators emit PyO3 `#[pyfunction]`/`#[pyclass]`, C-ABI `extern "C"` C++ wrappers, WASM, CUDA C, or JIT-synthesized emitters depending on the selected target.
8. **Scaffold** - A temporary Cargo crate, full workspace, or polyglot package is generated automatically, with `.cargo/config.toml` network resilience settings.
9. **Compile** - `cargo build --release`, `g++/clang++ -fPIC -shared`, `maturin build`, or the synthesized toolchain command produces the native artifact, depending on the selected target.
10. **Cache** - The compiled native artifact is keyed by the hash of the UAST node so identical functions re-execute without recompiling.
11. **Wavefront Schedule** - A Geometry-of-Interaction (GoI) matrix solver (`EX(M, U) = (I - U·M)⁻¹·U`) batches independent UAST nodes and functions into wavefronts. Incremental `ΔM` repairs avoid full DAG recomputation during multi-round generation and healing.
12. **Test** - `pytest` and `cargo test` run against the generated code in an isolated sandbox.
13. **Heal** - On failure, the orchestrator builds a workspace HINGraph, computes `ΔM` failure influence zones, and applies deterministic AST/pattern-based repairs. It escalates to focused, subgraph-limited LLM healing when static repairs are insufficient, and offers one-click "Regenerate Workspace from Blueprint" recovery.
14. **Explain** - Optional LLM-generated summaries are produced for human viewing after the build.

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

