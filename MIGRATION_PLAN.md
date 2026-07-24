# Aero-Forge / Aero-Topos Integration Migration Plan

## 1. Executive Summary

This plan defines how `aero-forge` can adopt `aero-topos` as the high-integrity build, pre-write validation, and artifact-generation substrate while preserving `aero-forge`'s existing prompt-driven Python-to-Rust flow. The goal is a single pipeline where:

1. Raw LLM output is still extracted and normalized inside `aero-forge`.
2. Extracted source flows through `aero-topos`'s **language routing**, **AST active merge / structural overlay**, **pre-write validation**, **precision shield**, **scaffold/repo generation**, and **active artifact promotion**.
3. The resulting compiled extension or standalone project is returned through `aero-forge`'s existing `BuildRunner` / `BuildSummary` surface.

No source code changes are proposed in this document; it is an architectural audit and migration matrix.

---

## 2. Current `aero-forge` Pipeline (Audited)

| Stage | Location | Responsibility |
| --- | --- | --- |
| Prompt construction & LLM call | `aero_forge/generate.py` (`generate_from_prompt`) | Builds user prompt, calls `get_llm_client`, returns raw markdown. |
| Code block extraction | `aero_forge/generate.py` (`parse_generated_response`, `extract_code_blocks`) | Splits response into implementation + pytest blocks. |
| Project write | `aero_forge/generate.py` (`write_generated_project`) | Writes `src/generated.py`, `tests/test_generated.py`, and a `blueprint.aero`. |
| Build orchestration | `aero_forge/orchestrator/orchestrator.py` (`Orchestrator.run`) | Parses source, classifies `HIN_COMPUTE` vs `GENERAL_PURPOSE`, transpiles to Rust, runs `cargo build`, tests in `Sandbox`, heals on failure. |
| Transpiler / HIN graph | `aero_forge/translator/aero_frontend.py`, `aero_forge/translator/translator.py` | Python AST → UAST → HIN graph → Rust crate (`Engine` in `aero_forge/scaffold/engine.py`). |
| Build runner | `aero_forge/build_runner.py` (`BuildRunner`, `BuildTaskDAG`) | Schedules source groups as DAG tasks with SHA-256 hashing, drives `Orchestrator`, aggregates results. |
| Scaffold templates | `aero_forge/scaffold/engine.py` (`ProjectScaffolder`) | Generates Axum/Clap/Python-hybrid project shells. |
| Sandbox | `aero_forge/sandbox/manager.py` (`Sandbox`, `SandboxManager`, `TraceVerifier`) | Isolated pytest execution and trace verification. |
| Cache | `aero_forge/cache/build_cache.py` | Artifact-level and task-level SHA-256 cache. |

### 2.1 Gaps Relative to `aero-topos`

- **No language router**: `aero-forge` always transpiles Python to Rust (or scaffolds Python projects). It does not route arbitrary source entries (Rust, C, C++, Fortran) to the correct backend.
- **No pre-write validation gate**: `aero-forge` builds in a temp crate, then copies artifacts back. It does not stage a full repository, run a user-defined `validation_cmd`, and *then* atomically promote.
- **No structural AST active merge**: User edits to generated files are not preserved across regenerations; there is no `OverlayManager`/3-way AST merge.
- **No standalone repo generator**: `aero-forge` produces a temp crate + `.so`/loader, not a turn-key GitHub-ready repository.
- **Limited precision shield**: `aero-forge` infers numeric types but does not inject `rug`/`pyo3` extension traits the way `aero-topos` `RustSemanticShield` does.
- **No execution-trace verifier until now**: `aero-forge` just added `TraceVerifier`; `aero-topos` has richer sandbox telemetry (`sandbox_runner.py`).

---

## 3. `aero-topos` Architectural Components (Deep Dive)

### 3.1 Language Routing & Target Emitters

- `src/scaffold/language_router.py`
  - `resolve_target_language(context, source_entry, source_path)` decides `rust` vs `python` (and conceptually `c`/`cpp`/`fortran`) from blueprint `frameworks.language`, a `SourceEntry` tag, or file extension.
  - `is_native_crate_language`, `is_python`, `cargo_bypass_warning` keep the cargo layer from firing on non-Rust targets.
- `builder_brains/emitters/`
  - `base.py`: `BaseEmitter` walks a UAST node list and dispatches to language-specific hooks (`_emit_function_declaration`, `_emit_type_declaration`, `_emit_import_declaration`, `_emit_statement`, `_emit_expression`).
  - `rust_emitter.py`: `RustEmitter` implements those hooks for Rust.
  - `python_emitter.py`, `cpp_emitter.py`: Same pattern for other targets.

### 3.2 Pre-Write Validation & Sandbox Gatekeeping

- `src/scaffold/pre_write_validator.py`
  - `PreWriteValidator` stages files in a workspace and runs `validation.validation_cmd` (e.g. `cargo test`, `pytest`, custom script). Only a zero exit code triggers promotion.
- `src/scaffold/workspace.py`
  - `OutOfTreeWorkspace` creates a temp staging directory, refuses to place it inside the tool tree, and atomically moves it to `distribution_directory` on success.
- `sandbox_runner.py`
  - `run_module` imports a module by name, invokes a callable with sample params, and records latency/accuracy traces. Used by the evolution / benchmark loops.

### 3.3 Precision Shield / Rust Shims

- `src/scaffold/rust_shield.py`
  - `RustSemanticShield.detect_anchors` looks for `rug`/`pyo3`.
  - `apply` injects `AeroNegMutExt` and `AeroNthRootExt` traits, fixes `let x = match` type cascades (`usize` annotation), and recovers mutability from `cannot borrow x as mutable` diagnostics.
- `src/build/cargo_manifest.py`
  - `CargoPlan` decides whether to reuse an existing `Cargo.toml` or synthesize one, ensures `pyo3` carries `extension-module` + `experimental-declarative-modules`, and discovers the correct crate root / target directory.

### 3.4 Structural AST Active Merge / Overlays

- `core/overlay/structural_merger.py`
  - `StructuralMerger` performs a 3-way AST merge: `base` (pristine generated), `left` (user edit), `right` (newly generated). Uses Tree-sitter via `core/parser/universal`.
  - Tier 1: coarse entity merge (imports, functions, types). Tier 2: fine-grained node alignment via identity signatures and content hashes. On conflict keeps the freshly generated version and flags `blueprint.aero`.
- `src/overlay/manager.py`
  - `OverlayManager.commit_overlay(file)` records a user edit as a patch against the pristine baseline.
  - `OverlayManager.reapply(file)` / `reapply_all()` reapplies patches after regeneration.
  - `OverlayManager.structural_reapply(...)` delegates to `StructuralMerger` for AST-level merging.
- `src/overlay/apply.py`
  - `apply_patch` is the line-level fallback; it locates hunks by content rather than line numbers so context shifts do not break patches.
- `src/overlay/store.py`
  - `OverlayStore` persists pristine baselines and overlay patches in `.build_cache`.

### 3.5 Repository & Artifact Generation

- `src/scaffold/repo_generator.py`
  - `RepoSpec` / `GeneratedRepo` describe a standalone repo.
  - `infer_dependencies(source)` scans for `use rug`, `use pyo3`, rayon parallel iterator patterns, etc., and produces a dependency table.
  - `generate_repo(root, spec)` writes `Cargo.toml`, `src/lib.rs`, `.gitignore`, `README.md`, `test_binding.py`.
- `translator/artifact_generator.py`
  - `ArtifactGenerator` is a template-driven registry for FFI artifacts (`templates/ffi`, `registry.json`). Renders `.pyi`, ctypes loaders, C headers, etc. idempotently.
- `src/scaffold/active_merge.py`
  - `find_compiled_library` locates the `cdylib` under `target/{release,debug}`.
  - `merge_active` copies the `.so`/`.dylib`/`.pyd` into the active extension layer (`core/extensions/`) and optionally loads it live into the running process.

### 3.6 Top-Level Orchestration

- `orchestrator.py` / `main.py`
  - `ScaffoldBuildPipeline` (in `src/scaffold/pipeline.py`) is the primary entry: resolve source → language router → rust shield → workspace provisioning → repo synthesis → isolation build → pre-write validation → active merge.

---

## 4. Migration Matrix

| `aero-topos` Component | Role | Current `aero-forge` Equivalent | Migration Action | Target `aero-forge` Location | Notes |
| --- | --- | --- | --- | --- | --- |
| `src/scaffold/language_router.py` | Decide `rust`/`python`/... target | None; always transpiles Python to Rust | **Import/adapt** as `aero_forge.orchestrator.language_router` | `aero_forge/orchestrator/language_router.py` | Use after LLM code extraction to route pure Python prompts vs. Rust/C source uploads. |
| `builder_brains/emitters/base.py` + `rust_emitter.py` | UAST → target source | `aero_forge/translator/translator.py` + `aero_forge/scaffold/engine.py` `Engine` | **Refactor** to emit Rust from `aero-topos` `BaseEmitter` API; keep `aero-forge` UAST as input | `aero_forge/translator/emitters/` | `RustEmitter` already matches `Engine.generate` output; replace ad-hoc string building. |
| `builder_brains/emitters/python_emitter.py` | UAST → Python | `aero_forge/translator/aero_frontend.py` (Python→UAST only) | **Add** reverse emitter for round-trip / scaffold use | `aero_forge/translator/emitters/python_emitter.py` | Useful for Python-hybrid templates. |
| `src/scaffold/rust_shield.py` | `rug`/`pyo3` trait & type fixes | `aero_forge/precision_shield/shield.py` | **Wrap** `Shield.analyze` with `RustSemanticShield.apply` before `Engine.generate` | `aero_forge/precision_shield/rust_shield.py` | `aero-forge` already has numeric trait inference; `RustSemanticShield` adds codegen-time shims. |
| `src/build/cargo_manifest.py` | `Cargo.toml` synthesis / reuse | Inline `Cargo.toml` string templates in `Engine.generate` | **Replace** with `CargoPlan` / `render_manifest` | `aero_forge/build/cargo_manifest.py` | Supports user-supplied manifests and correct `pyo3` feature injection. |
| `src/scaffold/pre_write_validator.py` + `workspace.py` | Staged build + validation before promotion | `Sandbox` + `BuildRunner` build directly in temp dirs | **Insert** `OutOfTreeWorkspace` + `PreWriteValidator` between transpile and artifact copy | `aero_forge/scaffold/workspace.py`, `aero_forge/scaffold/pre_write_validator.py` | Critical for high-integrity builds; prevents broken artifacts in `dist/`. |
| `src/scaffold/source_resolver.py` | Resolve source paths anywhere on disk | `aero_forge/blueprint.py` path resolution | **Use** `resolve_source_entry` for blueprint `file:` entries | `aero_forge/blueprint.py` or `aero_forge/scaffold/source_resolver.py` | Handles `~`, absolute, relative, and `base_dir` candidates. |
| `core/overlay/structural_merger.py` + `src/overlay/manager.py` | Preserve user edits across regenerations | None | **Add** overlay layer around generated `src/generated.py` and `Cargo.toml` | `aero_forge/overlay/` | Tree-sitter dependency required (`tree-sitter`, `tree-sitter-python`, `tree-sitter-rust`). |
| `src/scaffold/repo_generator.py` | Standalone repo generation | `ProjectScaffolder` in `aero_forge/scaffold/engine.py` | **Enhance** `ProjectScaffolder` with `RepoSpec`/`infer_dependencies` | `aero_forge/scaffold/repo_generator.py` | Produces GitHub-ready crates with `README.md`, `test_binding.py`, `.gitignore`. |
| `translator/artifact_generator.py` | Template-driven FFI loaders, `.pyi`, C headers | `Engine._write_loader`, `_generate_pyi` | **Replace** with `ArtifactGenerator` registry | `aero_forge/scaffold/artifact_generator.py` | Supports project-local `templates/ffi` overrides. |
| `src/scaffold/active_merge.py` | Promote compiled `.so` to active extension dir | `Orchestrator._merge_back` + `BuildRunner` output copy | **Extend** `Orchestrator._merge_back` with `find_compiled_library` / `merge_active` | `aero_forge/scaffold/active_merge.py` | Enables live loading of freshly built modules without restart. |
| `sandbox_runner.py` | Import + benchmark callable with latency traces | `TraceVerifier` in `aero_forge/sandbox/manager.py` | **Extend** `TraceVerifier` with `sandbox_runner.run_module` telemetry | `aero_forge/sandbox/manager.py` | Adds per-invocation latency/accuracy metrics for fitness-driven optimization. |
| `core/parser/universal.py` | Polyglot Tree-sitter parsing | `ast` + custom `aero_frontend` | **Use** for structural merge and language router; keep `aero_frontend` for Python-to-UAST | `aero_forge/parser/universal.py` | Required for AST overlay on Rust/Python sources. |

---

## 5. Proposed End-to-End Data Flow

```text
User prompt
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  aero_forge.generate.generate_from_prompt                           │
│  (keep: prompt construction, LLM client, algorithm library)        │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼ raw markdown
┌─────────────────────────────────────────────────────────────────────┐
│  aero_forge.generate.parse_generated_response                         │
│  (keep: code-fence extraction; output Python source + tests)        │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼ Python source + tests
┌─────────────────────────────────────────────────────────────────────┐
│  aero_forge.orchestrator.language_router (new, from topos)          │
│  • route: HIN_COMPUTE  → Rust/PyO3                                   │
│  • route: GENERAL_PURPOSE → native Python scaffold                  │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  aero_forge.overlay.manager.OverlayManager (new)                   │
│  • record pristine baseline                                          │
│  • reapply committed user overlays via StructuralMerger              │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼ merged source
┌─────────────────────────────────────────────────────────────────────┐
│  aero_forge.scaffold.workspace.OutOfTreeWorkspace (new)            │
│  • create temp staging dir outside tool tree                         │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼ in staging workspace
┌─────────────────────────────────────────────────────────────────────┐
│  aero_forge.translator.emitters.rust_emitter (new)                   │
│  • Python UAST → Rust source (replace Engine string building)        │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼ Rust source
┌─────────────────────────────────────────────────────────────────────┐
│  aero_forge.precision_shield.rust_shield (new)                      │
│  • inject AeroNegMutExt / AeroNthRootExt / mutability fixes         │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼ shielded source
┌─────────────────────────────────────────────────────────────────────┐
│  aero_forge.scaffold.repo_generator (new)                          │
│  • infer dependencies (rug, pyo3, rayon)                           │
│  • write Cargo.toml / src/lib.rs / .gitignore / README / tests     │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼ repo skeleton
┌─────────────────────────────────────────────────────────────────────┐
│  aero_forge.scaffold.pre_write_validator (new)                     │
│  • run cargo build / pytest in staging workspace                    │
│  • fail fast → do not promote                                       │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼ validation passed
┌─────────────────────────────────────────────────────────────────────┐
│  aero_forge.scaffold.workspace.commit() (atomic promote)             │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  aero_forge.scaffold.active_merge.merge_active (new)               │
│  • find cdylib under target/                                         │
│  • copy to aero_forge output dir / core/extensions                   │
│  • generate .pyi / loader via ArtifactGenerator                     │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  aero_forge.sandbox.TraceVerifier + sandbox_runner telemetry       │
│  • execute reference Python vs compiled target                     │
│  • assert Δ = 0 (stdout/stderr/exit/syscalls)                       │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  aero_forge.build_runner.BuildTaskDAG (keep)                        │
│  • cache by SHA-256 of merged source + repo spec + flags           │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.1 Raw LLM Output → AST Merge

1. `parse_generated_response` returns `(implementation, tests)`.
2. Before writing, `OverlayManager.record_generated(file, implementation)` stores the pristine baseline.
3. If the user (or a previous build) edited `src/generated.py`, `OverlayManager.structural_reapply` merges the new LLM output (`right`) with the user edit (`left`) against the previous baseline (`base`). Conflicts keep the generated version and are logged.
4. The merged source is then handed to the language router / emitter.

### 5.2 Pre-Write Validation → Repository Generation

1. `OutOfTreeWorkspace.create()` builds a staging directory under `/tmp` (or user `distribution_directory` with a `.staging` suffix).
2. `RustEmitter`/`RepoGenerator` write files inside the staging workspace.
3. `PreWriteValidator.validate(workspace_path, language="rust")` runs `cargo build --release` (or `pytest`, or a blueprint `validation_cmd`).
4. On zero exit, `workspace.commit()` atomically moves staging to the final deliverable path.
5. On failure, the staging directory is discarded and the error is returned without polluting `dist/`.

### 5.3 Artifact Generation → Active Merge

1. `CargoPlan` (from `src/build/cargo_manifest.py`) resolves `Cargo.toml` and target directory.
2. `cargo build --release` runs inside the staged crate.
3. `active_merge.find_compiled_library(workspace, crate_name)` locates `lib<crate>.so`/`.dylib`/`.pyd`.
4. `ArtifactGenerator` renders the Python loader and `.pyi` stub from `templates/ffi/`.
5. `merge_active` copies the library and loader into `aero-forge`'s output directory (or `core/extensions/` if running in a live self-hosting context).

---

## 6. Phased Rollout Recommendation

### Phase 0: Foundation (no behavior change)

- Vendor or symlink the following `aero-topos` modules under `aero_forge/vendor/topos/`:
  - `src/scaffold/language_router.py`
  - `src/build/cargo_manifest.py`
  - `src/scaffold/workspace.py`
  - `src/scaffold/pre_write_validator.py`
  - `src/scaffold/source_resolver.py`
- Add `tree-sitter` + grammar packages to `pyproject.toml`.
- Run full `pytest` to ensure imports work.

### Phase 1: Repository Generation

- Replace inline `Cargo.toml`/`src/lib.rs` string building in `aero_forge/scaffold/engine.py` with `aero-topos` `repo_generator` + `cargo_manifest`.
- Wire `PreWriteValidator` + `OutOfTreeWorkspace` into `Orchestrator._compile_to_native`.
- Add `ProjectScaffolder` integration to produce standalone repos.

### Phase 2: Structural Overlays

- Integrate `core/parser/universal` + `src/overlay/manager`.
- Hook `OverlayManager` into `generate.py` `write_generated_project` and `Orchestrator` regeneration loops.
- Add `MIGRATION_PLAN.md` migration tests for overlay persistence.

### Phase 3: Polyglot Emitters

- Introduce `aero_forge/translator/emitters/` (`base.py`, `rust_emitter.py`, `python_emitter.py`).
- Gradually move `Engine.generate` logic into `RustEmitter`, keeping `aero-forge` UAST as the canonical input.

### Phase 4: Active Merge & Telemetry

- Add `active_merge.merge_active` for live extension loading.
- Extend `TraceVerifier` with `sandbox_runner.run_module`-style latency/accuracy traces for fitness-driven optimization.

---

## 7. Risks & Open Questions

| Risk | Mitigation |
| --- | --- |
| `aero-topos` uses `src.*` and top-level `core.*` imports that collide with `aero-forge` layout. | Vendor under `aero_forge/vendor/topos/` with rewritten imports, or publish `aero-topos` as a package with a proper `topos.*` namespace. |
| Tree-sitter grammars add binary wheels and build time. | Pin `tree-sitter` + `tree-sitter-python` + `tree-sitter-rust`; make grammars optional if overlay feature not used. |
| `aero-forge` UAST differs from `aero-topos` UAST node shape. | Define a thin adapter in `aero_forge/translator/uast_adapter.py` mapping `aero-forge` UAST nodes to `BaseEmitter` node lists. |
| Pre-write validation may double build time. | Cache by SHA-256 (existing `BuildTaskDAG`) so unchanged sources skip validation. |
| User overlays on generated Rust require Rust Tree-sitter grammar. | Use Python-only overlays first; enable Rust structural merge as an opt-in flag. |
| `aero-topos` `Precision Shield` and `aero-forge` `Shield` may overlap. | Treat `aero-forge` `Shield.analyze` as input to `RustSemanticShield.apply` (type inference first, codegen shims second). |

---

## 8. Files to Create or Refactor (Summary)

### New modules (copied/adapted from `aero-topos`)

- `aero_forge/orchestrator/language_router.py`
- `aero_forge/parser/universal.py`
- `aero_forge/overlay/manager.py`
- `aero_forge/overlay/structural_adapter.py` (wraps `core/overlay/structural_merger`)
- `aero_forge/scaffold/workspace.py`
- `aero_forge/scaffold/pre_write_validator.py`
- `aero_forge/scaffold/source_resolver.py`
- `aero_forge/scaffold/repo_generator.py`
- `aero_forge/scaffold/active_merge.py`
- `aero_forge/scaffold/artifact_generator.py`
- `aero_forge/build/cargo_manifest.py`
- `aero_forge/precision_shield/rust_shield.py`
- `aero_forge/translator/emitters/base.py`
- `aero_forge/translator/emitters/rust_emitter.py`
- `aero_forge/translator/emitters/python_emitter.py`

### Existing modules to modify

- `aero_forge/generate.py` — call `OverlayManager` before writing.
- `aero_forge/orchestrator/orchestrator.py` — route through `language_router`, stage with `OutOfTreeWorkspace`, validate with `PreWriteValidator`, merge artifacts with `active_merge`.
- `aero_forge/scaffold/engine.py` — `Engine.generate` delegates to `RustEmitter` + `RepoGenerator` + `CargoPlan`.
- `aero_forge/sandbox/manager.py` — extend `TraceVerifier` with `sandbox_runner` telemetry.
- `pyproject.toml` — add `tree-sitter`, `tree-sitter-python`, `tree-sitter-rust`, optional `aero-topos` dependency.

---

## 9. Definition of Done

- `aero-forge` can accept a prompt, generate Python source, route it, merge user overlays, stage a repo, run pre-write validation, and promote a compiled artifact.
- The pipeline produces a `SemanticRegressionError` with a diff report when reference and target execution traces diverge.
- All existing `aero-forge` tests continue to pass; new tests cover overlay persistence, pre-write validation failure, and standalone repo generation.
