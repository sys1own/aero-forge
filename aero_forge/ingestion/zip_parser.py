"""ZIP archive ingestion and v3.0 draft blueprint generation."""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import tomllib  # type: ignore[import]
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]

import yaml

from aero_forge.blueprint.schema import (
    ArtifactType,
    BlueprintStatus,
    BlueprintV3,
    BuildArtifact,
    ContextState,
    ExecutionStrategyV3,
    GenerationMethod,
    LLMContext,
    Metadata,
    ToolchainSpec,
    VerificationNode,
    write_v3_blueprint,
)
from aero_forge.ingestion.command_inspector import _unwrap_single_root, detect_runnable_commands
from aero_forge.scaffold.module_guard import reify_missing_modules

logger = logging.getLogger("aero_forge.ingestion.zip_parser")


def extract_zip_safely(zip_bytes: bytes, dest: Path, archive_name: Optional[str] = None) -> None:
    """Extract a zip archive to *dest* while guarding against path traversal.

    If every member is inside a single top-level wrapper directory (and no files
    live at the archive root), that wrapper is stripped so files land directly
    under *dest*.
    """
    dest = Path(dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        members = [m for m in zf.namelist() if not m.startswith("__MACOSX/")]
        files = [m for m in members if not m.endswith("/")]
        if not files:
            return

        # Strip a single top-level wrapper directory only when it is not a
        # common source directory such as ``src`` or ``tests``.
        strip_prefix = ""
        top_dirs = {m.split("/")[0] for m in files if "/" in m}
        root_files = [m for m in files if "/" not in m]
        if len(top_dirs) == 1 and not root_files:
            prefix = next(iter(top_dirs))
            common_folders = {
                "src", "lib", "libs", "tests", "test", "app", "bin", "docs",
                "examples", "scripts", "pkg", "package", "include", "includes",
            }
            archive_stem = Path(archive_name).stem if archive_name else ""
            if prefix == archive_stem or prefix.lower() not in common_folders:
                strip_prefix = prefix + "/"

        for member in files:
            rel = member[len(strip_prefix):] if strip_prefix and member.startswith(strip_prefix) else member
            if not rel:
                continue
            if any(part in ("..", "") for part in Path(rel).parts):
                raise ValueError(f"Invalid or unsafe zip member path: {member}")

            target = dest / rel
            try:
                resolved = target.resolve()
                resolved.relative_to(dest.resolve())
            except ValueError as exc:
                raise ValueError(f"Zip member escapes extraction directory: {member}") from exc

            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(target, "wb") as dst:
                dst.write(src.read())

    # Flatten any remaining single-directory wrapper created by archive tools.
    _unwrap_single_root(dest)


def _detect_manifests(workspace: Path) -> Dict[str, Path]:
    manifests: Dict[str, Path] = {}
    for name in ("Cargo.toml", "pyproject.toml", "setup.py", "CMakeLists.txt", "Makefile", "package.json", "go.mod"):
        path = workspace / name
        if path.is_file():
            manifests[name] = path
    return manifests


def _detect_language_artifacts(workspace: Path) -> Tuple[bool, bool, bool, List[Path]]:
    has_python = bool(list(workspace.rglob("*.py")))
    has_rust = any((workspace / "Cargo.toml").is_file() for _ in [0]) or bool(list(workspace.rglob("*.rs")))
    cpp_sources = sorted(p for p in workspace.rglob("*") if p.suffix in {".cpp", ".c", ".h", ".hpp"})
    has_cpp = bool(cpp_sources)
    return has_python, has_rust, has_cpp, cpp_sources


def _cargo_members(workspace: Path) -> List[str]:
    cargo_toml = workspace / "Cargo.toml"
    if not cargo_toml.is_file():
        return []
    try:
        with cargo_toml.open("rb") as fh:
            data = tomllib.load(fh) or {}
    except Exception:
        return []
    members = data.get("workspace", {}).get("members", [])
    if not members and "package" in data:
        members = [data["package"].get("name", "rust_core")]
    return members


def _artifact_type_for_project(
    has_python: bool, has_rust: bool, has_cpp: bool
) -> Tuple[str, List[ToolchainSpec]]:
    toolchains: List[ToolchainSpec] = []
    if has_python:
        toolchains.append(ToolchainSpec(name="CPython", version="3.x"))
    if has_rust:
        toolchains.extend([ToolchainSpec(name="Rust", channel="stable"), ToolchainSpec(name="Cargo")])
    if has_cpp:
        toolchains.extend([ToolchainSpec(name="GCC"), ToolchainSpec(name="Clang")])

    if has_rust and has_cpp and has_python:
        return "tri_polyglot_rust_cpp_python", toolchains
    if has_rust and has_cpp:
        return "hybrid_cpp_rust", toolchains
    if has_rust and has_python:
        return "hybrid_rust_python", toolchains
    if has_cpp and has_python:
        return "hybrid_cpp_python", toolchains
    if has_rust:
        return "pure_rust", toolchains
    if has_cpp:
        return "hybrid_cpp_python", toolchains
    return "pure_python", toolchains


def generate_draft_v3_blueprint(workspace: Path) -> BlueprintV3:
    """Create a Blueprint v3.0.0 ``draft`` from an extracted workspace tree."""
    workspace = Path(workspace).resolve()
    manifests = _detect_manifests(workspace)
    has_python, has_rust, has_cpp, cpp_sources = _detect_language_artifacts(workspace)
    architecture, toolchains = _artifact_type_for_project(has_python, has_rust, has_cpp)

    build_pipeline: List[BuildArtifact] = []
    py_sources = sorted(p.relative_to(workspace) for p in workspace.rglob("*.py") if "/tests/" not in str(p.relative_to(workspace)) and not p.name.startswith("test_"))
    if py_sources:
        build_pipeline.append(
            BuildArtifact(
                id="python_app",
                type=ArtifactType.python_extension,
                source_files=[str(s) for s in py_sources],
                output_path="dist/python_app",
                description="Python application sources",
            )
        )

    if has_rust:
        members = _cargo_members(workspace)
        for member in members:
            build_pipeline.append(
                BuildArtifact(
                    id=f"rust_{member}",
                    type=ArtifactType.cargo_cdylib,
                    source_files=[f"{member}/src/lib.rs"],
                    output_path=f"target/release/lib{member.replace('-', '_')}.so",
                    dependencies=[],
                    description=f"Rust workspace member {member}",
                )
            )
        if not members:
            build_pipeline.append(
                BuildArtifact(
                    id="rust_core",
                    type=ArtifactType.cargo_cdylib,
                    source_files=["src/lib.rs"],
                    output_path="target/release/librust_core.so",
                    description="Rust core crate",
                )
            )

    if has_cpp:
        build_pipeline.append(
            BuildArtifact(
                id="cpp_native",
                type=ArtifactType.shared_library,
                source_files=[str(p.relative_to(workspace)) for p in cpp_sources],
                output_path="dist/libnative.so",
                description="C/C++ native shared library",
            )
        )

    primary_entrypoint = ""
    if py_sources:
        primary_entrypoint = str(py_sources[0])
    elif has_rust:
        primary_entrypoint = "src/main.rs" if (workspace / "src/main.rs").is_file() else "src/lib.rs"

    return BlueprintV3(
        metadata=Metadata(
            schema_version="3.0.0",
            project_name=workspace.name or "uploaded_project",
            status=BlueprintStatus.draft,
            generation_method=GenerationMethod.static_heuristic,
            transferable=False,
            description=f"Auto-generated draft blueprint for {architecture} project",
        ),
        llm_context=LLMContext(
            state=ContextState.raw,
            repository_summary=f"Auto-generated {architecture} draft; run synthesis to enrich context.",
            dependency_graph={},
            compute_hotspots=[],
        ),
        toolchains=toolchains,
        build_pipeline=build_pipeline,
        execution_strategy=ExecutionStrategyV3(
            primary_entrypoint=primary_entrypoint,
            runtime="python3" if has_python else ("cargo" if has_rust else "./binary"),
            working_dir="${WORKSPACE_ROOT}",
        ),
        verification_nodes=[
            VerificationNode(
                node_id="smoke_test",
                command="python3 ${WORKSPACE_ROOT}/" + (str(py_sources[0]) if py_sources else ""),
                expected_exit_code=0,
            )
        ]
        if py_sources
        else [],
    )


def ingest_zip_archive(
    zip_bytes: bytes,
    dest: Path,
    write_blueprint: bool = True,
) -> Tuple[Dict[str, Any], Optional[BlueprintV3]]:
    """Extract a ZIP archive and optionally generate a v3 draft ``blueprint.aero``."""
    extract_zip_safely(zip_bytes, dest)
    reify_missing_modules(dest)
    blueprint: Optional[BlueprintV3] = None
    if write_blueprint:
        blueprint = generate_draft_v3_blueprint(dest)
        write_v3_blueprint(blueprint, dest / "blueprint.aero")
    commands = detect_runnable_commands(dest)
    return {
        "status": "success",
        "extracted_to": str(dest),
        "commands": commands,
        "runnable_commands": commands,
    }, blueprint
