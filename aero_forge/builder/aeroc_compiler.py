"""Python-side wrapper to compile a Blueprint v3 into a workspace.aeroc binary."""

from __future__ import annotations

import base64
import json
import struct
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from aero_forge._native import compile_aeroc
from aero_forge.overlay import OverlayManager

if TYPE_CHECKING:
    from aero_forge.blueprint.schema import BlueprintV3, BuildArtifact


def _encode_source(path: Path, workspace: Path) -> Dict[str, Any]:
    """Read *path* relative to *workspace* and return a base64 source entry."""
    full = workspace / path
    data = full.read_bytes() if full.is_file() else b""
    return {
        "path": str(path.as_posix()).lstrip("/"),
        "content_base64": base64.b64encode(data).decode("ascii"),
    }


def _artifact_instruction(artifact: "BuildArtifact") -> Dict[str, Any] | None:
    """Map a Blueprint v3 artifact to an aeroc instruction."""
    first_src = str(artifact.source_files[0]) if artifact.source_files else ""
    out = str(artifact.output_path) if artifact.output_path else ""

    # Prefer an explicit Cargo.toml when present; otherwise default to the
    # workspace root manifest.
    manifest = next(
        (s for s in artifact.source_files if str(s).endswith("Cargo.toml")),
        "Cargo.toml",
    )
    if artifact.type == "cargo_cdylib":
        return {
            "op": "CARGO_BUILD",
            "manifest_ref": manifest,
            "flags": 0,
        }
    if artifact.type == "python_extension":
        return {
            "op": "PYO3_BIND",
            "src_ref": first_src or manifest,
            "out_ref": out,
        }
    if artifact.type == "custom_cmd":
        # Custom commands are treated as VM bytecode references; the command
        # string itself is stored in the string table by reference.
        return {
            "op": "VM_EXEC",
            "bytecode_ref": 0,
            "mem_limit": 0,
        }
    return None


def compile_blueprint_to_aeroc(
    blueprint: "BlueprintV3",
    output_path: str | Path,
    workspace: str | Path = ".",
) -> str:
    """Compile a Blueprint v3 build contract into a binary ``workspace.aeroc``."""
    workspace = Path(workspace).resolve()
    output_path = Path(output_path)

    # Flush any in-memory overlay edits so the exported IR reflects the real workspace.
    try:
        OverlayManager(workspace).flush_to_workspace(workspace)
    except Exception:
        pass

    nodes = [a.id for a in blueprint.build_pipeline]
    edges = {a.id: a.dependencies for a in blueprint.build_pipeline}

    instructions: List[Dict[str, Any]] = []
    for artifact in blueprint.build_pipeline:
        inst = _artifact_instruction(artifact)
        if inst:
            instructions.append(inst)

    for contract in blueprint.abi_contracts:
        if contract.header_path:
            instructions.append({
                "op": "CABI_CHECK",
                "header_ref": contract.header_path,
                "abi_hash": 0,
            })

    for node in blueprint.verification_nodes:
        instructions.append({
            "op": "UNIT_VERIFY",
            "test_bin_ref": node.command or node.node_id,
            "args": 0,
        })

    instructions.append({"op": "HALT"})

    # Collect every file in the workspace so the binary IR is a complete snapshot.
    exclude = {".aero", ".git", ".aero_core", "target", "__pycache__", "*.egg-info", ".pytest_cache"}
    sources: List[Dict[str, Any]] = []
    seen_paths: set[str] = set()

    def _add_file(path: Path) -> None:
        rel = path.relative_to(workspace).as_posix().lstrip("/")
        if rel in seen_paths or path.resolve() == output_path.resolve():
            return
        if any(p in exclude or p.endswith(".egg-info") or p == "__pycache__" for p in Path(rel).parts):
            return
        if path.is_file():
            seen_paths.add(rel)
            sources.append({
                "path": rel,
                "content_base64": base64.b64encode(path.read_bytes()).decode("ascii"),
            })

    for artifact in blueprint.build_pipeline:
        for src in artifact.source_files:
            _add_file(workspace / src)

    for path in sorted(workspace.rglob("*")):
        _add_file(path)

    # Always include the contract file itself so the container is self-describing.
    blueprint_path = workspace / "blueprint.aero"
    if blueprint_path.is_file():
        _add_file(blueprint_path)

    spec = {
        "nodes": nodes,
        "edges": edges,
        "instructions": instructions,
        "sources": sources,
        "flags": 0,
    }

    return compile_aeroc(json.dumps(spec), str(output_path))


def compile_directory_to_aeroc(
    directory: str | Path,
    output_path: str | Path,
    exclude: set[str] | None = None,
) -> str:
    """Compile an on-disk workspace tree into a binary ``workspace.aeroc`` container.

    The resulting container carries every file under *directory* (minus common
    build/cache directories) in the compressed payload and source map, so it can
    be unpacked later by the native ``aeroc_unpacker``.
    """
    directory = Path(directory).resolve()
    output_path = Path(output_path).resolve()
    exclude = set(exclude or {
        ".aero", ".git", ".aero_core", "target", "__pycache__", "*.egg-info", ".pytest_cache"
    })

    # Flush in-memory overlay edits so the exported IR is a complete snapshot.
    try:
        OverlayManager(directory).flush_to_workspace(directory)
    except Exception:
        pass

    def _keep(path: Path) -> bool:
        if path.resolve() == output_path:
            return False
        rel = path.relative_to(directory)
        parts = rel.parts
        if any(p in exclude or p.endswith(".egg-info") or p == "__pycache__" for p in parts):
            return False
        return path.is_file()

    sources: List[Dict[str, Any]] = []
    for path in sorted(directory.rglob("*")):
        if _keep(path):
            rel = path.relative_to(directory).as_posix().lstrip("/")
            sources.append({
                "path": rel,
                "content_base64": base64.b64encode(path.read_bytes()).decode("ascii"),
            })

    spec = {
        "nodes": ["workspace"],
        "edges": {},
        "instructions": [{"op": "HALT"}],
        "sources": sources,
        "flags": 0,
    }

    return compile_aeroc(json.dumps(spec), str(output_path))


def _find_runner_binary() -> Path:
    """Locate the pre-built ``aeroc-runner`` binary."""
    candidates = [
        Path(__file__).resolve().parents[1] / "_native" / "runner" / "target" / "release" / "aeroc-runner",
        Path(__file__).resolve().parents[1] / "_native" / "target" / "release" / "aeroc-runner",
    ]
    if env_runner := __import__("os").environ.get("AEROC_RUNNER"):
        candidates.insert(0, Path(env_runner))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("aeroc-runner binary not found; build it with cargo build --release in aero_forge/_native/runner")


def bundle_aeroc_executable(
    aeroc_path: str,
    output_path: str,
    runner_path: Optional[str] = None,
) -> str:
    """Create a self-extracting ``workspace.aeroc.bin`` from an ``aeroc-runner`` binary and a ``workspace.aeroc`` container.

    The emitted file is ``[runner] + [aeroc payload] + [AerocTrailerFooter]``.
    """
    runner = Path(runner_path) if runner_path else _find_runner_binary()
    aeroc = Path(aeroc_path).read_bytes()
    runner_bin = runner.read_bytes()

    aeroc_offset = len(runner_bin)
    payload_size = len(aeroc)
    footer = struct.pack("<QQ8s", payload_size, aeroc_offset, b"AEROCBIN")

    output = Path(output_path)
    output.write_bytes(runner_bin + aeroc + footer)
    return str(output.resolve())
