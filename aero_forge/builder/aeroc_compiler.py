"""Python-side wrapper to compile a Blueprint v3 into a workspace.aeroc binary."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

from aero_forge._native import compile_aeroc

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

    sources: List[Dict[str, Any]] = []
    seen_paths = set()
    for artifact in blueprint.build_pipeline:
        for src in artifact.source_files:
            path = Path(src)
            if path.as_posix() not in seen_paths:
                seen_paths.add(path.as_posix())
                sources.append(_encode_source(path, workspace))

    spec = {
        "nodes": nodes,
        "edges": edges,
        "instructions": instructions,
        "sources": sources,
        "flags": 0,
    }

    return compile_aeroc(json.dumps(spec), str(output_path))
