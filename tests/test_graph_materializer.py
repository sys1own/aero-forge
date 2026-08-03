"""Tests for the graph-driven polyglot materializer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aero_forge.blueprint.schema import PolyglotGraphBlueprint
from aero_forge.builder.materializers.graph_materializer import (
    GraphPolyglotMaterializer,
    MaterializationError,
    SystemToolchainRouter,
)
from aero_forge.builder.spec import EngineSpec, function, module, param


def _python_node_spec(node_id: str) -> dict:
    spec = EngineSpec(
        name=node_id,
        root=module(
            children=[
                function(
                    "add",
                    params=[param("a", "int"), param("b", "int")],
                    return_type="int",
                    body=None,
                )
            ]
        ),
    )
    return {"node_id": node_id, "lang": "python", "spec": spec}


def _rust_node_spec(node_id: str) -> dict:
    spec = EngineSpec(
        name=node_id,
        root=module(
            children=[
                function(
                    "add",
                    params=[param("a", "i64"), param("b", "i64")],
                    return_type="i64",
                    body=None,
                )
            ]
        ),
    )
    return {"node_id": node_id, "lang": "rust", "spec": spec, "toolchain": "cargo"}


def test_graph_blueprint_dag_validation() -> None:
    """A cycle-free graph blueprint validates; a cyclic one raises."""
    ok = PolyglotGraphBlueprint(
        project="p",
        nodes=[
            {"node_id": "a", "lang": "python"},
            {"node_id": "b", "lang": "rust"},
        ],
        edges=[{"source": "a", "target": "b", "symbol": "f"}],
    )
    assert ok.project == "p"

    with pytest.raises(ValueError, match="cycle"):
        PolyglotGraphBlueprint(
            project="p",
            nodes=[
                {"node_id": "a", "lang": "python"},
                {"node_id": "b", "lang": "rust"},
            ],
            edges=[
                {"source": "a", "target": "b", "symbol": "f"},
                {"source": "b", "target": "a", "symbol": "g"},
            ],
        )


def test_graph_materializer_emits_files(tmp_path: Path) -> None:
    """A two-node Rust -> Python graph produces artifacts in dependency order."""
    workspace = tmp_path / "graph_workspace"
    spec = {
        "project": "demo",
        "nodes": [
            _rust_node_spec("rust_core"),
            _python_node_spec("py_client"),
        ],
        "edges": [
            {
                "source": "rust_core",
                "target": "py_client",
                "boundary_type": "c_abi",
                "symbol": "add",
                "args": ["int64", "int64"],
                "return_type": "int64",
            }
        ],
    }

    materializer = GraphPolyglotMaterializer(workspace)
    result = materializer.materialize(spec)

    assert result["project"] == "demo"
    assert result["architecture"] == "graph_polyglot"
    assert result["stages"] == [["rust_core"], ["py_client"]]

    rust_dir = workspace / "rust_core"
    assert (rust_dir / "Cargo.toml").exists()
    assert (rust_dir / "src" / "lib.rs").exists()

    python_dir = workspace / "py_client"
    assert (python_dir / "pyproject.toml").exists()
    assert (python_dir / "py_client.py").exists()

    bridge_dir = workspace / "ffi_bridges" / "rust_core_py_client"
    assert bridge_dir.exists()
    assert any((bridge_dir / "bridge.h").exists() for _ in [None])


def test_graph_materializer_cyclic_graph_raises(tmp_path: Path) -> None:
    """A cyclic graph is rejected by the GoI wavefront solver."""
    workspace = tmp_path / "cyclic"
    spec = {
        "nodes": [
            _python_node_spec("a"),
            _python_node_spec("b"),
        ],
        "edges": [
            {"source": "a", "target": "b", "boundary_type": "c_abi", "symbol": "f"},
            {"source": "b", "target": "a", "boundary_type": "c_abi", "symbol": "g"},
        ],
    }
    materializer = GraphPolyglotMaterializer(workspace)
    with pytest.raises(MaterializationError, match="Cyclic dependency"):
        materializer.materialize(spec)


def test_system_toolchain_router_missing_toolchain() -> None:
    """Dispatching an unavailable toolchain raises a clear error."""
    with pytest.raises(RuntimeError, match="not found on PATH"):
        SystemToolchainRouter.dispatch_node_build(
            "x",
            {"toolchain": "nonexistent_toolchain_12345"},
            Path("."),
        )
