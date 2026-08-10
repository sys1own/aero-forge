"""End-to-end integration tests for the graph-driven polyglot build lifecycle.

These tests exercise the complete chain:
    blueprint parsing -> GoI matrix solving -> FFI contract synthesis
    -> plugin emission -> toolchain routing.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from aero_forge.blueprint.schema import PolyglotGraphBlueprint
from aero_forge.builder.emitters.base import BoundaryContract, EmitterRegistry
from aero_forge.builder.language_router import SystemToolchainRouter
from aero_forge.builder.materializers.graph_materializer import (
    GraphPolyglotMaterializer,
    MaterializationError,
)
from aero_forge.builder.spec import EngineSpec, function, module, param


def _engine_spec(node_id: str, lang: str) -> EngineSpec:
    """Return a tiny engine spec with one function."""
    return EngineSpec(
        name=node_id,
        root=module(
            children=[
                function(
                    f"compute_{node_id}",
                    params=[param("a", "int64"), param("b", "int64")],
                    return_type="int64",
                    body=None,
                )
            ]
        ),
    )


def _make_node(node_id: str, lang: str, toolchain: str) -> dict:
    return {
        "node_id": node_id,
        "lang": lang,
        "toolchain": toolchain,
        "spec": _engine_spec(node_id, lang),
    }


def test_tri_polyglot_dag_execution(tmp_path: Path) -> None:
    """Rust -> C++ -> Python is solved as three ordered wavefront stages."""
    workspace = tmp_path / "tri_polyglot"
    spec = {
        "project": "tri_polyglot_demo",
        "nodes": [
            _make_node("rust_core", "rust", "cargo"),
            _make_node("cpp_engine", "cpp", "clang++"),
            _make_node("py_client", "python", "python"),
        ],
        "edges": [
            {
                "source": "rust_core",
                "target": "cpp_engine",
                "boundary_type": "c_abi",
                "symbol": "rust_compute",
                "args": ["int64", "int64"],
                "return_type": "int64",
            },
            {
                "source": "cpp_engine",
                "target": "py_client",
                "boundary_type": "c_abi",
                "symbol": "cpp_compute",
                "args": ["int64", "int64"],
                "return_type": "int64",
            },
        ],
    }

    # Adjacency matrix built by the materializer: target depends on source.
    labels = ["rust_core", "cpp_engine", "py_client"]
    order = {nid: i for i, nid in enumerate(labels)}
    M = np.zeros((3, 3), dtype=np.float64)
    M[order["cpp_engine"], order["rust_core"]] = 1.0
    M[order["py_client"], order["cpp_engine"]] = 1.0

    materializer = GraphPolyglotMaterializer(workspace)
    result = materializer.materialize(spec)

    assert result["project"] == "tri_polyglot_demo"
    assert result["architecture"] == "tri_polyglot_rust_cpp_python"
    assert result["stages"] == [["rust_core"], ["cpp_engine"], ["py_client"]]

    # Rust node artifacts.
    assert (workspace / "rust_core" / "Cargo.toml").exists()
    assert (workspace / "rust_core" / "src" / "lib.rs").exists()

    # C++ node artifacts.
    assert (workspace / "cpp_engine" / "CMakeLists.txt").exists()
    assert (workspace / "cpp_engine" / "cpp_engine.cpp").exists()

    # Python node artifacts.
    assert (workspace / "py_client" / "pyproject.toml").exists()
    assert (workspace / "py_client" / "py_client.py").exists()

    # FFI bridges were emitted for every edge.
    bridge_1 = workspace / "ffi_bridges" / "rust_core_cpp_engine"
    bridge_2 = workspace / "ffi_bridges" / "cpp_engine_py_client"
    assert bridge_1.exists()
    assert bridge_2.exists()

    for bridge_dir in (bridge_1, bridge_2):
        header = bridge_dir / "bridge.h"
        source = bridge_dir / "bridge.c"
        loader = bridge_dir / "loader.py"
        assert header.exists()
        assert source.exists()
        assert loader.exists()
        header_text = header.read_text(encoding="utf-8")
        loader_text = loader.read_text(encoding="utf-8")
        assert "extern \"C\"" in header_text
        assert "#ifndef" in header_text
        assert "ctypes.CDLL" in loader_text


def test_cyclic_dependency_guard(tmp_path: Path) -> None:
    """A cyclic graph is caught by the GoI singularity check before disk writes."""
    workspace = tmp_path / "cyclic"
    spec = {
        "nodes": [
            _make_node("a", "python", "python"),
            _make_node("b", "python", "python"),
        ],
        "edges": [
            {
                "source": "a",
                "target": "b",
                "boundary_type": "c_abi",
                "symbol": "f",
                "args": [],
                "return_type": "",
            },
            {
                "source": "b",
                "target": "a",
                "boundary_type": "c_abi",
                "symbol": "g",
                "args": [],
                "return_type": "",
            },
        ],
    }

    materializer = GraphPolyglotMaterializer(workspace)
    with pytest.raises(MaterializationError, match="Cyclic dependency"):
        materializer.materialize(spec)

    assert not workspace.exists() or not any(workspace.iterdir())


def test_blueprint_graph_dag_validation() -> None:
    """``PolyglotGraphBlueprint`` rejects cycles and dangling edge endpoints."""
    PolyglotGraphBlueprint(
        project="ok",
        nodes=[
            {"node_id": "rust", "lang": "rust"},
            {"node_id": "py", "lang": "python"},
        ],
        edges=[{"source": "rust", "target": "py", "symbol": "f"}],
    )

    with pytest.raises(ValueError, match="cycle"):
        PolyglotGraphBlueprint(
            project="bad",
            nodes=[
                {"node_id": "rust", "lang": "rust"},
                {"node_id": "py", "lang": "python"},
            ],
            edges=[
                {"source": "rust", "target": "py", "symbol": "f"},
                {"source": "py", "target": "rust", "symbol": "g"},
            ],
        )

    with pytest.raises(ValueError, match="unknown"):
        PolyglotGraphBlueprint(
            project="dangling",
            nodes=[{"node_id": "rust", "lang": "rust"}],
            edges=[{"source": "rust", "target": "missing", "symbol": "f"}],
        )


def test_plugin_registry_all_runtimes() -> None:
    """``EmitterRegistry`` exposes plugins for every supported runtime."""
    registry = EmitterRegistry.get_instance()
    for lang in ("python", "rust", "cpp", "go", "csharp", "java"):
        plugin = registry.get_plugin(lang)
        assert plugin is not None
        assert plugin.descriptor.language_id == lang

    python_plugin = registry.get_plugin("python")
    assert BoundaryContract.C_ABI in python_plugin.descriptor.supported_boundaries
    assert BoundaryContract.PYO3_MATURIN in python_plugin.descriptor.supported_boundaries

    go_plugin = registry.get_plugin("go")
    assert BoundaryContract.CGO in go_plugin.descriptor.supported_boundaries

    csharp_plugin = registry.get_plugin("csharp")
    assert BoundaryContract.PINVOKE in csharp_plugin.descriptor.supported_boundaries

    java_plugin = registry.get_plugin("java")
    assert BoundaryContract.JNI in java_plugin.descriptor.supported_boundaries


def test_system_toolchain_router_command_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """``SystemToolchainRouter`` generates correct build flags for native toolchains."""
    monkeypatch.setattr(
        "aero_forge.builder.language_router.shutil.which",
        lambda name: f"/usr/bin/{name}",
    )
    ws = Path("/tmp/workspace")

    cargo_cmd = SystemToolchainRouter._build_command(
        "cargo", "mylib", [], ["--verbose"], ws
    )
    assert cargo_cmd == ["/usr/bin/cargo", "build", "--release", "--verbose"]

    gcc_cmd = SystemToolchainRouter._build_command(
        "gcc", "mylib", ["lib.c"], ["-O2"], ws
    )
    assert "/usr/bin/gcc" in gcc_cmd
    assert "-shared" in gcc_cmd
    assert "-fPIC" in gcc_cmd
    assert "lib.c" in gcc_cmd
    assert "-O2" in gcc_cmd

    go_cmd = SystemToolchainRouter._build_command(
        "go", "mylib", [], [], ws
    )
    assert "/usr/bin/go" in go_cmd
    assert "build" in go_cmd
    assert "-buildmode=c-shared" in go_cmd

    dotnet_cmd = SystemToolchainRouter._build_command(
        "dotnet", "mylib", [], [], ws
    )
    assert dotnet_cmd == ["/usr/bin/dotnet", "build"]
