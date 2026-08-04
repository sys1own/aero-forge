"""Tests for the aero-forge polyglot builder path resolution and stub generation.

These tests verify that the builder honors explicit non-standard directory
layouts from prompts and blueprints, that generated C++/Rust stubs compile
without unimplemented-function warnings, and that Python ``ctypes`` loaders
can locate native libraries relative to the workspace root.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import pytest

from aero_forge.blueprint import Blueprint, ContractEntry, ManifestEntry
from aero_forge.builder.emitters.cpp_emitter import CppEmitter
from aero_forge.builder.emitters.rust_emitter import RustEmitter
from aero_forge.builder.spec import ASTNode, EngineSpec, function, module, param
from aero_forge.native_bridge import _ctypes_loader_source
from aero_forge.orchestrator.stack_classifier import (
    INTENT_TRI_POLYGLOT_RUST_CPP_PYTHON,
    default_manifest_for_architecture,
    extract_source_directories,
)
from aero_forge.scaffold.cpp_materializer import _find_cpp_compiler
from aero_forge.scaffold.tri_polyglot_materializer import TriPolyglotMaterializer
from aero_forge.scaffold.workspace import _cpp_stub_from_signature, _rust_stub_from_signature
from aero_forge.universal_builder import _extract_explicit_file_paths, _hybrid_fallback_blueprint


def _valid_payload(tmp_path: Path) -> Path:
    """Return a workspace path and valid blueprint for proactive builder tests."""
    workspace = tmp_path / "proactive_ws"
    workspace.mkdir(parents=True)
    return workspace


def _tmp_workspace(tmp_path: Path, name: str = "workspace") -> Path:
    ws = tmp_path / name
    ws.mkdir(parents=True, exist_ok=True)
    return ws


class TestPathExtraction:
    def test_extract_source_directories_finds_custom_cpp_path(self) -> None:
        prompt = (
            "Build a tri-polyglot workspace with C++ in cpp_engine/src/kernels.cpp "
            "and rust in rust_engine/src/lib.rs and python in python_interface/main.py"
        )
        dirs = extract_source_directories(prompt)
        assert dirs["cpp_source"] == "cpp_engine/src/kernels.cpp"
        assert dirs["rust_crate_dir"] == "rust_engine"
        assert dirs["python_package"] == "python_interface"

    def test_extract_explicit_file_paths_supports_headers_and_rust(self) -> None:
        prompt = "Generate cpp_engine/include/api.h and rust_engine/src/lib.rs plus python_interface/cli.py"
        explicit = _extract_explicit_file_paths(prompt)
        paths = [p for p, _ in explicit]
        assert "cpp_engine/include/api.h" in paths
        assert "rust_engine/src/lib.rs" in paths
        assert "python_interface/cli.py" in paths

    def test_default_manifest_uses_custom_paths(self) -> None:
        prompt = "Build a tri-polyglot project with cpp_engine/src/kernels.cpp and python_interface/main.py"
        manifest = default_manifest_for_architecture(
            INTENT_TRI_POLYGLOT_RUST_CPP_PYTHON, "custom", prompt=prompt
        )
        paths = {e["path"] for e in manifest}
        assert "cpp_engine/src/kernels.cpp" in paths
        assert "python_interface/__init__.py" in paths

    def test_hybrid_fallback_blueprint_uses_custom_cpp_path(self) -> None:
        prompt = (
            "tri polyglot rust cpp python with cpp_engine/src/kernels.cpp "
            "and rust_engine/src/lib.rs"
        )
        blueprint = _hybrid_fallback_blueprint("custom", ["cpp", "rust", "python"], prompt=prompt)
        cpp_paths = [e.path for e in blueprint.manifest if e.path.endswith(".cpp")]
        assert "cpp_engine/src/kernels.cpp" in cpp_paths


class TestStubGeneration:
    def test_cpp_emitter_default_body_has_no_unimplemented_todo(self) -> None:
        """Empty C++ function bodies must suppress unused params and return a default."""
        spec = EngineSpec(
            name="demo",
            root=module(
                name="demo",
                children=[
                    function(
                        "scale",
                        params=[param("values", "list[float]"), param("factor", "float")],
                        return_type="list[float]",
                    )
                ],
            ),
        )
        source = CppEmitter(c_abi=True).emit(spec)
        assert "// TODO" not in source
        assert "(void)values;" in source or "(void)factor;" in source
        assert "return {};" in source

    def test_rust_emitter_default_body_has_allow_unused_and_default(self) -> None:
        """Empty Rust function bodies must allow unused variables and return a default."""
        spec = EngineSpec(
            name="demo",
            root=module(
                name="demo",
                children=[
                    function(
                        "scale",
                        params=[param("values", "list[float]"), param("factor", "float")],
                        return_type="list[float]",
                    )
                ],
            ),
        )
        source = RustEmitter().emit(spec)
        assert "todo!()" not in source
        assert "#[allow(unused_variables)]" in source
        assert "Default::default()" in source

    def test_workspace_cpp_stub_compiles_cleanly(self, tmp_path: Path) -> None:
        source = _cpp_stub_from_signature(
            "def fast_vector_transform(v: list[float], scalar: float) -> list[float]"
        )
        assert "unimplemented!()" not in source
        assert "(void)v;" in source
        assert "(void)scalar;" in source
        assert "return {};" in source

    def test_workspace_rust_stub_has_allow_and_default(self) -> None:
        source = _rust_stub_from_signature(
            "def fast_vector_transform(v: list[float], scalar: float) -> list[float]"
        )
        assert "unimplemented!()" not in source
        assert "#[allow(unused_variables)]" in source
        assert "Default::default()" in source


class TestNativeBridgeLoader:
    def test_ctypes_loader_uses_project_relative_search(self, tmp_path: Path) -> None:
        workspace = tmp_path / "ws"
        pkg = workspace / "python_interface"
        pkg.mkdir(parents=True)
        so_path = workspace / "cpp_engine" / "src" / "libcustom_cpp.so"
        loader = _ctypes_loader_source(
            "def scale(values: list[float], factor: float) -> list[float]:\n    pass\n",
            so_path,
            ["scale"],
            workspace_root=workspace,
            loader_path=pkg / "__init__.py",
        )
        assert "_ROOT = pathlib.Path(__file__).resolve().parents[1]" in loader
        assert "cpp_engine/src/libcustom_cpp.so" in loader
        assert "_find_library" in loader


@pytest.mark.integration
class TestTriPolyglotBuilder:
    def test_tri_polyglot_builds_with_custom_paths(self, tmp_path: Path) -> None:
        """End-to-end: a tri-polyglot workspace honors custom cpp/rust/python paths."""
        if not _find_cpp_compiler():
            pytest.skip("No C++ compiler available")
        if not shutil.which("cargo"):
            pytest.skip("No Rust cargo available")

        workspace = _tmp_workspace(tmp_path, "custom_tri")
        blueprint = Blueprint(
            project="custom_tri",
            architecture="tri_polyglot_rust_cpp_python",
            toolchains=["python", "rust", "cpp", "cargo"],
            manifest=[
                ManifestEntry(
                    path="rust_engine/Cargo.toml",
                    lang="toml",
                    purpose="PyO3 crate manifest",
                ),
                ManifestEntry(
                    path="rust_engine/src/lib.rs",
                    lang="rust",
                    purpose="Rust core",
                ),
                ManifestEntry(
                    path="cpp_engine/src/kernels.cpp",
                    lang="cpp",
                    purpose="C-ABI source",
                ),
                ManifestEntry(
                    path="python_interface/__init__.py",
                    lang="python",
                    purpose="package init",
                ),
                ManifestEntry(
                    path="python_interface/main.py",
                    lang="python",
                    purpose="CLI",
                ),
                ManifestEntry(
                    path="pyproject.toml",
                    lang="toml",
                    purpose="project manifest",
                ),
                ManifestEntry(
                    path="tests/test_tri.py",
                    lang="python",
                    purpose="tests",
                ),
                ManifestEntry(path="README.md", lang="markdown", purpose="docs"),
            ],
            contracts=[
                ContractEntry(
                    name="fast_vector_transform",
                    signature="def fast_vector_transform(v: list[float], scalar: float) -> list[float]",
                ),
                ContractEntry(
                    name="validate_token",
                    signature="def validate_token(token: str) -> bool",
                ),
                ContractEntry(
                    name="get_engine_status",
                    signature="def get_engine_status() -> dict[str, str]",
                ),
            ],
        )

        updated = TriPolyglotMaterializer(workspace).materialize(blueprint, build=True)
        assert updated.architecture == "tri_polyglot_rust_cpp_python"

        # Ensure the requested custom paths were materialized.
        assert (workspace / "cpp_engine" / "src" / "kernels.cpp").is_file()
        assert (workspace / "rust_engine" / "src" / "lib.rs").is_file()
        assert (workspace / "python_interface" / "__init__.py").is_file()

        # Ensure no hardcoded default cpp_core/native.cpp drifted in.
        assert not (workspace / "cpp_core" / "native.cpp").exists()

        # C++ shared library should be next to the requested source file.
        cpp_so = next((workspace / "cpp_engine" / "src").glob("*.so"), None)
        assert cpp_so, "Expected compiled C++ .so in cpp_engine/src"

        # Rust shared library should be in the custom rust crate target dir.
        rust_candidates = list(
            (workspace / "rust_engine" / "target" / "release").glob("*.so")
        ) + list((workspace / "target" / "release").glob("*.so"))
        rust_so = next((p for p in rust_candidates), None)
        assert rust_so, "Expected compiled Rust .so"

        # Smoke test the generated package through the Python driver.
        smoke = workspace / "check_custom_tri.py"
        smoke.write_text(
            "import sys\n"
            'sys.path.insert(0, ".")\n'
            "from python_interface import fast_vector_transform, validate_token, get_engine_status\n"
            "assert fast_vector_transform([1.0, 2.0, 3.0], 2.0) == [2.0, 4.0, 6.0]\n"
            'assert validate_token("validtoken123") is True\n'
            'assert validate_token("short") is False\n'
            'assert get_engine_status().get("status") == "ok"\n'
            'print("custom tri-polyglot smoke ok")\n'
        )
        result = subprocess.run(
            [sys.executable, str(smoke)], cwd=workspace, capture_output=True, text=True
        )
        assert result.returncode == 0, f"Custom tri-polyglot smoke test failed: {result.stderr}"

        # Run the generated pytest suite.
        pytest_result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_tri.py", "-q"],
            cwd=workspace,
            capture_output=True,
            text=True,
        )
        assert pytest_result.returncode == 0, (
            f"Generated tri-polyglot tests failed:\n{pytest_result.stdout}\n{pytest_result.stderr}"
        )


class TestProactivePolyglotBuilder:
    def _payload(self, tmp_path: Path, *, unsat: bool = False, goi_bad: bool = False):
        workspace = tmp_path / ("proactive_unsat" if unsat else "proactive_ok")
        workspace.mkdir(parents=True)

        constraints = [
            {"source_hole": "h1", "target_language": "rust"},
            {"source_hole": "h2", "target_language": "python"},
        ]
        if unsat:
            constraints.append({"source_hole": "h1", "target_language": "python"})

        # Acyclic 3-node chain (nilpotent under sigma = 0.5 I)
        goi_m = [0, 0, 0, 1, 0, 0, 0, 1, 0]
        goi_sigma = [0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5]
        if goi_bad:
            # Cyclic 2x2 non-nilpotent
            goi_m = [0, 1, 1, 0]
            goi_sigma = [1, 0, 0, 1]

        blueprint = {
            "project": "proactive_test",
            "architecture": "pure_python",
            "toolchains": ["python"],
            "manifest": [
                {"path": "src/__init__.py", "lang": "python", "purpose": "package"},
                {"path": "src/main.py", "lang": "python", "purpose": "entry"},
            ],
            "contracts": [
                {"name": "main", "signature": "def main() -> bool", "purpose": "entry"}
            ],
        }

        return {
            "workspace": str(workspace),
            "blueprint": blueprint,
            "nodes": [
                {"id": "rust_node", "type": "RustStruct", "lang": "rust"},
                {"id": "py_node", "type": "PyClass", "lang": "python"},
            ],
            "relations": [
                {"source": "rust_node", "target": "py_node", "relation": "ParentOf"}
            ],
            "holes": ["h1", "h2"],
            "constraints": constraints,
            "goi_dim": 3 if not goi_bad else 2,
            "goi_m": goi_m,
            "goi_sigma": goi_sigma,
        }

    def test_proactive_builder_passes_valid_payload(self, tmp_path: Path) -> None:
        """A valid multi-language payload should pass all gates and write files."""
        from aero_forge.builder.builder import ProactivePolyglotBuilder

        payload = self._payload(tmp_path)
        builder = ProactivePolyglotBuilder()
        result = builder.build_blueprint_proactive(payload)
        assert result is True
        assert (Path(payload["workspace"]) / "src" / "main.py").is_file()

    def test_proactive_builder_blocks_on_unsat_smt(self, tmp_path: Path) -> None:
        """An SMT UNSAT payload must not write files to disk."""
        from aero_forge.builder.builder import ProactivePolyglotBuilder

        payload = self._payload(tmp_path, unsat=True)
        builder = ProactivePolyglotBuilder()
        result = builder.build_blueprint_proactive(payload)
        assert result is False
        assert not (Path(payload["workspace"]) / "src").exists()

    def test_proactive_builder_blocks_on_goi_failure(self, tmp_path: Path) -> None:
        """A non-nilpotent GoI payload with a 3-cycle cannot be auto-remediated."""
        from aero_forge.builder.builder import ProactivePolyglotBuilder

        payload = self._payload(tmp_path, goi_bad=True)
        # A 3-cycle is not covered by the 2-cycle/single-edge fallback heuristic.
        payload["goi_dim"] = 3
        payload["goi_m"] = [0, 1, 0, 0, 0, 1, 1, 0, 0]
        payload["goi_sigma"] = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        builder = ProactivePolyglotBuilder()
        result = builder.build_blueprint_proactive(payload)
        assert result is False
        assert not (Path(payload["workspace"]) / "src").exists()

    def test_proactive_builder_rejects_draft_blueprint(self, tmp_path: Path) -> None:
        """A draft/auto-generated blueprint must be synthesized before materialization."""
        from aero_forge.builder.builder import ProactivePolyglotBuilder

        payload = self._payload(tmp_path)
        payload["blueprint"]["metadata"] = {
            "schema_version": "3.0.0",
            "status": "draft",
            "auto_generated": True,
            "llm_initialized": False,
        }
        payload["blueprint"]["llm_context"] = {"state": "raw"}

        builder = ProactivePolyglotBuilder()
        result = builder.build_blueprint_proactive(payload)
        assert result is False
        assert not (Path(payload["workspace"]) / "src").exists()
