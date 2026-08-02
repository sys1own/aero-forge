"""Integration stress-test matrix for the proactive polyglot builder.

Covers the five canonical cross-language inter-op scenarios and the three-tier
fallback remediation strategy, asserting that the builder reaches a 100%
first-pass materialization rate for valid or auto-remediable payloads.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

from aero_forge.builder.builder import ProactivePolyglotBuilder
from aero_forge.builder.fallback_manager import FallbackManager


def _workspace(tmp_path: Path, name: str) -> Path:
    ws = tmp_path / name
    ws.mkdir(parents=True, exist_ok=True)
    return ws


def _base_blueprint(project: str = "matrix_test") -> Dict[str, Any]:
    return {
        "project": project,
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


def _payload(
    tmp_path: Path,
    name: str,
    nodes: list,
    relations: list,
    holes: list,
    constraints: list,
    goi_dim: int,
    goi_m: list,
    goi_sigma: list,
) -> Dict[str, Any]:
    return {
        "workspace": str(_workspace(tmp_path, name)),
        "blueprint": _base_blueprint(name),
        "nodes": nodes,
        "relations": relations,
        "holes": holes,
        "constraints": constraints,
        "goi_dim": goi_dim,
        "goi_m": goi_m,
        "goi_sigma": goi_sigma,
    }


class TestProactiveBuilderMatrix:
    def test_tc_r2c_01_rust_to_cpp_coroutine_ffi(self, tmp_path: Path) -> None:
        """Rust async context calls a C++ coroutine over FFI; GoI must be nilpotent."""
        payload = _payload(
            tmp_path,
            "tc_r2c_01",
            nodes=[
                {"id": "rust_stream", "type": "AsyncContext", "lang": "rust"},
                {"id": "cpp_coroutine", "type": "CppClass", "lang": "cpp"},
            ],
            relations=[
                {
                    "source": "rust_stream",
                    "target": "cpp_coroutine",
                    "relation": "CallsFFI",
                }
            ],
            holes=["h1"],
            constraints=[{"source_hole": "h1", "target_language": "rust"}],
            goi_dim=3,
            goi_m=[0, 0, 0, 1, 0, 0, 0, 1, 0],
            goi_sigma=[0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5],
        )
        builder = ProactivePolyglotBuilder()
        assert builder.build_blueprint_proactive(payload) is True
        assert (Path(payload["workspace"]) / "src" / "main.py").is_file()

    def test_tc_c2p_02_cpp_to_python_raw_pointer_dpo(self, tmp_path: Path) -> None:
        """Rust -> C++ raw string FFI triggers DPO boundary wrapper injection."""
        payload = _payload(
            tmp_path,
            "tc_c2p_02",
            nodes=[
                {"id": "rust_caller", "type": "RustStruct", "lang": "rust", "props": {"arg_type": "&str"}},
                {"id": "cpp_target", "type": "CppClass", "lang": "cpp"},
            ],
            relations=[
                {
                    "source": "rust_caller",
                    "target": "cpp_target",
                    "relation": "CallsFFI",
                }
            ],
            holes=["h1"],
            constraints=[
                {"source_hole": "h1", "target_language": "python"},
                {"source_hole": "h1", "target_language": "cpp"},
            ],
            goi_dim=2,
            goi_m=[0, 0, 0, 0],
            goi_sigma=[0, 0, 0, 0],
        )
        # Pre-DPO: no boundary node.
        builder = ProactivePolyglotBuilder()
        builder._ingest_hin(payload)
        assert not any(
            builder.hin_engine.graph.nodes[n].get("node_type") == "FFIBoundary"
            for n in builder.hin_engine.graph.nodes
        )
        # After DPO, a boundary should be injected because the rust->cpp edge is
        # treated as a raw string FFI call.
        builder.hin_engine.apply_dpo_rewrite_ffi_strings()
        assert any(
            builder.hin_engine.graph.nodes[n].get("node_type") == "FFIBoundary"
            for n in builder.hin_engine.graph.nodes
        )
        # SMT is intentionally conflicting above; the fallback degrades h1 to a
        # safe Python byte-buffer and the build succeeds.
        payload["constraints"] = [
            {"source_hole": "h1", "target_language": "python"},
            {
                "source_hole": "h1",
                "ffi_layout": {
                    "struct": "Packet",
                    "field": "data",
                    "rust_offset": 0,
                    "cpp_offset": 4,
                    "rust_align": 8,
                    "cpp_align": 8,
                },
            },
        ]
        assert ProactivePolyglotBuilder().build_blueprint_proactive(payload) is True

    def test_tc_p2r_03_python_to_rust_async_trait_bounds(self, tmp_path: Path) -> None:
        """Python asyncio target resolves to Rust static trait bounds via SMT."""
        payload = _payload(
            tmp_path,
            "tc_p2r_03",
            nodes=[
                {"id": "py_async", "type": "AsyncContext", "lang": "python"},
                {"id": "rust_target", "type": "RustStruct", "lang": "rust"},
            ],
            relations=[
                {"source": "py_async", "target": "rust_target", "relation": "BindsTo"}
            ],
            holes=["h1"],
            constraints=[{"source_hole": "h1", "target_language": "rust"}],
            goi_dim=2,
            goi_m=[0, 0, 1, 0],
            goi_sigma=[0.3, 0, 0, 0.3],
        )
        from aero_forge.precision_shield import SMTASTEngine

        result = SMTASTEngine().solve_ast_sketch_holes(payload["holes"], payload["constraints"])
        assert result["h1"] == "RustType"
        assert ProactivePolyglotBuilder().build_blueprint_proactive(payload) is True

    def test_tc_mem_04_shared_memory_layout_equality(self, tmp_path: Path) -> None:
        """Rust and C++ structs share memory when FFI offsets/aligns match."""
        payload = _payload(
            tmp_path,
            "tc_mem_04",
            nodes=[
                {"id": "rust_struct", "type": "RustStruct", "lang": "rust"},
                {"id": "cpp_struct", "type": "CppClass", "lang": "cpp"},
            ],
            relations=[
                {
                    "source": "rust_struct",
                    "target": "cpp_struct",
                    "relation": "TransfersOwnershipTo",
                }
            ],
            holes=["h1"],
            constraints=[
                {"source_hole": "h1", "target_language": "rust"},
                {
                    "source_hole": "h1",
                    "ffi_layout": {
                        "struct": "SharedMem",
                        "field": "value",
                        "rust_offset": 0,
                        "cpp_offset": 0,
                        "rust_align": 8,
                        "cpp_align": 8,
                    },
                },
            ],
            goi_dim=2,
            goi_m=[0, 0, 0, 0],
            goi_sigma=[0, 0, 0, 0],
        )
        assert ProactivePolyglotBuilder().build_blueprint_proactive(payload) is True

    def test_tc_cyc_05_cyclic_import_visibility(self, tmp_path: Path) -> None:
        """Cross-language cyclic imports are SAT when all symbols are visible."""
        payload = _payload(
            tmp_path,
            "tc_cyc_05",
            nodes=[
                {"id": "py_mod", "type": "PyClass", "lang": "python"},
                {"id": "rs_mod", "type": "RustStruct", "lang": "rust"},
            ],
            relations=[
                {"source": "py_mod", "target": "rs_mod", "relation": "ImportsSymbol"},
                {"source": "rs_mod", "target": "py_mod", "relation": "ImportsSymbol"},
            ],
            holes=["h1"],
            constraints=[
                {"source_hole": "h1", "target_language": "python"},
                {
                    "imports": {
                        "module": "bridge",
                        "symbols": ["init", "step"],
                        "visible": True,
                    }
                },
            ],
            goi_dim=2,
            goi_m=[0, 0, 0, 0],
            goi_sigma=[0, 0, 0, 0],
        )
        from aero_forge.precision_shield import SMTASTEngine

        result = SMTASTEngine().solve_ast_sketch_holes(payload["holes"], payload["constraints"])
        assert result["h1"] == "PyType"
        assert ProactivePolyglotBuilder().build_blueprint_proactive(payload) is True


class TestFallbackStrategy:
    def test_level_1_safe_type_degradation(self, tmp_path: Path) -> None:
        """SMT UNSAT on FFI layout is auto-remediated to safe byte-buffer."""
        payload = {
            "workspace": str(_workspace(tmp_path, "level1")),
            "blueprint": _base_blueprint("level1"),
            "nodes": [{"id": "n1", "type": "RustStruct", "lang": "rust"}],
            "relations": [],
            "holes": ["h1"],
            "constraints": [
                {"source_hole": "h1", "target_language": "rust"},
                {
                    "source_hole": "h1",
                    "ffi_layout": {
                        "struct": "Buf",
                        "field": "ptr",
                        "rust_offset": 0,
                        "cpp_offset": 8,
                        "rust_align": 8,
                        "cpp_align": 8,
                    },
                },
            ],
            "goi_dim": 2,
            "goi_m": [0, 0, 0, 0],
            "goi_sigma": [0, 0, 0, 0],
        }
        # First attempt fails because FFI offsets mismatch.
        from aero_forge.precision_shield import SMTASTEngine

        with pytest.raises(ValueError):
            SMTASTEngine().solve_ast_sketch_holes(payload["holes"], payload["constraints"])

        # Fallback manager degrades the raw pointer contract.
        manager = FallbackManager()
        success, remediated = manager.remediate_smt_unsat(payload, "alignment mismatch")
        assert success is True
        assert manager.last_level == 1
        assert any("SerializationBuffer" in str(p) for p in manager.patches)

        # Builder now succeeds with the remediated payload.
        builder = ProactivePolyglotBuilder()
        assert builder.build_blueprint_proactive(remediated) is True

    def test_level_2_structural_pruning(self, tmp_path: Path) -> None:
        """GoI non-nilpotency is resolved by pruning cyclic async channels."""
        payload = {
            "workspace": str(_workspace(tmp_path, "level2")),
            "blueprint": _base_blueprint("level2"),
            "nodes": [{"id": "n1", "type": "RustStruct", "lang": "rust"}],
            "relations": [],
            "holes": ["h1"],
            "constraints": [{"source_hole": "h1", "target_language": "rust"}],
            "goi_dim": 2,
            "goi_m": [0, 1, 1, 0],
            "goi_sigma": [1, 0, 0, 1],
        }
        manager = FallbackManager()
        success, remediated = manager.remediate_goi_non_nilpotent(payload)
        assert success is True
        assert manager.last_level == 2
        assert any("DeadlockFreeChannel" in str(p) for p in manager.patches)

        # After pruning, the builder should pass GoI verification.
        builder = ProactivePolyglotBuilder()
        assert builder.build_blueprint_proactive(remediated) is True

    def test_level_3_business_logic_abort(self, tmp_path: Path) -> None:
        """SMT UNSAT on fundamental business logic aborts before disk write."""
        payload = {
            "workspace": str(_workspace(tmp_path, "level3")),
            "blueprint": _base_blueprint("level3"),
            "nodes": [{"id": "n1", "type": "RustStruct", "lang": "rust"}],
            "relations": [],
            "holes": ["h1"],
            "constraints": [
                {"source_hole": "h1", "target_language": "rust"},
                {"source_hole": "h1", "target_language": "python"},
            ],
            "goi_dim": 2,
            "goi_m": [0, 0, 0, 0],
            "goi_sigma": [0, 0, 0, 0],
        }
        result = ProactivePolyglotBuilder().build_blueprint_proactive(payload)
        assert result is False
        assert not (Path(payload["workspace"]) / "src").exists()


@pytest.mark.integration
class TestProactiveBuilderCompilation:
    def test_tri_polyglot_first_pass_compilation(self, tmp_path: Path) -> None:
        """A valid tri-polyglot payload compiles Rust and C++ on first pass."""
        if not shutil.which("cargo"):
            pytest.skip("cargo not available")
        if not _find_cpp():
            pytest.skip("no C++ compiler available")

        ws = _workspace(tmp_path, "tri_compile")
        payload = {
            "workspace": str(ws),
            "blueprint": {
                "project": "tri_compile",
                "architecture": "tri_polyglot_rust_cpp_python",
                "toolchains": ["python", "rust", "cpp", "cargo"],
                "manifest": [
                    {"path": "rust_engine/Cargo.toml", "lang": "toml"},
                    {"path": "rust_engine/src/lib.rs", "lang": "rust"},
                    {"path": "cpp_engine/src/kernels.cpp", "lang": "cpp"},
                    {"path": "cpp_engine/include/kernels.h", "lang": "cpp"},
                    {"path": "python_interface/__init__.py", "lang": "python"},
                    {"path": "python_interface/main.py", "lang": "python"},
                    {"path": "tests/test_orchestration.py", "lang": "python"},
                ],
                "contracts": [
                    {
                        "name": "run_scheduler",
                        "signature": "def run_scheduler(task_descrs: list[str]) -> list[int]",
                    },
                    {
                        "name": "multiply_matrices",
                        "signature": "def multiply_matrices(a: list[float], b: list[float], rows: int, cols: int, inner: int) -> list[float]",
                    },
                ],
            },
            "nodes": [
                {"id": "rust_scheduler", "type": "RustStruct", "lang": "rust"},
                {"id": "cpp_kernels", "type": "CppClass", "lang": "cpp"},
                {"id": "py_orchestrator", "type": "PyClass", "lang": "python"},
            ],
            "relations": [
                {"source": "rust_scheduler", "target": "cpp_kernels", "relation": "CallsFFI"},
                {"source": "py_orchestrator", "target": "rust_scheduler", "relation": "BindsTo"},
            ],
            "holes": ["h1"],
            "constraints": [{"source_hole": "h1", "target_language": "rust"}],
            "goi_dim": 3,
            "goi_m": [0, 0, 0, 1, 0, 0, 0, 1, 0],
            "goi_sigma": [0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5],
        }
        assert ProactivePolyglotBuilder().build_blueprint_proactive(payload) is True

        # C++ shared library must compile.
        cpp_src = ws / "cpp_engine" / "src" / "kernels.cpp"
        cpp_so = ws / "cpp_engine" / "src" / "libkernels.so"
        compiler = _find_cpp()
        result = subprocess.run(
            [compiler, "-O2", "-shared", "-fPIC", "-I.", "-o", str(cpp_so), str(cpp_src)],
            cwd=ws,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert cpp_so.is_file()

        # Rust crate must compile.
        result = subprocess.run(
            ["cargo", "build", "--release"],
            cwd=ws,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        rust_so = next((ws / "target" / "release").glob("*.so"), None)
        assert rust_so, "no Rust .so produced"


def _find_cpp() -> str | None:
    for cmd in ("g++", "clang++", "c++"):
        if shutil.which(cmd):
            return cmd
    return None
