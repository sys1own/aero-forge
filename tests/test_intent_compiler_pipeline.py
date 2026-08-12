"""Unit tests for the Six-Phase IntentCompiler pipeline helpers."""

import tempfile
from pathlib import Path

from aero_forge.builder.intent_compiler import IntentCompiler


def test_his_context_binding():
    compiler = IntentCompiler(llm_client=None)
    classification = {
        "functional_intent": [
            {"symbol_name": "foo", "type": "function"},
            {"symbol_name": "bar", "type": "algorithm"},
        ]
    }
    hctx = compiler._six_phase_bind_context(classification)
    assert hctx.hinv is not None
    assert hctx.dimension == 10000


def test_foge_topology_prefix(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("import lib\n")
    compiler = IntentCompiler(llm_client=None)
    topo = compiler._six_phase_topology_prefix(tmp_path)
    assert topo["encoded"]
    assert "main" in topo["nodes"]
    assert topo["dim"] == 256


def test_adjoint_bootstrap():
    compiler = IntentCompiler(llm_client=None)
    classification = {
        "architecture": "hybrid_rust_python",
        "functional_intent": [
            {"symbol_name": "calc", "type": "algorithm"},
            {"symbol_name": "main", "type": "function"},
        ],
    }
    skeleton = compiler._six_phase_bootstrap_skeleton(classification, {"encoded": False})
    assert skeleton["architecture"] == "hybrid_rust_python"
    assert any(n["node_id"] == "main" for n in skeleton["nodes"])


def test_bounded_completion_prompt_contains_skeleton():
    compiler = IntentCompiler(llm_client=None)
    classification = {
        "architecture": "pure_python",
        "functional_intent": [{"symbol_name": "main", "type": "function"}],
    }
    hctx = compiler._six_phase_bind_context(classification)
    topo = {"encoded": False, "nodes": [], "edges": [], "dim": 0}
    skeleton = compiler._six_phase_bootstrap_skeleton(classification, topo)
    prompt = compiler._six_phase_user_content("test", classification, hctx, topo, skeleton)
    assert "Manifest skeleton" in prompt
    assert "Topological prefix" in prompt


def test_formal_feedback_detects_invalid_boundary(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text('import rust_core\n')
    (tmp_path / "rust_core.rs").write_text('pub fn calc() {}\n')
    compiler = IntentCompiler(llm_client=None)
    bad_manifest = {
        "architecture": "hybrid_rust_python",
        "nodes": [
            {"node_id": "main", "lang": "python", "toolchain": "python", "exports": ["main"]},
            {"node_id": "rust_core", "lang": "rust", "toolchain": "cargo", "exports": ["calc"]},
        ],
        "edges": [{"source": "main", "target": "rust_core", "boundary_type": "C_ABI"}],
        "functional_intent": [{"symbol_name": "main"}],
    }
    feedback = compiler._six_phase_formal_feedback(bad_manifest, tmp_path)
    assert "C_ABI" in feedback or "SHACL" in feedback
