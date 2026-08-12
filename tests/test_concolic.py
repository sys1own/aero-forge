"""Unit tests for concolic manifest verification with Z3."""

import pytest

from aero_forge.builder.concolic import (
    ConcolicManifestVerifier,
    ConcolicResult,
    verify_manifest,
)


def _valid_manifest() -> dict:
    return {
        "architecture": "pure_python",
        "nodes": [
            {
                "node_id": "main",
                "lang": "python",
                "toolchain": "python",
                "exports": ["main"],
            }
        ],
        "edges": [],
        "functional_intent": [{"symbol_name": "main", "type": "function"}],
    }


def test_valid_manifest_is_sat():
    result = verify_manifest(_valid_manifest())
    assert result.satisfiable
    assert not result.unsat_core


def test_invalid_toolchain_is_unsat():
    manifest = _valid_manifest()
    manifest["nodes"][0]["toolchain"] = "cargo"
    result = verify_manifest(manifest)
    assert not result.satisfiable
    assert any("toolchain" in rule for rule in result.conflicting_rules)


def test_missing_functional_intent_is_unsat():
    manifest = _valid_manifest()
    manifest["functional_intent"].append(
        {"symbol_name": "missing", "type": "function"}
    )
    result = verify_manifest(manifest)
    assert not result.satisfiable
    assert any("missing" in rule for rule in result.conflicting_rules)


def test_cycle_is_unsat():
    manifest = {
        "architecture": "pure_python",
        "nodes": [
            {"node_id": "a", "lang": "python", "toolchain": "python"},
            {"node_id": "b", "lang": "python", "toolchain": "python"},
        ],
        "edges": [
            {"source": "a", "target": "b", "boundary_type": "internal"},
            {"source": "b", "target": "a", "boundary_type": "internal"},
        ],
        "functional_intent": [],
    }
    result = verify_manifest(manifest)
    assert not result.satisfiable
    assert any("cycle" in rule for rule in result.conflicting_rules)


def test_invalid_ffi_boundary_is_unsat():
    manifest = {
        "architecture": "hybrid_rust_python",
        "nodes": [
            {"node_id": "rust_core", "lang": "rust", "toolchain": "cargo"},
            {"node_id": "main", "lang": "python", "toolchain": "python"},
        ],
        "edges": [
            {
                "source": "main",
                "target": "rust_core",
                "boundary_type": "C_ABI",
            }
        ],
        "functional_intent": [],
    }
    result = verify_manifest(manifest)
    assert not result.satisfiable
    assert any("boundary" in rule for rule in result.conflicting_rules)


def test_refinement_feedback_contains_rules():
    manifest = _valid_manifest()
    manifest["functional_intent"].append(
        {"symbol_name": "orphan", "type": "function"}
    )
    verifier = ConcolicManifestVerifier(manifest)
    result = verifier.verify()
    feedback = verifier.refinement_feedback(result)
    assert "orphan" in feedback
    assert "logically inconsistent" in feedback


def test_concolic_result_defaults():
    result = ConcolicResult(satisfiable=True)
    assert result.unsat_core == []
    assert result.conflicting_rules == []
