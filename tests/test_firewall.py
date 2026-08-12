"""Unit tests for the Logical Firewall SHACL manifest validator."""

import pytest

from aero_forge.builder.firewall import LogicalFirewall, validate_manifest


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


def test_valid_manifest_conforms():
    report = validate_manifest(_valid_manifest())
    assert report.conforms
    assert not report.violations


def test_invalid_toolchain_detected():
    manifest = _valid_manifest()
    manifest["nodes"][0]["toolchain"] = "cargo"
    report = validate_manifest(manifest)
    assert not report.conforms
    assert any("cargo" in v["message"] for v in report.violations)


def test_missing_functional_intent_coverage():
    manifest = _valid_manifest()
    manifest["functional_intent"].append({"symbol_name": "missing"})
    report = validate_manifest(manifest)
    assert not report.conforms
    assert any("missing" in v["message"] for v in report.violations)


def test_invalid_ffi_boundary_detected():
    manifest = {
        "architecture": "hybrid_rust_python",
        "nodes": [
            {"node_id": "main", "lang": "python", "toolchain": "python", "exports": ["main"]},
            {"node_id": "rust", "lang": "rust", "toolchain": "cargo", "exports": ["calc"]},
        ],
        "edges": [{"source": "main", "target": "rust", "boundary_type": "C_ABI"}],
        "functional_intent": [{"symbol_name": "main"}],
    }
    report = validate_manifest(manifest)
    assert not report.conforms
    assert any("C_ABI" in v["message"] for v in report.violations)


def test_llm_feedback_format():
    report = validate_manifest({"architecture": "bad", "nodes": [], "edges": [], "functional_intent": []})
    text = report.to_llm_feedback()
    assert "SHACL validation failed" in text
    assert "Architecture" in text or "blueprint must contain" in text


def test_compact_rdf_summary_is_turtle():
    firewall = LogicalFirewall(_valid_manifest())
    summary = firewall.compact_rdf_summary()
    assert "@prefix" in summary or "aero:" in summary
