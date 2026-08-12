"""Unit tests for the Category-Theoretic Schema Bootstrapper."""

import pytest

from aero_forge.builder.adjoint import SchemaBootstrapper


def test_sigma_injects_node_stubs():
    intent = [
        {"symbol_name": "foo", "type": "function", "requirement_level": "required"},
        {"symbol_name": "bar", "type": "algorithm", "requirement_level": "required"},
    ]
    bs = SchemaBootstrapper(architecture_hint="hybrid_rust_python")
    stubs = bs.ΣF(intent)
    assert len(stubs) == 2
    assert any(s.node_id == "foo" and s.lang == "python" for s in stubs)
    assert any(s.node_id == "bar" and s.lang == "rust" for s in stubs)


def test_delta_merges_duplicate_nodes():
    bs = SchemaBootstrapper(architecture_hint="pure_python")
    stubs = [
        bs.ΣF([{"symbol_name": "foo", "type": "function"}])[0],
        bs.ΣF([{"symbol_name": "foo", "type": "test"}])[0],
    ]
    stubs[1].lang = "python"  # ensure same id
    merged, edges = bs.ΔF(stubs)
    assert len(merged) == 1
    assert "test" in merged[0].purpose


def test_pi_enforces_internal_edges_for_same_language():
    bs = SchemaBootstrapper(architecture_hint="pure_python")
    intent = [
        {"symbol_name": "a", "type": "function"},
        {"symbol_name": "b", "type": "function"},
    ]
    stubs = bs.ΣF(intent)
    edges = [{"source": "a", "target": "b", "relation": "imports"}]
    skeleton = bs.ΠF(stubs, edges, intent, "pure_python")
    assert skeleton["architecture"] == "pure_python"
    assert len(skeleton["nodes"]) == 2
    assert all(e["boundary_type"] == "internal" for e in skeleton["edges"])
    assert all(n["logic_sketch"] == "<TYPED_HOLE>" for n in skeleton["nodes"])


def test_pi_enforces_pyo3_boundary_for_rust_python():
    bs = SchemaBootstrapper(architecture_hint="hybrid_rust_python")
    intent = [
        {"symbol_name": "rust_core", "type": "algorithm"},
        {"symbol_name": "main", "type": "function"},
    ]
    stubs = bs.ΣF(intent)
    edges = [{"source": "rust_core", "target": "main", "relation": "uses"}]
    skeleton = bs.ΠF(stubs, edges, intent, "hybrid_rust_python")
    assert skeleton["edges"]
    assert skeleton["edges"][0]["boundary_type"] == "PYO3_MATURIN"


def test_grothendieck_bundle_pairs_intent_with_stubs():
    intent = [
        {"symbol_name": "x", "type": "function"},
        {"symbol_name": "y", "type": "function"},
    ]
    bs = SchemaBootstrapper(architecture_hint="pure_python")
    stubs = bs.ΣF(intent)
    bundle = bs.grothendieck_bundle(intent, stubs)
    assert len(bundle) == 2
    for entry, stub in bundle:
        assert stub.exports
        assert entry["symbol_name"] in stub.exports


def test_bootstrap_tri_polyglot_assigns_three_languages():
    intent = [
        {"symbol_name": "cpp_kernel", "type": "algorithm"},
        {"symbol_name": "rust_core", "type": "core"},
        {"symbol_name": "main", "type": "function"},
    ]
    bs = SchemaBootstrapper(architecture_hint="tri_polyglot_rust_cpp_python")
    skeleton = bs.bootstrap(intent)
    langs = {n["lang"] for n in skeleton["nodes"]}
    assert {"python", "rust", "cpp"} <= langs
