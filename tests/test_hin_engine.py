"""Tests for the Rust HIN (MELL interaction-net) engine bridge."""

import json
import os
import sys
from pathlib import Path

import pytest

from aero_forge.translator.aero_frontend import python_source_to_uast

# The native extension is optional until the workspace tool-chain builds it.
HIN = pytest.importorskip("aero_forge.hin_engine")


def _load_aero_future_modules():
    """Load aero-future's Python HIN VM for parity checks if present."""
    base = Path(__file__).resolve().parents[2] / "aero-future"
    if not base.is_dir():
        base = Path(os.environ.get("AERO_FUTURE_PATH", "/home/ubuntu/repos/aero-future"))
    if base.is_dir() and str(base) not in sys.path:
        sys.path.insert(0, str(base))
    try:
        from core.hin_vm import HINNetwork
        from core.translator import UASTToHINTranslator

        return UASTToHINTranslator, HINNetwork
    except Exception:
        return None, None


def test_native_bridge_available():
    assert HIN.native_available()


def test_identity_function_reduction():
    source = "def f(x):\n    return x\ny = f(42)\ny"
    uast = python_source_to_uast(source)
    result = HIN.reduce_uast(uast)
    assert result["native"]
    assert result["steps"] > 0
    assert isinstance(result["graph"], list)


def test_nested_function_reduction():
    source = "def f(x):\n    return x\ndef g(y):\n    return f(y)\ny = g(7)\ny"
    uast = python_source_to_uast(source)
    result = HIN.reduce_uast(uast)
    assert result["native"]
    assert result["steps"] > 0


def test_conditional_reduction_parity():
    source = "def f(x):\n    return x\nif f(True):\n    1\nelse:\n    0\n"
    uast = python_source_to_uast(source)
    result = HIN.reduce_uast(uast)
    assert result["native"]
    assert result["steps"] > 0

    PythonTranslator, PythonNetwork = _load_aero_future_modules()
    if PythonTranslator is None:
        pytest.skip("aero-future reference VM not available")

    t = PythonTranslator()
    net = t.translate_uast(uast)
    net.run_to_completion()
    assert len(net.nodes) == len(result["graph"])


def test_hin_engine_class_api():
    HinEngine = pytest.importorskip("aero_forge_native").HinEngine
    source = "def f(x):\n    return x\nf(42)"
    uast = python_source_to_uast(source)
    engine = HinEngine()
    engine.build_from_json(json.dumps(uast))
    steps = engine.reduce_to_completion(1_000_000)
    assert steps > 0
    graph = json.loads(engine.to_json())
    assert isinstance(graph, list)


def test_hin_node_defaults():
    from aero_forge.hin_graph import HINNode

    rust = HINNode("r", "RustStruct", "rust", {})
    python = HINNode("p", "PyClass", "python", {})
    cpp = HINNode("c", "CppClass", "cpp", {})
    ffi = HINNode("f", "FFIBoundary", "ffi", {})
    assert rust.ownership_level == "1"
    assert python.ownership_level == "!"
    assert cpp.ownership_level == r"$\bot$"
    assert ffi.ownership_level == "&"


def test_hingraph_add_node_and_relation():
    from aero_forge.hin_graph import HINGraph, HINNode

    g = HINGraph()
    n1 = HINNode("n1", "RustStruct", "rust", {"arg_type": "&str"})
    n2 = HINNode("n2", "CppClass", "cpp", {})
    g.add_node(n1)
    g.add_node(n2)
    g.add_relation("n1", "n2", "CallsFFI", arg_type="&str")
    assert g.node_count() == 2
    assert g.edge_count() == 1


def test_dpo_rewrite_inserts_ffi_boundary():
    from aero_forge.hin_engine import HINEngine, HINNode

    engine = HINEngine()
    rust = HINNode("rust_fn", "RustStruct", "rust", {"arg_type": "&str"})
    cpp = HINNode("cpp_fn", "CppClass", "cpp", {})
    engine.add_ast_node(rust)
    engine.add_ast_node(cpp)
    engine.add_relation("rust_fn", "cpp_fn", "CallsFFI", arg_type="&str")

    count = engine.apply_dpo_rewrite_ffi_strings()
    assert count == 1
    assert engine.graph.number_of_nodes() == 3
    assert engine.graph.has_edge("rust_fn", "ffi_bridge_rust_fn_cpp_fn_0")
    assert engine.graph.has_edge("ffi_bridge_rust_fn_cpp_fn_0", "cpp_fn")
    assert not engine.graph.has_edge("rust_fn", "cpp_fn", key="CallsFFI")
    bridge_data = engine.graph.nodes["ffi_bridge_rust_fn_cpp_fn_0"]
    assert bridge_data["node_type"] == "FFIBoundary"
    assert bridge_data["properties"]["wrapper"] == "C_String_Wrapper"
    assert bridge_data["properties"]["abi"] == "C"


def test_ownership_violation_for_linear_to_gc_transfer():
    from aero_forge.hin_engine import HINEngine, HINNode

    engine = HINEngine()
    rust = HINNode("rust_val", "RustStruct", "rust", {}, ownership_level="1")
    python = HINNode("py_val", "PyClass", "python", {}, ownership_level="!")
    engine.add_ast_node(rust)
    engine.add_ast_node(python)
    engine.add_relation("rust_val", "py_val", "TransfersOwnershipTo")

    violations = engine.propagate_ownership_constraints()
    assert len(violations) == 1
    assert "linear Rust node 'rust_val'" in violations[0]
    assert "Python GC node 'py_val'" in violations[0]


def test_ownership_no_violation_with_managed_intermediate():
    from aero_forge.hin_engine import HINEngine, HINNode

    engine = HINEngine()
    rust = HINNode("rust_val", "RustStruct", "rust", {}, ownership_level="1")
    bridge = HINNode("bridge", "FFIBoundary", "ffi", {}, ownership_level="&")
    python = HINNode("py_val", "PyClass", "python", {}, ownership_level="!")
    for n in (rust, bridge, python):
        engine.add_ast_node(n)
    engine.add_relation("rust_val", "bridge", "TransfersOwnershipTo")
    engine.add_relation("bridge", "py_val", "TransfersOwnershipTo")

    violations = engine.propagate_ownership_constraints()
    assert violations == []
