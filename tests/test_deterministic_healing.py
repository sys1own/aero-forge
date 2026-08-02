"""Tests for the pre-materialization deterministic AST healer."""

import json

import pytest


def test_pre_write_healer_ownership_mismatch_patch():
    """An ownership mismatch trace should queue an Arc<Mutex<T>> patch."""
    PreWriteHealer = pytest.importorskip("aero_forge._native").PreWriteHealer

    healer = PreWriteHealer()
    trace = "Ownership Mismatch detected at node_id \"rust_node_42\""
    healer.analyze_smt_unsat_core(trace)
    patches = healer.patches()
    assert len(patches) == 1
    assert patches[0][0] == "rust_node_42"
    assert patches[0][1] == "Arc<Mutex<T>>"
    assert patches[0][2] is True


def test_pre_write_healer_applies_patch_to_graph_json():
    """Patches should be applied in-memory without writing files."""
    PreWriteHealer = pytest.importorskip("aero_forge._native").PreWriteHealer

    healer = PreWriteHealer()
    healer.analyze_smt_unsat_core("Ownership Mismatch")
    graph = {
        "nodes": [
            {"id": "node_err_borrow", "type": "RustStruct"},
            {"id": "cpp_node", "type": "CppClass"},
        ]
    }
    patched_json = healer.apply_pre_write_patches(json.dumps(graph))
    patched = json.loads(patched_json)
    borrow = [n for n in patched["nodes"] if n["id"] == "node_err_borrow"][0]
    assert borrow["wrapped_type"] == "Arc<Mutex<T>>"


def test_pre_write_healer_ffi_layout_patch():
    """An FFI layout failure should queue a SerializationBuffer patch."""
    PreWriteHealer = pytest.importorskip("aero_forge._native").PreWriteHealer

    healer = PreWriteHealer()
    trace = "FFI layout alignment mismatch at id \"ffi_boundary_1\""
    healer.analyze_smt_unsat_core(trace)
    patches = healer.patches()
    assert patches[0][1] == "SerializationBuffer"
    assert patches[0][0] == "ffi_boundary_1"


def test_pre_write_healer_goi_deadlock_patch():
    """A GoI non-nilpotency failure should queue a DeadlockFreeChannel patch."""
    PreWriteHealer = pytest.importorskip("aero_forge._native").PreWriteHealer

    healer = PreWriteHealer()
    trace = "GoI non-nilpotent cycle detected; deadlock possible"
    healer.analyze_smt_unsat_core(trace)
    patches = healer.patches()
    assert patches[0][1] == "DeadlockFreeChannel"


def test_apply_pipeline_healer_python_api():
    """The Python bridge applies pre-materialization healing."""
    from aero_forge.apply_pipeline_healer import apply_pre_materialization_healing

    graph = {"nodes": [{"id": "node_err_borrow", "type": "RustStruct"}]}
    trace = "Ownership Mismatch"
    patched_json = apply_pre_materialization_healing(trace, json.dumps(graph))
    patched = json.loads(patched_json)
    assert patched["nodes"][0]["wrapped_type"] == "Arc<Mutex<T>>"


def test_apply_pipeline_healer_no_patch_for_unrelated_trace():
    """Unrelated traces should leave the graph unchanged."""
    from aero_forge.apply_pipeline_healer import apply_pre_materialization_healing

    graph = {"nodes": [{"id": "node_err_borrow", "type": "RustStruct"}]}
    trace = "Compilation succeeded"
    patched_json = apply_pre_materialization_healing(trace, json.dumps(graph))
    assert json.loads(patched_json) == graph


def test_build_pre_write_patches_python_fallback():
    """The patch builder returns sane defaults even without native binding details."""
    from aero_forge.apply_pipeline_healer import build_pre_write_patches

    trace = "Ownership Mismatch at node_id \"my_node\""
    patches = build_pre_write_patches(trace)
    assert len(patches) == 1
    assert patches[0]["target_node_id"] == "my_node"
    assert patches[0]["replacement_type"] == "Arc<Mutex<T>>"
    assert patches[0]["inject_wrapper"] is True
