"""Tests for the dynamic FFI contract synthesizer."""

from __future__ import annotations

import pytest

from aero_forge.scaffold.contract_synth import (
    FFI_TYPE_LAYOUTS,
    DynamicContractSynthesizer,
    FFIBoundaryEdge,
    GeneratedFFIBridge,
)


def test_ffi_type_layouts_cover_primitives() -> None:
    for primitive in ("int32", "int64", "float32", "float64", "pointer"):
        assert primitive in FFI_TYPE_LAYOUTS
        layout = FFI_TYPE_LAYOUTS[primitive]
        for key in ("c_type", "rust_type", "python_ctype", "csharp_type", "go_type", "size", "alignment"):
            assert key in layout


@pytest.mark.parametrize("boundary", ["c_abi", "pyo3", "cgo", "pinvoke"])
def test_synthesize_boundary_returns_generated_bridge(boundary: str) -> None:
    synth = DynamicContractSynthesizer()
    edge = FFIBoundaryEdge(
        edge_id="e1",
        source_node="rust_core",
        source_lang="rust",
        target_node="py_client",
        target_lang="python",
        boundary_type=boundary,
        symbol_name="add",
        argument_types=["int64", "int64"],
        return_type="int64",
        is_zero_copy=False,
    )
    bridge = synth.synthesize_boundary(edge)
    assert isinstance(bridge, GeneratedFFIBridge)
    assert bridge.edge_id == "e1"
    assert boundary in bridge.boundary_type
    assert bridge.source
    assert "add" in bridge.source
    if boundary == "c_abi":
        assert "extern \"C\"" in bridge.source or "extern" in bridge.source
        assert "__declspec(dllexport)" not in bridge.source
        assert "#ifndef" in bridge.header
        assert "ctypes.CDLL" in bridge.python_loader
    if boundary == "pyo3":
        assert "#[pyfunction]" in bridge.source
    if boundary == "cgo":
        assert "//export add" in bridge.source
        assert "package main" in bridge.source
    if boundary == "pinvoke":
        assert "[LibraryImport" in bridge.csharp_stub
        assert "__declspec(dllexport)" in bridge.source


def test_c_abi_header_guards() -> None:
    synth = DynamicContractSynthesizer()
    edge = FFIBoundaryEdge(
        edge_id="e2",
        source_node="cpp_lib",
        source_lang="cpp",
        target_node="py_client",
        target_lang="python",
        boundary_type="c_abi",
        symbol_name="process_buffer",
        argument_types=["pointer", "int64"],
        return_type="int64",
        is_zero_copy=True,
    )
    bridge = synth.synthesize_boundary(edge)
    assert "AERO_PROCESS_BUFFER_H" in bridge.header
    assert "void*" in bridge.header or "int64_t" in bridge.header
