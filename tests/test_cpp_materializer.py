"""Tests for the C++/ctypes materializer and C-ABI emitter helpers."""

from aero_forge.scaffold.cpp_materializer import (
    _extract_explicit_cpp_update,
    _generate_dtw_test_source,
    _is_c_abi_contract,
    _is_special_cpp_contract,
    _parse_signature,
    _special_cpp_source,
)


def test_extract_explicit_cpp_update_detects_dtw_prompt():
    prompt = (
        "Target: hybrid_cpp_python. Force Native Bridge. "
        "Add a hybrid_cpp_python sliding window DTW module. "
        "Create C++ implementation for sliding_window_dtw accepting two 1D double arrays and window size. "
        "Expose it via C-ABI / Native Bridge so Python can call it. "
        "Update tests/test_dtw.py to verify against naive Python."
    )
    result = _extract_explicit_cpp_update(prompt)
    assert result is not None
    assert result["function"] == "sliding_window_dtw"
    assert result["args"] == [("a", "list[float]"), ("b", "list[float]"), ("window", "int")]
    assert result["return_type"] == "float"
    assert result["test_path"] == "tests/test_dtw.py"


def test_is_special_cpp_contract_detects_dtw():
    signature = "def sliding_window_dtw(a: list[float], b: list[float], window: int) -> float"
    from aero_forge.blueprint import ContractEntry

    contract = ContractEntry(name="sliding_window_dtw", signature=signature)
    assert _is_c_abi_contract(contract)
    assert _is_special_cpp_contract(contract)


def test_special_cpp_source_emits_dtw_algorithm():
    signature = "def sliding_window_dtw(a: list[float], b: list[float], window: int) -> float"
    from aero_forge.blueprint import ContractEntry

    contract = ContractEntry(name="sliding_window_dtw", signature=signature)
    source = _special_cpp_source("test_pkg", contract)
    assert 'extern "C" AERO_EXPORT double sliding_window_dtw(' in source
    assert "sliding_window_dtw(const double* a, size_t a_len" in source
    assert "std::vector<double>" in source
    assert "return prev[" in source


def test_generate_dtw_test_source_imports_and_asserts():
    source = _generate_dtw_test_source("accelerator", "sliding_window_dtw", [("a", "list[float]"), ("b", "list[float]"), ("window", "int")])
    assert "from accelerator import sliding_window_dtw" in source
    assert "def _naive_dtw(a, b, window):" in source
    assert "math.isclose(got, expected" in source
    assert "sliding_window_dtw(a, b, 3)" in source
