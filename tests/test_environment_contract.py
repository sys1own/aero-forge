"""Environment contract verification tests for aero-forge."""

from __future__ import annotations

import shutil

import pytest

from aero_forge.environment import (
    ContractViolationError,
    VerifyDependencies,
)


def test_verify_language_rust_passes_when_toolchain_present() -> None:
    if not shutil.which("cargo"):
        pytest.skip("cargo not installed")
    VerifyDependencies.verify_language("rust")


def test_verify_language_fortran_fails_without_gfortran() -> None:
    if shutil.which("gfortran"):
        pytest.skip("gfortran is present")
    with pytest.raises(ContractViolationError) as exc_info:
        VerifyDependencies.verify_language("fortran")
    assert "gfortran" in str(exc_info.value).lower()


def test_missing_dependencies_detects_uninstalled_package() -> None:
    missing = VerifyDependencies.missing_dependencies({"nonexistent_abc_xyz": "nonexistent_abc_xyz"})
    assert "nonexistent_abc_xyz" in missing


def test_blueprint_contract_combines_languages_and_packages() -> None:
    blueprint = {
        "environment_contract": {
            "languages": ["rust", "python"],
            "required_python_packages": ["sys"],
        }
    }
    verifier = VerifyDependencies(blueprint)
    tools = verifier.required_tools()
    assert "cargo" in tools
    assert "rustc" in tools
    assert "python3" in tools
    # "sys" is a stdlib module and should already be importable, so no violation.
    assert "sys" not in verifier.missing_python_packages()


def test_cpp_compiler_detected_and_compiles_shared_library() -> None:
    if not shutil.which("g++") and not shutil.which("clang++"):
        pytest.skip("No C++ compiler available")
    compiler, so_path = VerifyDependencies.assert_cpp_shared_library()
    assert compiler in ("g++", "clang++", "c++")
    assert so_path.is_file()
    assert so_path.suffix == ".so"


def test_verify_language_cpp_fails_without_compiler(monkeypatch) -> None:
    def _no_compiler():
        return None

    monkeypatch.setattr(
        "aero_forge.environment.verify_dependencies._find_cpp_compiler", _no_compiler
    )
    with pytest.raises(ContractViolationError) as exc_info:
        VerifyDependencies.verify_language("cpp")
    assert "no c++ compiler" in str(exc_info.value).lower()
