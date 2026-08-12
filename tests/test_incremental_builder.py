"""Targeted incremental C++ extension generation tests."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from aero_forge.blueprint import Blueprint, ContractEntry, FunctionalIntent
from aero_forge.orchestrator.stack_classifier import INTENT_HYBRID_CPP_PYTHON
from aero_forge.scaffold.cpp_materializer import CppPolyglotMaterializer

PROMPT = (
    "Add a hybrid_cpp_python sliding window DTW module. "
    "Create C++ implementation for sliding_window_dtw accepting two 1D double arrays and window size. "
    "Expose it via C-ABI / Native Bridge so Python can call it. "
    "Update tests/test_dtw.py to verify against naive Python."
)


def _copy_accelerator_fixture(tmp_path: Path) -> Path:
    """Return a workspace that mirrors the aero-accelerator layout."""
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    pkg = workspace / "src" / "accelerator"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text('"""accelerator package."""\n__all__ = ["cli"]\n')
    (pkg / "cli.py").write_text(
        '"""Custom CLI preserved by incremental builder."""\n\n'
        'def main():\n    print("custom cli")\n'
    )
    (workspace / "pyproject.toml").write_text(
        '[build-system]\nrequires = ["setuptools>=64.0", "wheel"]\n'
        'build-backend = "setuptools.build_meta"\n\n'
        '[project]\nname = "accelerator"\nversion = "0.2.0"\n'
        'description = "Accelerator."\nrequires-python = ">=3.9"\n'
        'dependencies = ["numpy"]\n'
    )
    (workspace / "tests").mkdir(exist_ok=True)
    return workspace


def test_incremental_cpp_update_preserves_untouched_files(tmp_path: Path) -> None:
    """A targeted C++ extension build only creates/modifies the declared files."""
    workspace = _copy_accelerator_fixture(tmp_path)
    original_cli = (workspace / "src" / "accelerator" / "cli.py").read_text()
    original_pyproject = (workspace / "pyproject.toml").read_text()

    blueprint = Blueprint(
        project="accelerator",
        architecture=INTENT_HYBRID_CPP_PYTHON,
        toolchains=["python", "cpp"],
        prompt=PROMPT,
        metadata={"llm_initialized": "true"},
        functional_intent=[
            FunctionalIntent(symbol_name="sliding_window_dtw", type="function")
        ],
        contracts=[
            ContractEntry(name="sliding_window_dtw", signature="", language="cpp")
        ],
    )

    materializer = CppPolyglotMaterializer(workspace)
    updated = materializer.materialize(blueprint, build=False, force_overwrite=True)

    # Untouched file preserved exactly.
    assert (workspace / "src" / "accelerator" / "cli.py").read_text() == original_cli

    # New C++ source is created with functional algorithm and C-ABI export.
    cpp_path = workspace / "src" / "accelerator" / "sliding_window_dtw.cpp"
    assert cpp_path.is_file()
    cpp_text = cpp_path.read_text()
    assert len(cpp_text.encode("utf-8")) > 500
    assert 'extern "C" AERO_EXPORT double sliding_window_dtw(' in cpp_text
    assert "std::fabs(" in cpp_text

    # pyproject.toml retains existing fields and gains native package data.
    pyproject_text = (workspace / "pyproject.toml").read_text()
    assert original_pyproject.strip() in pyproject_text.strip()
    assert "[tool.setuptools.package-data]" in pyproject_text
    assert '"*.so"' in pyproject_text

    # __init__.py keeps existing exports and adds the new native bridge import.
    init_text = (workspace / "src" / "accelerator" / "__init__.py").read_text()
    assert "from .native_bridge import sliding_window_dtw" in init_text
    assert "sliding_window_dtw" in init_text

    # A modification plan was recorded.
    assert getattr(updated, "modification_plan", None)
    paths = {a["path"] for a in updated.modification_plan["actions"]}
    assert any("sliding_window_dtw.cpp" in p for p in paths)
    assert any(p == "src/accelerator/__init__.py" for p in paths)


def test_incremental_cpp_update_builds_and_passes_tests(tmp_path: Path) -> None:
    """The generated C++ extension compiles and its pytest suite passes."""
    if not shutil.which("g++"):
        pytest.skip("g++ compiler not available")

    workspace = _copy_accelerator_fixture(tmp_path)
    blueprint = Blueprint(
        project="accelerator",
        architecture=INTENT_HYBRID_CPP_PYTHON,
        toolchains=["python", "cpp"],
        prompt=PROMPT,
        metadata={"llm_initialized": "true"},
        functional_intent=[
            FunctionalIntent(symbol_name="sliding_window_dtw", type="function")
        ],
        contracts=[
            ContractEntry(name="sliding_window_dtw", signature="", language="cpp")
        ],
    )

    materializer = CppPolyglotMaterializer(workspace)
    materializer.materialize(blueprint, build=True, force_overwrite=True)

    so_path = workspace / "src" / "accelerator" / "libaccelerator.so"
    assert so_path.is_file(), "C-ABI shared library was not produced"

    env = {"PYTHONPATH": str(workspace / "src"), "PATH": "/usr/bin:/bin"}
    proc = subprocess.run(
        ["python", "-m", "pytest", "tests/test_dtw.py", "-v"],
        cwd=workspace,
        env=env,
        capture_output=True,
        text=True,
    )
    print(proc.stdout)
    print(proc.stderr)
    assert proc.returncode == 0, f"pytest failed:\n{proc.stdout}\n{proc.stderr}"
    assert "1 passed" in proc.stdout


AERO_ACCELERATOR_REPO = Path("/home/ubuntu/repos/aero-accelerator")


@pytest.mark.skipif(
    not AERO_ACCELERATOR_REPO.is_dir(), reason="aero-accelerator repo not present"
)
def test_incremental_cpp_update_on_accelerator_repo(tmp_path: Path) -> None:
    """End-to-end: add a C++ sliding-window DTW extension to aero-accelerator."""
    if not shutil.which("g++"):
        pytest.skip("g++ compiler not available")

    workspace = tmp_path / "aero-accelerator"
    shutil.copytree(AERO_ACCELERATOR_REPO, workspace)
    original_cli = (workspace / "src" / "accelerator" / "cli.py").read_text()

    prompt = (
        "Add a hybrid_cpp_python sliding window DTW module. "
        "Create C++ implementation for sliding_window_dtw accepting two 1D double arrays and window size. "
        "Expose it via C-ABI / Native Bridge so Python can call it. "
        "Update tests/test_swdtw.py to verify against naive Python."
    )
    blueprint = Blueprint(
        project="accelerator",
        architecture=INTENT_HYBRID_CPP_PYTHON,
        toolchains=["python", "cpp"],
        prompt=prompt,
        metadata={"llm_initialized": "true"},
    )

    materializer = CppPolyglotMaterializer(workspace)
    materializer.materialize(blueprint, build=True, force_overwrite=True)

    cpp_path = workspace / "src" / "accelerator" / "sliding_window_dtw.cpp"
    assert cpp_path.is_file() and len(cpp_path.read_bytes()) > 500
    assert 'extern "C" AERO_EXPORT double sliding_window_dtw(' in cpp_path.read_text()
    assert (workspace / "src" / "accelerator" / "cli.py").read_text() == original_cli

    env = {"PYTHONPATH": str(workspace / "src"), "PATH": "/usr/bin:/bin"}
    proc = subprocess.run(
        ["python", "-m", "pytest", "tests/test_swdtw.py", "-v"],
        cwd=workspace,
        env=env,
        capture_output=True,
        text=True,
    )
    print(proc.stdout)
    print(proc.stderr)
    assert proc.returncode == 0, f"pytest failed:\n{proc.stdout}\n{proc.stderr}"
    assert "1 passed" in proc.stdout
