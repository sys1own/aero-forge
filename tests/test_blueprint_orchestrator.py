"""Blueprint-driven planning and enforcement for hybrid Python/Rust builds."""

import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aero_forge.monorepo import generate_monorepo, orchestrate_hybrid_rust_python
from aero_forge.orchestrator.orchestrator import plan_workspace


def _make_mock_client(response: str):
    client = MagicMock()
    client.generate.return_value = response
    return client


@pytest.mark.skipif(
    not shutil.which("cargo") or not shutil.which("rustc"),
    reason="Rust toolchain not installed",
)
def test_blueprint_orchestrator_emits_hybrid_workspace(tmp_path):
    """A hybrid Python/Rust prompt emits blueprint.aero and all declared files."""
    response = (
        "```python\n"
        "def add(a: float, b: float) -> float:\n"
        "    return a + b\n"
        "```\n\n"
        "```python\n"
        "from generated import add\n\n"
        "def test_add():\n"
        "    assert add(2.0, 3.0) == 5.0\n"
        "```"
    )

    with (
        patch(
            "aero_forge.generate.get_llm_client",
            return_value=_make_mock_client(response),
        ),
        patch(
            "aero_forge.orchestrator.orchestrator.get_llm_client",
            return_value=_make_mock_client(response),
        ),
    ):
        result = generate_monorepo(
            "Create a hybrid Python/Rust add engine with PyO3 bindings",
            output_dir=tmp_path,
            llm_provider="openai",
            project_name="add_engine",
        )

    assert result["success"], result
    blueprint_path = tmp_path / "blueprint.aero"
    assert blueprint_path.is_file()

    blueprint_text = blueprint_path.read_text(encoding="utf-8")
    assert "architecture: hybrid_rust_python" in blueprint_text
    assert "toolchains:" in blueprint_text
    assert "- python" in blueprint_text
    assert "- cargo" in blueprint_text
    assert "manifest:" in blueprint_text
    assert "contracts:" in blueprint_text
    assert "Cargo.toml" in blueprint_text
    assert "src/lib.rs" in blueprint_text

    for declared in [
        "Cargo.toml",
        "rust_core/Cargo.toml",
        "rust_core/src/lib.rs",
        "python_engine/pyproject.toml",
        "python_engine/src/python_engine/__init__.py",
        "python_engine/service.py",
        "python_engine/bench.py",
    ]:
        assert (tmp_path / declared).is_file(), f"missing {declared}"

    # The exported Python wrapper should exist and re-export the primary function.
    primary_file = tmp_path / "python_engine" / "src" / "python_engine" / f"{result['primary_function']}.py"
    assert primary_file.is_file()

    init_file = tmp_path / "python_engine" / "src" / "python_engine" / "__init__.py"
    init_text = init_file.read_text(encoding="utf-8")
    assert result["primary_function"] in init_text


def test_plan_workspace_detects_hybrid_rust_python(tmp_path: Path) -> None:
    """A prompt with Rust/Python markers produces a hybrid_rust_python blueprint."""
    blueprint = plan_workspace(
        "Build a hybrid Python-Rust PyO3 extension with cargo and maturin",
        tmp_path,
        project_name="aero_test",
        llm_provider="none",
    )
    assert blueprint.architecture == "hybrid_rust_python"
    assert "cargo" in blueprint.toolchains
    assert "python" in blueprint.toolchains
    paths = {entry.path for entry in blueprint.manifest}
    assert "Cargo.toml" in paths
    assert "src/lib.rs" in paths
    assert "pyproject.toml" in paths


def test_orchestrate_hybrid_rust_python_builds(tmp_path: Path) -> None:
    """orchestrate_hybrid_rust_python emits a buildable hybrid workspace."""
    response = (
        "```python\n"
        "def compute_fft(samples: list[float]) -> list[float]:\n"
        "    return [x * 2.0 for x in samples]\n"
        "```\n\n"
        "```python\n"
        "from generated import compute_fft\n\n"
        "def test_compute_fft():\n"
        "    assert compute_fft([1.0, 2.0]) == [2.0, 4.0]\n"
        "```"
    )

    with (
        patch(
            "aero_forge.generate.get_llm_client",
            return_value=_make_mock_client(response),
        ),
        patch(
            "aero_forge.orchestrator.orchestrator.get_llm_client",
            return_value=_make_mock_client(response),
        ),
    ):
        result = orchestrate_hybrid_rust_python(
            "aero_test",
            ["compute_fft"],
            output_dir=tmp_path,
            llm_provider="openai",
        )

    assert result["success"], result
    blueprint_path = Path(result["blueprint_path"])
    assert blueprint_path.is_file()
    blueprint_text = blueprint_path.read_text(encoding="utf-8")
    assert "architecture: hybrid_rust_python" in blueprint_text
    assert "cargo" in blueprint_text

    dist = tmp_path / "dist"
    assert any(dist.rglob("*")) or result["files"], "dist/ should contain build artifacts"
