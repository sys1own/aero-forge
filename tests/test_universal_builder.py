"""Tests for the universal blueprint-driven builder entry point."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aero_forge.universal_builder import build_universal_project


def test_build_universal_dispatches_to_monorepo_for_hybrid(tmp_path: Path) -> None:
    """A hybrid Python/Rust prompt routes through the polyglot monorepo builder."""
    plan = MagicMock()
    plan.architecture = "hybrid_rust_python"
    plan.project = "hybrid_demo"
    plan.toolchains = ["python", "rust", "cargo"]
    plan.manifest = []
    plan.contracts = []
    plan.languages = ["python", "rust"]
    plan.features = ["data_pipeline"]
    plan.prompt = "test"
    plan.constraints = None

    with (
        patch("aero_forge.universal_builder.plan_workspace", return_value=plan),
        patch(
            "aero_forge.universal_builder.generate_monorepo",
            return_value={"success": True, "files": []},
        ) as mock_mono,
        patch(
            "aero_forge.universal_builder.generate_and_build",
            return_value={"success": True, "files": []},
        ) as mock_build,
    ):
        result = build_universal_project(
            "Build a Python-Rust hybrid batch processor",
            tmp_path,
            llm_provider="none",
        )

    assert result["success"] is True
    assert result["classification"]["architecture"] == "hybrid_rust_python"
    mock_mono.assert_called_once()
    mock_build.assert_not_called()


def test_build_universal_dispatches_to_python_builder_for_pure_python(tmp_path: Path) -> None:
    """A pure Python prompt routes through ``generate_and_build``."""
    plan = MagicMock()
    plan.architecture = "pure_python"
    plan.project = "pure_demo"
    plan.toolchains = ["python"]
    plan.manifest = []
    plan.contracts = []
    plan.languages = ["python"]
    plan.features = []
    plan.prompt = "test"
    plan.constraints = None

    with (
        patch("aero_forge.universal_builder.plan_workspace", return_value=plan),
        patch(
            "aero_forge.universal_builder.generate_monorepo",
            return_value={"success": True, "files": []},
        ) as mock_mono,
        patch(
            "aero_forge.universal_builder.generate_and_build",
            return_value={"success": True, "files": []},
        ) as mock_build,
    ):
        result = build_universal_project(
            "Implement a pure Python sorting function",
            tmp_path,
            llm_provider="none",
        )

    assert result["success"] is True
    assert result["classification"]["architecture"] == "pure_python"
    mock_build.assert_called_once()
    mock_mono.assert_not_called()


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(
    (not os.getenv("DEEPSEEK_API_KEY") and not os.getenv("AERO_FORGE_API_KEY"))
    or not shutil.which("cargo"),
    reason="Live LLM API key or cargo toolchain not available",
)
def test_universal_builder_hybrid_blueprint_and_build(tmp_path: Path) -> None:
    """End-to-end: a hybrid prompt produces a blueprint and compiles."""
    result = build_universal_project(
        (
            "Build a Python-Rust batch processor that doubles each float in a list. "
            "Expose a Rust core via PyO3 and wrap it with a Python package and tests."
        ),
        tmp_path,
        project_name="double_pipeline",
        llm_provider="deepseek",
        model="deepseek-chat",
    )

    blueprint = tmp_path / "blueprint.aero"
    assert blueprint.is_file()
    text = blueprint.read_text(encoding="utf-8")
    assert "architecture: hybrid_rust_python" in text
    assert "- cargo" in text or "- rust" in text

    assert result["success"], result
    assert (tmp_path / "Cargo.toml").is_file()
    assert (tmp_path / "rust_core" / "src" / "lib.rs").is_file()
    assert (tmp_path / "pyproject.toml").is_file()
    # The Python package may be named after the project rather than a hardcoded
    # template directory, so just check that at least one package __init__.py exists.
    assert list(tmp_path.rglob("__init__.py"))


@pytest.mark.integration
@pytest.mark.skipif(
    not shutil.which("cargo") or not shutil.which("rustc"),
    reason="Rust toolchain not installed",
)
def test_build_universal_honors_explicit_file_paths(tmp_path: Path) -> None:
    """The polyglot fallback uses the exact file paths requested in the prompt."""
    result = build_universal_project(
        (
            "Scaffold polyglot project named aero-orchestrator with "
            "aero_orchestrator/cli.py, run_shell.py, and tests/test_cli.py."
        ),
        tmp_path,
        project_name="aero-orchestrator",
        llm_provider="none",
    )

    assert result["success"], result.get("logs", "")
    assert (tmp_path / "aero_orchestrator" / "cli.py").is_file()
    assert (tmp_path / "run_shell.py").is_file()
    assert (tmp_path / "tests" / "test_cli.py").is_file()
    assert (tmp_path / "aero_orchestrator" / "native.py").is_file()
    # The old hardcoded template names must not appear.
    assert not (tmp_path / "aero_polyglot_runner").exists()
    assert not (tmp_path / "run_demo.py").is_file()

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{tmp_path}{os.pathsep}{env.get('PYTHONPATH', '')}".strip(os.pathsep)
    run_result = subprocess.run(
        ["python", str(tmp_path / "run_shell.py")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert run_result.returncode == 0, run_result.stderr
    assert "CLI ready" in run_result.stdout
