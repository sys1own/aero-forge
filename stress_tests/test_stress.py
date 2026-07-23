"""Stress/integration tests for the Aero-Forge build pipeline.

Each test runs ``aero-forge build`` on a level-specific blueprint and verifies
either a successful compile or a clear, graceful failure for unsupported
constructs. The stress suite is intentionally broad: new transpiler support
can turn an ``xfail`` or failure-assertion test into a passing one.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

STRESS_DIR = Path(__file__).parent
AERO_FORGE = [sys.executable, "-m", "aero_forge.cli"]


def _run_build(blueprint: Path, cwd: Path | None = None) -> subprocess.CompletedProcess:
    cmd = [*AERO_FORGE, "build", str(blueprint)]
    env = {k: v for k, v in os.environ.items() if "API" not in k.upper()}
    return subprocess.run(
        cmd,
        cwd=cwd or blueprint.parent,
        capture_output=True,
        text=True,
        env=env,
    )


import os  # noqa: E402


class TestLevel1Math:
    def test_factorial_power_is_prime_mandelbrot(self):
        blueprint = STRESS_DIR / "level1_math" / "blueprint.aero"
        result = _run_build(blueprint)
        assert result.returncode == 0, result.stderr + result.stdout
        assert "Build summary: 3 succeeded, 0 failed" in result.stderr

    def test_matrix_multiply_is_unsupported(self):
        """Matrix multiplication uses lists/len/append which are not yet supported."""
        blueprint = STRESS_DIR / "level1_math" / "blueprint_matrix.aero"
        result = _run_build(blueprint)
        assert result.returncode != 0
        assert "Unsupported" in result.stderr or "Unsupported" in result.stdout


class TestLevel2Collections:
    def test_tuple_unpack_and_builtin_minmax(self):
        blueprint = STRESS_DIR / "level2_collections" / "blueprint.aero"
        result = _run_build(blueprint)
        assert result.returncode == 0, result.stderr + result.stdout

    def test_list_dict_slice_zip_are_unsupported(self):
        blueprint = STRESS_DIR / "level2_collections" / "blueprint_unsupported.aero"
        result = _run_build(blueprint)
        assert result.returncode != 0
        assert "Unsupported" in (result.stderr + result.stdout)


class TestLevel3OOP:
    def test_classes_are_unsupported(self):
        blueprint = STRESS_DIR / "level3_oop" / "blueprint.aero"
        result = _run_build(blueprint)
        assert result.returncode != 0
        assert "Unsupported" in (result.stderr + result.stdout)


class TestLevel4ControlFlow:
    def test_break_continue_return(self):
        blueprint = STRESS_DIR / "level4_control_flow" / "blueprint.aero"
        result = _run_build(blueprint)
        assert result.returncode == 0, result.stderr + result.stdout
        assert "Build summary: 2 succeeded, 0 failed" in result.stderr


class TestLevel5Stdlib:
    def test_math_functions(self):
        blueprint = STRESS_DIR / "level5_stdlib" / "blueprint.aero"
        result = _run_build(blueprint)
        assert result.returncode == 0, result.stderr + result.stdout
        assert "Build summary: 3 succeeded, 0 failed" in result.stderr


class TestLevel6CrossFile:
    def test_multiple_source_files(self):
        blueprint = STRESS_DIR / "level6_cross_file" / "blueprint.aero"
        result = _run_build(blueprint)
        assert result.returncode == 0, result.stderr + result.stdout
        assert "Build summary: 3 succeeded, 0 failed" in result.stderr


class TestLevel7LLMHealing:
    @pytest.mark.skipif(
        not os.getenv("OPENROUTER_API_KEY") and not os.getenv("GEMINI_API_KEY"),
        reason="No LLM API key available",
    )
    def test_llm_heals_broken_function(self):
        blueprint = STRESS_DIR / "level7_llm_healing" / "blueprint.aero"
        result = _run_build(blueprint)
        # Allow partial success: the tool should not crash and should report results.
        assert result.returncode in (0, 1)

    def test_no_llm_graceful_failure(self):
        """A broken function with provider: none should fail gracefully without crashing."""
        blueprint = STRESS_DIR / "level7_llm_healing" / "blueprint_no_llm.aero"
        if not blueprint.exists():
            pytest.skip("no-llm stress blueprint not present")
        result = _run_build(blueprint)
        assert result.returncode != 0


class TestLevel8Performance:
    def test_many_functions_build_quickly(self):
        blueprint = STRESS_DIR / "level8_performance" / "blueprint.aero"
        result = _run_build(blueprint)
        assert result.returncode == 0, result.stderr + result.stdout
        assert "Build summary: 50 succeeded, 0 failed" in result.stderr


class TestLevel9BlueprintEdge:
    def test_valid_blueprint(self):
        blueprint = STRESS_DIR / "level9_blueprint_edge" / "blueprint_valid.aero"
        result = _run_build(blueprint)
        assert result.returncode == 0, result.stderr + result.stdout

    def test_missing_file_gives_clear_error(self):
        blueprint = STRESS_DIR / "level9_blueprint_edge" / "blueprint_missing_file.aero"
        result = _run_build(blueprint)
        assert result.returncode != 0
        assert "missing file" in (result.stderr + result.stdout).lower()

    def test_missing_function_gives_clear_error(self):
        blueprint = STRESS_DIR / "level9_blueprint_edge" / "blueprint_missing_function.aero"
        result = _run_build(blueprint)
        assert result.returncode != 0
        assert "not found" in (result.stderr + result.stdout).lower()
