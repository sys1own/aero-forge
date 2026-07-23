"""Stress/integration tests for the Aero-Forge build pipeline.

Each test runs ``aero-forge build`` on a level-specific blueprint and verifies
either a successful compile or a clear, graceful failure for unsupported
constructs. The stress suite is intentionally broad: new transpiler support
can turn an ``xfail`` or failure-assertion test into a passing one.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

STRESS_DIR = Path(__file__).parent
AERO_FORGE = [sys.executable, "-m", "aero_forge.cli"]


def _run_build(
    blueprint: Path,
    cwd: Path | None = None,
    env: dict | None = None,
) -> subprocess.CompletedProcess:
    cmd = [*AERO_FORGE, "build", str(blueprint)]
    # Start from the real environment but remove provider overrides that may
    # have been injected for this session. Tests that need a provider set it
    # explicitly via the ``env`` argument.
    run_env = os.environ.copy()
    run_env.pop("AERO_FORGE_LLM_PROVIDER", None)
    run_env.update(env or {})
    return subprocess.run(
        cmd,
        cwd=cwd or blueprint.parent,
        capture_output=True,
        text=True,
        env=run_env,
    )


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
    def test_simple_class(self):
        blueprint = STRESS_DIR / "level3_oop" / "blueprint.aero"
        result = _run_build(blueprint)
        assert result.returncode == 0, result.stderr + result.stdout
        assert "Build summary: 2 succeeded, 0 failed" in result.stderr


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
    @staticmethod
    def _is_api_limit_error(output: str) -> bool:
        lowered = output.lower()
        return any(
            marker in lowered
            for marker in (
                "rate limit",
                "quota",
                "key limit",
                "401",
                "403",
                "429",
            )
        )

    @pytest.mark.parametrize(
        "blueprint,summary",
        [
            ("blueprint_syntax.aero", "1 succeeded, 0 failed"),
            ("blueprint_type.aero", "1 succeeded, 0 failed"),
            ("blueprint_multi.aero", "3 succeeded, 0 failed"),
        ],
    )
    def test_openrouter_heals_broken_functions(self, blueprint, summary):
        if not os.getenv("OPENROUTER_API_KEY"):
            pytest.skip("OPENROUTER_API_KEY not set")
        bp = STRESS_DIR / "level7_llm_healing" / blueprint
        out_dir = bp.parent / "dist"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        result = _run_build(bp, env={"AERO_FORGE_LLM_PROVIDER": "openrouter"})
        output = result.stderr + result.stdout
        if result.returncode != 0 and self._is_api_limit_error(output):
            pytest.xfail(f"OpenRouter API limit hit: {output[:200]}")
        assert result.returncode == 0, output
        assert summary in result.stderr, output

    def test_gemini_heals_syntax_error(self):
        if not os.getenv("GEMINI_API_KEY"):
            pytest.skip("GEMINI_API_KEY not set")
        bp = STRESS_DIR / "level7_llm_healing" / "blueprint_gemini_syntax.aero"
        out_dir = bp.parent / "dist"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        result = _run_build(bp, env={"AERO_FORGE_LLM_PROVIDER": "gemini"})
        output = result.stderr + result.stdout
        if result.returncode != 0 and self._is_api_limit_error(output):
            pytest.xfail(f"Gemini API limit hit: {output[:200]}")
        assert result.returncode == 0, output
        assert "1 succeeded, 0 failed" in result.stderr, output

    def test_no_llm_graceful_failure(self):
        """A broken function with provider: none should fail gracefully without crashing."""
        blueprint = STRESS_DIR / "level7_llm_healing" / "blueprint_no_llm.aero"
        result = _run_build(blueprint, env={"AERO_FORGE_CACHE_ENABLED": "false"})
        assert result.returncode != 0
        output = result.stderr + result.stdout
        assert (
            "LLM disabled" in output
            or "could not be fixed" in output
            or "Syntax error" in output
        )


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
        blueprint = (
            STRESS_DIR / "level9_blueprint_edge" / "blueprint_missing_function.aero"
        )
        result = _run_build(blueprint)
        assert result.returncode != 0
        assert "not found" in (result.stderr + result.stdout).lower()


class TestLevel10Classes:
    def test_counter_class(self):
        blueprint = STRESS_DIR / "level10_classes" / "blueprint.aero"
        result = _run_build(blueprint)
        assert result.returncode == 0, result.stderr + result.stdout
        assert "Build summary: 1 succeeded, 0 failed" in result.stderr

    def test_calculator_class(self):
        blueprint = STRESS_DIR / "level10_classes" / "blueprint_calculator.aero"
        result = _run_build(blueprint)
        assert result.returncode == 0, result.stderr + result.stdout
        assert "Build summary: 1 succeeded, 0 failed" in result.stderr


class TestLevel11VecClass:
    def test_matrix_class_with_vec_fields(self):
        blueprint = STRESS_DIR / "level11_vec_class" / "blueprint.aero"
        result = _run_build(blueprint)
        assert result.returncode == 0, result.stderr + result.stdout
        assert "Build summary: 1 succeeded, 0 failed" in result.stderr


class TestLevel12Numpy:
    def test_numpy_vector_ops(self):
        blueprint = STRESS_DIR / "level12_numpy" / "blueprint.aero"
        result = _run_build(blueprint)
        assert result.returncode == 0, result.stderr + result.stdout
        assert "Build summary: 3 succeeded, 0 failed" in result.stderr
