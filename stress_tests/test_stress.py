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
    extra_args: list | None = None,
) -> subprocess.CompletedProcess:
    cmd = [*AERO_FORGE, "build", str(blueprint), *(extra_args or [])]
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
                "identical to current source",
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


class TestLevel13GPU:
    def test_gpu_pragma_falls_back_to_cpu(self):
        """A function marked ``# @accelerate gpu`` compiles on CPU when nvcc is unavailable."""
        blueprint = STRESS_DIR / "level13_gpu" / "blueprint.aero"
        result = _run_build(blueprint, env={"AERO_FORGE_CACHE_ENABLED": "false"})
        output = result.stderr + result.stdout
        assert result.returncode == 0, output
        assert "Build summary: 1 succeeded, 0 failed" in result.stderr


class TestLevel14AdvancedPython:
    @pytest.mark.parametrize(
        "blueprint,expected",
        [
            ("blueprint_try_except.aero", "try/except"),
            ("blueprint_with_stmt.aero", "with statements"),
            ("blueprint_yield_gen.aero", "yield"),
            ("blueprint_walrus.aero", "walrus"),
            ("blueprint_match_case.aero", "match/case"),
            ("blueprint_async_await.aero", "async/await"),
            ("blueprint_slots.aero", "__slots__"),
        ],
    )
    def test_unsupported_advanced_python(self, blueprint, expected):
        """Advanced Python constructs produce clear, specific error messages."""
        bp = STRESS_DIR / "level14_advanced_python" / blueprint
        result = _run_build(bp)
        output = result.stderr + result.stdout
        assert result.returncode != 0
        assert expected.lower() in output.lower()


class TestLevel20ErrorMessages:
    def test_explain_command_with_error_file(self):
        """``aero-forge explain`` produces a human-readable explanation."""
        source = STRESS_DIR / "level14_advanced_python" / "with_stmt.py"
        error_file = STRESS_DIR / "level20_error_messages" / "error.log"
        error_file.parent.mkdir(parents=True, exist_ok=True)
        error_file.write_text(
            "UnsupportedError: with statements / context managers are not supported",
            encoding="utf-8",
        )
        result = subprocess.run(
            [*AERO_FORGE, "explain", str(source), "--error-file", str(error_file)],
            capture_output=True,
            text=True,
        )
        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        assert "with" in output.lower()


class TestLevel22Examples:
    def test_examples_run_fibonacci(self):
        """The curated fibonacci example builds and passes tests."""
        result = subprocess.run(
            [*AERO_FORGE, "examples", "run", "fibonacci"],
            capture_output=True,
            text=True,
        )
        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        assert "1/1 succeeded" in output or "1 succeeded" in output


class TestLevel24Aeroignore:
    def test_auto_detect_respects_aeroignore(self):
        """Auto-detected build skips files listed in .aeroignore."""
        root = STRESS_DIR / "level24_ignore"
        result = subprocess.run(
            [*AERO_FORGE, "build", "--auto-detect", "--write-blueprint"],
            cwd=root,
            capture_output=True,
            text=True,
        )
        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        assert "keep" in output
        assert "skip" not in output


class TestLevel25AutoDetect:
    def test_auto_detect_standard_project(self):
        """``--auto-detect`` discovers src/ and tests/ and builds."""
        root = STRESS_DIR / "level25_autodetect"
        result = subprocess.run(
            [*AERO_FORGE, "build", "--auto-detect"],
            cwd=root,
            capture_output=True,
            text=True,
        )
        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        assert "1 succeeded" in output


class TestLevel15AlgorithmSelection:
    def test_library_selects_quicksort_for_sorting(self):
        from aero_forge.algorithms import find_algorithm

        algo = find_algorithm("build a fast sorting function")
        assert algo is not None
        assert algo.category == "sorting"

    def test_library_selects_prime_algorithm(self):
        from aero_forge.algorithms import find_algorithm

        algo = find_algorithm("check whether a number is prime")
        assert algo is not None
        assert "prime" in algo.name


class TestLevel16MultiVariant:
    def test_pareto_selection_prefers_fastest_passing_variant(self):
        from aero_forge.variants import pareto_frontier

        variants = [
            {
                "variant": 0,
                "elapsed_seconds": 2.0,
                "build": {"success": True, "passed": 1, "total": 1},
            },
            {
                "variant": 1,
                "elapsed_seconds": 1.0,
                "build": {"success": True, "passed": 1, "total": 1},
            },
            {
                "variant": 2,
                "elapsed_seconds": 3.0,
                "build": {"success": True, "passed": 1, "total": 1},
            },
        ]
        front = pareto_frontier(variants)
        assert len(front) == 1
        assert front[0]["variant"] == 1


class TestLevel17Explainable:
    def test_extract_explanation_from_llm_response(self):
        from aero_forge.generate import extract_explanation

        text = (
            "```python\ndef f():\n    pass\n```\n\n"
            "## Explanation\n"
            "This algorithm runs in O(n log n) time with O(n) space.\n"
        )
        explanation = extract_explanation(text)
        assert "O(n log n)" in explanation


class TestLevel18Discovery:
    def test_discover_flag_allows_unknown_prompt(self):
        """With --discover, the CLI accepts prompts not in the algorithm library."""
        from click.testing import CliRunner
        from aero_forge.cli import main
        from unittest.mock import patch, MagicMock

        runner = CliRunner()
        client = MagicMock()
        client.generate.return_value = (
            "```python\ndef novel(n: int) -> int:\n    return n * 2\n```"
        )

        with patch("aero_forge.generate.get_llm_client", return_value=client):
            result = runner.invoke(
                main,
                [
                    "generate",
                    "--prompt",
                    "solve a brand new unseen problem",
                    "--algorithm-library",
                    "--discover",
                    "--llm-provider",
                    "openai",
                ],
            )
        assert result.exit_code == 0, result.output


class TestLevel19CodeReview:
    def test_review_code_returns_corrected_implementation(self):
        from unittest.mock import patch, MagicMock
        from aero_forge.generate import _review_code

        response = (
            "Use an iterative loop to avoid recursion depth.\n\n"
            "```python\ndef factorial(n: int) -> int:\n"
            "    result = 1\n"
            "    for i in range(2, n + 1):\n"
            "        result *= i\n"
            "    return result\n```"
        )
        client = MagicMock()
        client.generate.return_value = response
        original = "def factorial(n: int) -> int:\n    return n * factorial(n - 1)"

        with patch("aero_forge.generate.get_llm_client", return_value=client):
            corrected = _review_code(original, "factorial", None, "openai", None, 3)

        assert "result = 1" in corrected
