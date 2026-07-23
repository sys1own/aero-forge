"""Tests for prompt-driven code generation."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aero_forge.generate import (
    GenerationError,
    extract_code_blocks,
    generate_and_build,
    parse_generated_response,
)


def test_extract_code_blocks():
    text = (
        "Here is the code:\n"
        "```python\n"
        "def square(n):\n    return n * n\n"
        "```\n"
        "And tests:\n"
        "```python\n"
        "def test_square():\n    assert square(4) == 16\n"
        "```"
    )
    blocks = extract_code_blocks(text)
    assert len(blocks) == 2
    assert blocks[0][0] == "python"
    assert "def square" in blocks[0][1]


def test_parse_generated_response_two_blocks():
    text = (
        "```python\ndef square(n):\n    return n * n\n```\n"
        "```python\ndef test_square():\n    assert square(4) == 16\n```"
    )
    impl, tests = parse_generated_response(text)
    assert "def square" in impl
    assert "def test_square" in tests


def test_parse_generated_response_single_block():
    text = (
        "```python\n"
        "def square(n):\n    return n * n\n\n"
        "def test_square():\n    assert square(4) == 16\n"
        "```"
    )
    impl, tests = parse_generated_response(text)
    assert "def square" in impl
    assert "def test_square" in tests


def test_parse_generated_response_no_blocks():
    with pytest.raises(GenerationError):
        parse_generated_response("no code here")


def test_find_algorithm_for_prompt():
    from aero_forge.algorithms import find_algorithm

    algo = find_algorithm("compute the fibonacci number")
    assert algo is not None and "def fibonacci" in algo.source


def test_list_algorithms():
    from aero_forge.algorithms import list_algorithms

    names = list_algorithms()
    assert "fibonacci" in names
    assert "gcd" in names
    assert "is_prime" in names
    assert "matrix_multiply" in names
    assert "quicksort" in names


def _make_mock_client(response: str):
    client = MagicMock()
    client.generate.return_value = response
    return client


def test_generate_and_build_writes_files(tmp_path):
    response = (
        "```python\n"
        "def square(n):\n"
        "    return n * n\n"
        "```\n\n"
        "```python\n"
        "from generated import square\n\n"
        "def test_square():\n"
        "    assert square(4) == 16\n"
        "```"
    )
    with patch(
        "aero_forge.generate.get_llm_client", return_value=_make_mock_client(response)
    ):
        result = generate_and_build(
            "write a function that squares a number",
            output_dir=tmp_path,
            llm_provider="openai",
            build_kwargs={"max_workers": 1, "cache_enabled": False},
        )

    assert result["source_path"] == str(tmp_path / "src" / "generated.py")
    assert result["test_path"] == str(tmp_path / "tests" / "test_generated.py")
    assert (tmp_path / "src" / "generated.py").is_file()
    assert (tmp_path / "tests" / "test_generated.py").is_file()
    assert (tmp_path / "blueprint.aero").is_file()
    assert result["build"]["success"] is True
    assert result["build"]["passed"] == 1


def test_optimize_runs_multiple_iterations(tmp_path):
    """``--optimize`` runs at least three iterations when the builds pass."""
    response = (
        "```python\n"
        "def square(n):\n"
        "    return n * n\n"
        "```\n\n"
        "```python\n"
        "from generated import square\n\n"
        "def test_square():\n"
        "    assert square(4) == 16\n"
        "```"
    )
    calls = []

    def mock_client(*args, **kwargs):
        m = MagicMock()
        m.generate.return_value = response
        calls.append(1)
        return m

    with patch("aero_forge.generate.get_llm_client", side_effect=mock_client):
        result = generate_and_build(
            "write a function that squares a number",
            output_dir=tmp_path,
            llm_provider="openai",
            optimize=True,
            max_iterations=3,
        )

    assert len(result["iterations"]) >= 3
    assert result["iterations"][-1]["build"]["success"] is True


def test_generate_and_build_no_llm_fails(tmp_path):
    with pytest.raises(GenerationError):
        generate_and_build(
            "do something",
            output_dir=tmp_path,
            llm_provider="none",
        )


def test_cli_generate_command(tmp_path):
    """The ``aero-forge generate`` CLI writes files and optionally builds."""
    response = (
        "```python\n"
        "def double(n):\n"
        "    return n * 2\n"
        "```\n\n"
        "```python\n"
        "from generated import double\n\n"
        "def test_double():\n"
        "    assert double(3) == 6\n"
        "```"
    )
    from click.testing import CliRunner
    from aero_forge.cli import main

    runner = CliRunner()
    with patch(
        "aero_forge.generate.get_llm_client", return_value=_make_mock_client(response)
    ):
        result = runner.invoke(
            main,
            [
                "generate",
                "--prompt",
                "write a function that doubles a number",
                "--build",
                "--output-dir",
                str(tmp_path),
                "--llm-provider",
                "openai",
            ],
        )

    assert result.exit_code == 0
    assert (tmp_path / "src" / "generated.py").is_file()
    assert (tmp_path / "tests" / "test_generated.py").is_file()
    assert "Build: 1/1 succeeded" in result.output


def test_blueprint_prompt_generates_and_builds(tmp_path):
    """A blueprint with a ``prompt`` field drives code generation and a build."""
    response = (
        "```python\n"
        "def cube(n):\n"
        "    return n * n * n\n"
        "```\n\n"
        "```python\n"
        "from generated import cube\n\n"
        "def test_cube():\n"
        "    assert cube(3) == 27\n"
        "```"
    )
    blueprint_file = tmp_path / "blueprint.aero"
    blueprint_file.write_text(
        "project: prompt_test\n"
        "prompt: write a function that cubes a number\n"
        "output_dir: ./dist\n"
        "llm:\n"
        "  provider: openai\n"
    )

    with patch(
        "aero_forge.generate.get_llm_client", return_value=_make_mock_client(response)
    ):
        from click.testing import CliRunner
        from aero_forge.cli import main

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["build", str(blueprint_file), "--no-llm"],
        )

    assert result.exit_code == 0
    assert (tmp_path / "dist" / "src" / "generated.py").is_file()
