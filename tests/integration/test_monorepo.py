"""Integration test for the prompt-driven Python-Rust monorepo generator.

Run explicitly with a DeepSeek API key:

    DEEPSEEK_API_KEY=... python -m pytest tests/integration/test_monorepo.py -v
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from aero_forge.monorepo import generate_monorepo


@pytest.mark.skipif(
    not os.getenv("DEEPSEEK_API_KEY") and not os.getenv("AERO_FORGE_API_KEY"),
    reason="Live LLM API key not available",
)
def test_monorepo_builds_and_tests(tmp_path: Path) -> None:
    """End-to-end: a computational prompt becomes a buildable Python-Rust monorepo."""
    output_dir = tmp_path / "monorepo"
    result = generate_monorepo(
        (
            "Implement a pure Python weighted decision matrix evaluator. "
            "It takes a scores matrix (list of list of float), weights (list of float), "
            "and criteria_types (list of 'benefit'/'cost' strings) and returns a list "
            "of weighted scores. This function will become the Rust core of a "
            "Python-Rust monorepo exposed via PyO3."
        ),
        constraints=(
            "Signature: def weighted_decision_matrix(scores: list[list[float]], "
            "weights: list[float], criteria_types: list[str]) -> list[float]\n"
            "On empty input compute the result by iterating over rows; do not use a "
            "top-level `return []`. Use explicit typed loops and no HTML, UI, or print statements."
        ),
        output_dir=output_dir,
        project_name="decision_matrix_monorepo",
        llm_provider=os.getenv("AERO_FORGE_LLM_PROVIDER", "deepseek"),
        model=os.getenv("AERO_FORGE_MODEL", "deepseek-chat"),
        max_retries=3,
        max_tokens=4096,
    )

    assert result["success"], (
        f"Monorepo failed: {result.get('error')}\n"
        f"cargo stderr: {result.get('cargo_error', '')}\n"
        f"pytest stderr: {result.get('pytest_error', '')}"
    )
    assert (output_dir / "rust_core" / "Cargo.toml").is_file()
    assert (output_dir / "python_engine" / "pyproject.toml").is_file()
    assert (output_dir / "Cargo.toml").is_file()
