"""Tests for the prompt engineering campaign harness."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aero_forge.prompt_engineering import (
    CampaignReport,
    CaseResult,
    DEFAULT_TEST_CASES,
    run_campaign,
    save_report,
)
from aero_forge.prompts import list_templates


def test_campaign_report_aggregation():
    report = CampaignReport(template="v5_balanced")
    report.cases = [
        CaseResult(
            template="v5_balanced",
            case="fibonacci",
            success=True,
            iterations=1,
            first_attempt_success=True,
            compile_passed=True,
            benchmark_seconds=0.5,
        ),
        CaseResult(
            template="v5_balanced",
            case="factorial",
            success=True,
            iterations=2,
            first_attempt_success=False,
            compile_passed=True,
            benchmark_seconds=1.0,
        ),
        CaseResult(
            template="v5_balanced",
            case="gcd",
            success=False,
            iterations=5,
            first_attempt_success=False,
            compile_passed=False,
            benchmark_seconds=0.2,
            error="compile failed",
        ),
    ]
    assert report.total == 3
    assert report.first_attempt_rate == pytest.approx(1 / 3)
    assert report.success_rate == pytest.approx(2 / 3)
    assert report.average_iterations == pytest.approx(8 / 3)
    assert report.average_benchmark_seconds == pytest.approx(1.7 / 3)


def test_run_campaign_with_mock_llm(tmp_path):
    response = (
        "```python\n"
        "def fibonacci(n):\n"
        "    if n <= 1:\n"
        "        return n\n"
        "    a, b = 0, 1\n"
        "    for _ in range(2, n + 1):\n"
        "        a, b = b, a + b\n"
        "    return b\n"
        "```\n\n"
        "```python\n"
        "from generated import fibonacci\n\n"
        "def test_fibonacci():\n"
        "    assert fibonacci(10) == 55\n"
        "```"
    )

    def mock_client(*args, **kwargs):
        m = MagicMock()
        m.generate.return_value = response
        return m

    with patch("aero_forge.generate.get_llm_client", side_effect=mock_client):
        reports = run_campaign(
            test_cases=DEFAULT_TEST_CASES[:1],
            templates=["v1_minimal", "v5_balanced"],
            llm_provider="openai",
            base_dir=tmp_path,
        )

    assert len(reports) == 2
    for report in reports:
        assert report.total == 1
        case = report.cases[0]
        assert case.case == "fibonacci"
        assert case.compile_passed is True


def test_save_report(tmp_path):
    report = CampaignReport(template="v5_balanced")
    report.cases.append(
        CaseResult(
            template="v5_balanced",
            case="factorial",
            success=True,
            iterations=1,
            first_attempt_success=True,
            compile_passed=True,
            benchmark_seconds=0.4,
        )
    )
    path = tmp_path / "report.json"
    save_report([report], path)
    assert path.is_file()
    assert '"template": "v5_balanced"' in path.read_text(encoding="utf-8")


@pytest.mark.skipif(
    not os.getenv("DEEPSEEK_API_KEY") and not os.getenv("AERO_FORGE_API_KEY"),
    reason="DeepSeek API key not configured",
)
def test_run_campaign_live_mini(tmp_path):
    """Run a tiny live campaign against DeepSeek when a key is available."""
    reports = run_campaign(
        test_cases=DEFAULT_TEST_CASES[:1],
        templates=["v5_balanced"],
        llm_provider="deepseek",
        base_dir=tmp_path,
        max_iterations=3,
    )
    assert len(reports) == 1
    assert reports[0].total == 1
