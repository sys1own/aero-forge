"""Prompt engineering campaign harness for Aero-Forge."""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from aero_forge.generate import generate_and_build
from aero_forge.prompts import list_templates


@dataclass
class CaseResult:
    """Result of one prompt-template × test-case run."""

    template: str
    case: str
    success: bool
    iterations: int
    first_attempt_success: bool
    compile_passed: bool
    benchmark_seconds: float = 0.0
    speedup: Optional[float] = None
    error: Optional[str] = None


@dataclass
class CampaignReport:
    """Aggregated report across templates and test cases."""

    template: str
    cases: List[CaseResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.cases)

    @property
    def first_attempt_rate(self) -> float:
        if not self.cases:
            return 0.0
        return sum(1 for c in self.cases if c.first_attempt_success) / len(self.cases)

    @property
    def success_rate(self) -> float:
        if not self.cases:
            return 0.0
        return sum(1 for c in self.cases if c.success) / len(self.cases)

    @property
    def average_iterations(self) -> float:
        if not self.cases:
            return 0.0
        return sum(c.iterations for c in self.cases) / len(self.cases)

    @property
    def average_benchmark_seconds(self) -> float:
        if not self.cases:
            return 0.0
        return sum(c.benchmark_seconds for c in self.cases) / len(self.cases)

    def summary(self) -> Dict[str, Any]:
        return {
            "template": self.template,
            "total": self.total,
            "first_attempt_rate": self.first_attempt_rate,
            "success_rate": self.success_rate,
            "average_iterations": self.average_iterations,
            "average_benchmark_seconds": self.average_benchmark_seconds,
            "cases": [asdict(c) for c in self.cases],
        }


DEFAULT_TEST_CASES: List[Dict[str, Any]] = [
    {
        "name": "fibonacci",
        "prompt": "Build a function that computes the nth Fibonacci number.",
        "constraints": "iterative only",
    },
    {
        "name": "factorial",
        "prompt": "Build a function that computes the factorial of a non-negative integer.",
        "constraints": "iterative only",
    },
    {
        "name": "gcd",
        "prompt": "Build a function that computes the greatest common divisor of two integers.",
    },
    {
        "name": "is_prime",
        "prompt": "Build a function that checks whether a number is prime.",
    },
    {
        "name": "mandelbrot",
        "prompt": "Build a function that computes Mandelbrot set iterations for a point (x, y).",
    },
    {
        "name": "matrix_multiply",
        "prompt": "Build a function that multiplies two matrices.",
    },
    {
        "name": "most_frequent",
        "prompt": "Build a function that finds the most frequent element in a list.",
    },
    {
        "name": "dot_product",
        "prompt": "Build a function that computes the dot product of two large vectors.",
    },
]


def run_campaign(
    test_cases: Optional[List[Dict[str, Any]]] = None,
    *,
    templates: Optional[List[str]] = None,
    llm_provider: str = "deepseek",
    model: Optional[str] = None,
    max_iterations: int = 5,
    base_dir: Optional[Path] = None,
) -> List[CampaignReport]:
    """Run the prompt engineering campaign and return per-template reports."""
    cases = test_cases or DEFAULT_TEST_CASES
    templates = templates or list_templates()
    base_dir = base_dir or Path(tempfile.mkdtemp(prefix="aero_forge_prompt_campaign_"))
    base_dir.mkdir(parents=True, exist_ok=True)

    reports: List[CampaignReport] = []
    for template in templates:
        report = CampaignReport(template=template)
        for case in cases:
            case_dir = base_dir / template / case["name"]
            case_dir.mkdir(parents=True, exist_ok=True)
            result = _run_case(
                case,
                template,
                case_dir,
                llm_provider,
                model,
                max_iterations,
            )
            report.cases.append(result)
        reports.append(report)
    return reports


def _run_case(
    case: Dict[str, Any],
    template: str,
    case_dir: Path,
    llm_provider: str,
    model: Optional[str],
    max_iterations: int,
) -> CaseResult:
    """Run a single test case and return its result."""
    start = time.perf_counter()
    try:
        result = generate_and_build(
            case["prompt"],
            constraints=case.get("constraints"),
            output_dir=case_dir,
            llm_provider=llm_provider,
            model=model,
            max_iterations=max_iterations,
            build_kwargs={"max_workers": 1, "cache_enabled": False},
            prompt_template=template,
        )
    except Exception as exc:  # pragma: no cover - defensive
        elapsed = time.perf_counter() - start
        return CaseResult(
            template=template,
            case=case["name"],
            success=False,
            iterations=0,
            first_attempt_success=False,
            compile_passed=False,
            benchmark_seconds=elapsed,
            error=str(exc),
        )

    elapsed = time.perf_counter() - start
    build = result.get("build") or {}
    success = bool(build.get("success"))
    iterations = 1 if success else max_iterations
    first_attempt = success and len(result.get("iterations", [])) <= 1

    return CaseResult(
        template=template,
        case=case["name"],
        success=success,
        iterations=iterations,
        first_attempt_success=first_attempt,
        compile_passed=success,
        benchmark_seconds=elapsed,
        error=build.get("logs") if not success else None,
    )


def print_report(reports: List[CampaignReport]) -> None:
    """Print a human-readable report to stdout."""
    for report in reports:
        print(f"PROMPT VERSION: {report.template}")
        print("-" * 40)
        for case in report.cases:
            print(f"Test {case.case}:")
            print(f"  - First-attempt success: {case.first_attempt_success}")
            print(f"  - Iterations: {case.iterations}")
            print(f"  - Compile passed: {case.compile_passed}")
            print(f"  - Benchmark seconds: {case.benchmark_seconds:.3f}")
            if case.error:
                print(f"  - Error: {case.error[:200]}")
        print()
        print(
            f"SUMMARY: success_rate={report.success_rate:.2%}, "
            f"first_attempt_rate={report.first_attempt_rate:.2%}, "
            f"avg_iterations={report.average_iterations:.2f}, "
            f"avg_benchmark_seconds={report.average_benchmark_seconds:.3f}"
        )
        print()


def save_report(reports: List[CampaignReport], path: Path) -> None:
    """Serialize the report to JSON."""
    payload = [report.summary() for report in reports]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    provider = os.getenv("AERO_FORGE_LLM_PROVIDER") or "deepseek"
    model = os.getenv("AERO_FORGE_MODEL")
    reports = run_campaign(llm_provider=provider, model=model)
    print_report(reports)
    out = Path("prompt_engineering_report.json")
    save_report(reports, out)
    print(f"Saved JSON report to {out}")
