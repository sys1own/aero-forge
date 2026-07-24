#!/usr/bin/env python3
"""Headless LLM build matrix for aero-forge end-to-end validation.

Runs a diverse set of prompts through the live DeepSeek-backed generation and
build pipeline and reports which ones pass all tests.  This is intentionally
run outside the normal pytest harness so it can use real API calls without
making CI flaky.

Usage:
    DEEPSEEK_API_KEY=... python scripts/test_llm_build_matrix.py
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure the repo root is on sys.path so the script can be invoked from anywhere.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aero_forge.generate import generate_and_build

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("test_llm_build_matrix")


def _default_output_dir(prompt_name: str) -> Path:
    """Return a unique output directory for this run."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    return Path(f"/tmp/llm_matrix_{prompt_name}_{ts}")


MATRIX: List[Tuple[str, str, str, str]] = [
    (
        "topological_sort",
        "Write a Python function named `topological_sort` that performs topological "
        "sort on a directed acyclic graph represented as an adjacency dict. Return "
        "a list of node labels. Use while loops and a queue/stack. Include strict type hints.",
        "def topological_sort(graph: dict[str, list[str]]) -> list[str]",
        "Example test: graph={'A':['B','C'],'B':['D'],'C':['D'],'D':[]}; expected one of ['A','B','C','D'] or ['A','C','B','D'].",
    ),
    (
        "decision_matrix_pro",
        "Write a pure Python function named `decision_matrix_pro` that evaluates a "
        "weighted decision matrix. Input is a list of options (list[list[float]]) "
        "and a list of criteria weights (list[float]). Output is a tuple of "
        "(list[float], int): the list of scores and the index of the best option. "
        "Emit only a typed function and tests, no HTML or UI.",
        "def decision_matrix_pro(options: list[list[float]], weights: list[float]) -> tuple[list[float], int]",
        "Example test: options=[[1.0,2.0],[3.0,4.0],[5.0,6.0]]; weights=[0.5,0.6]; expected scores=[1.7,3.9,6.1], best=2 (use pytest.approx for the scores).",
    ),
    (
        "dict_key_evaluator",
        "Write a Python function named `dict_key_evaluator` that takes a "
        "dict[str, float] of scores and a threshold float, iterates with "
        "dict.items(), and returns a list of keys whose values exceed the "
        "threshold. Include type hints.",
        "def dict_key_evaluator(scores: dict[str, float], threshold: float) -> list[str]",
        "Example test: scores={'a':1.0,'b':2.5,'c':0.5,'d':3.0}; threshold=1.5; expected ['b','d'].",
    ),
    (
        "command_palette_search",
        "Create a Python function named `command_palette_search` that filters a list "
        "of command names by a search query and returns matching commands. Use a "
        "simple linear scan and case-insensitive substring matching (query.lower() in "
        "cmd.lower()). Do NOT use binary search, chr, ord, or startswith. It should "
        "be a pure Python function, not a web app or HTML page.",
        "def command_palette_search(commands: list[str], query: str) -> list[str]",
        "Example test: commands=['Save As','Open File','Close Window']; query='open'; expected ['Open File'].",
    ),
]


def run_matrix() -> Dict[str, Any]:
    """Run every prompt through generate_and_build and collect results."""
    results: List[Dict[str, Any]] = []
    provider = os.getenv("AERO_FORGE_LLM_PROVIDER", "deepseek")
    model = os.getenv("AERO_FORGE_MODEL", "deepseek-chat")

    for name, prompt, signature_hint, example in MATRIX:
        output_dir = _default_output_dir(name)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        constraints = f"Signature: {signature_hint}"
        if example:
            constraints += f"\n{example}"

        logger.info("=" * 60)
        logger.info("Building %s with prompt:\n%s", name, prompt)
        logger.info("Expected signature hint: %s", signature_hint)
        logger.info("Constraints:\n%s", constraints)

        start = time.perf_counter()
        try:
            result = generate_and_build(
                prompt,
                constraints=constraints,
                output_dir=output_dir,
                project_name=name,
                llm_provider=provider,
                model=model,
                max_retries=3,
                build_kwargs={
                    "max_workers": 1,
                    "cache_enabled": False,
                    "max_iterations": 1,
                },
            )
        except Exception as exc:
            elapsed = time.perf_counter() - start
            logger.exception("Prompt %s raised an exception", name)
            result = {
                "source_path": "",
                "test_path": "",
                "implementation": "",
                "tests": "",
                "explanation": "",
                "build": {
                    "success": False,
                    "error": f"{exc.__class__.__name__}: {exc}",
                    "logs": getattr(exc, "output", ""),
                },
            }
        else:
            elapsed = time.perf_counter() - start

        build = result.get("build") or {}
        passed = bool(build.get("success"))

        summary = {
            "name": name,
            "prompt": prompt,
            "expected_signature": signature_hint,
            "elapsed_seconds": round(elapsed, 2),
            "success": passed,
            "source_path": result.get("source_path", ""),
            "test_path": result.get("test_path", ""),
            "implementation": result.get("implementation", ""),
            "tests": result.get("tests", ""),
            "build": {
                "success": build.get("success", False),
                "error": build.get("error", ""),
                "logs": build.get("logs", ""),
                "passed": build.get("passed", 0),
                "total": build.get("total", 0),
            },
        }
        results.append(summary)

        status = "PASSED" if passed else "FAILED"
        logger.info("[%s] %s in %.2fs", status, name, elapsed)
        if not passed:
            error = build.get("error", "") or build.get("logs", "")
            logger.error("Failure for %s:\n%s", name, error[:2000])

    total = len(results)
    passed = sum(1 for r in results if r["success"])
    report = {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "results": results,
    }
    return report


def main() -> int:
    if not os.getenv("DEEPSEEK_API_KEY") and not os.getenv("AERO_FORGE_API_KEY"):
        logger.error(
            "DEEPSEEK_API_KEY or AERO_FORGE_API_KEY must be set in the environment."
        )
        return 1

    report = run_matrix()
    report_path = Path("/tmp/llm_build_matrix_report.json")
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    logger.info("=" * 60)
    logger.info(
        "Matrix complete: %d/%d passed; report written to %s",
        report["passed"],
        report["total"],
        report_path,
    )
    for r in report["results"]:
        status = "PASS" if r["success"] else "FAIL"
        logger.info("  [%s] %s", status, r["name"])

    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
