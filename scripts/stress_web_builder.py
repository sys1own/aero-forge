#!/usr/bin/env python3
"""Stress-test the aero-forge web builder / generation pipeline end-to-end.

This harness runs three demanding prompt categories through the live
DeepSeek-backed pipeline:

1. Deep Recursive Module — a single module with multiple helper functions,
   queue/stack iteration, and set/dict operations.
2. Cross-Language FFI — a full Python-Rust monorepo (PyO3 core + Python
   async service + benchmarks + tests).
3. Complex Algorithmic Logic — a non-trivial numeric computation with
   matrix/vector boundaries and float comparisons.

Usage:
    DEEPSEEK_API_KEY=... python scripts/stress_web_builder.py
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aero_forge.generate import generate_and_build
from aero_forge.monorepo import generate_monorepo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("stress_web_builder")


Provider = os.getenv("AERO_FORGE_LLM_PROVIDER", "deepseek")
Model = os.getenv("AERO_FORGE_MODEL", "deepseek-chat")


def _out_dir(name: str) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    return Path(f"/tmp/aero_stress_{name}_{ts}")


def _run_case(
    name: str,
    fn: Callable[..., Dict[str, Any]],
    args: tuple,
    kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    out = _out_dir(name)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("STRESS CASE: %s", name)
    start = time.perf_counter()
    try:
        result = fn(*args, output_dir=out, **kwargs)
    except Exception as exc:
        elapsed = time.perf_counter() - start
        logger.exception("Case %s raised an exception", name)
        return {
            "name": name,
            "success": False,
            "elapsed_seconds": round(elapsed, 2),
            "error": f"{exc.__class__.__name__}: {exc}",
        }
    elapsed = time.perf_counter() - start

    success = result.get("success")
    if success is None:
        build = result.get("build") or {}
        success = bool(build.get("success"))
    success = bool(success)
    summary: Dict[str, Any] = {
        "name": name,
        "success": success,
        "elapsed_seconds": round(elapsed, 2),
        "output_dir": str(out),
    }
    if "primary_function" in result:
        summary["primary_function"] = result["primary_function"]
    if "files" in result:
        summary["files"] = result["files"]
    if "cargo_output" in result:
        summary["cargo_output"] = result["cargo_output"]
    if "pytest_output" in result:
        summary["pytest_output"] = result["pytest_output"]
    if "build" in result:
        summary["build"] = {
            "success": (result.get("build") or {}).get("success", False),
            "error": (result.get("build") or {}).get("error", ""),
        }
    if not success:
        summary["error"] = result.get("error", "")
        summary["cargo_error"] = result.get("cargo_error", "")
        summary["pytest_error"] = result.get("pytest_error", "")

    status = "PASSED" if success else "FAILED"
    logger.info("[%s] %s in %.2fs", status, name, elapsed)
    if not success:
        for key in ("error", "cargo_error", "pytest_error"):
            value = summary.get(key, "")
            if value:
                logger.error("%s: %s", key, value[:2000])
    return summary


def deep_recursive() -> Dict[str, Any]:
    """Prompt that exercises graph/queue loops and helper functions."""
    prompt = (
        "Implement a pure Python function named `topological_sort` that performs "
        "topological sorting on a directed acyclic graph represented as an "
        "adjacency dict (dict[str, list[str]]). Use Kahn's algorithm: a queue "
        "of nodes with zero in-degree, a dict for in-degree counts, and a "
        "set or list for the result. Include type hints and ensure the "
        "function works for empty input by returning an empty list."
    )
    constraints = (
        "Signature: def topological_sort(graph: dict[str, list[str]]) -> list[str]\n"
        "Example: graph={'A':['B','C'],'B':['D'],'C':['D'],'D':[]}; "
        "result is one of ['A','B','C','D'] or ['A','C','B','D']."
    )
    return _run_case(
        "deep_recursive_topological_sort",
        generate_and_build,
        (prompt,),
        {
            "constraints": constraints,
            "project_name": "topological_sort",
            "llm_provider": Provider,
            "model": Model,
            "max_retries": 3,
            "max_tokens": 4096,
            "build_kwargs": {
                "max_workers": 1,
                "cache_enabled": False,
                "max_iterations": 1,
            },
        },
    )


def ffi_monorepo() -> Dict[str, Any]:
    """Prompt that generates a full Python-Rust monorepo."""
    prompt = (
        "Implement a pure Python weighted decision matrix evaluator. It takes a "
        "scores matrix (list of list of float), weights (list of float), and "
        "criteria_types (list of 'benefit'/'cost' strings) and returns a list "
        "of weighted scores. This function will become the Rust core of a "
        "Python-Rust monorepo exposed via PyO3."
    )
    constraints = (
        "Signature: def weighted_decision_matrix(scores: list[list[float]], weights: list[float], criteria_types: list[str]) -> list[float]\n"
        "On empty input compute the result by iterating over rows; do not use a top-level `return []`. "
        "Use explicit typed loops and no HTML, UI, or print statements."
    )
    return _run_case(
        "ffi_monorepo",
        generate_monorepo,
        (prompt,),
        {
            "constraints": constraints,
            "project_name": "decision_matrix_monorepo",
            "llm_provider": Provider,
            "model": Model,
            "max_retries": 3,
            "max_tokens": 4096,
            "progress_callback": lambda m: logger.info("[monorepo] %s", m),
        },
    )


def complex_algorithmic() -> Dict[str, Any]:
    """Prompt that exercises matrix/vector math and float boundaries."""
    prompt = (
        "Implement a pure Python function named `weighted_moving_average` that "
        "computes a simple weighted moving average over a list of floats. "
        "Given a list `values` and an integer `window`, return a list where each "
        "element is the average of the previous `window` values (or fewer at the "
        "start). If `values` is empty or `window` is zero, return an empty list. "
        "Use explicit loops, no list comprehensions, and include type hints."
    )
    constraints = (
        "Signature: def weighted_moving_average(values: list[float], window: int) -> list[float]\n"
        "Example: values=[1.0,2.0,3.0,4.0], window=2 -> [1.0,1.5,2.5,3.5]."
    )
    return _run_case(
        "complex_weighted_moving_average",
        generate_and_build,
        (prompt,),
        {
            "constraints": constraints,
            "project_name": "weighted_moving_average",
            "llm_provider": Provider,
            "model": Model,
            "max_retries": 3,
            "max_tokens": 4096,
            "build_kwargs": {
                "max_workers": 1,
                "cache_enabled": False,
                "max_iterations": 1,
            },
        },
    )


def main() -> int:
    if not os.getenv("DEEPSEEK_API_KEY") and not os.getenv("AERO_FORGE_API_KEY"):
        logger.error(
            "DEEPSEEK_API_KEY or AERO_FORGE_API_KEY must be set in the environment."
        )
        return 1

    results: List[Dict[str, Any]] = [
        deep_recursive(),
        ffi_monorepo(),
        complex_algorithmic(),
    ]

    total = len(results)
    passed = sum(1 for r in results if r["success"])
    report = {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "results": results,
    }
    report_path = Path("/tmp/aero_stress_web_builder_report.json")
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    logger.info("=" * 60)
    logger.info(
        "Stress run complete: %d/%d passed; report written to %s",
        passed,
        total,
        report_path,
    )
    for r in results:
        status = "PASS" if r["success"] else "FAIL"
        logger.info("  [%s] %s (%.2fs)", status, r["name"], r["elapsed_seconds"])

    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
