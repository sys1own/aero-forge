"""Multi-variant generation, benchmarking, and Pareto selection."""

from __future__ import annotations

import logging
import shutil
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from aero_forge.config import ConfigOverride
from aero_forge.generate import generate_and_build

logger = logging.getLogger("aero_forge.variants")


def generate_variants(
    prompt: str,
    *,
    variants: int = 3,
    output_dir: Path = Path("."),
    project_name: str = "generated_project",
    constraints: Optional[str] = None,
    llm_provider: Optional[str] = None,
    model: Optional[str] = None,
    max_retries: int = 3,
    prompt_template: Optional[str] = None,
    algorithm_library: bool = False,
    selected_algorithm: Optional[str] = None,
    discover: bool = False,
    explain: bool = False,
    review: bool = False,
    config_override: Optional["ConfigOverride"] = None,
) -> List[Dict[str, Any]]:
    """Generate and build ``variants`` implementations of ``prompt``.

    Each variant is generated in its own subdirectory under ``output_dir`` and
    compiled independently.  Variants that fail generation, compilation, or
    benchmarking are captured as failed entries in the returned list rather
    than aborting the whole batch.
    """
    results: List[Dict[str, Any]] = []
    for i in range(variants):
        variant_dir = output_dir / f".variant_{i}"
        variant_dir.mkdir(parents=True, exist_ok=True)
        variant_constraints = (
            (f"{constraints}\n" if constraints else "")
            + f"Variant {i + 1}: use a different algorithm or optimization strategy than the other variants."
        )

        start = time.perf_counter()
        try:
            result = generate_and_build(
                prompt,
                constraints=variant_constraints,
                output_dir=variant_dir,
                project_name=f"{project_name}_variant_{i}",
                llm_provider=llm_provider,
                model=model,
                max_retries=max_retries,
                prompt_template=prompt_template,
                algorithm_library=algorithm_library,
                selected_algorithm=selected_algorithm,
                discover=discover,
                explain=explain,
                review=review,
                build_kwargs={"max_workers": 1, "cache_enabled": False},
                config_override=config_override,
            )
        except Exception as exc:
            logger.exception("Variant %d failed", i)
            elapsed = time.perf_counter() - start
            result = {
                "variant": i,
                "elapsed_seconds": elapsed,
                "source_path": "",
                "test_path": "",
                "blueprint_path": str(variant_dir / "blueprint.aero"),
                "implementation": "",
                "tests": "",
                "explanation": "",
                "build": {
                    "success": False,
                    "passed": 0,
                    "total": 0,
                    "error": f"{exc.__class__.__name__}: {exc}",
                    "logs": traceback.format_exc(),
                },
                "iterations": [],
            }
        else:
            elapsed = time.perf_counter() - start
            result["variant"] = i
            result["elapsed_seconds"] = elapsed
        results.append(result)
    return results


def pareto_frontier(
    results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return the Pareto-optimal variants.

    Objectives (maximize accuracy, minimize time, minimize memory):
      - accuracy = passed / total (or 0 if no tests)
      - time = elapsed_seconds
      - memory = a constant placeholder (0) because we do not measure RSS here

    A variant dominates another if it has >= accuracy, <= time, and <= memory
    with at least one strict improvement.
    """

    def metrics(r: Dict[str, Any]) -> Tuple[float, float, float]:
        build = r.get("build") or {}
        passed = build.get("passed", 0)
        total = build.get("total", 1) or 1
        accuracy = passed / total if build.get("success") else 0.0
        return (accuracy, r.get("elapsed_seconds", float("inf")), 0.0)

    def dominates(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> bool:
        return (
            a[0] >= b[0]
            and a[1] <= b[1]
            and a[2] <= b[2]
            and (a[0] > b[0] or a[1] < b[1] or a[2] < b[2])
        )

    front: List[Dict[str, Any]] = []
    for r in results:
        m = metrics(r)
        dominated = False
        for other in front[:]:
            om = metrics(other)
            if dominates(om, m):
                dominated = True
                break
            if dominates(m, om):
                front.remove(other)
        if not dominated:
            front.append(r)
    return front


def _accuracy(r: Dict[str, Any]) -> float:
    build = r.get("build") or {}
    if not build.get("success"):
        return 0.0
    passed = build.get("passed", 0)
    total = build.get("total", 1) or 1
    return passed / total


def _build_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-variant results into a single summary build dict."""
    successes = [r for r in results if (r.get("build") or {}).get("success")]
    any_success = bool(successes)
    total_passed = sum((r.get("build") or {}).get("passed", 0) for r in results)
    total_tests = sum((r.get("build") or {}).get("total", 0) for r in results)
    errors = [
        f"variant {r.get('variant', '?')}: {(r.get('build') or {}).get('error', 'unknown error')}"
        for r in results
        if not (r.get("build") or {}).get("success")
    ]
    return {
        "success": any_success,
        "passed": total_passed,
        "total": total_tests,
        "error": "; ".join(errors) if errors else "",
        "logs": "\n".join(
            (r.get("build") or {}).get("logs", "") for r in results if not (r.get("build") or {}).get("success")
        ),
    }


def select_best_variant(
    results: List[Dict[str, Any]],
    output_dir: Path = Path("."),
) -> Dict[str, Any]:
    """Pick the fastest variant that passes all tests.

    Falls back to the variant with the most passing tests if no variant passes
    everything.  If every variant fails, returns a failure payload with the
    per-variant error details.  The selected source and tests are copied to
    ``output_dir`` when a valid variant is available.
    """
    successful = [r for r in results if (r.get("build") or {}).get("success")]
    candidates = successful or results
    front = pareto_frontier(candidates)

    def sort_key(r: Dict[str, Any]) -> Tuple[int, float, float]:
        build = r.get("build") or {}
        passed = build.get("passed", 0)
        total = build.get("total", 1) or 1
        success = 1 if build.get("success") else 0
        accuracy = passed / total
        return (success, accuracy, -accuracy / max(r.get("elapsed_seconds", 1.0), 1e-9))

    best = max(front, key=sort_key)

    src_path = best.get("source_path") or ""
    test_path = best.get("test_path") or ""
    if src_path and Path(src_path).is_file():
        src = Path(src_path)
        tests = Path(test_path) if test_path and Path(test_path).is_file() else None
        out_src = output_dir / "src" / src.name
        out_tests = output_dir / "tests" / (tests.name if tests else src.stem + ".py")
        out_src.parent.mkdir(parents=True, exist_ok=True)
        out_tests.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out_src)
        if tests:
            shutil.copy2(tests, out_tests)
        best["selected"] = True
        best["final_source_path"] = str(out_src)
        best["final_test_path"] = str(out_tests)
    else:
        best["selected"] = False
        best["final_source_path"] = ""
        best["final_test_path"] = ""

    best.setdefault("build", _build_summary(results))
    best.setdefault("source_path", src_path)
    best.setdefault("test_path", test_path)
    best.setdefault("blueprint_path", str(output_dir / "blueprint.aero"))
    best.setdefault("implementation", "")
    best.setdefault("tests", "")
    best.setdefault("explanation", "")
    best.setdefault("iterations", [])
    return best
