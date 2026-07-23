"""Multi-variant generation, benchmarking, and Pareto selection."""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from aero_forge.generate import generate_and_build


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
) -> List[Dict[str, Any]]:
    """Generate and build ``variants`` implementations of ``prompt``.

    Each variant is generated in its own subdirectory under ``output_dir`` and
    compiled independently.  The returned list contains a result dict for each
    variant with an added ``variant`` index and ``elapsed_seconds`` field.
    """
    results: List[Dict[str, Any]] = []
    for i in range(variants):
        variant_dir = output_dir / f".variant_{i}"
        variant_dir.mkdir(parents=True, exist_ok=True)
        variant_constraints = (
            f"{constraints}\n" if constraints else ""
        ) + f"Variant {i + 1}: use a different algorithm or optimization strategy than the other variants."

        start = time.perf_counter()
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
        )
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


def select_best_variant(
    results: List[Dict[str, Any]],
    output_dir: Path = Path("."),
) -> Dict[str, Any]:
    """Pick the fastest variant that passes all tests.

    Falls back to the variant with the most passing tests if no variant passes
    everything.  The selected source and tests are copied to ``output_dir``.
    """
    front = pareto_frontier(results)

    def sort_key(r: Dict[str, Any]) -> Tuple[int, float, float]:
        build = r.get("build") or {}
        passed = build.get("passed", 0)
        total = build.get("total", 1) or 1
        success = 1 if build.get("success") else 0
        accuracy = passed / total
        return (success, accuracy, -accuracy / max(r.get("elapsed_seconds", 1.0), 1e-9))

    best = max(front, key=sort_key)
    src = Path(best["source_path"])
    tests = Path(best["test_path"])
    out_src = output_dir / "src" / "generated.py"
    out_tests = output_dir / "tests" / "test_generated.py"
    out_src.parent.mkdir(parents=True, exist_ok=True)
    out_tests.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, out_src)
    shutil.copy2(tests, out_tests)

    best["selected"] = True
    best["final_source_path"] = str(out_src)
    best["final_test_path"] = str(out_tests)
    return best
