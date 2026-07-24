"""Generate friendly, conversational summaries of build results."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from aero_forge.llm.clients import get_llm_client

logger = logging.getLogger("aero_forge.build_summary")


def format_build_summary(
    build_result: Dict[str, Any],
    *,
    output_dir: Path,
    prompt: Optional[str] = None,
    function_names: Optional[List[str]] = None,
    benchmark_seconds: Optional[float] = None,
    llm_provider: Optional[str] = None,
    model: Optional[str] = None,
    max_retries: int = 3,
) -> str:
    """Return a 2-4 sentence casual summary of a successful build.

    Falls back to a deterministic template if no LLM provider is available.
    """
    results = build_result.get("results") or []
    passed = sum(1 for r in results if r.get("success"))
    total = len(results)

    names = function_names or []
    if not names and results:
        for r in results:
            funcs = r.get("functions")
            if funcs:
                for name in funcs:
                    if name and name not in names:
                        names.append(str(name))
            else:
                name = r.get("function_name") or r.get("name")
                if name and name not in names:
                    names.append(str(name))

    if not names and prompt:
        # Best-effort extraction from the prompt: assume the first noun after
        # "function" is the function name.
        lowered = prompt.lower()
        if "function" in lowered:
            rest = lowered.split("function")[-1]
            tokens = rest.strip().split()
            if tokens:
                candidate = tokens[0].strip(" ,:()[]{}\"'")
                if candidate and candidate not in ("to", "that", "a", "an"):
                    names = [candidate]

    name_str = ", ".join(f"`{n}`" for n in names) if names else "the function"
    test_note = f"It passed {passed}/{total} tests." if total else "Tests passed."
    timing = (
        f" The build completed in {benchmark_seconds:.3f}s."
        if benchmark_seconds is not None
        else ""
    )

    client = get_llm_client(llm_provider, model=model, max_retries=max_retries)
    if client is not None:
        metrics = {
            "function": name_str,
            "tests_passed": passed,
            "tests_total": total,
            "output_dir": str(output_dir),
            "benchmark_seconds": benchmark_seconds,
            "prompt": prompt,
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a friendly coding assistant. Write a 2-4 sentence casual, "
                    "conversational summary of the build below. Mention what was built, "
                    "whether tests passed, any timing if available, and where the output is. "
                    "Do not use markdown headings or bullet points."
                ),
            },
            {
                "role": "user",
                "content": f"Build metrics: {json.dumps(metrics, indent=2)}",
            },
        ]
        try:
            summary = client.generate(messages, temperature=0.4)
            if summary and summary.strip():
                return summary.strip()
        except Exception as exc:
            logger.warning("LLM summary generation failed: %s", exc)

    return (
        f"Done! I generated {name_str}, compiled it to a Rust extension, and ran the tests. "
        f"{test_note}{timing} The compiled library is in `{output_dir}`. "
        f"You can import it with `from generated import <function_name>`."
    )
