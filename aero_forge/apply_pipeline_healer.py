"""Self-healing harness for ``apply_pipeline``.

The runner iterates over ``tests/test_apply_pipeline.py``. When a failure is
detected, it routes the failing context to a reasoning LLM (DeepSeek by
preference, with OpenRouter/DeepSeek as a fallback) and asks for a corrected
implementation of ``aero_forge/apply_pipeline.py``.
"""

from __future__ import annotations

import os
import subprocess
import textwrap
from pathlib import Path
from typing import Optional

from aero_forge.llm.clients import get_llm_client


_THIS_DIR = Path(__file__).resolve().parent
_SOURCE_PATH = _THIS_DIR / "apply_pipeline.py"
_TEST_TARGET = "tests/test_apply_pipeline.py"


def _deepseek_client(model: Optional[str] = None):
    """Return an LLM client routed to a DeepSeek reasoning model.

    Preference order:
      1. ``DEEPSEEK_API_KEY`` -> DeepSeek API directly.
      2. ``OPENROUTER_API_KEY`` -> OpenRouter with a DeepSeek model.
    """
    if os.getenv("DEEPSEEK_API_KEY"):
        return get_llm_client(provider="deepseek", model=model or "deepseek-reasoner")
    if os.getenv("OPENROUTER_API_KEY"):
        return get_llm_client(
            provider="openrouter",
            model=model or "deepseek/deepseek-reasoner",
        )
    return None


def _build_prompt(source: str, output: str) -> str:
    return textwrap.dedent(
        f"""\
        The following Python module processes packed RGBA pixel buffers.
        The pytest output below shows failures. Fix the source so all tests pass.

        Requirements:
        - Do not change the public function signature ``apply_pipeline(pixels, width, height, operation='grayscale')``.
        - Handle missing alpha channels, short buffers, extra trailing bytes, zero/negative dimensions, and bytes/bytearray inputs.
        - Return a flat list of ints of length ``width * height * 4``.
        - Supported operations: ``grayscale``, ``invert``, ``noop``.
        - Only return the corrected Python source code, no explanations or markdown fences.

        Current source:
        ```python
        {source}
        ```

        Pytest output:
        {output}
        """
    )


def self_heal_apply_pipeline(max_iterations: int = 3) -> bool:
    """Run ``tests/test_apply_pipeline.py`` and patch ``apply_pipeline.py`` via DeepSeek until it passes."""
    for iteration in range(max_iterations):
        result = subprocess.run(
            ["python", "-m", "pytest", _TEST_TARGET, "-q"],
            cwd=str(_THIS_DIR.parent),
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return True

        client = _deepseek_client()
        if client is None:
            raise RuntimeError(
                "No DeepSeek/OpenRouter API key available for self-healing."
            )

        original_source = _SOURCE_PATH.read_text(encoding="utf-8")
        prompt = _build_prompt(original_source, result.stdout + "\n" + result.stderr)
        fixed_source = client.generate(prompt, temperature=0.1)
        if not fixed_source:
            raise RuntimeError("LLM returned empty response for self-healing patch.")

        _SOURCE_PATH.write_text(fixed_source, encoding="utf-8")

    return False


if __name__ == "__main__":  # pragma: no cover
    ok = self_heal_apply_pipeline()
    print("apply_pipeline self-healing:", "PASSED" if ok else "FAILED")
