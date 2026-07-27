"""Explain build errors in plain English and suggest fixes.

If an LLM client is available, the error log and relevant source are sent to the
model with a request for plain-English explanation and actionable suggestions.
Without an LLM, a small set of local heuristics maps common Rust / transpiler
errors to clear messages.
"""

from __future__ import annotations

import re
from typing import Optional

from aero_forge.config import ConfigOverride, Tier
from aero_forge.errors import UnsupportedError, classify_cargo_error
from aero_forge.llm import get_llm_client


def explain_error(
    error_log: str,
    source: Optional[str] = None,
    llm_provider: Optional[str] = None,
    model: Optional[str] = None,
    config_override: Optional[ConfigOverride] = None,
) -> str:
    """Return a formatted explanation and fix suggestions for ``error_log``.

    The function first tries to use a configured LLM.  If no provider is
    available (or no API key is set), it falls back to local heuristics.
    """
    client = (
        get_llm_client(
            llm_provider, model=model, config_override=config_override, tier=Tier.FAST
        )
        if llm_provider
        else None
    )
    if client is not None:
        prompt = _build_explain_prompt(error_log, source)
        try:
            suggestion = client.generate(prompt, temperature=0.2)
            if suggestion:
                return _format_llm_explanation(error_log, suggestion)
        except Exception:
            # LLM unavailable or rate-limited; fall through to local heuristics.
            pass
    return _local_explanation(error_log)


def _build_explain_prompt(error_log: str, source: Optional[str]) -> str:
    context = f"\n\nPython source that produced the error:\n{source}" if source else ""
    return (
        "You are Aero-Forge, a fast, friendly coding co-pilot. "
        "Explain this build error in 2-3 short, punchy sentences. "
        "Translate the dry compiler log into plain language, identify the likely culprit, "
        "and suggest one concrete, minimal fix. Avoid bullet lists and raw JSON.\n\n"
        f"Error log:\n{error_log}{context}"
    )


def _format_llm_explanation(error_log: str, suggestion: str) -> str:
    lines = [
        "Error explanation:",
        "────────────────────────────────────────────────────────────",
    ]
    # Show the first meaningful Rust error line.
    first_error = _first_error_line(error_log)
    if first_error:
        lines.append(f"Error: {first_error}")
        lines.append("")
    lines.append(suggestion.strip())
    return "\n".join(lines)


def _first_error_line(error_log: str) -> Optional[str]:
    """Return the first 'error[...]:' or 'UnsupportedError' line."""
    for line in error_log.splitlines():
        if re.search(r"error\[E\w+\]:", line) or "UnsupportedError" in line:
            return line.strip()
    return None


def _local_explanation(error_log: str) -> str:
    """Fallback explanation when no LLM is available."""
    unsupported_match = re.search(r"UnsupportedError: (.+?)(?:\n|$)", error_log)
    if unsupported_match:
        reason = unsupported_match.group(1)
        return (
            f"Unsupported Python construct: {reason}\n"
            f"Suggestion: rewrite the code to avoid this construct, or use a supported equivalent."
        )
    # Pytest failures are distinct from Cargo/Rust compile failures.
    if re.search(r"test_.*::test_|\bPASSED\b|\bFAILED\b|\bERROR\b", error_log):
        return (
            "One or more generated tests failed.\n"
            "Suggestions:\n"
            "  - Check the traceback in the build log.\n"
            "  - Verify input/output types and edge-case handling.\n"
            "  - Use parity assertions or math.isclose for floating-point comparisons."
        )
    if "mismatched types" in error_log.lower():
        return (
            "Type mismatch between expected and actual Rust types.\n"
            "Suggestions:\n"
            "  - Add Python type annotations to function arguments and return values.\n"
            "  - Ensure numeric literals match the declared type (e.g. 4.0 for float).\n"
            "  - Avoid returning loop indices from a function that returns float."
        )
    if "cannot find value" in error_log.lower():
        return (
            "Rust could not resolve a name.\n"
            "Suggestions:\n"
            "  - Check that all variables are assigned before use.\n"
            "  - Avoid underscore-prefixed names for values you need to reference."
        )
    # Only invoke Cargo-specific classification when the log actually looks like
    # compiler output; otherwise fall back to a generic build error message.
    if re.search(r"(?:Compiling|Finished|error:|error\[|rustc|cargo)", error_log, re.I):
        cargo = classify_cargo_error(error_log)
        return f"Build error:\n{cargo}\nThe exact compiler output is included in the build log."
    return "Build/test failed. The exact output is included in the build log."


def explain_exception(exc: Exception, source: Optional[str] = None) -> str:
    """Explain a transpiler exception in plain English."""
    if isinstance(exc, UnsupportedError):
        return (
            f"Unsupported Python construct: {exc.message}\n"
            "Suggestion: rewrite the code to avoid this construct."
        )
    return explain_error(str(exc), source=source)
