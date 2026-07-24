"""Explain build errors in plain English and suggest fixes.

If an LLM client is available, the error log and relevant source are sent to the
model with a request for plain-English explanation and actionable suggestions.
Without an LLM, a small set of local heuristics maps common Rust / transpiler
errors to clear messages.
"""

from __future__ import annotations

import re
from typing import Optional

from aero_forge.config import ConfigOverride
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
            llm_provider, model=model, config_override=config_override
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
        "You are an expert Rust/Python engineer helping a user fix an error from a "
        "Python-to-Rust transpiler called Aero-Forge.\n\n"
        "Explain the following build error in plain English.  Identify the likely cause, "
        "the line/construct involved, and give one or two concrete, minimal fixes the "
        "user can apply.  Keep the response concise and use bullet points.\n\n"
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
    cargo = classify_cargo_error(error_log)
    unsupported_match = re.search(r"UnsupportedError: (.+?)(?:\n|$)", error_log)
    if unsupported_match:
        reason = unsupported_match.group(1)
        return (
            f"Unsupported Python construct: {reason}\n"
            f"Suggestion: rewrite the code to avoid this construct, or use a supported equivalent."
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
    return f"Build error:\n{cargo}\nUse --verbose to see the full compiler output."


def explain_exception(exc: Exception, source: Optional[str] = None) -> str:
    """Explain a transpiler exception in plain English."""
    if isinstance(exc, UnsupportedError):
        return (
            f"Unsupported Python construct: {exc.message}\n"
            "Suggestion: rewrite the code to avoid this construct."
        )
    return explain_error(str(exc), source=source)
