"""Extract structured build actions from Co-pilot Markdown responses."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional, Tuple

import yaml

logger = logging.getLogger("aero_forge.copilot.action_parser")


_CODE_FENCE_RE = re.compile(
    r"```(?:yaml|json)\s+blueprint\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


def _normalize_target(raw: Any) -> Optional[str]:
    """Return a recognized Aero-Forge target name or None."""
    if not isinstance(raw, str):
        return None
    value = raw.strip().lower().replace(" ", "_").replace("-", "_")
    valid = {
        "pure_python",
        "pure_rust",
        "hybrid_rust_python",
        "hybrid_cpp_python",
        "hybrid_cpp_rust",
        "multi_crate_rust",
        "tri_polyglot_rust_cpp_python",
        "wasm",
    }
    if value in valid:
        return value
    synonyms = {
        "rust_python": "hybrid_rust_python",
        "python_rust": "hybrid_rust_python",
        "cpp_python": "hybrid_cpp_python",
        "python_cpp": "hybrid_cpp_python",
        "c++_python": "hybrid_cpp_python",
        "rust_cpp": "hybrid_cpp_rust",
        "cpp_rust": "hybrid_cpp_rust",
        "tri_polyglot": "tri_polyglot_rust_cpp_python",
    }
    return synonyms.get(value)


def _normalize_acceleration(raw: Any) -> str:
    """Return a recognized acceleration policy string."""
    if not isinstance(raw, str):
        return "Selective Acceleration (Auto-Detect Heavy Compute)"
    lowered = raw.lower()
    if "force" in lowered or "force native" in lowered:
        return "Force Native Bridge"
    if "bypass" in lowered or "standard" in lowered:
        return "Standard Runtime (Bypass Bridge)"
    return "Selective Acceleration (Auto-Detect Heavy Compute)"


def extract_build_contract(text: str) -> Optional[Dict[str, Any]]:
    """Parse a YAML or JSON build contract from a Markdown code fence.

    Only explicitly tagged blocks (```yaml blueprint ...``` or
    ```json blueprint ...```) are treated as build contracts. This avoids
    matching broad document keys like ``architecture_overview``.
    """
    if not text or not text.strip():
        return None

    match = _CODE_FENCE_RE.search(text)
    if not match:
        return None

    payload = match.group(1).strip()
    if not payload:
        return None

    parsed: Any = None
    try:
        parsed = yaml.safe_load(payload)
    except yaml.YAMLError:
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            logger.debug("Build contract block is not valid YAML/JSON")
            return None

    if not isinstance(parsed, dict):
        return None

    prompt = parsed.get("prompt") or parsed.get("build_prompt")
    target = _normalize_target(parsed.get("target", parsed.get("architecture")))
    if not prompt or not target:
        return None

    return {
        "prompt": str(prompt).strip(),
        "target": target,
        "acceleration": _normalize_acceleration(parsed.get("acceleration")),
    }


def _infer_target_from_text(text: str) -> Optional[str]:
    """Infer a build target from prose when no explicit contract is present."""
    lowered = text.lower()
    has_python = "python" in lowered or "pyo3" in lowered
    has_rust = "rust" in lowered or "cargo" in lowered or "pyo3" in lowered
    has_cpp = (
        "cpp" in lowered
        or "c++" in lowered
        or "cxx" in lowered
        or "clang" in lowered
        or "g++" in lowered
    )

    if has_python and has_rust and has_cpp:
        return "tri_polyglot_rust_cpp_python"
    if has_python and has_rust:
        return "hybrid_rust_python"
    if has_python and has_cpp:
        return "hybrid_cpp_python"
    if has_rust and has_cpp:
        return "hybrid_cpp_rust"
    if has_rust:
        return "pure_rust"
    if has_cpp:
        return "hybrid_cpp_python"
    return "pure_python"


def _has_build_intent(text: str) -> bool:
    """Return True when the text clearly requests a build/feature."""
    lowered = text.lower()
    keywords = (
        "build",
        "create",
        "implement",
        "generate",
        "function",
        "module",
        "project",
        "fast",
        "speed",
        "accelerate",
        "optimize",
        "blueprint",
    )
    return any(k in lowered for k in keywords)


def parse_action_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort extraction of a PROPOSE_BUILD action from any assistant text."""
    contract = extract_build_contract(text)
    if contract:
        return {
            "type": "PROPOSE_BUILD",
            "params": {
                "prompt": contract["prompt"],
                "target": contract["target"],
                "acceleration": contract["acceleration"],
            },
        }

    target = _infer_target_from_text(text)
    if not target or not _has_build_intent(text):
        return None

    prompt = text.strip()
    if len(prompt) > 500:
        prompt = prompt[:500].rsplit(" ", 1)[0] + "..."
    if prompt.startswith("{"):
        prompt = prompt[:200]

    return {
        "type": "PROPOSE_BUILD",
        "params": {
            "prompt": prompt,
            "target": target,
            "acceleration": _normalize_acceleration(text),
        },
    }


def _maybe_parse_json_object(response: str) -> Optional[Tuple[str, Optional[Dict[str, Any]]]]:
    """Handle legacy top-level JSON responses with ``reply`` and ``action``."""
    text = response.strip()
    if not text.startswith("{"):
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    reply = str(parsed.get("reply", response)).strip()
    action = parsed.get("action")
    if action and isinstance(action, dict):
        params = action.get("params") or {}
        prompt = params.get("prompt") or reply
        target = _normalize_target(params.get("target", params.get("architecture")))
        if prompt and target:
            action = {
                "type": "PROPOSE_BUILD",
                "params": {
                    "prompt": str(prompt).strip(),
                    "target": target,
                    "acceleration": _normalize_acceleration(params.get("acceleration")),
                },
            }
            return reply, action
    return reply, None


def parse_copilot_response(response: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Split an assistant response into a Markdown reply and an optional action.

    Supports, in order:
    - ``yaml blueprint`` / ``json blueprint`` fenced build contracts
    - legacy top-level JSON objects with ``reply``/``action``
    - best-effort extraction from plain prose
    """
    if not response or not response.strip():
        return "", None

    contract = extract_build_contract(response)
    if contract:
        reply = _CODE_FENCE_RE.sub("\n", response).strip()
        action = {
            "type": "PROPOSE_BUILD",
            "params": {
                "prompt": contract["prompt"],
                "target": contract["target"],
                "acceleration": contract["acceleration"],
            },
        }
        return reply, action

    legacy = _maybe_parse_json_object(response)
    if legacy is not None:
        return legacy

    action = parse_action_from_text(response)
    return response.strip(), action
