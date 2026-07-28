"""Co-pilot response post-processor.

Guarantees that every Co-pilot reply sent to the UI is human-readable
Markdown with an optional YAML build contract, even when the underlying
model emits raw JSON.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional, Tuple

import yaml

from aero_forge.copilot.action_parser import (
    _normalize_acceleration,
    _normalize_target,
    parse_action_from_text,
    parse_copilot_response,
)

logger = logging.getLogger("aero_forge.copilot.agent")


def _looks_like_json(text: str) -> bool:
    """Return True if the text appears to be a raw JSON object/list."""
    if not text:
        return False
    stripped = text.strip()
    return stripped.startswith(("{", "["))


def _extract_action_from_json(data: Any) -> Optional[Dict[str, Any]]:
    """Build a PROPOSE_BUILD action from a parsed JSON payload."""
    if not isinstance(data, dict):
        return None

    action_data = data.get("action")
    if not action_data or not isinstance(action_data, dict):
        return None

    params = action_data.get("params") or {}
    prompt = (
        params.get("prompt")
        or data.get("reply")
        or data.get("build_prompt")
    )
    target = _normalize_target(
        params.get("target")
        or params.get("architecture")
        or data.get("target")
        or data.get("architecture")
    )
    if not prompt or not target:
        return None

    return {
        "type": "PROPOSE_BUILD",
        "params": {
            "prompt": str(prompt).strip(),
            "target": target,
            "acceleration": _normalize_acceleration(params.get("acceleration")),
        },
    }


def _build_yaml_contract(action: Dict[str, Any]) -> str:
    """Render a YAML blueprint contract from a PROPOSE_BUILD action."""
    params = action.get("params") or {}
    contract = {
        "prompt": str(params.get("prompt", "")).strip(),
        "target": params.get("target", "pure_python"),
        "acceleration": params.get(
            "acceleration", "Selective Acceleration (Auto-Detect Heavy Compute)"
        ),
    }
    return yaml.safe_dump(contract, sort_keys=False, default_flow_style=False).strip()


def _markdown_reply_from_action(action: Dict[str, Any], fallback_text: str = "") -> str:
    """Generate a Markdown explanation from a build action."""
    params = action.get("params") or {}
    target = params.get("target", "pure_python")
    prompt = str(params.get("prompt", fallback_text)).strip()

    target_label = target.replace("_", " ").title()
    lines = [
        "### Architecture Overview",
        f"I propose a **{target_label}** build for this request.",
        "",
        "#### Components & Strategy",
        f"- The selected target mode is `{target}`.",
        f"- The build prompt is designed for the Aero-Forge engine.",
        "",
        "#### Build Contract",
        "```yaml blueprint",
        _build_yaml_contract(action),
        "```",
    ]
    return "\n".join(lines)


def _wrap_raw_json(response: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Convert a raw JSON response into Markdown + YAML blueprint."""
    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        # Not valid JSON; treat as plain text and best-effort extract an action.
        action = parse_action_from_text(response)
        if action:
            return _markdown_reply_from_action(action, response), action
        return response.strip(), None

    action = _extract_action_from_json(data) if isinstance(data, dict) else None
    if action:
        # Always render a structured Markdown explanation for raw JSON actions.
        return _markdown_reply_from_action(action, response), action

    # No actionable payload. Prefer a human reply field if present.
    if isinstance(data, dict):
        raw_reply = str(data.get("reply", "")).strip()
        if raw_reply and not _looks_like_json(raw_reply):
            return raw_reply, None

    # No reply: pretty-print the JSON inside a Markdown code block.
    return (
        "### Response\n```json\n" + json.dumps(data, indent=2) + "\n```",
        None,
    )


def _has_markdown_heading(text: str) -> bool:
    """Return True if the text already contains a Markdown heading."""
    return bool(re.search(r"^#{1,4} ", text, re.MULTILINE))


def _ensure_yaml_fenced_contract(
    reply: str, action: Dict[str, Any]
) -> str:
    """Append a YAML blueprint code block if the reply is Markdown but lacks one."""
    if re.search(r"```(?:yaml|json)\s+blueprint", reply, re.IGNORECASE):
        return reply

    return (
        reply.rstrip()
        + "\n\n#### Build Contract\n```yaml blueprint\n"
        + _build_yaml_contract(action)
        + "\n```"
    )


def format_copilot_response(response: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Return a Markdown-formatted reply and an optional PROPOSE_BUILD action.

    This function is the single point of sanitization for Co-pilot chat output:
    - Markdown + YAML code-fenced responses pass through unchanged.
    - Legacy JSON responses are converted to Markdown + YAML.
    - Plain text with build intent is wrapped with a Markdown explanation and a
      YAML build contract.
    """
    if not response or not response.strip():
        return "", None

    if _looks_like_json(response):
        return _wrap_raw_json(response)

    reply, action = parse_copilot_response(response)

    # If the reply is empty but we extracted an action, synthesize Markdown.
    if not reply and action:
        reply = _markdown_reply_from_action(action, response)

    # If there is still no action, try a best-effort parse of the original text.
    if not action:
        action = parse_action_from_text(response)
        if action:
            reply = _markdown_reply_from_action(action, response)

    # If we have an action and no Markdown heading, replace with our structured
    # explanation so the UI never shows raw prose for a build proposal.
    if action and not _has_markdown_heading(reply):
        reply = _markdown_reply_from_action(action, response)

    # If we have Markdown with an action but no code-fenced contract, append one.
    if action:
        reply = _ensure_yaml_fenced_contract(reply or "", action)

    return (reply or response).strip(), action
