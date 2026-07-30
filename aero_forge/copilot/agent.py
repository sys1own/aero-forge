"""Co-pilot response post-processor.

Guarantees that every Co-pilot reply sent to the UI is human-readable
Markdown with an isolated, sanitized action payload.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

from aero_forge.blueprint.core import (
    BlueprintCore,
    ensure_workspace_blueprint,
    parse_aero,
)
from aero_forge.copilot.action_parser import (
    ActionParser,
    _infer_target_from_text,
    _normalize_acceleration,
    _normalize_target,
    clean_explanation_text,
    sanitize_builder_prompt,
)

logger = logging.getLogger("aero_forge.copilot.agent")


def _legacy_action_type(action: Dict[str, Any]) -> str:
    """Map the canonical ActionParser type to the legacy UI action type."""
    new_type = action.get("type", "build")
    source = action.get("source", "")
    if action.get("contract") or source in ("blueprint_contract", "legacy_json", "plain_text") or new_type in ("apply_blueprint", "PROPOSE_BUILD"):
        return "PROPOSE_BUILD"
    if new_type in ("suggest_build_prompt", "trigger_build") or source in ("build_prompt_fence", "xml_tag", "structured_json", "action_trigger_build"):
        return "SUGGEST_BUILD_PROMPT"
    return "SUGGEST_BUILD_PROMPT"


def _to_legacy_action(parsed: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert an ActionParser result to the historical action shape used by the UI."""
    action = parsed.get("action")
    if not action:
        return None

    params = action.get("parameters") or {}
    clean = sanitize_builder_prompt(action.get("clean_prompt", ""))
    return {
        "type": _legacy_action_type(action),
        "params": {
            "prompt": clean,
            "target": params.get("target") or _normalize_target(clean) or _infer_target_from_text(clean),
            "acceleration": params.get("acceleration") or _normalize_acceleration(clean),
            "explanation": parsed.get("display_text", ""),
            "parameters": params,
        },
    }


def _has_markdown_heading(text: str) -> bool:
    """Return True if the text already contains a Markdown heading."""
    return bool(re.search(r"^#{1,4} ", text, re.MULTILINE))


def format_copilot_response(response: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Return a Markdown-formatted reply and an optional action.

    The action payload is isolated, meta-preamble-free, and never wrapped in a
    YAML/JSON code block inside the Markdown reply.
    """
    if not response or not response.strip():
        return "", None

    parsed = ActionParser().parse(response)
    action = parsed.get("action") or {}
    clean_prompt = sanitize_builder_prompt(action.get("clean_prompt", "") or "")
    display_text = clean_explanation_text(parsed.get("display_text", ""), clean_prompt)
    legacy_action = _to_legacy_action(parsed)

    # If the model returned raw JSON without a human display_text, pretty-print it
    # for the chat timeline but do not expose it as an action.
    if not display_text and _looks_like_json(response):
        try:
            data = json.loads(response.strip())
            display_text = "### Response\n```json\n" + json.dumps(data, indent=2) + "\n```"
        except json.JSONDecodeError:
            display_text = response.strip()

    if legacy_action:
        prompt_text = clean_prompt or legacy_action["params"]["prompt"]
        # Avoid duplicating the executable prompt in the chat bubble.
        if not display_text or display_text.strip() == prompt_text.strip():
            target = legacy_action["params"]["target"]
            target_label = target.replace("_", " ").title()
            display_text = f"### Architecture Overview\n\nI propose a **{target_label}** build. Use the Action Card to edit or trigger it."

    if not _has_markdown_heading(display_text) and display_text:
        display_text = "### Architecture Overview\n\n" + display_text

    if not display_text:
        display_text = clean_explanation_text(parsed.get("display_text", response.strip()), clean_prompt)

    return display_text.strip(), legacy_action


def _looks_like_json(text: str) -> bool:
    """Return True if the text appears to be a raw JSON object/list."""
    if not text:
        return False
    stripped = text.strip()
    return stripped.startswith(("{", "["))


def _load_workspace_blueprint(workspace_path: Path) -> Dict[str, Any]:
    """Read and parse the workspace blueprint, auto-detecting one if absent.

    Tries ``blueprint.aero`` and ``workspace_blueprint.yaml`` in that order.
    If neither exists, calls ``ensure_workspace_blueprint`` to synthesize a
    minimal default from standard templates and ``BlueprintCore.autodetect``
    to build an in-memory schema from the workspace contents.
    """
    workspace_path = Path(workspace_path)
    ensure_workspace_blueprint(workspace_path)
    candidates = [
        workspace_path / "blueprint.aero",
        workspace_path / "workspace_blueprint.yaml",
    ]
    for candidate in candidates:
        if candidate.is_file():
            try:
                text = candidate.read_text(encoding="utf-8")
                if candidate.suffix.lower() in (".yaml", ".yml"):
                    return yaml.safe_load(text) or {}
                return parse_aero(text)
            except Exception as exc:
                logger.warning("Could not parse blueprint %s: %s", candidate, exc)
                return {}
    return BlueprintCore.autodetect(workspace_path)


def workspace_blueprint_tag(workspace_path: Path) -> str:
    """Return a formatted ``<workspace_blueprint>`` context tag for prompts.

    The blueprint is rendered as YAML (or JSON as a fallback) so the model
    can reason about the current workspace contract, manifest, and functions.
    """
    blueprint = _load_workspace_blueprint(workspace_path)
    if not blueprint:
        return ""
    try:
        payload = yaml.safe_dump(blueprint, sort_keys=False, default_flow_style=False)
    except Exception:
        payload = json.dumps(blueprint, indent=2, default=str)
    return f"<workspace_blueprint>\n{payload}</workspace_blueprint>"
