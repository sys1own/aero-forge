"""Extract structured build actions from Co-pilot Markdown and JSON responses."""

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

_SUGGEST_JSON_FENCE_RE = re.compile(
    r"```(?:json)\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)

_BUILD_PROMPT_FENCE_RE = re.compile(
    r"```build_prompt\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)

_BUILD_PROMPT_TAG_RE = re.compile(
    r"<build_prompt>(.*?)</build_prompt>",
    re.DOTALL | re.IGNORECASE,
)

_EXPLANATION_TAG_RE = re.compile(
    r"<explanation>(.*?)</explanation>",
    re.DOTALL | re.IGNORECASE,
)

# Meta wrappers and conversational preambles that must not leak into a build prompt.
_META_PROMPT_PATTERNS = [
    re.compile(r"^\s*Here(?:'s| is)?\s+(?:a\s+)?(?:detailed\s+)?prompt[^\n]*\n*", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*You\s+can\s+paste(?:\s+this)?(?:\s+into|\s+in)?(?:\s+the\s+(?:builder|prompt))?[^\n]*\n*", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*Below\s+is\s+(?:the\s+)?(?:build\s+)?prompt[^\n]*\n*", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*Build\s+Contract\s*[:\-]?[^\n]*\n*", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*Build\s+Prompt\s*[:\-]?[^\n]*\n*", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*yaml\s+blueprint\s*[:\-]?[^\n]*\n*", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s* acceleration\s*[:=]\s*[^\n]+\n*", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s* target\s*[:=]\s*[^\n]+\n*", re.IGNORECASE | re.MULTILINE),
    re.compile(r'^\s*["\']{1,2}', re.MULTILINE),
    re.compile(r'["\']{1,2}\s*$', re.MULTILINE),
]


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


def _strip_outer_quotes(text: str) -> str:
    """Remove surrounding single or double quotes and common escapes."""
    text = text.strip()
    while len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'"):
        text = text[1:-1].strip()
    text = text.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"').replace("\\'", "'")
    return text.strip()


class ActionParser:
    """Parse a Co-pilot response into a clean display text and an isolated action."""

    def __init__(self) -> None:
        self.meta_patterns = _META_PROMPT_PATTERNS

    def sanitize(self, text: str) -> str:
        """Remove meta wrappers and collapse surrounding whitespace."""
        if not text:
            return ""
        cleaned = _strip_outer_quotes(text)
        for pattern in self.meta_patterns:
            cleaned = pattern.sub("", cleaned)
        cleaned = cleaned.strip()
        cleaned = re.sub(r"\n\s*\n+", "\n\n", cleaned)
        return cleaned.strip()

    def _try_parse_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse a raw or fenced JSON response, returning None on failure."""
        if not text:
            return None
        stripped = text.strip()
        if not stripped.startswith(("{", "[")):
            fence = _SUGGEST_JSON_FENCE_RE.search(stripped)
            if fence:
                stripped = fence.group(1).strip()
        if not stripped.startswith(("{", "[")):
            return None
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return None

    def _extract_prompt_from_json(self, data: Dict[str, Any]) -> Optional[str]:
        """Pull the executable prompt from a structured JSON object."""
        if not isinstance(data, dict):
            return None
        action = data.get("action") or {}
        for key in ("clean_prompt", "target_prompt", "build_prompt", "prompt"):
            candidate = action.get(key) if isinstance(action, dict) else None
            if not candidate:
                candidate = data.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        return None

    def _extract_display_text(self, data: Dict[str, Any], fallback: str = "") -> str:
        """Return the conversational display_text/explanation or the fallback."""
        if not isinstance(data, dict):
            return fallback.strip()
        for key in ("display_text", "explanation", "reply", "message"):
            candidate = data.get(key)
            if isinstance(candidate, str):
                return candidate.strip()
        return fallback.strip()

    def _extract_parameters(self, prompt: str, data: Any) -> Dict[str, Any]:
        """Build a parameter bag from explicit JSON fields or inferred from prompt text."""
        params: Dict[str, Any] = {}
        action_data: Optional[Dict[str, Any]] = None
        if isinstance(data, dict):
            action_data = data.get("action") or {}
            if isinstance(action_data, dict):
                params = dict(action_data.get("parameters") or action_data.get("params") or {})

        target = (
            params.get("target")
            or params.get("architecture")
            or (action_data.get("target") if isinstance(action_data, dict) else None)
            or (data.get("target") if isinstance(data, dict) else None)
        )
        acceleration = (
            params.get("acceleration")
            or (action_data.get("acceleration") if isinstance(action_data, dict) else None)
            or (data.get("acceleration") if isinstance(data, dict) else None)
        )

        if not target:
            target = _normalize_target(prompt) or _infer_target_from_text(prompt)
        if not acceleration:
            acceleration = _normalize_acceleration(prompt)

        target = _normalize_target(target) or target
        acceleration = _normalize_acceleration(acceleration)

        return {
            "target": target or "pure_python",
            "acceleration": acceleration,
        }

    def _extract_blueprint(self, data: Any) -> Optional[str]:
        """Return a raw blueprint string if one is embedded in the action."""
        if not isinstance(data, dict):
            return None
        action = data.get("action") or {}
        if isinstance(action, dict):
            for key in ("blueprint", "blueprint_yaml", "blueprint_json"):
                candidate = action.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    return candidate.strip()
        candidate = data.get("blueprint")
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
        return None

    def extract_clean_prompt(self, text: str) -> Optional[str]:
        """Return only the distilled inner prompt or blueprint string.

        The returned string has meta-preambles, outer quotes, and YAML/JSON
        blueprint wrappers removed.
        """
        if not text or not text.strip():
            return None

        # 1. Structured JSON response with a clean prompt field.
        data = self._try_parse_json(text)
        if data:
            candidate = self._extract_prompt_from_json(data)
            if candidate:
                return self.sanitize(candidate)

        # 2. New ```build_prompt fence format.
        fence = _BUILD_PROMPT_FENCE_RE.search(text)
        if fence:
            return self.sanitize(fence.group(1))

        # 3. XML-style <build_prompt> fallback.
        tag = _BUILD_PROMPT_TAG_RE.search(text)
        if tag:
            return self.sanitize(tag.group(1))

        # 4. YAML/JSON blueprint contract block: return the builder prompt inside it.
        contract_match = _CODE_FENCE_RE.search(text)
        if contract_match:
            raw_contract = contract_match.group(1).strip()
            try:
                parsed_contract = yaml.safe_load(raw_contract)
                if isinstance(parsed_contract, dict):
                    prompt = parsed_contract.get("prompt") or parsed_contract.get("build_prompt")
                    if isinstance(prompt, str) and prompt.strip():
                        return self.sanitize(prompt)
            except yaml.YAMLError:
                try:
                    parsed_contract = json.loads(raw_contract)
                    if isinstance(parsed_contract, dict):
                        prompt = parsed_contract.get("prompt") or parsed_contract.get("build_prompt")
                        if isinstance(prompt, str) and prompt.strip():
                            return self.sanitize(prompt)
                except json.JSONDecodeError:
                    pass
            return self.sanitize(raw_contract)

        # 5. Plain text with build intent: return sanitized text as-is.
        if _has_build_intent(text):
            return self.sanitize(text)

        return None

    def parse(self, text: str) -> Dict[str, Any]:
        """Parse a Co-pilot response into a structured packet.

        Returns
        -------
        {
            "display_text": str,
            "action": None | {
                "type": str,
                "clean_prompt": str,
                "parameters": {"target": str, "acceleration": str},
                "blueprint": str | None,
            },
        }
        """
        if not text or not text.strip():
            return {"display_text": "", "action": None}

        data = self._try_parse_json(text)

        # New structured JSON response with explicit display_text and action.
        if data and ("display_text" in data or "action" in data):
            display_text = self._extract_display_text(data, "")
            clean = self._extract_prompt_from_json(data)
            if clean:
                clean = self.sanitize(clean)
                params = self._extract_parameters(clean, data)
                raw_action = data.get("action")
                action_type = "build"
                if isinstance(raw_action, dict):
                    action_type = raw_action.get("type") or action_type
                elif isinstance(raw_action, str):
                    action_type = raw_action
                blueprint = self._extract_blueprint(data)
                return {
                    "display_text": display_text,
                    "action": {
                        "type": action_type,
                        "source": "structured_json",
                        "clean_prompt": clean,
                        "parameters": params,
                        "blueprint": blueprint,
                    },
                }

        # Legacy JSON reply/action with prompt/target/acceleration inside params.
        if data:
            legacy_action = data.get("action")
            if isinstance(legacy_action, dict):
                legacy_params = legacy_action.get("params") or legacy_action.get("parameters") or {}
                prompt = (
                    legacy_params.get("prompt")
                    or legacy_params.get("build_prompt")
                    or legacy_params.get("clean_prompt")
                    or data.get("build_prompt")
                    or data.get("prompt")
                    or data.get("reply")
                    or ""
                )
                if isinstance(prompt, str) and prompt.strip():
                    clean = self.sanitize(prompt)
                    params = self._extract_parameters(clean, data)
                    action_type = legacy_action.get("type")
                    if not action_type or action_type in ("PROPOSE_BUILD", "SUGGEST_BUILD_PROMPT"):
                        action_type = "build"
                    return {
                        "display_text": self._extract_display_text(data, ""),
                        "action": {
                            "type": action_type,
                            "source": "legacy_json",
                            "clean_prompt": clean,
                            "parameters": params,
                            "blueprint": self._extract_blueprint(data),
                        },
                    }

            # JSON with reply only.
            display_text = self._extract_display_text(data, "")
            if display_text:
                return {"display_text": display_text, "action": None}
            # Non-action raw JSON should be left for the caller to pretty-print.
            if text.strip().startswith(("{", "[")):
                return {"display_text": "", "action": None}

        # Markdown/fenced fallback.
        display_text = text

        # Build prompt fence (new clean format).
        fence = _BUILD_PROMPT_FENCE_RE.search(text)
        if fence:
            clean = self.sanitize(fence.group(1))
            display_text = _BUILD_PROMPT_FENCE_RE.sub("", display_text)
            display_text = self.sanitize(display_text)
            params = self._extract_parameters(clean, {})
            return {
                "display_text": display_text,
                "action": {
                    "type": "build",
                    "source": "build_prompt_fence",
                    "clean_prompt": clean,
                    "parameters": params,
                    "blueprint": None,
                },
            }

        # XML-style <build_prompt> fallback.
        tag = _BUILD_PROMPT_TAG_RE.search(text)
        if tag:
            clean = self.sanitize(tag.group(1))
            display_text = _BUILD_PROMPT_TAG_RE.sub("", display_text)
            display_text = self.sanitize(display_text)
            params = self._extract_parameters(clean, {})
            return {
                "display_text": display_text,
                "action": {
                    "type": "build",
                    "source": "xml_tag",
                    "clean_prompt": clean,
                    "parameters": params,
                    "blueprint": None,
                },
            }

        # YAML/JSON blueprint contract block.
        contract_match = _CODE_FENCE_RE.search(text)
        if contract_match:
            raw_contract = contract_match.group(1).strip()
            contract_data: Any = None
            try:
                contract_data = yaml.safe_load(raw_contract)
            except yaml.YAMLError:
                try:
                    contract_data = json.loads(raw_contract)
                except json.JSONDecodeError:
                    pass
            prompt = ""
            if isinstance(contract_data, dict):
                prompt = contract_data.get("prompt") or contract_data.get("build_prompt") or ""
                if not isinstance(prompt, str):
                    prompt = str(prompt)
            clean = self.sanitize(prompt) if prompt else self.sanitize(raw_contract)
            display_text = _CODE_FENCE_RE.sub("", display_text)
            display_text = self.sanitize(display_text)
            params = self._extract_parameters(clean, contract_data if isinstance(contract_data, dict) else {})
            return {
                "display_text": display_text,
                "action": {
                    "type": "build",
                    "source": "blueprint_contract",
                    "contract": True,
                    "clean_prompt": clean,
                    "parameters": params,
                    "blueprint": raw_contract,
                },
            }

        # Plain text with build intent.
        if _has_build_intent(text):
            clean = self.sanitize(text)
            params = self._extract_parameters(clean, {})
            return {
                "display_text": "",
                "action": {
                    "type": "build",
                    "source": "plain_text",
                    "clean_prompt": clean,
                    "parameters": params,
                    "blueprint": None,
                },
            }

        # No actionable payload.
        return {"display_text": self.sanitize(text), "action": None}


def extract_clean_prompt(text: str) -> Optional[str]:
    """Module-level helper for the ActionParser clean-prompt extraction."""
    return ActionParser().extract_clean_prompt(text)


def extract_build_prompt(text: str) -> Tuple[str, Optional[str]]:
    """Split a copilot response into conversational reply and build prompt.

    Priority:
      1. A fenced `` ```build_prompt ... ``` `` block.
      2. A legacy JSON code-fence containing a ``suggest_build_prompt`` action.
      3. XML-style ``<build_prompt>`` / ``<explanation>`` tags.
      4. No prompt found: return the full text as the reply.

    Returns
    -------
    (conversational_reply, extracted_prompt)
    """
    raw = text or ""
    stripped = raw.strip()

    # 1. New ``build_prompt`` fence format.
    fence = _BUILD_PROMPT_FENCE_RE.search(stripped)
    if fence:
        extracted = fence.group(1).strip()
        reply = _BUILD_PROMPT_FENCE_RE.sub("", stripped).strip()
        return reply, extracted

    # 2. JSON code fence containing a suggest_build_prompt payload.
    json_fence = _SUGGEST_JSON_FENCE_RE.search(stripped)
    if json_fence:
        payload = json_fence.group(1).strip()
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict) and (
            parsed.get("action") == "suggest_build_prompt" or "build_prompt" in parsed
        ):
            reply = str(parsed.get("explanation") or "").strip()
            extracted = str(parsed.get("build_prompt") or "").strip() or None
            if not reply:
                reply = _SUGGEST_JSON_FENCE_RE.sub("", stripped).strip()
            if extracted:
                return reply, extracted

    # 3. XML-style fallback tags.
    bp_match = _BUILD_PROMPT_TAG_RE.search(stripped)
    if bp_match:
        extracted = bp_match.group(1).strip()
        reply = ""
        ex_match = _EXPLANATION_TAG_RE.search(stripped)
        if ex_match:
            reply = ex_match.group(1).strip()
        if not reply:
            reply = _BUILD_PROMPT_TAG_RE.sub("", stripped).strip()
        return reply, extracted

    # 4. No prompt found.
    return stripped, None


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


def parse_suggested_build_prompt(text: str) -> Dict[str, Any]:
    """Extract the new ``suggest_build_prompt`` structured payload.

    Returns a dict with ``explanation``, ``has_suggestion``, ``build_prompt``,
    and ``raw``.  Tries ``build_prompt`` fenced blocks first, then JSON code
    fences, then XML-style ``<build_prompt>`` / ``<explanation>`` fallbacks.
    """
    raw = text or ""
    stripped = raw.strip()

    reply, build_prompt = extract_build_prompt(stripped)
    if build_prompt:
        return {
            "explanation": reply,
            "has_suggestion": True,
            "build_prompt": build_prompt,
            "raw": stripped,
        }

    return {"explanation": stripped, "has_suggestion": False, "build_prompt": None, "raw": stripped}


def _build_suggestion(parsed: Dict[str, Any], raw: str) -> Dict[str, Any]:
    build_prompt = str(parsed.get("build_prompt") or "").strip()
    explanation = str(parsed.get("explanation") or "").strip()
    return {
        "explanation": explanation,
        "has_suggestion": bool(build_prompt),
        "build_prompt": build_prompt or None,
        "raw": raw,
    }


def parse_action_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort extraction of a build action from any assistant text."""
    suggestion = parse_suggested_build_prompt(text)
    if suggestion["has_suggestion"]:
        build_prompt = suggestion["build_prompt"] or ""
        return {
            "type": "SUGGEST_BUILD_PROMPT",
            "params": {
                "prompt": build_prompt,
                "explanation": suggestion["explanation"],
                "target": _normalize_target(build_prompt) or _infer_target_from_text(build_prompt),
                "acceleration": _normalize_acceleration(build_prompt),
            },
        }

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
    - ``build_prompt`` fenced blocks (new clean format)
    - ``suggest_build_prompt`` JSON payloads
    - ``yaml blueprint`` / ``json blueprint`` fenced build contracts
    - legacy top-level JSON objects with ``reply``/``action``
    - best-effort extraction from plain prose
    """
    if not response or not response.strip():
        return "", None

    # New ``build_prompt`` fence format.
    reply, build_prompt = extract_build_prompt(response)
    if build_prompt:
        target = _normalize_target(build_prompt) or _infer_target_from_text(build_prompt)
        action = {
            "type": "SUGGEST_BUILD_PROMPT",
            "params": {
                "prompt": build_prompt,
                "explanation": reply,
                "target": target,
                "acceleration": _normalize_acceleration(build_prompt),
            },
        }
        return reply, action

    # New structured action-card format.
    suggestion = parse_suggested_build_prompt(response)
    if suggestion["has_suggestion"]:
        build_prompt = suggestion["build_prompt"] or ""
        target = _normalize_target(build_prompt) or _infer_target_from_text(build_prompt)
        reply = suggestion["explanation"] or _SUGGEST_JSON_FENCE_RE.sub("", response).strip()
        action = {
            "type": "SUGGEST_BUILD_PROMPT",
            "params": {
                "prompt": build_prompt,
                "explanation": suggestion["explanation"],
                "target": target,
                "acceleration": _normalize_acceleration(build_prompt),
            },
        }
        return reply, action

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
