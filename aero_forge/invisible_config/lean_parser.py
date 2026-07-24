"""Ultra-lean blueprint dialect parser for aero-forge."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

_PROJECT_RE = re.compile(r'^\s*project\s+"(?P<name>[^"]*)"\s*$')
_ASSIGN_RE = re.compile(
    r"^\s*(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<value>.+?)\s*$"
)
_LEAN_ASSIGN_SIGNATURE = re.compile(
    r'^\s*(?:project\s+"|[A-Za-z_][A-Za-z0-9_]*\s*=)', re.MULTILINE
)


class LeanBlueprintError(ValueError):
    """Raised when an ultra-lean blueprint cannot be parsed."""


@dataclass
class LeanBlueprint:
    """The parsed semantic intent of a lean blueprint."""

    project: str = ""
    ingest: List[str] = field(default_factory=list)
    targets: List[str] = field(default_factory=list)
    optimize: str = "balanced"
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project": self.project,
            "ingest": list(self.ingest),
            "targets": list(self.targets),
            "optimize": self.optimize,
            "extras": dict(self.extras),
        }


def _is_comment_or_blank(line: str) -> bool:
    stripped = line.strip()
    return not stripped or stripped.startswith("#") or stripped.startswith("//")


def _parse_literal(raw_value: str) -> Any:
    """Parse a lean blueprint literal value."""
    raw_value = raw_value.strip()
    lowered = raw_value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in ("none", "null"):
        return None
    try:
        return ast.literal_eval(raw_value)
    except Exception:
        return raw_value


def _as_str_list(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value)]


def looks_like_lean_blueprint(content: str) -> bool:
    """Heuristically detect the lean dialect (vs. INI / JSON / block DSL)."""
    first_meaningful = ""
    for raw in content.splitlines():
        if _is_comment_or_blank(raw):
            continue
        first_meaningful = raw.strip()
        break
    if not first_meaningful:
        return False
    if first_meaningful[0] in "[{":
        return False
    if "{" in content:
        return False
    return bool(_LEAN_ASSIGN_SIGNATURE.search(content))


def parse_lean_blueprint(content: str) -> LeanBlueprint:
    """Parse the lean dialect into a :class:`LeanBlueprint`."""
    blueprint = LeanBlueprint()
    seen_keys: Dict[str, int] = {}

    for lineno, raw in enumerate(content.splitlines(), start=1):
        if _is_comment_or_blank(raw):
            continue
        line = raw

        project_match = _PROJECT_RE.match(line)
        if project_match:
            if blueprint.project:
                raise LeanBlueprintError(f"line {lineno}: duplicate 'project' declaration")
            blueprint.project = project_match.group("name")
            continue

        assign_match = _ASSIGN_RE.match(line)
        if not assign_match:
            raise LeanBlueprintError(
                f"line {lineno}: expected 'project \"name\"' or 'key = value', got: {line.strip()!r}"
            )

        key = assign_match.group("key")
        raw_value = assign_match.group("value")
        seen_keys[key] = seen_keys.get(key, 0) + 1
        if seen_keys[key] > 1:
            raise LeanBlueprintError(f"line {lineno}: duplicate key '{key}'")

        value = _parse_literal(raw_value)
        if key == "ingest":
            blueprint.ingest = _as_str_list(value)
        elif key == "targets":
            blueprint.targets = _as_str_list(value)
        elif key == "optimize":
            blueprint.optimize = str(value)
        elif key == "project":
            blueprint.project = str(value)
        else:
            blueprint.extras[key] = value

    if not blueprint.project:
        raise LeanBlueprintError("missing required 'project \"name\"' declaration")
    if not blueprint.targets:
        raise LeanBlueprintError("missing required 'targets = [...]' declaration")
    return blueprint
