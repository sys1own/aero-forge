"""Structural 3-way AST merge for Python and Rust source files.

Applies diff overlays at the function/class/struct level instead of raw line
patches.  For Python we use the standard ``ast`` module; for Rust we use a
best-effort regex-based block extractor with an optional tree-sitter fallback.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple


class MergeConflictError(Exception):
    """Raised when left and right both modify the same entity incompatibly."""


@dataclass
class Entity:
    """A top-level structural entity (function, class, struct, import, ...)."""

    kind: str
    name: str
    text: str
    node: Optional[ast.AST] = None


@dataclass
class Overlay:
    """Result of a structural merge: new source and any conflicts."""

    source: str
    conflicts: List[Tuple[str, str, str]] = field(default_factory=list)


def _entity_key(entity: Entity) -> str:
    """Identity key for aligning entities across versions."""
    return f"{entity.kind}:{entity.name}"


def _extract_python_entities(source: str) -> Dict[str, Entity]:
    """Parse Python source and return a mapping of top-level entities by key."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise MergeConflictError(f"Cannot parse base source: {exc}") from exc

    lines = source.splitlines(keepends=True)

    def node_text(node: ast.AST) -> str:
        start = getattr(node, "lineno", 1) - 1
        end = getattr(node, "end_lineno", start + 1)
        return "".join(lines[start:end])

    entities: Dict[str, Entity] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                entities[f"import:{alias.name}"] = Entity("import", alias.name, node_text(node))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                entities[f"import:{module}.{alias.name}"] = Entity("import", alias.name, node_text(node))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            entities[f"function:{node.name}"] = Entity("function", node.name, node_text(node), node)
        elif isinstance(node, ast.ClassDef):
            entities[f"class:{node.name}"] = Entity("class", node.name, node_text(node), node)
        elif isinstance(node, ast.Assign):
            # Try to capture simple global assignments by target name.
            for target in node.targets:
                if isinstance(target, ast.Name):
                    entities[f"variable:{target.id}"] = Entity("variable", target.id, node_text(node))
    return entities


def _python_imports_text(source: str) -> str:
    """Return the leading comments/imports block before the first function/class."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ""
    for i, node in enumerate(tree.body):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start_line = getattr(node, "lineno", 1)
            return "".join(source.splitlines(keepends=True)[: start_line - 1])
    return source


def _merge_python(base: str, left: str, right: str) -> Overlay:
    """3-way structural merge for Python source.

    For each top-level entity, if ``left`` differs from ``base``, apply the left
    version onto ``right`` unless ``right`` also differs from ``base`` in a
    different way, which is reported as a conflict.
    """
    base_entities = _extract_python_entities(base)
    left_entities = _extract_python_entities(left)
    right_entities = _extract_python_entities(right)

    conflicts: List[Tuple[str, str, str]] = []
    merged = dict(right_entities)

    for key, base_ent in base_entities.items():
        left_ent = left_entities.get(key)
        right_ent = right_entities.get(key)
        left_changed = left_ent is not None and left_ent.text != base_ent.text
        right_changed = right_ent is not None and right_ent.text != base_ent.text

        if left_changed and right_changed and left_ent.text != right_ent.text:
            conflicts.append((key, "both modified", base_ent.name))
            continue
        if left_changed:
            merged[key] = left_ent

    # Add entities newly created in left.
    for key, left_ent in left_entities.items():
        if key not in base_entities and key not in right_entities:
            merged[key] = left_ent

    if conflicts:
        return Overlay(base, conflicts=conflicts)

    # Preserve original ordering from ``right`` as much as possible, appending
    # new left entities at the end.
    order = [k for k in right_entities if k in merged]
    for key in merged:
        if key not in order:
            order.append(key)

    header = _python_imports_text(right)
    body_parts: List[str] = []
    seen_header_keys = set()
    for key in order:
        ent = merged[key]
        if ent.kind == "import" and key in seen_header_keys:
            continue
        if ent.kind == "import":
            seen_header_keys.add(key)
        body_parts.append(ent.text)

    # Re-assemble with the right-hand header and two trailing newlines.
    body = "\n".join(body_parts)
    if header and body:
        source = header.rstrip("\n") + "\n\n" + body + "\n"
    elif header:
        source = header
    else:
        source = body + "\n"
    return Overlay(source, conflicts=[])


# Rust block patterns: name followed by `{...}` body (balanced) or `;` terminated.
_RUST_BLOCK_RE = re.compile(
    r"(?P<kind>fn|struct|enum|impl|trait|mod|type|use)\s+(?P<name>[A-Za-z_][A-Za-z0-9_:<>]*)"
    r"(?P<head>[^;{]*)(?P<body_or_semi>[;{])",
    re.DOTALL,
)


def _balanced_braces(text: str, start: int) -> str:
    """Return the brace-balanced block starting at ``start`` (inclusive)."""
    depth = 0
    in_string = False
    string_char = ""
    escaped = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == string_char:
                in_string = False
        else:
            if ch in ('"', "'"):
                in_string = True
                string_char = ch
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        i += 1
    return text[start:]


def _extract_rust_entities(source: str) -> Dict[str, Entity]:
    """Best-effort Rust entity extraction via regex and brace balancing."""
    entities: Dict[str, Entity] = {}
    seen = set()
    for match in _RUST_BLOCK_RE.finditer(source):
        kind = match.group("kind")
        name = match.group("name").strip()
        if not name:
            continue
        key = f"{kind}:{name}"
        if key in seen:
            continue
        seen.add(key)
        start = match.start()
        body_or_semi = match.group("body_or_semi")
        if body_or_semi == "{":
            block = _balanced_braces(source, match.end("body_or_semi") - 1)
            end = start + len(source[start : match.end("body_or_semi") - 1]) + len(block)
            text = source[start:end]
        else:
            # Semi-terminated item: extend to the next semicolon.
            semi = source.find(";", match.end())
            end = semi + 1 if semi != -1 else len(source)
            text = source[start:end]
        entities[key] = Entity(kind, name, text)
    return entities


def _merge_rust(base: str, left: str, right: str) -> Overlay:
    """3-way structural merge for Rust source using regex block extraction."""
    base_entities = _extract_rust_entities(base)
    left_entities = _extract_rust_entities(left)
    right_entities = _extract_rust_entities(right)

    conflicts: List[Tuple[str, str, str]] = []
    merged = dict(right_entities)

    for key, base_ent in base_entities.items():
        left_ent = left_entities.get(key)
        right_ent = right_entities.get(key)
        left_changed = left_ent is not None and left_ent.text != base_ent.text
        right_changed = right_ent is not None and right_ent.text != base_ent.text

        if left_changed and right_changed and left_ent.text != right_ent.text:
            conflicts.append((key, "both modified", base_ent.name))
            continue
        if left_changed:
            merged[key] = left_ent

    for key, left_ent in left_entities.items():
        if key not in base_entities and key not in right_entities:
            merged[key] = left_ent

    if conflicts:
        return Overlay(base, conflicts=conflicts)

    # Preserve ordering from right, append new left entities.
    order = [k for k in right_entities if k in merged]
    for key in merged:
        if key not in order:
            order.append(key)

    parts = [merged[k].text for k in order]
    return Overlay("\n\n".join(parts) + "\n", conflicts=[])


def structural_merge(base: str, left: str, right: str, language: str = "python") -> Overlay:
    """Perform a 3-way structural merge of ``base``, ``left``, ``right``.

    ``language`` may be ``"python"`` or ``"rust"``. The returned ``Overlay``
    contains the merged source and any conflicts detected.
    """
    if language == "python":
        return _merge_python(base, left, right)
    if language == "rust":
        return _merge_rust(base, left, right)
    raise ValueError(f"Unsupported language for structural merge: {language}")


def apply_overlay(source: str, overlay: str, language: str = "python") -> str:
    """Apply a single overlay (``overlay``) onto ``source`` structurally.

    This is a convenience wrapper around ``structural_merge(source, overlay, source)``.
    """
    return structural_merge(source, overlay, source, language=language).source
