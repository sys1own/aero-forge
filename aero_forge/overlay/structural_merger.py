"""Structural AST 3-way merge for regenerated source files."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from aero_forge.overlay.apply import apply_patch
from aero_forge.overlay.patch import make_patch


@dataclass
class MergeOutcome:
    """Result of merging a user's manual edits (left) onto a fresh generation (right)."""

    merged: bool
    text: str
    conflicts: List[str] = field(default_factory=list)
    changed: bool = False

    def to_dict(self) -> dict:
        return {
            "merged": self.merged,
            "conflicts": list(self.conflicts),
            "changed": self.changed,
            "text": self.text,
        }


class StructuralMerger:
    """Merge hand-edited overlays into regenerated code without destroying user changes."""

    def __init__(self, language: str = "python") -> None:
        self.language = language

    @staticmethod
    def _entity_name(node: ast.AST) -> str:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return node.name
        if isinstance(node, ast.Import):
            return ",".join(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = ",".join(a.asname or a.name for a in node.names)
            return f"{module}:{names}"
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            return node.targets[0].id
        return ""

    @staticmethod
    def _entity_id(node: ast.AST) -> str:
        return f"{type(node).__name__}#{StructuralMerger._entity_name(node)}"

    @classmethod
    def _split_python(cls, text: str) -> Dict[str, Tuple[str, ast.AST]]:
        tree = ast.parse(text)
        lines = text.splitlines()
        entities: Dict[str, Tuple[str, ast.AST]] = {}
        for i, node in enumerate(tree.body):
            start = node.lineno - 1
            end = getattr(node, "end_lineno", start + 1) or start + 1
            entity_text = "\n".join(lines[start:end])
            # Ensure every top-level entity is terminated so concatenation never
            # collapses distinct statements onto a single line.
            entity_text = entity_text.rstrip("\n") + "\n"
            entities[cls._entity_id(node)] = (entity_text, node)
        return entities

    def _merge_python(self, base: str, left: str, right: str) -> MergeOutcome:
        base_entities = self._split_python(base)
        left_entities = self._split_python(left)
        right_entities = self._split_python(right)

        conflicts: List[str] = []
        changed = False
        result_lines: List[str] = []

        for ent_id, (right_text, _node) in right_entities.items():
            base_text, _ = base_entities.get(ent_id, ("", None))
            left_text, _ = left_entities.get(ent_id, ("", None))

            if ent_id not in left_entities:
                # User did not touch this entity; keep fresh generated version.
                result_lines.append(right_text)
                continue

            if base_text == left_text:
                # User did not change this entity (or identical to base); keep right.
                result_lines.append(right_text)
                continue

            if base_text == right_text:
                # Generation did not change this entity; apply user's edit wholesale.
                result_lines.append(left_text)
                changed = True
                continue

            # Both sides changed the same entity. Try a line-level 3-way patch.
            patch = make_patch(base_text, left_text, fromfile="base", tofile="left")
            merged_text, patch_conflict = apply_patch(right_text, patch)
            if patch_conflict:
                conflicts.append(f"conflict in {ent_id}")
                result_lines.append(right_text)
            else:
                if merged_text != right_text:
                    changed = True
                result_lines.append(merged_text)

        # Append entities present in left but removed in right (preservation rule).
        for ent_id, (left_text, _) in left_entities.items():
            if ent_id not in right_entities and ent_id in base_entities:
                result_lines.append(left_text)
                changed = True

        result = "".join(result_lines)
        if base.endswith("\n") and not result.endswith("\n"):
            result += "\n"
        return MergeOutcome(
            merged=not conflicts,
            text=result,
            conflicts=conflicts,
            changed=changed or result != right,
        )

    @staticmethod
    def _rust_entity_pattern() -> re.Pattern:
        return re.compile(
            r"^(?:pub\s+)?(?:fn|struct|enum|impl|trait|mod|use)\s+",
            re.MULTILINE,
        )

    def _merge_rust(self, base: str, left: str, right: str) -> MergeOutcome:
        patch = make_patch(base, left, fromfile="base", tofile="left")
        merged, conflict = apply_patch(right, patch)
        return MergeOutcome(
            merged=not conflict,
            text=merged,
            conflicts=["line-level merge conflict"] if conflict else [],
            changed=merged != right,
        )

    def merge(self, base: str, left: str, right: str) -> MergeOutcome:
        """Three-way merge: base -> left (user edit) onto right (new generation)."""
        if self.language == "python":
            return self._merge_python(base, left, right)
        return self._merge_rust(base, left, right)

    def merge_file(
        self,
        base_path,
        left_path,
        right_path,
        language: Optional[str] = None,
    ) -> MergeOutcome:
        lang = language or self.language
        base = Path(base_path).read_text(encoding="utf-8")
        left = Path(left_path).read_text(encoding="utf-8")
        right = Path(right_path).read_text(encoding="utf-8")
        merger = StructuralMerger(lang)
        return merger.merge(base, left, right)


# ---------------------------------------------------------------------------
# Legacy structural-merge primitives (kept in overlay layer for consumers that
# still reference `apply_overlay` / `structural_merge`).
# ---------------------------------------------------------------------------


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
        end = getattr(node, "end_lineno", start + 1) or start + 1
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
    """Return the leading imports/docstring/comments block before code."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ""
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        # Allow a leading module docstring to remain in the header.
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        start_line = getattr(node, "lineno", 1)
        return "".join(source.splitlines(keepends=True)[: start_line - 1])
    return ""


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


def _is_complete_python_overlay(source: str, overlay: str) -> bool:
    """Return True when *overlay* is a full rewrite containing every source entity."""
    try:
        source_tree = ast.parse(source)
        overlay_tree = ast.parse(overlay)
    except SyntaxError:
        return False

    def named_names(tree: ast.AST) -> set[str]:
        names: set[str] = set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.add(f"function:{node.name}")
            elif isinstance(node, ast.ClassDef):
                names.add(f"class:{node.name}")
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        names.add(f"variable:{target.id}")
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names.add(f"variable:{node.target.id}")
        return names

    source_named = named_names(source_tree)
    overlay_named = named_names(overlay_tree)
    if source_named - overlay_named:
        return False

    def generic_count(tree: ast.AST) -> int:
        return sum(
            1
            for node in tree.body
            if not isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom),
            )
        )

    return generic_count(overlay_tree) >= generic_count(source_tree)


def apply_overlay(source: str, overlay: str, language: str = "python") -> str:
    """Apply a single overlay (``overlay``) onto ``source`` structurally.

    For complete Python rewrites we return the overlay directly to avoid line-
    shift artifacts from the 3-way merge. Otherwise we fall back to a structural
    3-way merge so partial overlays (e.g. a single changed function) do not erase
    unrelated code.
    """
    if language == "python" and _is_complete_python_overlay(source, overlay):
        return overlay
    return structural_merge(source, overlay, source, language=language).source
