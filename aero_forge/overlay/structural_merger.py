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
        append_newline = text.endswith("\n")
        entities: Dict[str, Tuple[str, ast.AST]] = {}
        for i, node in enumerate(tree.body):
            start = node.lineno - 1
            end = getattr(node, "end_lineno", start + 1) or start + 1
            entity_text = "\n".join(lines[start:end])
            if append_newline:
                entity_text += "\n"
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


