"""High-level overlay orchestration for aero-forge."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Union

from aero_forge.overlay.apply import apply_patch
from aero_forge.overlay.patch import is_empty_patch, make_patch
from aero_forge.overlay.store import OverlayStore
from aero_forge.overlay.structural_merger import StructuralMerger

_PathLike = Union[str, Path]


class OverlayError(Exception):
    """Raised when an overlay operation cannot proceed."""


class ReapplyStatus(str, Enum):
    APPLIED = "applied"
    CONFLICT = "conflict"
    MISSING = "missing"
    SKIPPED = "skipped"


class OverlayManager:
    """Tie together patch, apply, store, and structural merge layers."""

    def __init__(self, workspace: _PathLike, store: Optional[OverlayStore] = None) -> None:
        self.workspace = Path(workspace).resolve()
        self.store = store or OverlayStore(self.workspace)

    def record_generated(self, file: _PathLike, content: str = None) -> None:
        """Snapshot a freshly generated file as the pristine baseline."""
        self.store.record_generated(file, content)

    def commit_overlay(self, file: _PathLike) -> Optional[str]:
        """Persist the diff between *file* and its pristine baseline."""
        path = Path(file).resolve()
        if not path.is_file():
            raise OverlayError(f"File not found: {path}")

        baseline = self.store.read_cache(path)
        if baseline is None:
            raise OverlayError(
                f"No pristine baseline in {self.store.build_cache_dir} for {path}. "
                "Generate/build the file before committing an overlay."
            )

        key = self.store.relkey(path)
        current = path.read_text(encoding="utf-8")
        patch = make_patch(baseline, current, fromfile=key, tofile=key)
        if is_empty_patch(patch):
            self.store.remove_overlay(path)
            return None
        self.store.save_overlay(path, patch)
        return patch

    def reapply(self, file: _PathLike) -> ReapplyStatus:
        """Re-apply the committed overlay to a (re)generated *file* (line-based)."""
        path = Path(file).resolve()
        overlay = self.store.read_overlay(path)
        if overlay is None:
            return ReapplyStatus.APPLIED
        if not path.is_file():
            return ReapplyStatus.MISSING

        pristine = path.read_text(encoding="utf-8")
        self.store.record_generated(path, pristine)

        merged, conflict = apply_patch(pristine, overlay)
        if conflict:
            return ReapplyStatus.CONFLICT
        path.write_text(merged, encoding="utf-8")
        return ReapplyStatus.APPLIED

    def reapply_all(self) -> Dict[str, ReapplyStatus]:
        """Re-apply every committed overlay; returns ``{relkey: status}``."""
        results: Dict[str, ReapplyStatus] = {}
        for key in self.store.list_overlays():
            results[key] = self.reapply(self.store.file_for_key(key))
        return results

    def structural_reapply(
        self,
        file: _PathLike,
        regenerated_text: str,
        *,
        language: Optional[str] = None,
    ) -> ReapplyStatus:
        """Re-apply user edits with the structural 3-way AST merge engine.

        ``base`` is the pristine snapshot in ``.build_cache``, *Left* is the
        user's current on-disk file, and *Right* is ``regenerated_text``.  On
        conflict the freshly generated text is kept as the pristine baseline.
        """
        path = Path(file).resolve()
        if not path.is_file():
            return ReapplyStatus.MISSING

        base = self.store.read_cache(path)
        if base is None:
            base = path.read_text(encoding="utf-8")

        left = path.read_text(encoding="utf-8")
        lang = language or self._detect_language(path)
        merger = StructuralMerger(lang)
        outcome = merger.merge(base, left, regenerated_text)

        self.store.record_generated(path, regenerated_text)
        if outcome.conflicts:
            return ReapplyStatus.CONFLICT

        path.write_text(outcome.text, encoding="utf-8")
        return ReapplyStatus.APPLIED

    @staticmethod
    def _detect_language(path: Path) -> str:
        suffix = path.suffix
        if suffix == ".py":
            return "python"
        if suffix in {".rs"}:
            return "rust"
        if suffix in {".cpp", ".cc", ".cxx", ".hpp"}:
            return "cpp"
        return "rust"

    def commit_all(self) -> Dict[str, Optional[str]]:
        """Commit overlays for every file that has a pristine baseline."""
        results: Dict[str, Optional[str]] = {}
        if not self.store.build_cache_dir.is_dir():
            return results
        for cache_file in sorted(self.store.build_cache_dir.rglob("*")):
            if not cache_file.is_file():
                continue
            rel = cache_file.relative_to(self.store.build_cache_dir).as_posix()
            target = self.store.file_for_key(rel)
            if target.is_file():
                try:
                    results[rel] = self.commit_overlay(target)
                except OverlayError:
                    results[rel] = None
        return results
