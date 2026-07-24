"""Non-destructive overlay and structural AST merge for aero-forge."""

from __future__ import annotations

from aero_forge.overlay.apply import apply_patch
from aero_forge.overlay.manager import OverlayManager, ReapplyStatus
from aero_forge.overlay.patch import is_empty_patch, make_patch
from aero_forge.overlay.store import OverlayStore
from aero_forge.overlay.structural_merger import StructuralMerger, MergeOutcome

__all__ = [
    "OverlayManager",
    "OverlayStore",
    "ReapplyStatus",
    "apply_patch",
    "is_empty_patch",
    "make_patch",
    "StructuralMerger",
    "MergeOutcome",
]
