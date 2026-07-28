"""Auto-materialize workspaces from standalone ``workspace.aeroc`` artifacts."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from aero_forge._native import unpack_aeroc
from aero_forge.scaffold.module_guard import reify_missing_modules

if TYPE_CHECKING:
    from os import PathLike

logger = logging.getLogger("aero_forge.materializer")


def unpack_aeroc_file(aeroc_path: str | Path, output_dir: str | Path) -> int:
    """Extract ``workspace.aeroc`` into *output_dir*.

    Returns the number of source files reconstructed.
    """
    count = unpack_aeroc(str(aeroc_path), str(output_dir))
    reify_missing_modules(Path(output_dir))
    return count


def workspace_requires_materialization(workspace: str | Path) -> bool:
    """Return True when *workspace* contains a ``workspace.aeroc`` but no source tree/blueprint."""
    workspace = Path(workspace)
    aeroc = workspace / "workspace.aeroc"
    blueprint = workspace / "blueprint.aero"
    if not aeroc.is_file():
        return False
    if blueprint.is_file():
        return False
    # If any non-hidden, non-aeroc files exist, assume it is already materialized.
    for child in workspace.iterdir():
        if child.name in {".aero", ".git", "workspace.aeroc"}:
            continue
        if child.is_file() or (child.is_dir() and child.name != "__pycache__"):
            return False
    return True


def auto_materialize(workspace: str | Path) -> bool:
    """Unpack ``workspace.aeroc`` into *workspace* if the directory is empty."""
    workspace = Path(workspace).resolve()
    aeroc = workspace / "workspace.aeroc"
    if not workspace_requires_materialization(workspace):
        return False

    logger.info("Auto-materializing workspace from %s", aeroc)
    count = unpack_aeroc_file(aeroc, workspace)
    logger.info("Extracted %d source files", count)
    return True
