"""On-disk storage for overlay patches and pristine build snapshots."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Union

_PathLike = Union[str, Path]

DEFAULT_BUILD_CACHE = ".build_cache"
DEFAULT_OVERLAYS = ".overlays"


class OverlayStore:
    def __init__(
        self,
        workspace: _PathLike,
        build_cache_dir: str = DEFAULT_BUILD_CACHE,
        overlays_dir: str = DEFAULT_OVERLAYS,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.build_cache_dir = self.workspace / build_cache_dir
        self.overlays_dir = self.workspace / overlays_dir

    def relkey(self, file: _PathLike) -> str:
        """Return the workspace-relative key for *file* (POSIX separators)."""
        resolved = Path(file).resolve()
        try:
            rel = resolved.relative_to(self.workspace)
        except ValueError:
            rel = Path(os.path.splitdrive(str(resolved))[1].lstrip("/\\"))
        return rel.as_posix()

    def cache_path(self, file: _PathLike) -> Path:
        return self.build_cache_dir / self.relkey(file)

    def overlay_path(self, file: _PathLike) -> Path:
        return self.overlays_dir / (self.relkey(file) + ".patch")

    def record_generated(self, file: _PathLike, content: str = None) -> None:
        if content is None:
            content = Path(file).read_text(encoding="utf-8")
        dest = self.cache_path(file)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")

    def read_cache(self, file: _PathLike):
        path = self.cache_path(file)
        if not path.is_file():
            return None
        return path.read_text(encoding="utf-8")

    def save_overlay(self, file: _PathLike, patch: str) -> Path:
        path = self.overlay_path(file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(patch, encoding="utf-8")
        return path

    def read_overlay(self, file: _PathLike):
        path = self.overlay_path(file)
        if not path.is_file():
            return None
        return path.read_text(encoding="utf-8")

    def has_overlay(self, file: _PathLike) -> bool:
        return self.overlay_path(file).is_file()

    def remove_overlay(self, file: _PathLike) -> bool:
        path = self.overlay_path(file)
        if path.is_file():
            path.unlink()
            return True
        return False

    def list_overlays(self) -> List[str]:
        if not self.overlays_dir.is_dir():
            return []
        keys: List[str] = []
        for patch in sorted(self.overlays_dir.rglob("*.patch")):
            rel = patch.relative_to(self.overlays_dir).as_posix()
            keys.append(rel[: -len(".patch")])
        return keys

    def file_for_key(self, key: str) -> Path:
        return self.workspace / key
