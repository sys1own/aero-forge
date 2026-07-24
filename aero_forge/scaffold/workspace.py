"""Out-of-tree workspace isolation for pre-write validation."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Optional

TOOL_ROOT = Path(__file__).resolve().parents[1]


class WorkspaceLocationError(ValueError):
    """Raised when a requested workspace would land inside the tool tree."""


def _is_inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


class OutOfTreeWorkspace:
    """A staging/build workspace guaranteed to live outside the tool tree.

    1. ``create()`` materialises a temporary staging directory.
    2. Generated files and build artifacts are written into the staging dir.
    3. A delegated validation command runs in the staging dir.
    4. ``commit()`` atomically moves the staging directory to the final
       ``distribution_directory`` only when validation succeeded.
    """

    def __init__(
        self,
        distribution_directory: Optional[Path] = None,
        prefix: str = "aero-build-",
        keep: Optional[bool] = None,
    ) -> None:
        self._distribution = (
            Path(distribution_directory).expanduser() if distribution_directory else None
        )
        if self._distribution is not None and _is_inside(self._distribution, TOOL_ROOT):
            raise WorkspaceLocationError(
                f"distribution directory {self._distribution} must be outside {TOOL_ROOT}"
            )
        self._prefix = prefix
        self.keep = keep if keep is not None else (self._distribution is not None)
        self._root: Optional[Path] = None
        self._committed = False

    @property
    def root(self) -> Path:
        if self._root is None:
            raise RuntimeError("workspace not created; use `with OutOfTreeWorkspace(...) as ws:`")
        return self._root

    @property
    def is_temporary(self) -> bool:
        return self._distribution is None

    @property
    def is_committed(self) -> bool:
        return self._committed

    def create(self) -> Path:
        if self._distribution is not None:
            staging = self._distribution.with_suffix(".staging")
            staging.mkdir(parents=True, exist_ok=True)
            self._root = staging
            return self._root
        self._root = Path(tempfile.mkdtemp(prefix=self._prefix))
        return self._root

    def commit(self) -> Optional[Path]:
        """Promote the staging workspace to the final distribution directory."""
        if self._committed:
            return self._distribution
        if self._distribution is None:
            return self._root
        if self._distribution.exists():
            shutil.rmtree(self._distribution, ignore_errors=True)
        self._root.rename(self._distribution)
        self._committed = True
        return self._distribution

    def discard(self) -> None:
        """Remove the staging workspace without committing."""
        if self._root is not None and self._root.exists() and not self._committed:
            shutil.rmtree(self._root, ignore_errors=True)
        self._root = None

    def __enter__(self) -> "OutOfTreeWorkspace":
        self.create()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is None:
            self.commit()
        else:
            self.discard()
