"""Active in-tree merger for verified out-of-tree scaffold builds."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

SHARED_LIB_SUFFIXES = (".so", ".dylib", ".pyd")


@dataclass
class MergeResult:
    """Outcome of an active in-tree merge."""

    merged: bool
    source: Optional[str] = None
    destination: Optional[str] = None
    module_name: Optional[str] = None
    loaded: bool = False
    reason: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "merged": self.merged,
            "source": self.source,
            "destination": self.destination,
            "module_name": self.module_name,
            "loaded": self.loaded,
            "reason": self.reason,
            "notes": list(self.notes),
        }


def _candidate_names(crate_name: str) -> List[str]:
    names: List[str] = []
    for suffix in SHARED_LIB_SUFFIXES:
        names.append(f"lib{crate_name}{suffix}")
        names.append(f"{crate_name}{suffix}")
    return names


def find_compiled_library(workspace_root: Path, crate_name: str) -> Optional[Path]:
    """Locate the compiled shared library for *crate_name* under ``target/``."""
    profiles = []
    for profile in ("release", "debug"):
        directory = Path(workspace_root) / "target" / profile
        if directory.is_dir():
            profiles.append(directory)

    wanted = _candidate_names(crate_name)
    for directory in profiles:
        for name in wanted:
            candidate = directory / name
            if candidate.is_file():
                return candidate

    for directory in profiles:
        libs = sorted(
            p for p in directory.iterdir() if p.is_file() and p.suffix in SHARED_LIB_SUFFIXES
        )
        if libs:
            return libs[0]
    return None


def active_extensions_dir(base_dir: Optional[Path] = None) -> Path:
    """Return the live extensions directory."""
    if base_dir is None:
        return Path.cwd() / ".active_extensions"
    return base_dir / ".active_extensions"


def merge_active(
    workspace_root: Path,
    crate_name: str,
    module_name: Optional[str] = None,
    *,
    dest_dir: Optional[Path] = None,
    load: bool = True,
    project_root: Optional[Path] = None,
) -> MergeResult:
    """Copy the crate's compiled library into the active extension layer."""
    import_name = module_name or crate_name

    library = find_compiled_library(workspace_root, crate_name)
    if library is None:
        return MergeResult(
            merged=False,
            module_name=import_name,
            reason=(
                f"no compiled shared library found under "
                f"{Path(workspace_root) / 'target'}"
            ),
        )

    target_dir = dest_dir if dest_dir is not None else active_extensions_dir(project_root)
    target_dir.mkdir(parents=True, exist_ok=True)

    suffix = library.suffix
    if suffix == ".dll":
        suffix = ".pyd"
    destination = target_dir / f"{import_name}{suffix}"
    shutil.copy2(library, destination)

    result = MergeResult(
        merged=True,
        source=str(library),
        destination=str(destination),
        module_name=import_name,
        notes=[f"copied {library.name} -> {destination}"],
    )

    if load:
        loaded_module = _load_extension_file(destination, import_name)
        result.loaded = loaded_module is not None
        if loaded_module is not None:
            result.notes.append(f"loaded '{import_name}' into the live process")
        else:
            result.notes.append(
                f"copied but could not load '{import_name}' "
                "(ABI/symbol mismatch?); it will load on next interpreter start"
            )

    return result


def _load_extension_file(path: Path, module_name: str) -> Optional[Any]:
    """Best-effort load of a C-ABI shared library using ctypes."""
    try:
        import ctypes

        return ctypes.CDLL(str(path))
    except Exception:
        return None
