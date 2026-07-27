"""Workspace export orchestration combining target source, native crate source,
the embedded wavefront micro-runtime, and standalone ``.aeroc`` artifacts."""

from __future__ import annotations

import io
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

from aero_forge.bundle_repo import (
    ExportProfile,
    _native_crate_source_files,
    create_project_zip,
)
from aero_forge.scaffold.aeroc_export import export_aeroc_project, package_aeroc


def _source_files(session_dir: Path) -> Dict[str, str]:
    """Return a mapping of relative path -> content for human-readable source files."""
    files: Dict[str, str] = {}
    if not session_dir.is_dir():
        return files
    for path in sorted(session_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(session_dir)
        # Skip build artifacts and hidden caches.
        skip_prefixes = (
            "target",
            ".venv",
            "__pycache__",
            ".pytest_cache",
            ".aero",
            ".cargo",
            "crates",
        )
        if any(part in rel.parts[:1] for part in skip_prefixes):
            continue
        if rel.name.startswith("."):
            continue
        try:
            files[str(rel.as_posix())] = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
    return files


def export_workspace(
    session_dir: Path,
    options: Optional[Dict[str, Any]] = None,
    project_name: str = "aero-forge-export",
) -> tuple[bytes, str]:
    """Create a workspace export archive according to ``options``.

    ``options`` keys:
        * ``pure_target`` (bool)              - include target source files
        * ``include_native_crate`` (bool)    - include PyO3 native crate source
        * ``include_wavefront_runtime`` (bool) - include embedded aero_core runtime
        * ``standalone_aeroc`` (bool)        - include a pre-packaged ``.aeroc`` project

    Returns ``(archive_bytes, filename)``.
    """
    options = options or {}
    pure_target = options.get("pure_target", True)
    include_native = options.get("include_native_crate", False)
    include_wavefront = options.get("include_wavefront_runtime", False)
    standalone_aeroc = options.get("standalone_aeroc", False)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if pure_target:
            for rel, content in _source_files(session_dir).items():
                zf.writestr(rel, content)

        if include_native:
            for arc, content in _native_crate_source_files():
                zf.writestr(str(arc.as_posix()), content)
            zf.writestr("pyproject.toml", _pyproject_toml_for_maturin(project_name))

        if include_wavefront:
            embedded_src = Path(__file__).resolve().parent / "embedded" / "aero_core"
            if embedded_src.is_dir():
                for path in sorted(embedded_src.rglob("*")):
                    if not path.is_file():
                        continue
                    rel = path.relative_to(embedded_src)
                    if rel.name == "__pycache__" or "__pycache__" in rel.parts:
                        continue
                    arcname = f"crates/aero_core/{rel.as_posix()}"
                    zf.writestr(arcname, path.read_text(encoding="utf-8"))

        if standalone_aeroc:
            with tempfile.TemporaryDirectory() as tmpdir:
                aeroc_dir = Path(tmpdir) / "aeroc-export"
                export_aeroc_project(session_dir, aeroc_dir, project_name=project_name)
                aeroc_archive = package_aeroc(aeroc_dir)
                zf.writestr(
                    f"{project_name}.aeroc",
                    aeroc_archive.read_bytes(),
                )

    filename = f"{project_name}.zip"
    return buf.getvalue(), filename


def _pyproject_toml_for_maturin(project_name: str) -> str:
    return f"""[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[project]
name = "{project_name}"
version = "0.1.0"
description = "Aero-Forge generated project with optional PyO3 native acceleration."
requires-python = ">=3.9"
dependencies = ["blake3"]

[tool.maturin]
manifest-path = "crates/native_core/Cargo.toml"
module-name = "aero_forge_native"
"""
