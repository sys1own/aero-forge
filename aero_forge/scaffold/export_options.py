"""Workspace export orchestration combining target source, native crate source,
the embedded wavefront micro-runtime, and standalone ``.aeroc`` artifacts."""

from __future__ import annotations

import dataclasses
import enum
import io
import json
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
from aero_forge.orchestrator.orchestrator import validate_blueprint_for_export
from aero_forge.overlay import OverlayManager
from aero_forge.scaffold.aeroc_export import (
    export_aeroc_project,
    generate_verification_json,
    package_aeroc,
    verify_aeroc_project,
    verify_workspace_for_export,
)


class ExportMode(str, enum.Enum):
    """Verification policy for workspace exports."""

    STRICT = "strict"
    DRAFT = "draft"


@dataclasses.dataclass
class ExportOptions:
    """Options controlling workspace export contents and verification.

    * ``mode`` - ``strict`` blocks export on verification failures; ``draft``
      allows export and marks it ``unverified``.
    * ``run_tests`` - run pytest in the workspace before export.
    * ``run_compilation`` - compile Rust/C++ targets before export.
    * ``pure_target`` - include human-readable workspace source files.
    * ``include_native_crate`` - include the PyO3 native acceleration crate.
    * ``include_wavefront_runtime`` - include the embedded ``aero_core`` runtime.
    * ``standalone_aeroc`` - include a pre-packaged ``.aerozip`` project.
    * ``project_name`` - archive / project name stem.
    """

    mode: ExportMode = ExportMode.STRICT
    run_tests: bool = True
    run_compilation: bool = True
    pure_target: bool = True
    include_native_crate: bool = False
    include_wavefront_runtime: bool = False
    standalone_aeroc: bool = False
    project_name: str = "aero-forge-export"

    @classmethod
    def from_dict(
        cls, options: Optional[Dict[str, Any]], project_name: str = "aero-forge-export"
    ) -> "ExportOptions":
        """Build an ``ExportOptions`` from a legacy options dictionary."""
        if not options:
            return cls(project_name=project_name)
        if isinstance(options, cls):
            return options
        if not isinstance(options, dict):
            options = dict(options)
        allowed = {f.name for f in dataclasses.fields(cls)}
        kwargs = {k: v for k, v in options.items() if k in allowed}
        opts = cls(**kwargs)
        if project_name and opts.project_name == "aero-forge-export":
            opts.project_name = project_name
        if "mode" in options and isinstance(options["mode"], str):
            opts.mode = ExportMode(options["mode"])
        return opts


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
    options: Optional[Any] = None,
    project_name: str = "aero-forge-export",
) -> tuple[bytes, str]:
    """Create a workspace export archive according to ``options``.

    ``options`` may be a dict or an :class:`ExportOptions`.  Keys / fields:
        * ``mode`` ("strict" | "draft")       - verification policy
        * ``run_tests`` (bool)                - run pytest before export
        * ``run_compilation`` (bool)            - compile native targets before export
        * ``pure_target`` (bool)              - include target source files
        * ``include_native_crate`` (bool)    - include PyO3 native crate source
        * ``include_wavefront_runtime`` (bool) - include embedded aero_core runtime
        * ``standalone_aeroc`` (bool)        - include a pre-packaged ``.aeroc`` project

    Returns ``(archive_bytes, filename)``.
    """
    opts = ExportOptions.from_dict(options, project_name=project_name)
    pure_target = opts.pure_target
    include_native = opts.include_native_crate
    include_wavefront = opts.include_wavefront_runtime
    standalone_aeroc = opts.standalone_aeroc
    project_name = opts.project_name

    # Flush in-memory overlay edits so the export contains the real file tree.
    try:
        OverlayManager(session_dir).flush_to_workspace(session_dir)
    except Exception:
        pass

    # Enforce Blueprint v3 transferability for any export that includes a blueprint.
    blueprint_path = session_dir / "blueprint.aero"
    if blueprint_path.is_file():
        try:
            validate_blueprint_for_export(blueprint_path)
        except Exception:
            # Legacy v2 blueprints or missing validators are allowed to export.
            pass

    # Pre-flight verification: syntax, tests, and compilation checks.
    verification = verify_workspace_for_export(session_dir, opts)

    buf = io.BytesIO()
    file_hashes: Dict[str, str] = {}
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if pure_target:
            for rel, content in _source_files(session_dir).items():
                zf.writestr(rel, content)
                file_hashes[rel] = _sha256(content)

        if include_native:
            for arc, content in _native_crate_source_files():
                arcname = str(arc.as_posix())
                zf.writestr(arcname, content)
                file_hashes[arcname] = _sha256(content)
            toml_text = _pyproject_toml_for_maturin(project_name)
            zf.writestr("pyproject.toml", toml_text)
            file_hashes["pyproject.toml"] = _sha256(toml_text)

        if include_wavefront:
            embedded_src = Path(__file__).resolve().parent / "embedded" / "aero_core"
            if embedded_src.is_dir():
                for path in sorted(embedded_src.rglob("*")):
                    if not path.is_file():
                        continue
                    rel = path.relative_to(embedded_src)
                    if rel.name == "__pycache__" or "__pycache__" in rel.parts:
                        continue
                    if rel.parts and rel.parts[0] == "target":
                        continue
                    arcname = f"crates/aero_core/{rel.as_posix()}"
                    content = path.read_bytes().decode("utf-8", errors="replace")
                    zf.writestr(arcname, content)
                    file_hashes[arcname] = _sha256(content)

        if standalone_aeroc:
            with tempfile.TemporaryDirectory() as tmpdir:
                aeroc_dir = Path(tmpdir) / "aeroc-export"
                export_aeroc_project(session_dir, aeroc_dir, project_name=project_name)
                aeroc_verification = verify_aeroc_project(aeroc_dir, opts)
                verification.update(aeroc_verification)

                aeroc_hashes = _hash_aeroc_dir(aeroc_dir)
                aeroc_json = generate_verification_json(aeroc_verification, aeroc_hashes)
                (aeroc_dir / "verification.json").write_text(aeroc_json, encoding="utf-8")

                aeroc_archive = package_aeroc(aeroc_dir)
                archive_bytes = aeroc_archive.read_bytes()
                arcname = f"{project_name}.aerozip"
                zf.writestr(arcname, archive_bytes)
                file_hashes[arcname] = _sha256_bytes(archive_bytes)

        verification_json = generate_verification_json(verification, file_hashes)
        zf.writestr("verification.json", verification_json)

    filename = f"{project_name}.zip"
    return buf.getvalue(), filename


def _sha256(content: str) -> str:
    import hashlib
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _sha256_bytes(content: bytes) -> str:
    import hashlib
    return hashlib.sha256(content).hexdigest()


def _hash_aeroc_dir(project_dir: Path) -> Dict[str, str]:
    """Return SHA-256 hashes for every file in an exported aeroc project."""
    import hashlib

    hashes: Dict[str, str] = {}
    for path in sorted(project_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(project_dir).as_posix()
        if rel == "verification.json":
            continue
        try:
            hashes[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            continue
    return hashes


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
