"""Compact workspace bundler for LLM chat context.

Exports ``bundle_workspace`` and helpers that turn the current project tree into
a token-efficient, machine-readable payload (dictionary and serialized XML/JSON)
so the chat assistant can reason about source files, blueprints, and test status
during multi-turn sessions.
"""

from __future__ import annotations

import enum
import io
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from html import escape as html_escape
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from aero_forge.scaffold.cargo_config import CARGO_CONFIG_TOML

SOURCE_SUFFIXES = {
    ".rs",
    ".py",
    ".toml",
    ".aero",
    ".md",
    ".json",
    ".yaml",
    ".yml",
    ".ini",
    ".cfg",
    ".sh",
    ".txt",
}

BINARY_SUFFIXES = {
    ".so",
    ".pyd",
    ".dll",
    ".dylib",
    ".wasm",
    ".zip",
    ".tar",
    ".gz",
    ".bz2",
    ".xz",
    ".whl",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".pdf",
    ".o",
    ".rlib",
    ".rmeta",
    ".d",
    ".exe",
}

SKIP_DIRS = {
    "target",
    "dist",
    "build",
    "__pycache__",
    ".pytest_cache",
    ".cargo",
    ".git",
    ".aero-forge-cache",
    ".build_cache",
    ".overlays",
    ".mypy_cache",
    ".tox",
    ".venv",
    "venv",
    "node_modules",
    ".eggs",
    "*.egg-info",
}


def _is_source_file(path: Path) -> bool:
    """Return True for human-relevant source/configuration files."""
    if path.name.startswith("."):
        return False
    ext = path.suffix.lower()
    if ext in BINARY_SUFFIXES or not ext:
        # Allow shebang scripts without an extension.
        if path.suffix == "":
            try:
                with open(path, "rb") as f:
                    first = f.read(16)
            except OSError:
                return False
            return first.startswith(b"#!")
        return False
    return ext in SOURCE_SUFFIXES


def _should_skip_dir(rel: Path) -> bool:
    """Return True if a directory path should be skipped entirely."""
    parts = set(rel.parts)
    if parts & SKIP_DIRS:
        return True
    for part in rel.parts:
        if part.endswith(".egg-info"):
            return True
    return False


def _compact(content: str) -> str:
    """Reduce whitespace noise without altering semantics.

    - Strip trailing whitespace from each line.
    - Collapse runs of more than two consecutive blank lines to two.
    """
    lines = [line.rstrip() for line in content.splitlines()]
    compacted: List[str] = []
    blank_count = 0
    for line in lines:
        if line == "":
            blank_count += 1
            if blank_count <= 2:
                compacted.append(line)
        else:
            blank_count = 0
            compacted.append(line)
    return "\n".join(compacted).strip("\n") + "\n"


def _read_test_status(workspace_dir: Path) -> Optional[Dict[str, Any]]:
    """Return the latest test execution status if it is available on disk."""
    status_candidates = [
        workspace_dir / ".aero" / "test_status.json",
        workspace_dir / ".pytest_cache" / "v" / "cache" / "lastfailed",
        workspace_dir / "test-results.json",
    ]
    for candidate in status_candidates:
        if not candidate.is_file():
            continue
        try:
            data = candidate.read_text(encoding="utf-8")
        except OSError:
            continue
        if candidate.name == "lastfailed":
            failed = [line for line in data.splitlines() if line.strip()]
            return {"status": "failed" if failed else "passed", "failed_tests": failed}
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return {"raw": data}

    tests_dir = workspace_dir / "tests"
    if tests_dir.is_dir():
        test_files = sorted(
            p.relative_to(workspace_dir).as_posix()
            for p in tests_dir.rglob("test_*.py")
        )
        if test_files:
            return {"status": "unknown", "test_files": test_files}
    return None


def bundle_workspace(
    workspace_dir: Path,
    max_file_size_kb: int = 100,
) -> Dict[str, Any]:
    """Package *workspace_dir* into a compact, LLM-friendly dictionary.

    Returns a dictionary with:
    - ``workspace``: absolute path to the workspace.
    - ``files``: mapping from relative POSIX path to compacted source content.
    - ``blueprint``: contents of ``blueprint.aero`` if present.
    - ``test_status``: latest test execution status if available.
    """
    workspace_dir = Path(workspace_dir).resolve()
    max_bytes = max_file_size_kb * 1024

    bundle: Dict[str, Any] = {
        "workspace": str(workspace_dir),
        "files": {},
        "blueprint": None,
        "test_status": None,
    }

    for path in sorted(workspace_dir.rglob("*")):
        try:
            rel = path.relative_to(workspace_dir)
        except ValueError:
            continue
        if _should_skip_dir(rel):
            # Prune the whole directory tree by skipping its contents.
            continue
        if not path.is_file():
            continue
        if not _is_source_file(rel):
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > max_bytes:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        bundle["files"][rel.as_posix()] = _compact(content)

    blueprint = workspace_dir / "blueprint.aero"
    if blueprint.is_file() and blueprint.stat().st_size <= max_bytes:
        try:
            bundle["blueprint"] = _compact(blueprint.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, OSError):
            pass

    bundle["test_status"] = _read_test_status(workspace_dir)
    return bundle


def bundle_to_xml(bundle: Dict[str, Any]) -> str:
    """Serialize a bundle into an XML-like string with ``<file path="...">`` tags."""
    lines = [
        '<workspace path="{}">'.format(html_escape(str(bundle["workspace"]), quote=True)),
    ]
    if bundle.get("blueprint"):
        lines.append("<blueprint>{}</blueprint>".format(html_escape(bundle["blueprint"], quote=True)))
    if bundle.get("test_status"):
        status = json.dumps(bundle["test_status"], indent=2)
        lines.append("<test_status>{}</test_status>".format(html_escape(status, quote=True)))
    for path, content in bundle["files"].items():
        lines.append('<file path="{}">'.format(html_escape(path, quote=True)))
        lines.append(html_escape(content, quote=True))
        lines.append("</file>")
    lines.append("</workspace>")
    return "\n".join(lines)


def bundle_to_json(bundle: Dict[str, Any]) -> str:
    """Serialize a bundle as a compact JSON string."""
    return json.dumps(bundle, indent=2)


def format_context_block(bundle: Dict[str, Any], fmt: str = "xml") -> str:
    """Return a string suitable for injection into a system/chat prompt."""
    if fmt == "json":
        body = bundle_to_json(bundle)
    else:
        body = bundle_to_xml(bundle)
    return f"CURRENT_PROJECT_CONTEXT ({fmt.upper()} workspace bundle):\n{body}\n---END PROJECT CONTEXT---"


class ExportProfile(str, enum.Enum):
    """Bundle export profiles for generated project archives."""

    STANDARD = "standard"
    ACCELERATED_PYO3 = "accelerated_py03"


# Source directory for the in-repo PyO3 native acceleration crate.
_NATIVE_CRATE_SOURCE = Path(__file__).resolve().parent / "_native"


def _zip_skip(rel: Path) -> bool:
    """Return True when ``rel`` should be omitted from an exported zip archive."""
    if not rel.parts:
        return False
    name = rel.name
    if name.startswith("."):
        return True
    if name in SKIP_DIRS or any(part in SKIP_DIRS for part in rel.parts):
        return True
    if name.endswith(".egg-info"):
        return True
    ext = name.split(".")[-1].lower() if "." in name else ""
    if ext and f".{ext}" in BINARY_SUFFIXES:
        return True
    return False


def _pyproject_toml_for_maturin(
    project_name: str = "generated-native",
    module_name: str = "aero_forge_native",
    manifest_path: str = "crates/native_core/Cargo.toml",
) -> str:
    """Return a root ``pyproject.toml`` configured to build the native core with maturin."""
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
manifest-path = "{manifest_path}"
module-name = "{module_name}"
"""


def _native_crate_source_files() -> List[Tuple[str, str]]:
    """Return (arcname, content) pairs for the PyO3 native acceleration crate source.

    Only textual source/configuration files are included; compiled artifacts and
    build directories are ignored.
    """
    if not _NATIVE_CRATE_SOURCE.is_dir():
        return []

    files: List[Tuple[str, str]] = []
    for path in sorted(_NATIVE_CRATE_SOURCE.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(_NATIVE_CRATE_SOURCE)
        if _zip_skip(rel):
            continue
        try:
            files.append((Path("crates") / "native_core" / rel, path.read_text(encoding="utf-8")))
        except (UnicodeDecodeError, OSError):
            continue
    return files


def create_project_zip(
    workspace_dir: Path,
    profile: ExportProfile = ExportProfile.STANDARD,
    project_name: str = "generated-native",
    native_crate_source: Optional[Path] = None,
) -> bytes:
    """Return a zip archive of ``workspace_dir`` according to ``profile``.

    * ``STANDARD`` includes human-relevant source/config files only.
    * ``ACCELERATED_PYO3`` additionally bundles ``crates/native_core/`` with the
      PyO3 acceleration crate source and a root ``pyproject.toml`` configured for
      ``maturin develop`` / ``pip install .``.
    """
    workspace_dir = Path(workspace_dir).resolve()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(workspace_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(workspace_dir)
            if _zip_skip(rel):
                continue
            # Exclude the accelerator crate if it happens to be inside the workspace.
            if rel.parts[:2] == ("crates", "native_core"):
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            zf.writestr(str(rel.as_posix()), content)

        if profile == ExportProfile.ACCELERATED_PYO3:
            crate_source = native_crate_source or _NATIVE_CRATE_SOURCE
            source_files = _native_crate_source_files() if not native_crate_source else []
            if native_crate_source:
                for path in sorted(native_crate_source.rglob("*")):
                    if not path.is_file():
                        continue
                    rel = path.relative_to(native_crate_source)
                    if _zip_skip(rel):
                        continue
                    try:
                        source_files.append(
                            (Path("crates") / "native_core" / rel, path.read_text(encoding="utf-8"))
                        )
                    except (UnicodeDecodeError, OSError):
                        continue
            for arc, content in source_files:
                zf.writestr(str(arc.as_posix()), content)
            zf.writestr("pyproject.toml", _pyproject_toml_for_maturin(project_name))
            zf.writestr(".cargo/config.toml", CARGO_CONFIG_TOML)

    return buf.getvalue()


def zip_export_filename(profile: ExportProfile) -> str:
    """Return the suggested download filename for an export profile."""
    if profile == ExportProfile.ACCELERATED_PYO3:
        return "project-accelerated.zip"
    return "project-standard.zip"


def scaffold_native_crate(
    workspace_dir: Path,
    project_name: str = "generated-native",
    native_crate_source: Optional[Path] = None,
) -> None:
    """Write the PyO3 native acceleration crate source into ``workspace_dir``.

    The crate is placed at ``crates/native_core/`` and a root ``pyproject.toml``
    configured for ``maturin`` is written only if those files do not already exist.
    Compiled artifacts (``.so``, ``target/``, etc.) are ignored when copying.
    """
    workspace_dir = Path(workspace_dir).resolve()
    crate_dest = workspace_dir / "crates" / "native_core"
    crate_source = native_crate_source or _NATIVE_CRATE_SOURCE

    if crate_dest.is_dir() and (crate_dest / "Cargo.toml").is_file():
        # Leave an existing crate in place to avoid overwriting user work.
        pass
    else:
        crate_dest.mkdir(parents=True, exist_ok=True)
        for path in sorted(crate_source.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(crate_source)
            if _zip_skip(rel):
                continue
            # Never copy workspace/build artifacts from the source crate tree.
            if rel.name == "Cargo.lock" or rel.name.endswith(".egg-info"):
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            target = crate_dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.name == "Cargo.toml":
                # Use the canonical directory name as the package name so that
                # workspace-scoped cargo commands like ``cargo test -p native_core``
                # resolve correctly without relying on the source crate's name.
                content = re.sub(
                    r'^name\s*=\s*"[^"]+"',
                    'name = "native_core"',
                    content,
                    flags=re.MULTILINE,
                )
            target.write_text(content, encoding="utf-8")

    pyproject = workspace_dir / "pyproject.toml"
    if not pyproject.is_file():
        pyproject.write_text(_pyproject_toml_for_maturin(project_name), encoding="utf-8")

    from aero_forge.scaffold.cargo_config import write_cargo_config
    from aero_forge.scaffold.cargo_manifest import ensure_workspace_cargo_toml

    ensure_workspace_cargo_toml(workspace_dir, crate_member="crates/native_core")
    write_cargo_config(workspace_dir)
