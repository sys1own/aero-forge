"""Wavefront scaffold exporter.

A scaffold bundle (``.aerozip`` / ``-scaffold.zip``) is a self-contained
project archive containing the embedded ``aero_core`` zero-dependency Rust
wavefront micro-runtime, Python wrapper entrypoints, and the original
workspace source files / compiled ``workspace.aeroc`` binary.  The bundle can
be built and run with ``cargo build --release`` or ``pip install .`` without
any ``aero_forge`` dependency.
"""

from __future__ import annotations

import io
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

from aero_forge.scaffold.cargo_config import write_cargo_config


_AERO_CORE_TEMPLATE = Path(__file__).resolve().parent / "embedded" / "aero_core"

_PYPROJECT_TOML = """\
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "aeroc"
version = "0.1.0"
description = "Standalone Aero-Forge compiled artifact runner"
requires-python = ">=3.9"

[project.scripts]
aeroc-runner = "aeroc.cli:main"

[tool.setuptools.packages.find]
where = ["."]
include = ["aeroc*"]

[tool.setuptools.package-data]
aeroc = ["aero_core/**/*"]
"""

_AEROC_INIT = """\
\"\"\"Aero-Forge standalone compiled artifact runner.\"\"\"

from pathlib import Path

def crate_dir() -> Path:
    return Path(__file__).resolve().parent / "aero_core"
"""

_AEROC_CLI = '''\
\"\"\"Console entrypoint for the exported aeroc runner.\"\"\"

import os
import subprocess
import sys
from pathlib import Path


def _crate_dir() -> Path:
    return Path(__file__).resolve().parent / "aero_core"


def _binary_path(crate: Path) -> Path:
    return crate / "target" / "release" / "aeroc-runner"


def main() -> int:
    crate = _crate_dir()
    binary = _binary_path(crate)
    if not binary.is_file():
        subprocess.run(
            ["cargo", "build", "--release"],
            cwd=str(crate),
            check=True,
        )
    cmd = [str(binary)] + sys.argv[1:]
    return subprocess.run(cmd, cwd=os.getcwd()).returncode


if __name__ == "__main__":
    sys.exit(main())
'''


def _copy_template(src: Path, dst: Path) -> None:
    if not src.is_dir():
        raise FileNotFoundError(f"Embedded aero_core template not found at {src}")
    shutil.copytree(src, dst, dirs_exist_ok=True)
    # Remove any __pycache__ left over from template traversal.
    for path in dst.rglob("__pycache__"):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)


def export_aeroc_project(
    workspace_dir: Path,
    output_dir: Path,
    project_name: str = "aeroc-export",
) -> Path:
    """Create a standalone ``.aeroc`` project directory at ``output_dir``.

    The returned path is the project root; it contains ``pyproject.toml``,
    the ``aeroc`` Python package with the embedded ``aero_core`` crate, and
    an optimized ``.cargo/config.toml`` inside the crate.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pkg_dir = output_dir / "aeroc"
    pkg_dir.mkdir(parents=True, exist_ok=True)

    (pkg_dir / "__init__.py").write_text(_AEROC_INIT, encoding="utf-8")
    (pkg_dir / "cli.py").write_text(_AEROC_CLI, encoding="utf-8")

    crate_dir = pkg_dir / "aero_core"
    _copy_template(_AERO_CORE_TEMPLATE, crate_dir)

    # Ensure hardware-optimized cargo config is present in the exported crate.
    write_cargo_config(crate_dir)

    (output_dir / "pyproject.toml").write_text(_PYPROJECT_TOML, encoding="utf-8")
    (output_dir / "README.md").write_text(
        f"# {project_name}\n\nWavefront scaffold bundle.\n\n"
        "Build with `cargo build --release` inside `aeroc/aero_core/`,\n"
        "or install the Python wrapper with `pip install .` and run `aeroc-runner`.\n"
        "If a `workspace.aeroc` binary IR container is present, the runner will\n"
        "execute it directly without extracting files.\n",
        encoding="utf-8",
    )
    return output_dir


def package_aeroc(project_dir: Path, output_path: Optional[Path] = None) -> Path:
    """Zip the exported project directory into ``{project_dir}.aerozip``."""
    project_dir = Path(project_dir).resolve()
    output = output_path or (project_dir.with_suffix(".aerozip"))
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(project_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(project_dir))
    return output


def export_scaffold_zip(
    workspace_dir: Path,
    output_path: Optional[Path] = None,
    project_name: str = "aero-forge-export",
) -> Path:
    """Package the workspace and the aero_core runtime into a single scaffold zip.

    The resulting ``.aerozip`` archive contains the workspace files, the
    ``aero_core`` Rust runtime, Python wrapper entrypoints, and the compiled
    ``workspace.aeroc`` binary if it exists.
    """
    workspace_dir = Path(workspace_dir).resolve()
    with tempfile.TemporaryDirectory() as tmpdir:
        scaffold_dir = Path(tmpdir) / "scaffold"
        export_aeroc_project(workspace_dir, scaffold_dir, project_name=project_name)

        skip_prefixes = {
            "target",
            ".venv",
            "__pycache__",
            ".pytest_cache",
            ".aero",
            ".cargo",
            "dist",
            "build",
        }
        for src in sorted(workspace_dir.rglob("*")):
            if not src.is_file():
                continue
            rel = src.relative_to(workspace_dir)
            if any(part in skip_prefixes for part in rel.parts[:1]):
                continue
            if rel.name.startswith("."):
                continue
            dst = scaffold_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

        aeroc_path = workspace_dir / "workspace.aeroc"
        if aeroc_path.is_file():
            shutil.copy2(aeroc_path, scaffold_dir / "workspace.aeroc")

        output = output_path or (workspace_dir / f"{project_name}.aerozip")
        return package_aeroc(scaffold_dir, output)


def compile_aeroc(
    project_dir: Path,
    timeout: Optional[float] = 300.0,
) -> Path:
    """Build the standalone runner binary inside ``project_dir``.

    Returns the path to the compiled ``aeroc-runner`` binary.
    """
    project_dir = Path(project_dir).resolve()
    crate_dir = project_dir / "aeroc" / "aero_core"
    if not crate_dir.is_dir():
        raise FileNotFoundError(f"aero_core crate missing in {project_dir}")

    cmd = ["cargo", "build", "--release"]
    subprocess.run(
        cmd,
        cwd=str(crate_dir),
        check=True,
        timeout=timeout,
    )
    binary = crate_dir / "target" / "release" / "aeroc-runner"
    if not binary.is_file():
        raise FileNotFoundError(f"Compiled binary not found at {binary}")
    return binary


def export_and_compile_aeroc(
    workspace_dir: Path,
    output_dir: Path,
    project_name: str = "aeroc-export",
) -> Path:
    """Export and compile the standalone runner in one step."""
    project_dir = export_aeroc_project(workspace_dir, output_dir, project_name)
    return compile_aeroc(project_dir)
