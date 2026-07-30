"""Wavefront scaffold exporter.

A scaffold bundle (``.aerozip`` / ``-scaffold.zip``) is a self-contained
project archive containing the embedded ``aero_core`` zero-dependency Rust
wavefront micro-runtime, Python wrapper entrypoints, and the original
workspace source files / compiled ``workspace.aeroc`` binary.  The bundle can
be built and run with ``cargo build --release`` or ``pip install .`` without
any ``aero_forge`` dependency.
"""

from __future__ import annotations

import datetime
import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from aero_forge.errors import AeroForgeError, ExportVerificationError
from aero_forge.overlay import OverlayManager
from aero_forge.scaffold.cargo_config import write_cargo_config
from aero_forge.scaffold.pre_write_validator import (
    PreWriteValidator,
    ValidationError,
)


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


def _opt(options: Optional[Any], name: str, default: Any) -> Any:
    """Return option *name* from a dict or dataclass-like object."""
    if options is None:
        return default
    if isinstance(options, dict):
        return options.get(name, default)
    return getattr(options, name, default)


def _git_commit_or_version() -> str:
    """Return the current git commit hash, or ``unknown`` when not in a repo."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    except FileNotFoundError:
        pass
    return "unknown"


def _file_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _hash_project_files(project_dir: Path) -> Dict[str, str]:
    """Return a mapping of relative file paths to SHA-256 hashes."""
    hashes: Dict[str, str] = {}
    for path in sorted(project_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(project_dir).as_posix()
        if rel == "verification.json":
            continue
        try:
            hashes[rel] = _file_hash(path.read_bytes())
        except OSError:
            continue
    return hashes


def _parse_pytest_summary(output: str) -> Dict[str, Any]:
    """Parse pytest terminal summary for pass/fail/skip counts."""
    summary: Dict[str, Any] = {
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "total": 0,
        "success": True,
    }
    # Match lines like "3 passed, 1 failed, 2 skipped in 1.23s"
    match = re.search(
        r"(\d+) passed(?:, (\d+) failed)?(?:, (\d+) skipped)?(?:, (\d+) error)? in ",
        output,
    )
    if match:
        summary["passed"] = int(match.group(1) or 0)
        summary["failed"] = int(match.group(2) or 0)
        summary["skipped"] = int(match.group(3) or 0)
        summary["error"] = int(match.group(4) or 0)
        summary["total"] = (
            summary["passed"] + summary["failed"] + summary["skipped"] + summary["error"]
        )
        summary["success"] = summary["failed"] == 0 and summary["error"] == 0
    return summary


def _find_test_files(workspace_dir: Path) -> List[Path]:
    """Return Python test files under *workspace_dir*, skipping build caches."""
    excluded = {"target", ".venv", "__pycache__", ".pytest_cache", ".git", ".aero_core"}
    test_files: List[Path] = []
    for path in workspace_dir.rglob("test_*.py"):
        if not path.is_file():
            continue
        if any(part in excluded for part in path.parts):
            continue
        test_files.append(path)
    return test_files


def _find_cargo_manifest(workspace_dir: Path) -> Optional[Path]:
    """Return the nearest ``Cargo.toml`` owned by the workspace (not vendored deps)."""
    excluded = {"target", ".cargo", "vendor", "node_modules"}
    for path in workspace_dir.rglob("Cargo.toml"):
        if any(part in excluded for part in path.parts):
            continue
        return path
    return None


def verify_workspace_for_export(
    workspace_dir: Path,
    options: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run pre-flight verification and return a verification manifest.

    The manifest contains test telemetry, a performance baseline, and any
    verification errors encountered.  In ``strict`` mode an
    ``ExportVerificationError`` is raised when verification fails.
    """
    workspace_dir = Path(workspace_dir).resolve()
    mode = _opt(options, "mode", "strict")
    run_tests = _opt(options, "run_tests", True)
    run_compilation = _opt(options, "run_compilation", True)

    start = time.time()
    errors: List[str] = []
    test_summary: Dict[str, Any] = {
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "total": 0,
        "success": True,
    }

    # Static / syntax validation for Python and Rust workspaces.
    py_files = [p for p in workspace_dir.rglob("*.py") if p.is_file()]
    cargo_toml = _find_cargo_manifest(workspace_dir)
    cpp_files = [p for p in workspace_dir.rglob("*.cpp") if p.is_file()]

    if py_files:
        try:
            PreWriteValidator().validate(workspace_dir, language="python")
        except ValidationError as exc:
            errors.append(str(exc))

    if cargo_toml is not None and cargo_toml.is_file():
        if run_compilation and shutil.which("cargo"):
            try:
                result = subprocess.run(
                    ["cargo", "check"],
                    cwd=str(cargo_toml.parent),
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                if result.returncode != 0:
                    errors.append(result.stdout + result.stderr)
            except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
                errors.append(f"Rust compilation check failed: {exc}")

    if cpp_files and run_compilation and shutil.which("g++"):
        # Lightweight syntax-only check for the first C++ source file.
        try:
            result = subprocess.run(
                ["g++", "-fsyntax-only", "-std=c++17", str(cpp_files[0])],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode != 0:
                errors.append(result.stderr or result.stdout)
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            errors.append(f"C++ syntax check failed: {exc}")

    # Run pytest if the workspace contains Python tests.
    if run_tests and py_files:
        test_files = _find_test_files(workspace_dir)
        if test_files:
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pytest", "-q", "--tb=no"],
                    cwd=str(workspace_dir),
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                test_summary = _parse_pytest_summary(result.stdout + result.stderr)
                if result.returncode != 0 and test_summary["success"]:
                    test_summary["success"] = False
            except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
                test_summary = {"passed": 0, "failed": 0, "skipped": 0, "total": 0, "success": False}
                errors.append(f"pytest execution failed: {exc}")

    elapsed = time.time() - start
    success = not errors and test_summary.get("success", True)

    verification = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "commit_or_version": _git_commit_or_version(),
        "test_summary": test_summary,
        "performance_baseline": {"verification_seconds": round(elapsed, 4)},
        "errors": errors,
        "success": success,
    }

    if not success and str(mode).lower() == "strict":
        raise ExportVerificationError(
            f"Export verification failed: {errors or 'tests failed'}",
            verification=verification,
        )

    verification["unverified"] = not success
    return verification


def verify_aeroc_project(
    project_dir: Path,
    options: Optional[Any] = None,
) -> Dict[str, Any]:
    """Verify the exported ``aero_core`` standalone runner project compiles."""
    project_dir = Path(project_dir).resolve()
    run_compilation = _opt(options, "run_compilation", True)
    mode = _opt(options, "mode", "strict")

    errors: List[str] = []
    start = time.time()

    crate_dir = project_dir / "aeroc" / "aero_core"
    if run_compilation and crate_dir.is_dir() and shutil.which("cargo"):
        try:
            result = subprocess.run(
                ["cargo", "build", "--release"],
                cwd=str(crate_dir),
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                errors.append(result.stdout + result.stderr)
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            errors.append(f"aero_core compilation failed: {exc}")

    elapsed = time.time() - start
    success = not errors
    verification = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "commit_or_version": _git_commit_or_version(),
        "test_summary": {"passed": 0, "failed": 0, "skipped": 0, "total": 0, "success": success},
        "performance_baseline": {"verification_seconds": round(elapsed, 4)},
        "errors": errors,
        "success": success,
    }

    if not success and str(mode).lower() == "strict":
        raise ExportVerificationError(
            "Standalone aeroc project failed compilation",
            verification=verification,
        )

    verification["unverified"] = not success
    return verification


def generate_verification_json(
    verification: Dict[str, Any],
    file_hashes: Optional[Dict[str, str]] = None,
) -> str:
    """Return a JSON string for ``verification.json``.

    The payload contains the verification telemetry plus a SHA-256 hash map for
    every bundled file.  In ``draft`` mode or when verification failed, the
    ``unverified`` flag is set to ``true``.
    """
    payload = dict(verification)
    if file_hashes is not None:
        payload["file_hashes"] = file_hashes
    payload.setdefault("unverified", not payload.get("success", True))
    return json.dumps(payload, indent=2, sort_keys=True)


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
    options: Optional[Any] = None,
) -> Path:
    """Package the workspace and the aero_core runtime into a single scaffold zip.

    The resulting ``.aerozip`` archive contains the workspace files, the
    ``aero_core`` Rust runtime, Python wrapper entrypoints, a ``verification.json``
    attestation, and the compiled ``workspace.aeroc`` binary if it exists.
    """
    workspace_dir = Path(workspace_dir).resolve()

    # Flush any in-memory overlay edits so the exported scaffold reflects real files.
    try:
        OverlayManager(workspace_dir).flush_to_workspace(workspace_dir)
    except Exception:
        pass

    verification = verify_workspace_for_export(workspace_dir, options)

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

        # Verify the standalone aero_core runner compiles before packaging.
        aeroc_verification = verify_aeroc_project(scaffold_dir, options)
        verification.update(aeroc_verification)

        file_hashes = _hash_project_files(scaffold_dir)
        verification_json = generate_verification_json(verification, file_hashes)
        (scaffold_dir / "verification.json").write_text(
            verification_json, encoding="utf-8"
        )

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
