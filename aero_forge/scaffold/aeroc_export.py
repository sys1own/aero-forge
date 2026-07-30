"""Wavefront scaffold exporter.

A scaffold bundle (``.aerozip`` / ``-scaffold.zip``) is a self-contained
project archive containing the embedded ``aero_core`` zero-dependency Rust
wavefront micro-runtime, Python wrapper entrypoints, and the original
workspace source files / compiled ``workspace.aeroc`` binary.  The bundle can
be built and run with ``cargo build --release`` or ``pip install .`` without
any ``aero_forge`` dependency.
"""

from __future__ import annotations

import base64
import datetime
import hashlib
import io
import json
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
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


# ---------------------------------------------------------------------------
# Hybrid binary + source .aeroc packing and unpacking
# ---------------------------------------------------------------------------


def _host_platform_tag() -> str:
    """Return a Rust-compatible ``{os}_{arch}`` tag for the current host."""
    os_name = sys.platform
    if os_name.startswith("linux"):
        os_name = "linux"
    elif os_name == "darwin":
        os_name = "macos"
    elif os_name.startswith("win"):
        os_name = "windows"
    elif os_name.startswith("freebsd"):
        os_name = "freebsd"

    arch = platform.machine().lower()
    arch_aliases = {
        "amd64": "x86_64",
        "x86_64": "x86_64",
        "x86": "x86",
        "i386": "x86",
        "i686": "x86",
        "arm64": "aarch64",
        "aarch64": "aarch64",
        "arm": "arm",
    }
    arch = arch_aliases.get(arch, arch)
    return f"{os_name}_{arch}"


def _is_native_artifact(path: Path) -> bool:
    """Return True if *path* is a pre-compiled native binary or wheel."""
    if not path.is_file():
        return False
    return path.suffix.lower() in {".so", ".dylib", ".dll", ".pyd", ".whl"}


def _read_pyproject_dependencies(workspace_dir: Path) -> List[str]:
    """Parse Python project dependencies from ``pyproject.toml``."""
    pyproject = workspace_dir / "pyproject.toml"
    if not pyproject.is_file():
        return []
    try:
        import tomli as tomllib
    except ImportError:
        return []
    try:
        with pyproject.open("rb") as f:
            data = tomllib.load(f)
    except Exception:
        return []
    return data.get("project", {}).get("dependencies", []) or []


def _read_cargo_dependencies(workspace_dir: Path) -> Dict[str, Any]:
    """Parse Rust crate dependencies from ``Cargo.toml``."""
    cargo = workspace_dir / "Cargo.toml"
    if not cargo.is_file():
        return {}
    try:
        import tomli as tomllib
    except ImportError:
        return {}
    try:
        with cargo.open("rb") as f:
            data = tomllib.load(f)
    except Exception:
        return {}
    return data.get("dependencies", {}) or {}


def _build_environment_lock(
    workspace_dir: Path,
    platform_tag: str,
    native_binaries: List[Path],
) -> Dict[str, Any]:
    """Build a pinned dependency / toolchain lock file for the .aeroc bundle."""
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    py_deps = _read_pyproject_dependencies(workspace_dir)
    rust_deps = _read_cargo_dependencies(workspace_dir)

    toolchains: List[str] = ["python>=3.9"]
    if (workspace_dir / "Cargo.toml").is_file() or any(
        p.name == "Cargo.toml" for p in workspace_dir.rglob("Cargo.toml")
    ):
        toolchains.append("cargo>=1.70")
    if any(p.suffix.lower() in {".cpp", ".cxx", ".c", ".h", ".hpp"} for p in workspace_dir.rglob("*")):
        toolchains.append("g++")

    return {
        "schema_version": 1,
        "source_root": "src",
        "artifact_root": f"artifacts/native/{platform_tag}",
        "platforms": [platform_tag],
        "toolchains": toolchains,
        "python": {
            "version": python_version,
            "dependencies": py_deps,
        },
        "rust": {
            "dependencies": rust_deps,
        },
        "native_binaries": [f"artifacts/native/{platform_tag}/{p.name}" for p in native_binaries],
    }


def compile_hybrid_aeroc(
    workspace_dir: Path,
    output_path: Union[str, Path],
    platform_tag: Optional[str] = None,
) -> str:
    """Compile a workspace into a hybrid .aeroc container with native binaries.

    The container layout is:

        src/                          - raw source files
        artifacts/native/{os}_{arch}/ - pre-compiled .so / .dylib / .dll
        environment.lock              - pinned dependencies and toolchains

    Native binaries are detected from the workspace root (and ``target/release/``,
    ``dist/`` and ``build/`` subdirectories).  Source files are placed under
    ``src/`` so the unpacker can strip the prefix during extraction.
    """
    workspace_dir = Path(workspace_dir).resolve()
    output_path = Path(output_path).resolve()
    platform_tag = platform_tag or _host_platform_tag()

    from aero_forge._native import compile_aeroc

    exclude_prefixes = {
        ".aero",
        ".git",
        ".aero_core",
        "target",
        ".venv",
        "__pycache__",
        ".pytest_cache",
        ".cargo",
        "node_modules",
    }

    native_binaries: List[Path] = []
    source_files: List[Path] = []

    # Native artifact search paths.
    native_search_dirs = {
        workspace_dir,
        workspace_dir / "target" / "release",
        workspace_dir / "dist",
        workspace_dir / "build",
    }
    for search_dir in native_search_dirs:
        if not search_dir.is_dir():
            continue
        for path in sorted(search_dir.rglob("*")):
            if _is_native_artifact(path):
                native_binaries.append(path)

    for path in sorted(workspace_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(workspace_dir)
        if any(part in exclude_prefixes for part in rel.parts):
            continue
        if rel.name.startswith("."):
            continue
        if _is_native_artifact(path):
            # Already accounted for above.
            continue
        source_files.append(path)

    sources: List[Dict[str, str]] = []

    # Layout marker so the unpacker knows to strip the ``src/`` prefix.
    marker = json.dumps({"layout": "hybrid", "version": 1}, indent=2)
    sources.append({
        "path": ".aeroc_hybrid",
        "content_base64": base64.b64encode(marker.encode("utf-8")).decode("ascii"),
    })

    # Add source files under the ``src/`` prefix.
    for path in source_files:
        rel = path.relative_to(workspace_dir).as_posix().lstrip("/")
        try:
            data = path.read_bytes()
        except OSError:
            continue
        sources.append({
            "path": f"src/{rel}",
            "content_base64": base64.b64encode(data).decode("ascii"),
        })

    # Add native binaries under the platform-specific artifact directory.
    for path in native_binaries:
        try:
            data = path.read_bytes()
        except OSError:
            continue
        sources.append({
            "path": f"artifacts/native/{platform_tag}/{path.name}",
            "content_base64": base64.b64encode(data).decode("ascii"),
        })

    # Add environment.lock at the root.
    lock = _build_environment_lock(workspace_dir, platform_tag, native_binaries)
    lock_json = json.dumps(lock, indent=2, sort_keys=True)
    sources.append({
        "path": "environment.lock",
        "content_base64": base64.b64encode(lock_json.encode("utf-8")).decode("ascii"),
    })

    spec = {
        "nodes": ["workspace"],
        "edges": {},
        "instructions": [{"op": "HALT"}],
        "sources": sources,
        "flags": 0,
    }

    return compile_aeroc(json.dumps(spec), str(output_path))


def _cargo_manifest_in(output_dir: Path) -> Optional[Path]:
    """Locate the Cargo.toml to use for source fallback compilation."""
    for candidate in [output_dir / "Cargo.toml", output_dir / "src" / "Cargo.toml"]:
        if candidate.is_file():
            return candidate
    return None


def _fallback_compile_command(output_dir: Path, env: Dict[str, str]) -> Optional[List[str]]:
    """Return a compile command appropriate for the extracted workspace."""
    cargo_toml = _cargo_manifest_in(output_dir)
    if cargo_toml:
        args = ["cargo", "build", "--release"]
        if cargo_toml.parent != output_dir:
            args.extend(["--manifest-path", str(cargo_toml)])
        return args

    pyproject = output_dir / "pyproject.toml"
    if pyproject.is_file():
        return ["pip", "install", "--no-cache-dir", "-e", "."]

    setup = output_dir / "setup.py"
    if setup.is_file():
        return ["pip", "install", "--no-cache-dir", "-e", "."]

    return None


def unpack_aeroc_with_fallback(
    aeroc_path: Union[str, Path],
    output_dir: Union[str, Path],
    run_fallback: bool = True,
) -> Dict[str, Any]:
    """Unpack a hybrid .aeroc container and optionally compile from source.

    The Rust unpacker extracts matching native binaries and ``src/`` source
    files.  If no native artifact matches the host platform, it leaves a
    ``.aeroc_fallback`` marker in *output_dir*.  When *run_fallback* is True
    the Python side invokes :class:`ToolchainManager` to build from source.
    """
    from aero_forge._native import unpack_aeroc
    from aero_forge.toolchain import ToolchainManager

    aeroc_path = Path(aeroc_path).resolve()
    output_dir = Path(output_dir).resolve()

    count = unpack_aeroc(str(aeroc_path), str(output_dir))
    result: Dict[str, Any] = {
        "extracted": count,
        "output_dir": str(output_dir),
        "fallback": False,
        "compiled": False,
    }

    fallback_marker = output_dir / ".aeroc_fallback"
    if fallback_marker.is_file():
        result["fallback"] = True
        reason = fallback_marker.read_text(encoding="utf-8").strip()
        result["fallback_reason"] = reason

        if run_fallback:
            command = _fallback_compile_command(output_dir, {})
            if command:
                toolchain = ToolchainManager(output_dir)
                toolchain.prepare_environment(command[0])
                proc = subprocess.run(
                    command,
                    cwd=str(output_dir),
                    env=toolchain.env,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                result["compiled"] = proc.returncode == 0
                result["compile_stdout"] = proc.stdout
                result["compile_stderr"] = proc.stderr
                if proc.returncode == 0:
                    fallback_marker.unlink()
                    result["fallback"] = False
            else:
                result["compile_stderr"] = "No supported source manifest found for fallback compilation"

    return result


# ---------------------------------------------------------------------------
# LLM lineage tracking
# ---------------------------------------------------------------------------


def _build_lineage(workspace_dir: Path) -> Dict[str, Any]:
    """Collect synthesis and healing metadata for ``lineage.json``."""
    workspace_dir = Path(workspace_dir).resolve()
    blueprint_path = workspace_dir / "blueprint.aero"
    blueprint_data: Any = None
    prompt_history: List[str] = []

    if blueprint_path.is_file():
        try:
            text = blueprint_path.read_text(encoding="utf-8")
            if text.strip().startswith(("{", "[")):
                blueprint_data = json.loads(text)
            else:
                import yaml

                blueprint_data = yaml.safe_load(text) or {}
        except Exception:
            blueprint_data = blueprint_path.read_text(encoding="utf-8")
        if isinstance(blueprint_data, dict):
            description = (
                blueprint_data.get("metadata", {}).get("description")
                or blueprint_data.get("description")
                or ""
            )
            if description:
                prompt_history.append(str(description))

    prompt_history_path = workspace_dir / ".aero" / "prompt_history.json"
    if prompt_history_path.is_file():
        try:
            extra = json.loads(prompt_history_path.read_text(encoding="utf-8"))
            if isinstance(extra, list):
                prompt_history.extend(str(p) for p in extra if p)
        except Exception:
            pass

    attempts: List[Dict[str, Any]] = []
    attempts_path = workspace_dir / ".aero" / "healing_attempts.json"
    if attempts_path.is_file():
        try:
            attempts = json.loads(attempts_path.read_text(encoding="utf-8"))
            if not isinstance(attempts, list):
                attempts = []
        except Exception:
            pass

    def _count(strategy: str) -> int:
        return sum(1 for a in attempts if a.get("strategy") == strategy)

    source_files: List[str] = []
    for path in sorted(workspace_dir.rglob("*")):
        if path.is_file() and not any(part in {".aero", "__pycache__", ".venv", "target"} for part in path.relative_to(workspace_dir).parts):
            rel = path.relative_to(workspace_dir).as_posix()
            if not rel.startswith(".") and rel != "workspace.aeroc":
                source_files.append(rel)

    return {
        "schema_version": 1,
        "generated_by": "aero-forge",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "workspace_root": str(workspace_dir),
        "blueprint": blueprint_data,
        "prompt_history": prompt_history,
        "healing": {
            "attempts_total": len(attempts),
            "ast_attempts": _count("ast"),
            "llm_attempts": _count("llm"),
            "successful": sum(1 for a in attempts if a.get("success") is True),
            "failed": sum(1 for a in attempts if a.get("success") is False),
            "attempts": attempts,
        },
        "source_files": source_files,
    }


# ---------------------------------------------------------------------------
# Delta export
# ---------------------------------------------------------------------------


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    try:
        h.update(path.read_bytes())
    except OSError:
        return ""
    return h.hexdigest()


def _compute_workspace_delta(workspace_dir: Path, base_dir: Path) -> Dict[str, Any]:
    """Return a delta manifest between *workspace_dir* and an unpacked *base_dir*."""
    excluded = {
        ".aero",
        ".git",
        ".venv",
        "__pycache__",
        ".pytest_cache",
        "target",
        ".cargo",
        "dist",
        "build",
        "node_modules",
    }

    def _collect_files(root: Path) -> Dict[str, Path]:
        files: Dict[str, Path] = {}
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(root).as_posix()
            if rel == "lineage.json":
                continue
            parts = path.relative_to(root).parts
            if any(part in excluded or part.startswith(".") for part in parts):
                continue
            files[rel] = path
        return files

    current_files = _collect_files(workspace_dir)
    base_files = _collect_files(base_dir)

    operations: List[Dict[str, Any]] = []
    changed: List[Path] = []

    for rel, cur_path in current_files.items():
        if rel not in base_files:
            operations.append({"op": "add", "path": rel, "sha256": _file_sha256(cur_path)})
            changed.append(cur_path)
        else:
            cur_hash = _file_sha256(cur_path)
            base_hash = _file_sha256(base_files[rel])
            if cur_hash != base_hash:
                operations.append({"op": "modify", "path": rel, "sha256": cur_hash})
                changed.append(cur_path)

    for rel, base_path in base_files.items():
        if rel not in current_files:
            operations.append({"op": "delete", "path": rel, "sha256": _file_sha256(base_path)})

    return {"operations": operations, "changed_files": changed, "base_dir": base_dir}


def compile_delta_aeroc(
    workspace_dir: Path,
    base_aeroc: Union[str, Path],
    output_path: Union[str, Path],
    platform_tag: Optional[str] = None,
) -> str:
    """Build a differential .aeroc package against a base bundle.

    The resulting archive contains only added/modified source files plus a
    ``delta.json`` manifest describing delete operations and base hash.
    """
    from aero_forge._native import compile_aeroc
    from aero_forge.materializer import unpack_aeroc_file

    workspace_dir = Path(workspace_dir).resolve()
    base_aeroc = Path(base_aeroc).resolve()
    output_path = Path(output_path).resolve()

    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir) / "base"
        base_dir.mkdir(parents=True, exist_ok=True)
        unpack_aeroc_file(base_aeroc, base_dir)

        delta = _compute_workspace_delta(workspace_dir, base_dir)

        sources: List[Dict[str, str]] = []

        # Mark the package as hybrid so the unpacker strips the ``src/`` prefix.
        marker = json.dumps({"layout": "hybrid", "version": 1}, indent=2)
        sources.append({
            "path": ".aeroc_hybrid",
            "content_base64": base64.b64encode(marker.encode("utf-8")).decode("ascii"),
        })

        # Add the delta manifest.
        base_hash = _file_sha256(base_aeroc)
        delta_manifest = {
            "schema_version": 1,
            "base_aeroc_hash": base_hash,
            "base_file_count": len(list(base_dir.rglob("*"))),
            "delta_version": 1,
            "operations": delta["operations"],
        }
        sources.append({
            "path": "delta.json",
            "content_base64": base64.b64encode(
                json.dumps(delta_manifest, indent=2, sort_keys=True).encode("utf-8")
            ).decode("ascii"),
        })

        # Include changed/added source files under the hybrid ``src/`` prefix.
        for path in delta["changed_files"]:
            rel = path.relative_to(workspace_dir).as_posix().lstrip("/")
            try:
                data = path.read_bytes()
            except OSError:
                continue
            sources.append({
                "path": f"src/{rel}",
                "content_base64": base64.b64encode(data).decode("ascii"),
            })

        # Embed the lineage manifest so delta bundles know how they were produced.
        lineage = _build_lineage(workspace_dir)
        sources.append({
            "path": "lineage.json",
            "content_base64": base64.b64encode(
                json.dumps(lineage, indent=2, sort_keys=True).encode("utf-8")
            ).decode("ascii"),
        })

        spec = {
            "nodes": ["workspace"],
            "edges": {},
            "instructions": [{"op": "HALT"}],
            "sources": sources,
            "flags": 0,
        }

        return compile_aeroc(json.dumps(spec), str(output_path))


def compile_hybrid_aeroc(
    workspace_dir: Path,
    output_path: Union[str, Path],
    platform_tag: Optional[str] = None,
    base_bundle: Optional[Union[str, Path]] = None,
) -> str:
    """Compile a workspace into a hybrid .aeroc container with native binaries.

    The container layout is:

        src/                          - raw source files
        artifacts/native/{os}_{arch}/ - pre-compiled .so / .dylib / .dll
        environment.lock              - pinned dependencies and toolchains
        lineage.json                  - synthesis metadata and healing counts

    If *base_bundle* is provided, a delta package is produced instead.
    Native binaries are detected from the workspace root (and ``target/release/``,
    ``dist/`` and ``build/`` subdirectories).  Source files are placed under
    ``src/`` so the unpacker can strip the prefix during extraction.
    """
    if base_bundle is not None:
        return compile_delta_aeroc(workspace_dir, base_bundle, output_path, platform_tag)

    workspace_dir = Path(workspace_dir).resolve()
    output_path = Path(output_path).resolve()
    platform_tag = platform_tag or _host_platform_tag()

    from aero_forge._native import compile_aeroc

    exclude_prefixes = {
        ".aero",
        ".git",
        ".aero_core",
        "target",
        ".venv",
        "__pycache__",
        ".pytest_cache",
        ".cargo",
        "node_modules",
    }

    native_binaries: List[Path] = []
    source_files: List[Path] = []

    # Native artifact search paths.
    native_search_dirs = {
        workspace_dir,
        workspace_dir / "target" / "release",
        workspace_dir / "dist",
        workspace_dir / "build",
    }
    for search_dir in native_search_dirs:
        if not search_dir.is_dir():
            continue
        for path in sorted(search_dir.rglob("*")):
            if _is_native_artifact(path):
                native_binaries.append(path)

    for path in sorted(workspace_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(workspace_dir)
        if any(part in exclude_prefixes for part in rel.parts):
            continue
        if rel.name.startswith("."):
            continue
        if _is_native_artifact(path):
            # Already accounted for above.
            continue
        source_files.append(path)

    sources: List[Dict[str, str]] = []

    # Layout marker so the unpacker knows to strip the ``src/`` prefix.
    marker = json.dumps({"layout": "hybrid", "version": 1}, indent=2)
    sources.append({
        "path": ".aeroc_hybrid",
        "content_base64": base64.b64encode(marker.encode("utf-8")).decode("ascii"),
    })

    # Add source files under the ``src/`` prefix.
    for path in source_files:
        rel = path.relative_to(workspace_dir).as_posix().lstrip("/")
        try:
            data = path.read_bytes()
        except OSError:
            continue
        sources.append({
            "path": f"src/{rel}",
            "content_base64": base64.b64encode(data).decode("ascii"),
        })

    # Add native binaries under the platform-specific artifact directory.
    for path in native_binaries:
        try:
            data = path.read_bytes()
        except OSError:
            continue
        sources.append({
            "path": f"artifacts/native/{platform_tag}/{path.name}",
            "content_base64": base64.b64encode(data).decode("ascii"),
        })

    # Add environment.lock at the root.
    lock = _build_environment_lock(workspace_dir, platform_tag, native_binaries)
    lock_json = json.dumps(lock, indent=2, sort_keys=True)
    sources.append({
        "path": "environment.lock",
        "content_base64": base64.b64encode(lock_json.encode("utf-8")).decode("ascii"),
    })

    # Embed LLM/healing lineage.
    lineage = _build_lineage(workspace_dir)
    sources.append({
        "path": "lineage.json",
        "content_base64": base64.b64encode(
            json.dumps(lineage, indent=2, sort_keys=True).encode("utf-8")
        ).decode("ascii"),
    })

    spec = {
        "nodes": ["workspace"],
        "edges": {},
        "instructions": [{"op": "HALT"}],
        "sources": sources,
        "flags": 0,
    }

    return compile_aeroc(json.dumps(spec), str(output_path))


# ---------------------------------------------------------------------------
# Single-command aero run execution
# ---------------------------------------------------------------------------


def _detect_entrypoint(workdir: Path) -> Optional[List[str]]:
    """Return the best-effort command to execute the unpacked workspace."""
    pyproject = workdir / "pyproject.toml"
    if pyproject.is_file():
        try:
            import tomli as tomllib
        except ImportError:
            tomllib = None  # type: ignore[assignment]
        try:
            with pyproject.open("rb") as f:
                data = tomllib.load(f)  # type: ignore[union-attr]
            scripts = data.get("project", {}).get("scripts", {})
            for name in scripts:
                return [name]
        except Exception:
            pass

    if (workdir / "main.py").is_file():
        return [sys.executable, "main.py"]

    if (workdir / "Cargo.toml").is_file() and shutil.which("cargo"):
        return ["cargo", "run", "--release"]

    runner = workdir / "aeroc-runner"
    if runner.is_file() and os.access(runner, os.X_OK):
        return [str(runner)]

    # Fallback: run the first Python file that looks like an entrypoint.
    for candidate in sorted(workdir.glob("*.py")):
        if candidate.name.startswith("test_"):
            continue
        return [sys.executable, candidate.name]

    return None


def run_aeroc_archive(
    aeroc_path: Union[str, Path],
    args: Optional[List[str]] = None,
    keep_workdir: bool = False,
    json_output: bool = False,
) -> Dict[str, Any]:
    """Unpack a ``.aeroc`` archive, prepare an isolated environment, and run it.

    The workspace is extracted into a sandbox directory managed by
    :class:`SandboxManager`.  If the container has no matching native binary
    for the host, the source tree is compiled via ``ToolchainManager`` before
    execution.  Output is streamed to the caller's stdout/stderr unless
    *json_output* is True, in which case it is captured for the returned
    payload.
    """
    from aero_forge.sandbox.manager import SandboxManager
    from aero_forge.toolchain import ToolchainManager

    aeroc_path = Path(aeroc_path).resolve()
    args = args or []

    manager = SandboxManager()
    session_id = uuid.uuid4().hex[:16]
    workdir = manager.create_session_sandbox(session_id)

    result = unpack_aeroc_with_fallback(aeroc_path, workdir, run_fallback=True)

    cmd = _detect_entrypoint(workdir)
    if cmd is None:
        if not keep_workdir:
            manager.clean_session_sandbox(session_id)
        return {
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": "No runnable entrypoint found in .aeroc archive",
            **result,
        }

    toolchain = ToolchainManager(workdir)
    toolchain.prepare_environment(cmd[0])

    try:
        if json_output:
            proc = subprocess.run(
                cmd + args,
                cwd=str(workdir),
                env=toolchain.env,
                capture_output=True,
                text=True,
                timeout=300,
            )
            returncode = proc.returncode
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
        else:
            proc = subprocess.Popen(
                cmd + args,
                cwd=str(workdir),
                env=toolchain.env,
            )
            returncode = proc.wait(timeout=300)
            stdout = ""
            stderr = ""
    except subprocess.TimeoutExpired:
        proc.kill()
        returncode = -1
        stdout = ""
        stderr = "Execution timed out after 300s"

    if not keep_workdir:
        manager.clean_session_sandbox(session_id)

    return {
        "success": returncode == 0,
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
        "command": cmd + args,
        "workdir": str(workdir),
        **result,
    }
