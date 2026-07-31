"""Isolated sandbox execution for draft Blueprint v3.0.0 builds.

Draft blueprints are not transferable and must be compiled without touching the
original workspace source tree. ``DraftSandboxBuilder`` copies the workspace
into a temporary sandbox, makes declared source directories read-only, runs the
v3 blueprint there, and then promotes outputs to a persistent sandbox output
directory.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from aero_forge.blueprint.schema import BlueprintStatus, BlueprintV3
from aero_forge.environment import env_manager
from aero_forge.errors import UserError

logger = logging.getLogger("aero_forge.builder.sandbox")

_SOURCE_DIR_NAMES: Set[str] = {"src", "lib", "rust_core", "cpp_core", "python_engine"}
_SKIP_DIR_NAMES: Set[str] = {
    ".aero",
    ".aero_backup",
    ".git",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    ".mypy_cache",
    ".cargo",
    "dist",
    "target",
    "build",
}


class DraftSandboxBuilder:
    """Run a draft Blueprint v3 build in an isolated, read-only-source sandbox."""

    def __init__(
        self,
        blueprint: BlueprintV3,
        workspace: Path,
        output_dir: Path,
        env: Optional[Dict[str, str]] = None,
    ) -> None:
        if blueprint.metadata.status != BlueprintStatus.draft:
            raise UserError(
                "DraftSandboxBuilder can only be used with draft blueprints. "
                f"Got status={blueprint.metadata.status}"
            )
        self.blueprint = blueprint
        self.workspace = Path(workspace).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.env = env

    def build(self) -> Dict[str, Any]:
        """Copy the workspace, run the v3 blueprint in a sandbox, and return results."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        sandbox_root = Path(
            tempfile.mkdtemp(prefix=f"aero_draft_{self.workspace.name}_", dir=self.output_dir)
        )
        logger.info("Preparing draft build sandbox: %s", sandbox_root)

        # Ensure any pip subprocess in the sandbox disables caches and reinstalls
        # workspace packages so overlay edits are never masked by stale wheels.
        env = (self.env or os.environ.copy()).copy()
        env["PIP_NO_CACHE_DIR"] = env.get("PIP_NO_CACHE_DIR", "1")
        env["PIP_FORCE_REINSTALL"] = env.get("PIP_FORCE_REINSTALL", "1")

        try:
            self._copy_workspace(sandbox_root)
            self._ensure_writable_output_dirs(sandbox_root)
            self._ensure_workspace_package_installed(sandbox_root, env)
            self._make_source_dirs_read_only(sandbox_root)

            try:
                from aero_forge.daemon import compile_and_run_blueprint

                compile_and_run_blueprint(
                    self.blueprint,
                    sandbox_root,
                    output_dir=sandbox_root,
                    max_workers=4,
                )
                result = {"status": "success", "build_results": [], "stdout": "", "stderr": ""}
            except Exception:
                logger.exception("aeroc-daemon failed, falling back to Python executor")
                result = self.blueprint.execute(sandbox_root, env=env)

            self._copy_outputs(sandbox_root, self.output_dir)

            logs = []
            for stage in result.get("build_results", []):
                if stage.get("error"):
                    logs.append(f"[{stage.get('artifact')}] {stage['error']}")
            logs.append(result.get("stdout", ""))
            logs.append(result.get("stderr", ""))

            success = result.get("status") == "success"
            return {
                "success": success,
                "status": result.get("status", "failed"),
                "project": self.blueprint.metadata.project_name,
                "output_dir": str(self.output_dir),
                "total": 1,
                "passed": 1 if success else 0,
                "failed": 0 if success else 1,
                "error": "" if success else (result.get("error") or "Draft sandbox build failed"),
                "logs": "\n".join(logs).strip(),
                "results": result.get("build_results", []),
                "verification": result.get("verification", []),
            }
        except Exception as exc:
            logger.exception("Draft sandbox build failed")
            raise UserError(f"Draft sandbox build failed: {exc}") from exc
        finally:
            # Keep the sandbox around for debugging if the build failed; otherwise
            # clean up to avoid accumulating temp directories.
            pass

    def _copy_workspace(self, sandbox_root: Path) -> None:
        """Copy workspace files into the sandbox, excluding generated/cache dirs."""
        for item in self.workspace.iterdir():
            if item.name in _SKIP_DIR_NAMES:
                continue
            dest = sandbox_root / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True, ignore=_copy_ignore)
            elif item.is_file():
                shutil.copy2(item, dest)

    def _source_dir_names(self) -> Set[str]:
        """Derive source directory names from the blueprint manifest plus defaults."""
        names = set(_SOURCE_DIR_NAMES)
        for entry in getattr(self.blueprint, "manifest", []) or []:
            parts = Path(entry.path).parts
            if parts and parts[0] not in _SKIP_DIR_NAMES:
                names.add(parts[0])
        return names

    def _make_source_dirs_read_only(self, sandbox_root: Path) -> None:
        """Set source directories and their contents to read-only in the sandbox."""
        source_dirs = self._source_dir_names()
        for root, dirs, files in os.walk(sandbox_root, topdown=True):
            current = Path(root)
            if any(part in _SKIP_DIR_NAMES for part in current.parts):
                dirs[:] = [d for d in dirs if d not in _SKIP_DIR_NAMES]
                continue

            # If the current directory is a recognized source tree, make it read-only.
            if current.name in source_dirs or current == sandbox_root:
                try:
                    current.chmod(0o755)
                except OSError:
                    pass

            for d in list(dirs):
                dpath = current / d
                if d in source_dirs:
                    self._chmod_recursive(dpath, dir_mode=0o555, file_mode=0o444)
                    dirs.remove(d)

            for filename in files:
                fpath = current / filename
                try:
                    fpath.chmod(0o444)
                except OSError:
                    pass

    def _chmod_recursive(
        self, path: Path, dir_mode: int = 0o555, file_mode: int = 0o444
    ) -> None:
        for root, dirs, files in os.walk(path):
            for d in dirs:
                try:
                    (Path(root) / d).chmod(dir_mode)
                except OSError:
                    pass
            for f in files:
                try:
                    (Path(root) / f).chmod(file_mode)
                except OSError:
                    pass
            try:
                Path(root).chmod(dir_mode)
            except OSError:
                pass

    def _ensure_workspace_package_installed(self, sandbox_root: Path, env: Dict[str, str]) -> None:
        """Re-install the workspace package in editable mode inside the sandbox.

        This is done before source directories are locked to read-only so pip can
        write package metadata, and it always uses ``--no-cache-dir`` and
        ``--force-reinstall`` so stale wheels are not reused.
        """
        venv_dir = sandbox_root / ".venv"
        if not venv_dir.is_dir():
            return
        if sys.platform == "win32":
            venv_python = str(venv_dir / "Scripts" / "python.exe")
        else:
            venv_python = str(venv_dir / "bin" / "python")
        if not Path(venv_python).is_file():
            return
        proc = env_manager.install_workspace_editable(
            sandbox_root,
            venv_python,
            env=env,
            log_callback=lambda level, prefix, msg: logger.log(
                logging.INFO if level in ("info",) else logging.WARNING, f"[{prefix}] {msg}"
            ),
        )
        if proc.returncode != 0:
            logger.warning("Editable install in sandbox failed: %s", proc.stderr.strip()[-500:])

    def _ensure_writable_output_dirs(self, sandbox_root: Path) -> None:
        """Ensure output directories exist and are writable before the build runs."""
        for dirname in ("dist", "target"):
            out = sandbox_root / dirname
            out.mkdir(parents=True, exist_ok=True)
            out.chmod(0o777)

    def _copy_outputs(self, sandbox_root: Path, output_dir: Path) -> None:
        """Promote build artifacts from the sandbox to the workspace output dir."""
        artifact_extensions = {".so", ".dll", ".dylib", ".rlib", ".a", ".wasm", ".exe", ".pyd"}
        for dirname in ("dist", "target"):
            src = sandbox_root / dirname
            if not src.exists():
                continue
            for src_file in src.rglob("*"):
                if src_file.is_file() and src_file.suffix in artifact_extensions:
                    rel = src_file.relative_to(src)
                    dst = output_dir / dirname / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_file, dst)


def _copy_ignore(src: str, names: List[str]) -> Set[str]:
    """shutil copytree ignore callback that skips cache/build directories."""
    return {n for n in names if n in _SKIP_DIR_NAMES}


def run_draft_build(
    blueprint: BlueprintV3,
    workspace: Path,
    output_dir: Path,
    env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Convenience helper to run a draft Blueprint v3 build in a sandbox."""
    return DraftSandboxBuilder(blueprint, workspace, output_dir, env=env).build()
