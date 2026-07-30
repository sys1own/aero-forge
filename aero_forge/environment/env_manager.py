"""Environment manager for isolated workspace virtual environments."""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("aero_forge.environment.env_manager")


def run_pip_in_workspace(
    args: List[str],
    workspace: Path,
    venv_python: str,
    env: Optional[Dict[str, str]] = None,
    log_callback: Optional[Callable[[str, str, str], None]] = None,
    check: bool = False,
) -> subprocess.CompletedProcess:
    """Run ``pip <args>`` inside *workspace* using the ``.venv`` interpreter.

    This always passes ``cwd=workspace`` so that ``pip install -e .`` resolves
    the package manifest in the session sandbox, not in the host process cwd.
    """
    log = log_callback or (lambda _level, _prefix, _msg: None)
    cmd = [venv_python, "-m", "pip", *args]
    log("info", "ENV", f"Running {' '.join(cmd)} in {workspace}")
    proc = subprocess.run(
        cmd,
        cwd=str(workspace),
        capture_output=True,
        text=True,
        env=env or os.environ.copy(),
        check=False,
    )
    if proc.returncode != 0:
        log("error", "ENV", proc.stderr.strip()[-500:])
    elif proc.stdout:
        log("info", "ENV", proc.stdout.strip()[-500:])
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(
            proc.returncode, cmd, output=proc.stdout, stderr=proc.stderr
        )
    return proc


def install_workspace_editable(
    workspace: Path,
    venv_python: str,
    env: Optional[Dict[str, str]] = None,
    log_callback: Optional[Callable[[str, str, str], None]] = None,
    extra_args: Optional[List[str]] = None,
) -> subprocess.CompletedProcess:
    """Install the workspace package in editable mode inside *workspace*.

    Detects ``pyproject.toml``, ``setup.py``, or ``setup.cfg`` and runs
    ``pip install -e . --no-deps`` from *workspace*.
    """
    manifest = None
    for name in ("pyproject.toml", "setup.py", "setup.cfg"):
        if (workspace / name).is_file():
            manifest = name
            break
    if not manifest:
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="",
            stderr="No packaging manifest found; skipping editable install.",
        )
    args = ["install", "-e", "."]
    if extra_args:
        args.extend(extra_args)
    else:
        args.append("--no-deps")
    log = log_callback or (lambda _level, _prefix, _msg: None)
    log("info", "ENV", f"Installing workspace package in editable mode from {manifest}...")
    return run_pip_in_workspace(args, workspace, venv_python, env=env, log_callback=log_callback)
