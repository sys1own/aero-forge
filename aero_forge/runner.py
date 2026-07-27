"""Sandbox command runner with automatic toolchain resolution and fallback."""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from aero_forge.toolchain import ToolchainManager


async def resolve_command(
    command: str,
    env: Optional[Dict[str, str]] = None,
    sandbox_dir: Optional[Path] = None,
) -> Tuple[str, Dict[str, str], List[Tuple[str, str, str]]]:
    """Resolve ``command`` for a sandbox and prepare its environment.

    Returns a tuple of ``(resolved_command, environment_dict, log_messages)``.
    Each log message is ``(level, prefix, text)``.
    """
    env = env or os.environ.copy()
    sandbox_dir = sandbox_dir or Path(os.environ.get("AERO_FORGE_SESSION_DIR", "."))
    logs: List[Tuple[str, str, str]] = []

    def log_callback(level: str, prefix: str, message: str) -> None:
        logs.append((level, prefix, message))

    manager = ToolchainManager(sandbox_dir, log_callback=log_callback)
    manager.env = env
    loop = asyncio.get_event_loop()

    await loop.run_in_executor(None, manager.prepare_environment, command)
    resolved = await loop.run_in_executor(None, manager.resolve_command, command)

    return resolved, manager.env, logs


def ensure_cargo() -> bool:
    """Backward-compatible helper to ensure ``cargo`` is on PATH."""
    manager = ToolchainManager(Path("."))
    return manager.ensure_rust_toolchain()


def _maturin_module_ok(env: Dict[str, str]) -> bool:
    python = sys.executable if shutil.which(sys.executable) else "python3"
    proc = subprocess.run(
        [python, "-m", "maturin", "--version"],
        capture_output=True,
        text=True,
        env=env,
    )
    return proc.returncode == 0


def maturin_available(env: Optional[Dict[str, str]] = None) -> bool:
    """Backward-compatible helper; check whether ``maturin`` is available."""
    env = env or os.environ.copy()
    manager = ToolchainManager(Path("."))
    manager.env = env
    return shutil.which("maturin", path=manager.env.get("PATH", "")) is not None or _maturin_module_ok(env)


def install_maturin_sync(env: Optional[Dict[str, str]] = None) -> bool:
    """Backward-compatible helper; attempt to install maturin into ``.venv``."""
    env = env or os.environ.copy()
    manager = ToolchainManager(Path("."))
    manager.env = env
    try:
        manager.ensure_python_packages(["maturin"])
        return manager.maturin_available()
    except Exception:
        return False
