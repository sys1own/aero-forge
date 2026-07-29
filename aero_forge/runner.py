"""Sandbox command runner with automatic toolchain resolution and fallback."""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from aero_forge.scheduler.wavefront import Task, WavefrontScheduler
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


def run_wavefront_tasks(
    commands: List[str],
    sandbox_dir: Path,
    dependencies: Optional[Dict[int, List[int]]] = None,
    env: Optional[Dict[str, str]] = None,
    log_callback: Optional[Callable[[str, str, str], None]] = None,
) -> List[Dict[str, Any]]:
    """Execute a set of build/test commands through the WavefrontScheduler.

    ``dependencies`` maps command index -> list of dependency indices. Commands
    without dependencies run in wave $W_0$; each wave completes before the
    next begins. Logs are streamed via ``log_callback`` as ``[WAVE]`` messages.
    """
    env = env or os.environ.copy()
    dependencies = dependencies or {}
    scheduler = WavefrontScheduler(log_callback=log_callback)

    tasks = {str(i): Task(name=f"task-{i}", command=cmd, cwd=sandbox_dir, env=env) for i, cmd in enumerate(commands)}
    adj: Dict[str, List[str]] = {}
    for i, deps in dependencies.items():
        adj[str(i)] = [str(d) for d in deps]
    for key in tasks:
        adj.setdefault(key, [])

    return scheduler.execute_sync(tasks, adj)


async def run_wavefront_tasks_async(
    commands: List[str],
    sandbox_dir: Path,
    dependencies: Optional[Dict[int, List[int]]] = None,
    env: Optional[Dict[str, str]] = None,
    log_callback: Optional[Callable[[str, str, str], None]] = None,
) -> List[Dict[str, Any]]:
    """Async variant of ``run_wavefront_tasks``."""
    env = env or os.environ.copy()
    dependencies = dependencies or {}
    scheduler = WavefrontScheduler(log_callback=log_callback)
    tasks = {str(i): Task(name=f"task-{i}", command=cmd, cwd=sandbox_dir, env=env) for i, cmd in enumerate(commands)}
    adj: Dict[str, List[str]] = {}
    for i, deps in dependencies.items():
        adj[str(i)] = [str(d) for d in deps]
    for key in tasks:
        adj.setdefault(key, [])
    return await scheduler.execute(tasks, adj)


async def _execute_async(
    command: str,
    sandbox_dir: Path,
    env: Optional[Dict[str, str]],
    timeout: int,
) -> Tuple[Dict[str, Any], List[Tuple[str, str, str]]]:
    """Resolve ``command`` and run it through the wavefront scheduler."""
    resolved, command_env, logs = await resolve_command(command, env, sandbox_dir)
    results = await run_wavefront_tasks_async(
        [resolved],
        sandbox_dir=sandbox_dir,
        env=command_env,
        log_callback=None,
    )
    return results[0] if results else {"returncode": -1, "stdout": "", "stderr": "No results"}, logs


def execute(
    command: str,
    sandbox_dir: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    timeout: int = 120,
) -> Dict[str, Any]:
    """Synchronously resolve and execute ``command`` inside a prepared sandbox."""
    sandbox_dir = Path(sandbox_dir or os.environ.get("AERO_FORGE_SESSION_DIR", "."))
    env = env or os.environ.copy()
    result, _logs = asyncio.run(_execute_async(command, sandbox_dir, env, timeout))
    return result


async def execute_async(
    command: str,
    sandbox_dir: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    timeout: int = 120,
) -> Dict[str, Any]:
    """Asynchronously resolve and execute ``command`` inside a prepared sandbox."""
    sandbox_dir = Path(sandbox_dir or os.environ.get("AERO_FORGE_SESSION_DIR", "."))
    env = env or os.environ.copy()
    result, _logs = await _execute_async(command, sandbox_dir, env, timeout)
    return result
