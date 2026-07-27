"""Sandbox command runner with automatic toolchain resolution and fallback."""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
from typing import Dict, Optional

from aero_forge.sandbox.manager import ensure_cargo_in_path


def _python_executable() -> str:
    """Return the Python interpreter to use for ``-m`` invocations."""
    return sys.executable if shutil.which(sys.executable) else "python3"


def ensure_cargo() -> bool:
    """Ensure ``cargo`` is available in the process PATH, returning success."""
    if shutil.which("cargo"):
        return True
    ensure_cargo_in_path()
    return shutil.which("cargo") is not None


def ensure_pip(env: Dict[str, str]) -> bool:
    """Return True when a usable ``pip`` module is available."""
    python = _python_executable()
    proc = subprocess.run(
        [python, "-m", "pip", "--version"],
        capture_output=True,
        text=True,
        env=env,
    )
    return proc.returncode == 0


async def _install_maturin(env: Dict[str, str]) -> bool:
    """Attempt to install ``maturin`` with pip in the provided environment."""
    python = _python_executable()
    proc = await asyncio.create_subprocess_exec(
        python,
        "-m",
        "pip",
        "install",
        "maturin",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    await proc.communicate()
    return proc.returncode == 0


def maturin_available(env: Optional[Dict[str, str]] = None) -> bool:
    """Return True when ``maturin`` is callable either as a binary or as a module."""
    env = env or os.environ.copy()
    if shutil.which("maturin"):
        return True
    python = _python_executable()
    proc = subprocess.run(
        [python, "-m", "maturin", "--version"],
        capture_output=True,
        text=True,
        env=env,
    )
    return proc.returncode == 0


# Backwards-compatible alias for internal callers.
_maturin_available = maturin_available


def _maturin_command(args: str, env: Dict[str, str]) -> str:
    """Return the best available maturin invocation for ``args``."""
    if shutil.which("maturin"):
        return f"maturin {args}"
    python = _python_executable()
    return f"{python} -m maturin {args}"


def _cargo_available() -> bool:
    """Return True when ``cargo`` can be found."""
    return ensure_cargo()


def install_maturin_sync(env: Optional[Dict[str, str]] = None) -> bool:
    """Synchronously attempt to install ``maturin`` via pip."""
    env = env or os.environ.copy()
    python = _python_executable()
    proc = subprocess.run(
        [python, "-m", "pip", "install", "maturin"],
        capture_output=True,
        text=True,
        env=env,
    )
    return proc.returncode == 0


async def resolve_command(command: str, env: Optional[Dict[str, str]] = None) -> str:
    """Resolve a sandbox command, installing missing toolchains when possible.

    * ``cargo``-based commands require ``cargo`` in ``PATH``; if absent the function
      attempts to activate the local Rustup installation.
    * ``maturin``-based commands attempt a ``pip install maturin`` fallback when the
      executable or module is not present, then rewrite the command to use the
      module form if needed.

    Raises ``RuntimeError`` when a required toolchain cannot be resolved.
    """
    env = env or os.environ.copy()
    original = command.strip()
    lowered = original.lower()

    if lowered.startswith("cargo ") or lowered.startswith("cargo\t"):
        if not _cargo_available():
            raise RuntimeError(
                "Rust toolchain (cargo) is not available in the sandbox environment"
            )
        return original

    if lowered.startswith("maturin ") or lowered.startswith("maturin\t"):
        if not _cargo_available():
            raise RuntimeError(
                "Rust toolchain (cargo) is required for maturin builds but is not available"
            )
        if not _maturin_available(env):
            if not ensure_pip(env):
                raise RuntimeError(
                    "maturin is missing and pip is unavailable to install it"
                )
            installed = await _install_maturin(env)
            if not installed:
                raise RuntimeError("failed to install maturin via pip")
        # Rewrite the command to use the executable or module form that works.
        args = original.split(None, 1)[1]
        return _maturin_command(args, env)

    return original
