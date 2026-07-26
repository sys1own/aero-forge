"""Resilient Cargo execution and workspace configuration for aero-forge."""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

logger = logging.getLogger("aero_forge.cargo")

CARGO_CONFIG_TOML = """# aero-forge: resilient Cargo network configuration
[net]
retry = 5
git-fetch-with-cli = true

[http]
timeout = 60
multiplexing = false
"""

def _cargo_bin_dirs() -> List[Path]:
    """Return candidate Rustup cargo bin directories that exist on disk."""
    candidates = [Path.home() / ".cargo" / "bin", Path("/root/.cargo/bin")]
    return [p for p in candidates if p.is_dir()]


def _env_with_cargo(env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Return an environment dict with ``~/.cargo/bin`` on PATH if cargo is missing."""
    merged = dict(os.environ)
    if env:
        merged.update(env)
    if shutil.which("cargo", path=merged.get("PATH")):
        return merged
    extra = [str(p) for p in _cargo_bin_dirs()]
    if extra:
        merged["PATH"] = os.pathsep.join([*extra, merged.get("PATH", "")])
        logger.debug("Prepending Rust toolchain directories to PATH: %s", extra)
    return merged


# Network / IO failure patterns that warrant a retry or offline fallback.
_NETWORK_FAILURE_PATTERNS = [
    re.compile(r"failed to download", re.I),
    re.compile(r"failed to fetch", re.I),
    re.compile(r"unable to get packages", re.I),
    re.compile(r"timeout", re.I),
    re.compile(r"connection\s+.*(?:reset|refused|timed out|abort|closed)", re.I),
    re.compile(r"io\s+error", re.I),
    re.compile(r"no\s+such\s+host", re.I),
    re.compile(r"could\s+not\s+resolve", re.I),
    re.compile(r"network\s+.*unreachable", re.I),
    re.compile(r"ssl\s+.*error", re.I),
    re.compile(r"curl\s+error", re.I),
]


def write_cargo_config(crate_root: Union[str, Path]) -> Path:
    """Write ``.cargo/config.toml`` with network retry/time-out defaults."""
    root = Path(crate_root).resolve()
    cargo_dir = root / ".cargo"
    cargo_dir.mkdir(parents=True, exist_ok=True)
    config_path = cargo_dir / "config.toml"
    config_path.write_text(CARGO_CONFIG_TOML, encoding="utf-8")
    logger.debug("Wrote Cargo config to %s", config_path)
    return config_path


def _looks_like_network_failure(output: str) -> bool:
    return any(pattern.search(output) for pattern in _NETWORK_FAILURE_PATTERNS)


def _is_offline_viable(command: Sequence[str]) -> bool:
    """Return ``True`` if ``--offline`` can be appended to the cargo command."""
    return "--offline" not in command


def run_cargo(
    command: Sequence[Union[str, os.PathLike[str]]],
    *,
    cwd: Union[str, Path],
    env: Optional[Dict[str, str]] = None,
    retries: int = 3,
    timeout: float = 600,
    allow_offline: bool = True,
    retry_delay: float = 2.0,
) -> subprocess.CompletedProcess[str]:
    """Run a Cargo sub-command with retries and optional offline fallback.

    A ``.cargo/config.toml`` is written to *cwd* before invoking Cargo so the
    build inherits resilient network defaults.  If the command fails with a
    network/IO pattern, it is retried up to *retries* times.  When retries
    are exhausted, an ``--offline`` attempt is made if the local Cargo cache
    looks like it already contains the required crates.
    """
    workdir = Path(cwd).resolve()
    write_cargo_config(workdir)

    base_env = _env_with_cargo(env)

    str_command = [str(arg) for arg in command]
    if not str_command or str_command[0] != "cargo":
        str_command = ["cargo", *str_command]

    last_error: Optional[str] = None
    for attempt in range(1, retries + 1):
        logger.info("Running %s (attempt %d/%d) in %s", " ".join(str_command), attempt, retries, workdir)
        try:
            result = subprocess.run(
                str_command,
                cwd=workdir,
                env=base_env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            last_error = f"Cargo command timed out after {exc.timeout}s"
            logger.warning("%s; retrying...", last_error)
            time.sleep(retry_delay)
            continue

        combined = f"{result.stdout}\n{result.stderr}".strip()
        if result.returncode == 0 or not _looks_like_network_failure(combined):
            return result

        last_error = combined
        logger.warning("Cargo failed with network/IO symptoms on attempt %d; retrying...", attempt)
        if attempt < retries:
            time.sleep(retry_delay)

    # Final attempt: try offline if the cache contains pre-downloaded crates.
    if allow_offline and _is_offline_viable(str_command):
        cache_registry = Path.home() / ".cargo" / "registry"
        if cache_registry.is_dir() and any(cache_registry.rglob("*.crate")):
            offline_command = [*str_command, "--offline"]
            logger.info("Retrying with --offline: %s", " ".join(offline_command))
            offline_result = subprocess.run(
                offline_command,
                cwd=workdir,
                env=base_env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            offline_combined = f"{offline_result.stdout}\n{offline_result.stderr}".strip()
            if offline_result.returncode == 0:
                return offline_result
            if not _looks_like_network_failure(offline_combined):
                # Offline attempt failed for a non-network reason; preserve that log.
                return offline_result
            logger.warning("Offline fallback also failed: %s", offline_combined[:500])

    assert last_error is not None
    logger.error("Cargo command failed after %d attempts: %s", retries, last_error[:500])
    return subprocess.CompletedProcess(
        args=str_command,
        returncode=1,
        stdout="",
        stderr=last_error,
    )


def cargo_check(
    crate_root: Union[str, Path],
    *,
    env: Optional[Dict[str, str]] = None,
    timeout: float = 600,
) -> subprocess.CompletedProcess[str]:
    """Run ``cargo check`` with resilience defaults."""
    return run_cargo(["cargo", "check"], cwd=crate_root, env=env, timeout=timeout)


def cargo_build(
    crate_root: Union[str, Path],
    *,
    release: bool = True,
    target: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    timeout: float = 600,
) -> subprocess.CompletedProcess[str]:
    """Run ``cargo build`` with resilience defaults."""
    cmd = ["cargo", "build"]
    if release:
        cmd.append("--release")
    if target:
        cmd.extend(["--target", target])
    return run_cargo(cmd, cwd=crate_root, env=env, timeout=timeout)


def maturin_build(
    crate_root: Union[str, Path],
    *,
    env: Optional[Dict[str, str]] = None,
    timeout: float = 600,
) -> subprocess.CompletedProcess[str]:
    """Run ``maturin build --release`` with resilience defaults."""
    return run_cargo(
        ["maturin", "build", "--release"],
        cwd=crate_root,
        env=env,
        timeout=timeout,
    )
