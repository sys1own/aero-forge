"""Resilient Cargo execution and workspace configuration for aero-forge."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

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


def _toolchain_bin_dirs() -> List[Path]:
    """Return candidate per-user toolchain bin directories that exist on disk."""
    candidates = [Path.home() / ".cargo" / "bin", Path("/root/.cargo/bin"), Path("/usr/local/cargo/bin")]
    return [p for p in candidates if p.is_dir()]


def _is_rustup_cargo(cargo_path: Path) -> bool:
    """Detect whether ``cargo`` is the Rustup shim installed under ``$CARGO_HOME/bin``."""
    try:
        return cargo_path.name == "cargo" and cargo_path.parent.name == "bin"
    except Exception:
        return False


def _env_with_cargo(env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Return an environment dict with Rust toolchain on PATH and rustup env set."""
    merged = dict(os.environ)
    if env:
        merged.update(env)
    if not shutil.which("cargo", path=merged.get("PATH")):
        extra = [str(p) for p in _toolchain_bin_dirs()]
        if extra:
            merged["PATH"] = os.pathsep.join([*extra, merged.get("PATH", "")])
            logger.debug("Prepending Rust toolchain directories to PATH: %s", extra)

    cargo = shutil.which("cargo", path=merged.get("PATH"))
    if cargo and _is_rustup_cargo(Path(cargo)):
        cargo_home = Path(cargo).parent.parent
        if not merged.get("CARGO_HOME"):
            merged["CARGO_HOME"] = str(cargo_home)
        if not merged.get("RUSTUP_HOME"):
            rustup_home = cargo_home.parent / ".rustup"
            if rustup_home.is_dir():
                merged["RUSTUP_HOME"] = str(rustup_home)
    return merged


def _bootstrap_rust(env: Dict[str, str]) -> Dict[str, str]:
    """Install a minimal stable Rust toolchain into an isolated directory."""
    if os.environ.get("AERO_FORGE_NO_RUST_BOOTSTRAP"):
        return env
    rust_dir = Path.home() / ".aero_forge" / "toolchains" / "rust"
    rust_dir.mkdir(parents=True, exist_ok=True)
    cargo_home = rust_dir / "cargo"
    rustup_home = rust_dir / "rustup"
    bin_dir = cargo_home / "bin"

    env = dict(env)
    env["CARGO_HOME"] = str(cargo_home)
    env["RUSTUP_HOME"] = str(rustup_home)
    env["PATH"] = os.pathsep.join([str(bin_dir), env.get("PATH", "")])

    if (bin_dir / "cargo").is_file():
        logger.debug("Using previously bootstrapped Rust toolchain at %s", rust_dir)
        return env

    logger.warning("Rust toolchain not found; bootstrapping rustup into %s", rust_dir)
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".sh", delete=False) as f:
        script_path = Path(f.name)
        try:
            subprocess.run(
                ["curl", "--proto", "=https", "--tlsv1.2", "-sSf", "https://sh.rustup.rs"],
                stdout=f,
                check=True,
                timeout=120,
            )
        except Exception:
            f.close()
            import urllib.request
            with urllib.request.urlopen("https://sh.rustup.rs", timeout=120) as resp:
                script_path.write_bytes(resp.read())

    install_env = dict(env)
    install_env.pop("CARGO_TARGET_DIR", None)
    result = subprocess.run(
        ["sh", str(script_path), "-y", "--default-toolchain", "stable", "--profile", "minimal"],
        env=install_env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    script_path.unlink(missing_ok=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Rust toolchain bootstrap failed (exit {result.returncode}):\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    if not (bin_dir / "cargo").is_file():
        subprocess.run(
            [str(bin_dir / "rustup"), "toolchain", "install", "stable"],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
    return env


def ensure_rust_toolchain(env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Return an env dict with cargo/rustc available, bootstrapping if necessary."""
    env = _env_with_cargo(env)
    if shutil.which("cargo", path=env.get("PATH")) and shutil.which("rustc", path=env.get("PATH")):
        return env
    if os.environ.get("AERO_FORGE_NO_RUST_BOOTSTRAP"):
        return env
    try:
        return _bootstrap_rust(env)
    except Exception:
        logger.exception("Rust toolchain bootstrap failed")
        return env


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

    base_env = ensure_rust_toolchain(env)

    str_command = [str(arg) for arg in command]
    if not str_command:
        str_command = ["cargo", "build"]
    elif str_command[0] not in ("cargo", "maturin"):
        str_command = ["cargo", *str_command]

    def _run(cmd: List[str]) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                cmd,
                cwd=workdir,
                env=base_env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except FileNotFoundError as exc:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=127,
                stdout="",
                stderr=f"Cargo command not found: {exc}",
            )

    last_error: Optional[str] = None
    for attempt in range(1, retries + 1):
        logger.info("Running %s (attempt %d/%d) in %s", " ".join(str_command), attempt, retries, workdir)
        try:
            result = _run(str_command)
        except subprocess.TimeoutExpired as exc:
            last_error = f"Cargo command timed out after {exc.timeout}s"
            logger.warning("%s; retrying...", last_error)
            if attempt < retries:
                time.sleep(retry_delay)
            continue

        if result.returncode == 0:
            return result

        combined = f"{result.stdout}\n{result.stderr}".strip()
        if not _looks_like_network_failure(combined):
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
            offline_result = _run(offline_command)
            offline_combined = f"{offline_result.stdout}\n{offline_result.stderr}".strip()
            if offline_result.returncode == 0:
                return offline_result
            if not _looks_like_network_failure(offline_combined):
                return offline_result
            logger.warning("Offline fallback also failed: %s", offline_combined[:500])

    # Emit verbose diagnostics for the failing command so the UI can surface
    # the exact compiler output instead of a generic "use --verbose" message.
    if str_command and str_command[0] in ("cargo", "maturin") and last_error is not None:
        verbose_command = [*str_command, "-v"]
        logger.info("Re-running with verbose output: %s", " ".join(verbose_command))
        verbose_result = _run(verbose_command)
        if verbose_result.returncode != 0:
            return verbose_result

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


def _crate_name(crate_root: Path) -> str:
    """Read the crate name from ``Cargo.toml`` (``[lib]`` name, then ``[package]``)."""
    cargo_toml = crate_root / "Cargo.toml"
    if cargo_toml.is_file():
        text = cargo_toml.read_text(encoding="utf-8")
        lib_match = re.search(r"\[lib\][^\[]*?name\s*=\s*\"([^\"]+)\"", text, re.DOTALL)
        if lib_match:
            return lib_match.group(1)
        pkg_match = re.search(r"\[package\][^\[]*?name\s*=\s*\"([^\"]+)\"", text, re.DOTALL)
        if pkg_match:
            return pkg_match.group(1)
    return crate_root.name


def _discover_pyo3_functions(crate_root: Path) -> List[Tuple[str, str, str]]:
    """Scan ``src/**/*.rs`` for ``#[pyfunction(name = ...)]`` declarations.

    Returns a list of ``(module_path, rust_name, python_name)`` tuples.  The
    module path is the Rust path relative to ``src`` (e.g. ``ops::matmul``).
    """
    src = crate_root / "src"
    functions: List[Tuple[str, str, str]] = []
    if not src.is_dir():
        return functions
    for rs in src.rglob("*.rs"):
        if rs.name == "lib.rs" or rs.name == "mod.rs":
            continue
        rel = rs.relative_to(src)
        mod_path = "::".join(rel.with_suffix("").parts)
        text = rs.read_text(encoding="utf-8")
        # Find every #[pyfunction(...)] attribute and the function it decorates.
        for m in re.finditer(r"#\[pyfunction(?:\s*\((.*?)\))?\]", text, re.DOTALL):
            attr_body = m.group(1) or ""
            py_name = ""
            name_m = re.search(r'name\s*=\s*"([^"]+)"', attr_body)
            if name_m:
                py_name = name_m.group(1)
            fn_m = re.search(r"(?:pub\s+)?fn\s+(\w+)\s*\(", text[m.end():])
            if not fn_m:
                continue
            rust_name = fn_m.group(1)
            if not py_name:
                py_name = rust_name
            functions.append((mod_path, rust_name, py_name))
    return functions


def _regenerate_lib_rs(crate_root: Path, crate_name: str, functions: List[Tuple[str, str, str]]) -> Path:
    """Rewrite ``src/lib.rs`` so it references the real exported function symbols."""
    lib_rs = crate_root / "src" / "lib.rs"
    lib_rs.parent.mkdir(parents=True, exist_ok=True)
    top_mods = sorted({path.split("::")[0] for path, _, _ in functions if path})
    mod_lines = [f"mod {m};" for m in top_mods]
    use_lines = [f"use {path}::{rust_name};" for path, rust_name, _ in functions]
    reg_lines = [f"    m.add_wrapped(wrap_pyfunction!({rust_name}));?" for _, rust_name, _ in functions]
    source = "use pyo3::prelude::*;\n\n"
    if mod_lines:
        source += "\n".join(mod_lines) + "\n\n"
    if use_lines:
        source += "\n".join(use_lines) + "\n\n"
    source += f"""#[pymodule]
fn {crate_name}(_py: Python, m: &PyModule) -> PyResult<()> {{
"""
    if reg_lines:
        source += "\n".join(reg_lines) + "\n"
    source += "    Ok(())\n}\n"
    lib_rs.write_text(source, encoding="utf-8")
    return lib_rs


def cargo_check_and_repair(
    crate_root: Union[str, Path],
    *,
    max_retries: int = 2,
    env: Optional[Dict[str, str]] = None,
    timeout: float = 600,
) -> subprocess.CompletedProcess[str]:
    """Run ``cargo check --message-format=json`` and repair ``src/lib.rs`` for E0432/E0425.

    If an unresolved import or symbol error is detected, ``src/lib.rs`` is
    regenerated from the actual ``#[pyfunction]`` functions found in the crate
    source, and ``cargo check`` is retried up to *max_retries* additional times.
    """
    root = Path(crate_root).resolve()
    lib_rs = root / "src" / "lib.rs"
    last_result: Optional[subprocess.CompletedProcess[str]] = None
    for attempt in range(max_retries + 1):
        result = run_cargo(
            ["cargo", "check", "--message-format=json"],
            cwd=root,
            env=env,
            timeout=timeout,
            retries=1,
        )
        last_result = result
        if result.returncode == 0:
            return result

        messages: List[Dict[str, Any]] = []
        for line in (result.stdout + "\n" + result.stderr).splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("reason") == "compiler-message":
                msg = obj.get("message", obj)
                if msg.get("level") == "error":
                    messages.append(msg)
            elif obj.get("level") == "error":
                messages.append(obj)

        has_unresolved = any(
            _resolve_error_code(msg) in ("E0432", "E0425")
            for msg in messages
        )
        if not has_unresolved or not lib_rs.is_file():
            return result

        functions = _discover_pyo3_functions(root)
        if not functions:
            return result
        crate_name = _crate_name(root)
        _regenerate_lib_rs(root, crate_name, functions)
        logger.info("Repaired src/lib.rs on attempt %d (E0432/E0425) in %s", attempt, root)

    assert last_result is not None
    return last_result


def _resolve_error_code(msg: Dict[str, Any]) -> Optional[str]:
    code = msg.get("code")
    if isinstance(code, dict):
        return code.get("code")
    if isinstance(code, str):
        return code
    return None
