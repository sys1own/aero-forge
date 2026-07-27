"""Runtime native acceleration: build and load a workspace-local native_core crate."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, List, Optional


def _log(callback: Optional[Callable[[str, str, str], None]], level: str, prefix: str, message: str) -> None:
    if callback:
        callback(level, prefix, message)


def _shared_object_names() -> List[str]:
    if sys.platform == "win32":
        return ["aero_forge_native.pyd", "aero_forge_native.dll", "libaero_forge_native.dll"]
    if sys.platform == "darwin":
        return ["aero_forge_native.so", "libaero_forge_native.dylib", "libaero_forge_native.so"]
    return ["aero_forge_native.so", "libaero_forge_native.so"]


def _find_artifact(search_dir: Path) -> Optional[Path]:
    for name in _shared_object_names():
        for candidate in search_dir.rglob(name):
            if candidate.is_file():
                return candidate
    return None


def build_native_core(
    session_dir: Path,
    log_callback: Optional[Callable[[str, str, str], None]] = None,
) -> Optional[Path]:
    """Build ``crates/native_core`` and return the path to the built shared library."""
    crate_dir = Path(session_dir).resolve() / "crates" / "native_core"
    if not crate_dir.is_dir() or not (crate_dir / "Cargo.toml").is_file():
        _log(log_callback, "info", "ACCEL", "No crates/native_core found; skipping native build.")
        return None

    if not shutil.which("cargo"):
        _log(log_callback, "warning", "ACCEL", "Cargo not found; cannot build crates/native_core.")
        return None

    env = os.environ.copy()
    env.setdefault("CARGO_TARGET_DIR", str(crate_dir / "target"))

    _log(log_callback, "info", "ACCEL", f"Building native crate at {crate_dir}...")
    proc = subprocess.run(
        ["cargo", "build", "--release"],
        cwd=str(crate_dir),
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        _log(log_callback, "error", "ACCEL", f"cargo build failed (exit {proc.returncode}):")
        for line in (proc.stdout + proc.stderr).splitlines():
            _log(log_callback, "error", "ACCEL", line)
        return None

    artifact = _find_artifact(crate_dir / "target" / "release")
    if not artifact:
        _log(log_callback, "error", "ACCEL", "Native crate built but no shared library artifact found.")
        return None

    _log(log_callback, "info", "ACCEL", f"Native artifact found: {artifact}")
    return artifact


def _runtime_native_dir(session_dir: Path) -> Path:
    dest = Path(session_dir).resolve() / ".aero" / "native"
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def activate_runtime_native_acceleration(
    session_dir: Path,
    log_callback: Optional[Callable[[str, str, str], None]] = None,
) -> bool:
    """Build ``crates/native_core`` and load the resulting extension into Python."""
    artifact = build_native_core(session_dir, log_callback)
    if not artifact:
        return False

    native_dir = _runtime_native_dir(session_dir)
    extension = ".pyd" if sys.platform == "win32" else (".dylib" if sys.platform == "darwin" else ".so")
    dest = native_dir / f"aero_forge_native{extension}"

    try:
        shutil.copy2(artifact, dest)
        _log(log_callback, "info", "ACCEL", f"Copied native artifact to {dest}")
    except OSError as exc:
        _log(log_callback, "error", "ACCEL", f"Could not stage native artifact: {exc}")
        return False

    os.environ["AERO_FORGE_NATIVE_PATH"] = str(dest)
    if str(native_dir) not in sys.path:
        sys.path.insert(0, str(native_dir))

    try:
        spec = importlib.util.find_spec("aero_forge_native")
        if spec is None:
            _log(log_callback, "error", "ACCEL", "Could not import aero_forge_native from staged artifact.")
            return False

        import aero_forge.accelerator as accel

        native = importlib.import_module("aero_forge_native")
        accel._NATIVE = {
            "Hasher": getattr(native, "Hasher"),
            "GraphEngine": getattr(native, "GraphEngine"),
            "hash_bytes": getattr(native, "hash_bytes"),
            "hash_file": getattr(native, "hash_file"),
        }
        accel.Hasher = accel._NATIVE["Hasher"]
        accel.GraphEngine = accel._NATIVE["GraphEngine"]
        accel.hash_bytes = accel._NATIVE["hash_bytes"]
        accel.hash_file = accel._NATIVE["hash_file"]
        _log(log_callback, "info", "ACCEL", "Native acceleration active (crates/native_core loaded)")
        return True
    except Exception as exc:
        _log(log_callback, "error", "ACCEL", f"Native acceleration load failed: {exc}")
        return False
