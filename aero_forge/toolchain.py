"""Universal toolchain manager: detect, install, and configure build tools."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

from aero_forge.sandbox.manager import ensure_cargo_in_path


class ToolchainManager:
    """Prepare a sandbox environment for Python, Rust, and Node commands."""

    def __init__(
        self,
        sandbox_dir: Path,
        log_callback: Optional[Callable[[str, str, str], None]] = None,
    ) -> None:
        self.sandbox_dir = Path(sandbox_dir).resolve()
        self.log_callback = log_callback or (lambda _level, _prefix, _msg: None)
        self.env: Dict[str, str] = os.environ.copy()
        self._prepared = False

    def _log(self, level: str, prefix: str, message: str) -> None:
        self.log_callback(level, prefix, message)

    def _venv_path(self) -> Path:
        return self.sandbox_dir / ".venv"

    def _venv_bin(self) -> Path:
        if sys.platform == "win32":
            return self._venv_path() / "Scripts"
        return self._venv_path() / "bin"

    def _venv_python(self) -> str:
        if sys.platform == "win32":
            return str(self._venv_bin() / "python.exe")
        return str(self._venv_bin() / "python")

    def ensure_virtualenv(self) -> Dict[str, str]:
        """Ensure a ``.venv`` exists in ``sandbox_dir`` and return updated env vars."""
        venv_path = self._venv_path()
        python_exe = "python.exe" if sys.platform == "win32" else "bin/python"
        if not (venv_path / python_exe).is_file():
            self._log("info", "ENV", f"Creating Python virtual environment (.venv) in {self.sandbox_dir}...")
            python = sys.executable if shutil.which(sys.executable) else "python3"
            subprocess.run(
                [python, "-m", "venv", str(venv_path)],
                capture_output=True,
                text=True,
                check=False,
            )
        else:
            self._log("info", "ENV", "Using existing Python virtual environment (.venv).")

        venv_bin = self._venv_bin()
        self.env["VIRTUAL_ENV"] = str(venv_path)
        self.env["PATH"] = f"{venv_bin}{os.pathsep}{self.env.get('PATH', '')}"
        self._log("info", "ENV", "Injected VIRTUAL_ENV and updated PATH for command execution.")
        return {"VIRTUAL_ENV": str(venv_path), "PATH": self.env["PATH"]}

    def _run_venv_python(
        self,
        args: List[str],
        check: bool = True,
        capture: bool = True,
    ) -> subprocess.CompletedProcess:
        """Run a command using the virtual environment's Python interpreter."""
        cmd = [self._venv_python(), *args]
        return subprocess.run(
            cmd,
            capture_output=capture,
            text=True,
            env=self.env,
            check=check,
        )

    def ensure_python_packages(self, required_packages: List[str]) -> None:
        """Install ``required_packages`` into ``.venv`` if missing."""
        self.ensure_virtualenv()
        for package in required_packages:
            try:
                self._run_venv_python(["-m", "pip", "show", package], check=True)
                self._log("info", "TOOLCHAIN", f"{package} detected (ok) in .venv")
            except (subprocess.CalledProcessError, FileNotFoundError):
                self._log("info", "TOOLCHAIN", f"{package} missing in .venv -> Auto-installing `pip install {package}`...")
                self._run_venv_python(
                    ["-m", "pip", "install", "--quiet", package],
                    check=True,
                )
                self._log("info", "TOOLCHAIN", f"{package} installed (ok) in .venv")

    def ensure_rust_toolchain(self) -> bool:
        """Verify ``cargo`` and ``rustc`` are available; return success."""
        if shutil.which("cargo"):
            self._log("info", "TOOLCHAIN", "Cargo / Rustc detected (ok)")
            return True
        self._log("info", "TOOLCHAIN", "Cargo not found; searching local rustup installation...")
        ensure_cargo_in_path()
        if shutil.which("cargo"):
            self._log("info", "TOOLCHAIN", "Cargo / Rustc detected (ok)")
            return True
        self._log("error", "TOOLCHAIN", "Cargo / Rustc not available in the sandbox environment")
        return False

    def prepare_environment(self, command: str) -> Dict[str, str]:
        """Inspect ``command`` and prepare the environment for it.

        Returns a dictionary suitable for ``subprocess.Popen(..., env=...)``
        containing the updated ``PATH`` and ``VIRTUAL_ENV`` variables.
        """
        self._log("info", "TOOLCHAIN", "Inspecting required toolchains for workspace...")
        lowered = command.lower().strip()

        python = shutil.which("python3") or shutil.which("python") or sys.executable
        if python and shutil.which(python):
            proc = subprocess.run([python, "--version"], capture_output=True, text=True)
            version = proc.stdout.strip() or "unknown"
            self._log("info", "TOOLCHAIN", f"Python {version} detected (ok)")
        else:
            self._log("error", "TOOLCHAIN", "Python not detected")

        if lowered.startswith("cargo ") or lowered.startswith("maturin "):
            if not self.ensure_rust_toolchain():
                raise RuntimeError("Rust toolchain (cargo) is not available")

        if any(
            lowered.startswith(prefix)
            for prefix in ("python ", "python3 ", "pytest", "maturin ", "pip ")
        ) or any(tool in lowered for tool in ("pytest", "maturin", "pip")):
            self.ensure_virtualenv()

        packages: List[str] = []
        if "maturin" in lowered:
            packages.append("maturin")
        if "pytest" in lowered:
            packages.append("pytest")
        if packages:
            self.ensure_python_packages(packages)

        if lowered.startswith("maturin "):
            if not shutil.which("maturin", path=self.env.get("PATH", "")):
                self._log(
                    "warning",
                    "TOOLCHAIN",
                    "maturin binary not found on PATH; will use `python -m maturin` fallback.",
                )

        self._prepared = True
        self._log("info", "ENV", "Environment ready for command execution.")
        return self.env

    def resolve_command(self, command: str) -> str:
        """Rewrite ``command`` so it can run inside the prepared environment.

        For ``maturin`` commands, if the binary is missing but the module is
        installed, convert to ``python -m maturin <args>``.  If the virtual
        environment cannot be created, fall back to
        ``maturin build --release && pip install target/wheels/*.whl``.
        """
        lowered = command.lower().strip()
        if not lowered.startswith("maturin "):
            return command

        args = command.split(None, 1)[1] if " " in command else ""
        try:
            if not self._prepared:
                self.prepare_environment(command)
        except Exception as exc:
            self._log("error", "ENV", f"Virtual environment setup failed: {exc}; falling back to wheel build.")
            return "maturin build --release && pip install target/wheels/*.whl"

        maturin_bin = shutil.which("maturin", path=self.env.get("PATH", ""))
        if maturin_bin:
            return f"{maturin_bin} {args}"
        return f"{self._venv_python()} -m maturin {args}"
