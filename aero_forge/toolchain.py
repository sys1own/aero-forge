"""Universal toolchain manager: detect, install, and configure build tools."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

from aero_forge.environment import env_manager
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
        self._maturin_available = False
        self._workspace_package_installed = False

    def _log(self, level: str, prefix: str, message: str) -> None:
        self.log_callback(level, prefix, message)

    def _which(self, name: str) -> Optional[str]:
        """Locate ``name`` on PATH, preferring the prepared environment PATH."""
        env_path = self.env.get("PATH", "")
        try:
            return shutil.which(name, path=env_path)
        except TypeError:
            return shutil.which(name)

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

    def _venv_site_packages(self) -> Path:
        """Return the venv purelib directory."""
        proc = self._run_subprocess(
            [self._venv_python(), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
            check=True,
        )
        return Path(proc.stdout.strip())

    def _inject_pythonpath(self) -> None:
        """Prepend workspace root and ``src/`` to ``PYTHONPATH`` in ``self.env``."""
        parts: List[str] = [str(self.sandbox_dir)]
        src_dir = self.sandbox_dir / "src"
        if src_dir.is_dir():
            parts.append(str(src_dir))
        existing = self.env.get("PYTHONPATH", "")
        if existing:
            parts.append(existing)
        self.env["PYTHONPATH"] = os.pathsep.join(p for p in parts if p).strip(os.pathsep)
        self._log("info", "ENV", f"Injected PYTHONPATH: {self.env['PYTHONPATH']}")

    def _workspace_package_manifest(self) -> Optional[Path]:
        """Return the first packaging manifest found in the workspace root, if any."""
        for name in ("pyproject.toml", "setup.py", "setup.cfg"):
            path = self.sandbox_dir / name
            if path.is_file():
                return path
        return None

    def _ensure_workspace_package_installed(self) -> None:
        """Install the workspace package in editable mode when a manifest exists."""
        if self._workspace_package_installed:
            return
        proc = env_manager.install_workspace_editable(
            self.sandbox_dir,
            self._venv_python(),
            env=self.env,
            log_callback=self.log_callback,
        )
        if proc.returncode == 0:
            self._workspace_package_installed = True
            self._log("info", "ENV", "Workspace package installed in editable mode (ok)")
        else:
            self._log(
                "warning",
                "ENV",
                f"Editable install failed; falling back to PYTHONPATH. stderr: {proc.stderr.strip()[-500:]}",
            )

    def _host_package_dir(self, package: str) -> Optional[Path]:
        """Locate the host installation directory of ``package``."""
        try:
            import importlib.util

            spec = importlib.util.find_spec(package)
            if spec and spec.origin:
                return Path(spec.origin).parent
        except Exception:
            pass
        return None

    def _host_package_metadata_dirs(self, package: str, package_dir: Path) -> List[Path]:
        """Locate host metadata directories (dist-info / egg-info) for *package*."""
        candidates = []
        parent = package_dir.parent
        for item in parent.iterdir():
            lower = item.name.lower()
            if lower == package.lower() or lower.replace("-", "_") == package.lower():
                continue
            if lower.startswith(package.lower()) and (
                lower.endswith(".dist-info") or lower.endswith(".egg-info")
            ):
                candidates.append(item)
        return candidates

    def _symlink_host_package(self, package: str) -> None:
        """Symlink a host Python package and its metadata into the venv site-packages."""
        src = self._host_package_dir(package)
        if not src or not src.is_dir():
            self._log("warning", "TOOLCHAIN", f"Host package {package} not found; cannot symlink")
            return
        dest = self._venv_site_packages() / src.name
        if dest.is_symlink() or dest.exists():
            if dest.is_symlink() or dest.is_file():
                dest.unlink()
            elif dest.is_dir():
                shutil.rmtree(dest)
        try:
            os.symlink(src, dest, target_is_directory=True)
            self._log("info", "TOOLCHAIN", f"Linked host {package} package into {dest}")
        except OSError as exc:
            self._log("warning", "TOOLCHAIN", f"Could not symlink {package}: {exc}; copying instead")
            shutil.copytree(src, dest, dirs_exist_ok=True)
            self._log("info", "TOOLCHAIN", f"Copied host {package} package into {dest}")

        # Also link metadata so ``pip show`` recognizes the package and avoids
        # redundant (and permission-prone) installs.
        for meta in self._host_package_metadata_dirs(package, src):
            meta_dest = self._venv_site_packages() / meta.name
            if meta_dest.is_symlink() or meta_dest.exists():
                if meta_dest.is_symlink() or meta_dest.is_file():
                    meta_dest.unlink()
                elif meta_dest.is_dir():
                    shutil.rmtree(meta_dest)
            try:
                os.symlink(meta, meta_dest, target_is_directory=True)
                self._log("info", "TOOLCHAIN", f"Linked host {package} metadata into {meta_dest}")
            except OSError as exc:
                self._log("warning", "TOOLCHAIN", f"Could not symlink {package} metadata: {exc}")

    def _write_wrapper(self, name: str, module: str, command: Optional[str] = None) -> None:
        """Write a ``.venv/bin`` wrapper that invokes ``python -m <module>``."""
        venv_bin = self._venv_bin()
        venv_bin.mkdir(parents=True, exist_ok=True)
        wrapper = venv_bin / name
        if sys.platform == "win32":
            wrapper = wrapper.with_suffix(".bat")
            wrapper.write_text(
                f"@echo off\n{self._venv_python()} -m {module} %*\n",
                encoding="utf-8",
            )
        else:
            wrapper.write_text(
                f"#!/bin/sh\nexec {self._venv_python()} -m {module} {command or ''} \"$@\"\n",
                encoding="utf-8",
            )
            wrapper.chmod(0o755)

    def _bootstrap_venv_from_host(self) -> None:
        """Populate a ``--without-pip`` venv with host pip, setuptools, wheel and binaries."""
        self._log("info", "TOOLCHAIN", "Bootstrapping .venv from host toolchains...")

        # Symlink core packaging packages from the host so pip works offline.
        for package in ("pip", "setuptools", "wheel"):
            self._symlink_host_package(package)

        # Create pip/pip3/wheel wrappers.
        self._write_wrapper("pip", "pip")
        self._write_wrapper("pip3", "pip")
        self._write_wrapper("wheel", "wheel")

        # Refresh PATH so the newly-added wrappers are found.
        self.env["PATH"] = f"{self._venv_bin()}{os.pathsep}{self.env.get('PATH', '')}"

    def _run_subprocess(
        self,
        cmd: List[str],
        check: bool = True,
        capture: bool = True,
    ) -> subprocess.CompletedProcess:
        """Run ``cmd`` and return the completed process, logging diagnostics on failure."""
        proc = subprocess.run(
            cmd,
            capture_output=capture,
            text=True,
            env=self.env,
            check=False,
        )
        if check and proc.returncode != 0:
            if proc.stdout:
                self._log("info", "TOOLCHAIN", proc.stdout.strip())
            if proc.stderr:
                self._log("error", "TOOLCHAIN", proc.stderr.strip())
            raise subprocess.CalledProcessError(
                proc.returncode, cmd, output=proc.stdout, stderr=proc.stderr
            )
        return proc

    def _run_pip(self, args: List[str], check: bool = True) -> subprocess.CompletedProcess:
        """Run ``pip`` inside the virtual environment, streaming captured diagnostics."""
        return self._run_subprocess([self._venv_python(), "-m", "pip", *args], check=check)

    def _create_venv(self, python: str) -> None:
        """Create a virtual environment that can see host packages and has no bundled pip."""
        venv_path = self._venv_path()
        proc = self._run_subprocess(
            [python, "-m", "venv", "--system-site-packages", "--without-pip", str(venv_path)],
            check=False,
        )
        if proc.returncode != 0:
            # Some older venv modules do not support --without-pip; try without it.
            if "unrecognized arguments" in (proc.stderr or ""):
                self._log("warning", "ENV", "--without-pip not supported; retrying without it")
                proc = self._run_subprocess(
                    [python, "-m", "venv", str(venv_path)],
                    check=False,
                )
            if proc.returncode != 0:
                self._log("error", "ENV", f"Failed to create virtual environment: {proc.stderr or proc.stdout}")
                raise RuntimeError("Virtual environment creation failed")

    def ensure_virtualenv(self) -> Dict[str, str]:
        """Ensure a ``.venv`` exists in ``sandbox_dir`` and return updated env vars."""
        venv_path = self._venv_path()
        python_exe = "python.exe" if sys.platform == "win32" else "bin/python"
        if not (venv_path / python_exe).is_file():
            self._log("info", "ENV", f"Creating Python virtual environment (.venv) in {self.sandbox_dir}...")
            python = sys.executable if shutil.which(sys.executable) else "python3"
            self._create_venv(python)
            self._log("info", "ENV", "Python virtual environment created (ok)")
            self._bootstrap_venv_from_host()
        else:
            self._log("info", "ENV", "Using existing Python virtual environment (.venv).")
            # Ensure wrappers exist even for pre-existing venvs.
            if not (self._venv_bin() / "pip").exists():
                self._bootstrap_venv_from_host()

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

    def _install_python_package(self, package: str) -> None:
        """Install ``package`` with ``pip``, logging full diagnostics on failure."""
        self._log("info", "TOOLCHAIN", f"{package} missing in .venv -> Auto-installing `pip install {package}`...")
        try:
            self._run_pip(["install", package])
            self._log("info", "TOOLCHAIN", f"{package} installed (ok) in .venv")
        except subprocess.CalledProcessError as exc:
            self._log("error", "TOOLCHAIN", f"`pip install {package}` failed (exit {exc.returncode}):")
            if exc.output:
                for line in exc.output.strip().splitlines():
                    self._log("info", "TOOLCHAIN", line)
            if exc.stderr:
                for line in exc.stderr.strip().splitlines():
                    self._log("error", "TOOLCHAIN", line)
            raise

    def _alias_host_maturin_into_venv(self, host_maturin: str) -> None:
        """Create a symlink/copy of the host ``maturin`` binary inside ``.venv/bin``."""
        venv_bin = self._venv_bin()
        venv_maturin = venv_bin / "maturin"
        real_host = Path(host_maturin).resolve()
        if real_host.resolve() == venv_maturin.resolve():
            # Already points to itself; nothing to do.
            return
        try:
            if sys.platform == "win32":
                shutil.copy2(host_maturin, venv_maturin)
            else:
                if venv_maturin.is_symlink() or venv_maturin.exists():
                    venv_maturin.unlink()
                os.symlink(host_maturin, venv_maturin)
            self._log("info", "TOOLCHAIN", f"Linked host maturin into {venv_maturin}")
        except OSError as exc:
            self._log("warning", "TOOLCHAIN", f"Could not link host maturin into venv: {exc}; using host path directly")

    def ensure_maturin(self) -> bool:
        """Ensure ``maturin`` is available, preferring host binary or falling back."""
        if self._maturin_available:
            return True

        self.ensure_virtualenv()

        # 1. Prefer a host maturin binary and alias it into the venv.
        host_maturin = self._which("maturin") or shutil.which("maturin")
        if host_maturin:
            self._alias_host_maturin_into_venv(host_maturin)
            self._maturin_available = True
            self._log("info", "TOOLCHAIN", f"Using host maturin binary: {host_maturin}")
            return True

        # 2. Try pip install with full diagnostic logging.
        try:
            self._install_python_package("maturin")
            self._maturin_available = True
            return True
        except subprocess.CalledProcessError:
            self._log("warning", "TOOLCHAIN", "pip install maturin failed; trying cargo install maturin --locked...")

        # 3. Fall back to cargo install.
        if shutil.which("cargo"):
            try:
                cargo_home = Path(self.env.get("CARGO_HOME", Path.home() / ".cargo"))
                cargo_bin = cargo_home / "bin"
                cargo_bin.mkdir(parents=True, exist_ok=True)
                self._run_subprocess(
                    ["cargo", "install", "maturin", "--locked"],
                    check=True,
                )
                self.env["PATH"] = f"{cargo_bin}{os.pathsep}{self.env.get('PATH', '')}"
                self._log("info", "TOOLCHAIN", f"maturin installed via cargo into {cargo_bin}")
                self._maturin_available = True
                return True
            except subprocess.CalledProcessError as exc:
                self._log("error", "TOOLCHAIN", f"cargo install maturin failed: {exc.stderr}")

        self._log("error", "TOOLCHAIN", "maturin could not be provisioned; will fall back to cargo build workflows")
        return False

    def ensure_python_packages(self, required_packages: List[str]) -> None:
        """Install ``required_packages`` into ``.venv`` if missing."""
        self.ensure_virtualenv()
        for package in required_packages:
            if package == "maturin":
                self.ensure_maturin()
                continue
            try:
                self._run_venv_python(["-m", "pip", "show", package], check=True)
                self._log("info", "TOOLCHAIN", f"{package} detected (ok) in .venv")
            except (subprocess.CalledProcessError, FileNotFoundError):
                self._install_python_package(package)

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

        has_rust = True
        if lowered.startswith("cargo ") or lowered.startswith("maturin "):
            has_rust = self.ensure_rust_toolchain()
            if not has_rust:
                self._log("warning", "TOOLCHAIN", "Rust toolchain (cargo) is not available; proceeding with Python-only fallback")

        python_like = lowered in ("python", "python3") or lowered.startswith(("python ", "python3 ", "pytest", "maturin ", "pip "))
        if python_like or any(tool in lowered for tool in ("pytest", "maturin", "pip")):
            self.ensure_virtualenv()

        packages: List[str] = []
        if "pytest" in lowered:
            packages.append("pytest")
        if packages:
            self.ensure_python_packages(packages)

        if "maturin" in lowered and has_rust:
            self._maturin_available = self.ensure_maturin()
            if not self._maturin_available and lowered.startswith("maturin "):
                self._log(
                    "warning",
                    "TOOLCHAIN",
                    "maturin not available; will rewrite command to a cargo fallback.",
                )
        elif "maturin" in lowered and not has_rust:
            self._maturin_available = False
            self._log(
                "warning",
                "TOOLCHAIN",
                "maturin skipped because Rust toolchain is unavailable; will rewrite command to a cargo fallback.",
            )

        if Path(self._venv_python()).is_file():
            self._ensure_workspace_package_installed()
        self._inject_pythonpath()

        self._prepared = True
        self._log("info", "ENV", "Environment ready for command execution.")
        return self.env

    def _find_cargo_manifest(self) -> Optional[Path]:
        """Locate the nearest ``Cargo.toml`` starting at ``self.sandbox_dir``."""
        root = self.sandbox_dir
        if (root / "Cargo.toml").is_file():
            return root / "Cargo.toml"
        exclude = {"target", ".cargo"}
        for path in root.rglob("Cargo.toml"):
            if any(part in exclude for part in path.parts):
                continue
            return path
        return None

    def _cargo_command(self, base: str) -> str:
        """Return ``base`` with an automatic ``--manifest-path`` if needed."""
        lowered = base.lower()
        if "--manifest-path" in lowered:
            return base
        manifest = self._find_cargo_manifest()
        if manifest and manifest.parent != self.sandbox_dir:
            rel = manifest.parent.relative_to(self.sandbox_dir).as_posix()
            return f"{base} --manifest-path {rel}/Cargo.toml"
        return base

    def resolve_command(self, command: str) -> str:
        """Rewrite ``command`` so it can run inside the prepared environment.

        For ``maturin`` commands, prefer the resolved binary path. If maturin
        cannot be provisioned, fall back to a plain ``cargo build --release``
        or ``cargo test`` workflow so workspace acceleration is not halted.
        """
        lowered = command.lower().strip()
        if lowered.startswith("cargo "):
            return self._cargo_command(command)
        if not lowered.startswith("maturin "):
            return command

        args = command.split(None, 1)[1] if " " in command else ""
        try:
            if not self._prepared:
                self.prepare_environment(command)
        except Exception as exc:
            self._log("error", "ENV", f"Virtual environment setup failed: {exc}; falling back to cargo.")
            self._maturin_available = False

        if self._maturin_available:
            maturin_bin = self._which("maturin")
            if maturin_bin:
                return f"{maturin_bin} {args}"
            return f"{self._venv_python()} -m maturin {args}"

        self._log("warning", "TOOLCHAIN", "maturin unavailable; rewriting to cargo fallback")
        if "test" in lowered or "pytest" in lowered:
            return self._cargo_command("cargo test")
        if "develop" in lowered or "build" in lowered:
            return self._cargo_command("cargo build --release")
        return self._cargo_command("cargo build")
