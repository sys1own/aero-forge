"""Auto-bootstrap missing toolchains for universal polyglot builds."""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import shutil
import subprocess
import tarfile
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from aero_forge.config import get_toolchains_dir

logger = logging.getLogger("aero_forge.toolchain_bootstrap")


class ToolchainNotFoundError(RuntimeError):
    """Raised when a required toolchain is missing and cannot be bootstrapped."""

    def __init__(self, toolchain: str, message: str, install_command: str = "") -> None:
        super().__init__(message)
        self.toolchain = toolchain
        self.install_command = install_command


def _detect_platform() -> Tuple[str, str]:
    """Return (system, machine) normalized for download URLs."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    # Normalize common synonyms.
    if machine in ("x86_64", "amd64", "x64"):
        machine = "x86_64"
    elif machine in ("aarch64", "arm64"):
        machine = "aarch64"
    return system, machine


def _http_get_json(url: str, timeout: int = 30) -> Any:
    """Fetch and parse JSON from *url*."""
    req = urllib.request.Request(url, headers={"User-Agent": "aero-forge-toolchain-bootstrap"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _download(url: str, dest: Path, timeout: int = 120) -> None:
    """Download *url* to *dest* with a simple progress log."""
    logger.info("Downloading toolchain from %s", url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "aero-forge-toolchain-bootstrap"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response, dest.open("wb") as f:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    except urllib.error.URLError as exc:
        raise ToolchainNotFoundError("", f"Download failed for {url}: {exc}") from exc


def _extract(archive: Path, dest_dir: Path) -> None:
    """Extract a tar.xz / tar.gz / zip archive to *dest_dir*."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    suffix = archive.suffix.lower()
    if suffix == ".zip":
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(dest_dir)
    elif ".tar" in archive.name or suffix in (".gz", ".xz", ".bz2"):
        with tarfile.open(archive, "r:*") as tf:
            tf.extractall(dest_dir, filter="data")
    else:
        raise ToolchainNotFoundError("", f"Unsupported archive format: {archive}")


def _add_to_path(bin_dir: Path) -> None:
    """Prepend *bin_dir* to the process PATH so shutil.which can find it."""
    if not bin_dir.is_dir():
        return
    current = os.environ.get("PATH", "")
    path_str = str(bin_dir)
    if path_str in current.split(os.pathsep):
        return
    os.environ["PATH"] = path_str + os.pathsep + current


class ToolchainBootstrap:
    """Verify and, when possible, download portable toolchain binaries."""

    _KNOWN: Dict[str, str] = {
        "zig": "Zig",
        "go": "Go",
        "mojo": "Mojo",
        "maturin": "maturin",
        "cmake": "CMake",
        "dotnet": ".NET SDK",
        "nvcc": "NVIDIA CUDA Toolkit",
    }

    @classmethod
    def diagnostic(cls, toolchain: str) -> str:
        """Return a human-facing installation command for *toolchain*."""
        name = cls._KNOWN.get(toolchain, toolchain)
        system, machine = _detect_platform()

        if toolchain == "zig":
            return (
                "Install Zig:\n"
                "  Debian/Ubuntu: sudo snap install zig --classic  # or download from https://ziglang.org/download/\n"
                "  macOS: brew install zig\n"
                "  Other: wget https://ziglang.org/download/index.json and extract the tarball for your platform."
            )
        if toolchain == "go":
            return (
                "Install Go:\n"
                "  Linux/macOS: https://go.dev/dl/  (extract the tarball and add bin/ to PATH)\n"
                "  Debian/Ubuntu: sudo apt-get install golang-go\n"
                "  macOS: brew install go"
            )
        if toolchain == "mojo":
            return (
                "Install Mojo SDK:\n"
                "  1. Install the Modular CLI: curl -s https://get.modular.com/magic | python3 -\n"
                "  2. modular install mojo\n"
                "  See https://docs.modular.com/mojo/manual/install/ for platform requirements."
            )
        if toolchain == "maturin":
            return "Install maturin: pip install maturin"
        if toolchain == "cmake":
            return "Install CMake: pip install cmake  # or apt-get/brew install cmake"
        if toolchain == "dotnet":
            return "Install .NET SDK: https://dotnet.microsoft.com/download"
        if toolchain == "nvcc":
            return "Install the NVIDIA CUDA Toolkit: https://developer.nvidia.com/cuda-downloads"
        if toolchain in ("gcc", "clang", "g++", "clang++"):
            return (
                f"Install a C/C++ compiler:\n"
                "  Debian/Ubuntu: sudo apt-get install build-essential\n"
                "  macOS: xcode-select --install"
            )
        return (
            f"Toolchain {toolchain!r} ({name}) was not found on PATH. "
            f"Install a {system}-{machine} build for {name} and ensure its bin directory is on PATH."
        )

    @classmethod
    def is_available(cls, toolchain: str) -> bool:
        """Return ``True`` if the host can already invoke *toolchain*."""
        return shutil.which(toolchain) is not None

    @classmethod
    def ensure(cls, toolchain: str, bootstrap: bool = True) -> Optional[str]:
        """Return the absolute path to *toolchain*, bootstrapping if needed.

        Raises:
            ToolchainNotFoundError: when the toolchain is missing and cannot be
                auto-installed on this platform.
        """
        name = toolchain.strip().lower()
        exec_map = {"python": "python3", "py": "python3", "cpython": "python3"}
        exec_name = exec_map.get(name, name)

        path = shutil.which(exec_name)
        if path:
            return path

        if not bootstrap:
            raise ToolchainNotFoundError(
                name,
                f"Toolchain {name!r} not found on PATH.",
                cls.diagnostic(name),
            )

        bootstrapped = cls._try_bootstrap(name)
        if bootstrapped:
            _add_to_path(bootstrapped.parent)
            return str(bootstrapped)

        raise ToolchainNotFoundError(
            name,
            f"Toolchain {name!r} not found on PATH and could not be auto-bootstrapped.",
            cls.diagnostic(name),
        )

    @classmethod
    def _try_bootstrap(cls, name: str) -> Optional[Path]:
        """Attempt to download and extract a portable binary for *name*."""
        if name == "zig":
            return cls._bootstrap_zig()
        if name == "go":
            return cls._bootstrap_go()
        return None

    @classmethod
    def _bootstrap_zig(cls) -> Optional[Path]:
        """Download the latest stable Zig release for the host platform."""
        system, machine = _detect_platform()
        if system not in ("linux", "darwin", "windows"):
            return None
        zig_os = {"linux": "linux", "darwin": "macos", "windows": "windows"}[system]
        zig_arch = "x86_64" if machine == "x86_64" else "aarch64"

        try:
            data = _http_get_json("https://ziglang.org/download/index.json")
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as exc:
            logger.warning("Could not query Zig download index: %s", exc)
            return None

        # Prefer the newest stable (non-master) release.
        candidates: List[Tuple[Tuple[int, ...], Any, str]] = []
        for version, info in data.items():
            if not isinstance(info, dict):
                continue
            if version.lower() in ("master", "dev"):
                continue
            arch_info = info.get(f"{zig_arch}-{zig_os}")
            if not isinstance(arch_info, dict):
                continue
            tarball = arch_info.get("tarball")
            if tarball:
                try:
                    parts = [int(p) for p in re.split(r"[^0-9]", version) if p.isdigit()]
                    candidates.append((tuple(parts), arch_info, version))
                except ValueError:
                    candidates.append(((0,), arch_info, version))
        if not candidates:
            # Fall back to master if no stable build is listed.
            master = data.get("master", {})
            if isinstance(master, dict):
                arch_info = master.get(f"{zig_arch}-{zig_os}")
                if isinstance(arch_info, dict):
                    candidates.append(((0,), arch_info, "master"))
        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0], reverse=True)
        chosen = candidates[0][1]
        tarball_url = chosen.get("tarball")
        if not tarball_url:
            return None

        return cls._install_from_url(name="zig", url=tarball_url, binary_relpath="zig")

    @classmethod
    def _bootstrap_go(cls) -> Optional[Path]:
        """Download the latest stable Go release for the host platform."""
        system, machine = _detect_platform()
        go_os = system
        if go_os == "darwin":
            go_os = "darwin"
        go_arch = "amd64" if machine == "x86_64" else "arm64"

        try:
            releases = _http_get_json("https://go.dev/dl/?mode=json")
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as exc:
            logger.warning("Could not query Go download index: %s", exc)
            return None

        for release in releases:
            version = release.get("version", "")
            if re.search(r"(rc|beta|alpha)", version, re.I):
                continue
            for file_info in release.get("files", []):
                if (
                    file_info.get("os") == go_os
                    and file_info.get("arch") == go_arch
                    and file_info.get("kind") == "archive"
                ):
                    filename = file_info.get("filename")
                    if filename:
                        url = f"https://go.dev/dl/{filename}"
                        return cls._install_from_url(
                            name="go",
                            url=url,
                            binary_relpath="bin/go",
                        )
        return None

    @classmethod
    def _install_from_url(cls, name: str, url: str, binary_relpath: str) -> Optional[Path]:
        """Download *url*, extract it, and return the path to the binary."""
        root = get_toolchains_dir() / name
        if root.is_dir() and any(root.rglob("*")):
            # If the toolchain was already downloaded, prefer the cached copy.
            cached = root / binary_relpath
            if cached.is_file():
                logger.info("Using cached %s toolchain at %s", name, cached)
                return cached

        archive_name = Path(url).name or f"{name}-archive.tar.gz"
        with tempfile.TemporaryDirectory() as tmp:
            archive_path = Path(tmp) / archive_name
            _download(url, archive_path)
            extract_root = Path(tmp) / "extracted"
            _extract(archive_path, extract_root)
            # The archive usually contains a single top-level directory.
            top_dirs = [p for p in extract_root.iterdir() if p.is_dir()]
            if not top_dirs:
                return None
            src_dir = top_dirs[0]
            # Move the extracted tree into the cached toolchain root.
            if root.exists():
                shutil.rmtree(root, ignore_errors=True)
            shutil.move(str(src_dir), str(root))

        binary = root / binary_relpath
        if not binary.is_file():
            return None
        binary.chmod(0o755)
        logger.info("Bootstrapped %s toolchain at %s", name, binary)
        return binary


