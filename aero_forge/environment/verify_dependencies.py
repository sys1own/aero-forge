"""Strict environment-contract verification for aero-forge."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


def _rust_aware_path(env: Optional[Dict[str, str]] = None) -> str:
    """Return PATH, bootstrapping Rust if cargo/rustc are missing and needed."""
    from aero_forge.scaffold.cargo_runner import ensure_rust_toolchain

    merged = dict(os.environ)
    if env:
        merged.update(env)
    result = ensure_rust_toolchain(merged)
    # Persist discovered rustup locations so plain subprocess.run calls inherit them.
    for key in ("PATH", "CARGO_HOME", "RUSTUP_HOME"):
        if result.get(key) and not os.environ.get(key):
            os.environ[key] = result[key]
    return result.get("PATH", os.environ.get("PATH", ""))


class ContractViolationError(Exception):
    """Raised when the host environment does not satisfy the active contract."""


def _find_cpp_compiler() -> Optional[str]:
    """Return the first available C++ compiler driver, preferring clang++ then g++."""
    for name in ("clang++", "g++", "c++"):
        if shutil.which(name):
            return name
    return None


def _compile_test_shared_library(compiler: str) -> Optional[Path]:
    """Compile a minimal C++ dynamic library to verify the toolchain works."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="aero_forge_cpp_check_"))
    src_path = tmp_dir / "test.cpp"
    so_path = tmp_dir / "libaero_forge_cpp_test.so"
    src_path.write_text(
        '#include <cstdint>\n'
        '#ifdef _WIN32\n'
        '#define AERO_EXPORT __declspec(dllexport)\n'
        '#else\n'
        '#define AERO_EXPORT __attribute__((visibility("default")))\n'
        '#endif\n'
        'extern "C" {\n'
        'AERO_EXPORT int32_t aero_forge_cpp_test() { return 42; }\n'
        '}\n',
        encoding="utf-8",
    )
    cmd = [
        compiler,
        "-shared",
        "-fPIC",
        "-O2",
        "-std=c++17",
        "-o",
        str(so_path),
        str(src_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if proc.returncode != 0:
            return None
        return so_path
    except Exception:
        return None


SYSTEM_TOOLCHAINS: Dict[str, List[str]] = {
    "rust": ["rustc", "cargo"],
    "auto": ["rustc", "cargo"],
    "c": ["cc", "gcc", "clang"],
    "cpp": ["g++", "clang++", "c++"],
    "fortran": ["gfortran"],
    "python": ["python3"],
    "node": ["node"],
}

DEFAULT_PYTHON_PACKAGES: Dict[str, str] = {
    "numpy": "numpy",
    "pybind11": "pybind11",
}

_PYTHON_PACKAGE_INSTALL_HINTS: Dict[str, str] = {
    "numpy": "pip install numpy",
    "pybind11": "pip install pybind11",
    "tomli": "pip install tomli",
    "tomlkit": "pip install tomlkit",
}

_TOOL_INSTALL_HINTS: Dict[str, str] = {
    "cargo": "Install Rust (https://rustup.rs/) or use your system package manager",
    "rustc": "Install Rust (https://rustup.rs/) or use your system package manager",
    "cc": "Install a C compiler such as gcc or clang",
    "c++": "Install a C++ compiler such as g++ or clang++",
    "gfortran": "Install gfortran via your system package manager",
    "python3": "Python 3 must be available on PATH",
    "node": "Install Node.js via your system package manager or https://nodejs.org/",
}


def _ensure_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


class VerifyDependencies:
    """Query the host environment against a blueprint-defined contract."""

    def __init__(self, blueprint: Optional[Dict[str, Any]] = None) -> None:
        self.blueprint: Dict[str, Any] = dict(blueprint) if blueprint else {}

    @staticmethod
    def missing_dependencies(packages: Optional[Dict[str, str]] = None) -> List[str]:
        """Return importable names from *packages* that are not installed."""
        packages = packages or DEFAULT_PYTHON_PACKAGES
        return [name for name in packages if importlib.util.find_spec(name) is None]

    @staticmethod
    def missing_toolchain_binaries(language: str) -> List[str]:
        """Return required binaries for *language* that are absent from PATH.

        For C/C++ the list is treated as alternatives: any one compiler on PATH
        satisfies the toolchain contract.
        """
        required = SYSTEM_TOOLCHAINS.get(language, [])
        if not required:
            return []
        path = _rust_aware_path() if language in ("rust", "auto") else os.environ.get("PATH", "")
        if language in ("c", "cpp"):
            if any(shutil.which(binary, path=path) for binary in required):
                return []
        return [binary for binary in required if shutil.which(binary, path=path) is None]

    @classmethod
    def verify_toolchain(cls, language: str) -> bool:
        """Return ``True`` when every binary for *language* is on PATH."""
        return not cls.missing_toolchain_binaries(language)

    @classmethod
    def for_language(cls, language: str) -> "VerifyDependencies":
        """Build a contract verifier for a single source language."""
        return cls({"context_registry": {"_source": {"language": language}}})

    def _contract_section(self) -> Dict[str, Any]:
        ec = self.blueprint.get("environment_contract")
        return dict(ec) if isinstance(ec, dict) else {}

    def _default_packages(self) -> Dict[str, str]:
        if self._contract_section().get("skip_defaults", False):
            return {}
        return dict(DEFAULT_PYTHON_PACKAGES)

    def _languages(self) -> Set[str]:
        languages: Set[str] = set()

        inferred = self.blueprint.get("inferred_language")
        if isinstance(inferred, str):
            languages.add(inferred)

        for entry in self.blueprint.get("context_registry", {}).values():
            if isinstance(entry, dict):
                lang = entry.get("language")
                if isinstance(lang, str):
                    languages.add(lang)

        for target in self.blueprint.get("compilation_targets", []):
            if isinstance(target, dict):
                lang = target.get("language")
                if isinstance(lang, str):
                    languages.add(lang)

        system = self.blueprint.get("system", {})
        if isinstance(system, dict):
            strategy = str(system.get("strategy", "")).strip().lower()
            if strategy in SYSTEM_TOOLCHAINS:
                languages.add(strategy)

        for lang in _ensure_str_list(self._contract_section().get("languages")):
            if lang:
                languages.add(lang)

        return languages

    def required_tools(self) -> List[str]:
        """Return the deduced set of binaries required by the contract."""
        tools: Set[str] = set()
        for lang in self._languages():
            for binary in SYSTEM_TOOLCHAINS.get(lang, []):
                tools.add(binary)
        for tool in _ensure_str_list(self._contract_section().get("required_tools")):
            if tool:
                tools.add(tool)
        return sorted(tools)

    def required_python_packages(self) -> Dict[str, str]:
        """Return the map ``{import_name: pip_name}`` to verify."""
        packages = self._default_packages()
        extra = self._contract_section().get("required_python_packages")
        if isinstance(extra, dict):
            for key, val in extra.items():
                packages[str(key)] = str(val)
        elif isinstance(extra, list):
            for pkg in extra:
                pkg = str(pkg)
                packages[pkg] = pkg
        return packages

    def missing_tools(self) -> List[str]:
        """Return required tools that are not on PATH."""
        tools = self.required_tools()
        path = _rust_aware_path() if {"cargo", "rustc"} & set(tools) else os.environ.get("PATH", "")
        missing = []
        seen_languages: set = set()
        for lang, binaries in SYSTEM_TOOLCHAINS.items():
            if any(t in binaries for t in tools):
                if lang in seen_languages:
                    continue
                seen_languages.add(lang)
                if lang in ("c", "cpp"):
                    if not any(shutil.which(b, path=path) for b in binaries):
                        missing.extend(binaries)
                else:
                    missing.extend(b for b in binaries if shutil.which(b, path=path) is None)
        # Also check any ad-hoc tools that are not part of a known toolchain.
        known = {b for binaries in SYSTEM_TOOLCHAINS.values() for b in binaries}
        missing.extend(t for t in tools if t not in known and shutil.which(t, path=path) is None)
        return sorted(set(missing))

    def missing_python_packages(self) -> List[str]:
        """Return required Python packages that are not importable."""
        return self.missing_dependencies(self.required_python_packages())

    def violations(self) -> List[str]:
        """Build human-readable violation lines for missing requirements."""
        messages: List[str] = []
        for tool in self.missing_tools():
            hint = _TOOL_INSTALL_HINTS.get(tool, f"Install {tool}")
            messages.append(f"tool '{tool}' is missing -- {hint}")
        for pkg in self.missing_python_packages():
            hint = _PYTHON_PACKAGE_INSTALL_HINTS.get(pkg, f"pip install {pkg}")
            messages.append(f"python package '{pkg}' is missing -- {hint}")
        return messages

    def verify(self) -> None:
        """Raise :class:`ContractViolationError` if the contract is not met."""
        violations = self.violations()
        if violations:
            raise ContractViolationError(
                "Contract Violation: environment does not satisfy the active blueprint:\n  - "
                + "\n  - ".join(violations)
            )
        if "cpp" in self._languages() or self.blueprint.get("architecture") == "hybrid_cpp_python":
            self.assert_cpp_shared_library()

    @classmethod
    def assert_cpp_shared_library(cls) -> Tuple[str, Path]:
        """Find a C++ compiler and verify it can build a ``.so``/``.dylib``/``.dll``.

        Returns a ``(compiler, so_path)`` tuple on success or raises
        :class:`ContractViolationError`.
        """
        compiler = _find_cpp_compiler()
        if compiler is None:
            raise ContractViolationError(
                "No C++ compiler found on PATH. "
                "Install build-essential (g++) or clang (clang++) to build C++ extensions."
            )
        so_path = _compile_test_shared_library(compiler)
        if so_path is None:
            raise ContractViolationError(
                f"'{compiler}' was found but failed to compile a test shared library. "
                "Ensure a working C++ toolchain and standard library are installed."
            )
        return compiler, so_path

    @classmethod
    def verify_language(cls, language: str) -> None:
        """Check the toolchain binaries for a single language and, for C++, compile a test library."""
        if language == "cpp":
            cls.assert_cpp_shared_library()
            return
        missing = cls.missing_toolchain_binaries(language)
        if missing:
            hints = [f"'{b}': {_TOOL_INSTALL_HINTS.get(b, f'Install {b}')}" for b in missing]
            raise ContractViolationError(
                f"Contract Violation: language '{language}' requires missing tools -- "
                + ", ".join(hints)
            )


__all__ = [
    "ContractViolationError",
    "VerifyDependencies",
    "SYSTEM_TOOLCHAINS",
    "DEFAULT_PYTHON_PACKAGES",
]


def main() -> int:
    """CLI entrypoint for a quick environment check."""
    import sys

    try:
        VerifyDependencies().verify()
        compiler, so_path = VerifyDependencies.assert_cpp_shared_library()
        print(f"Environment OK. C++ compiler: {compiler}, test .so: {so_path}")
        return 0
    except ContractViolationError as exc:
        print(f"Environment check failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
