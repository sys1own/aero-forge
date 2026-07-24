"""Strict environment-contract verification for aero-forge."""

from __future__ import annotations

import importlib.util
import shutil
from typing import Any, Dict, List, Optional, Set


class ContractViolationError(Exception):
    """Raised when the host environment does not satisfy the active contract."""


SYSTEM_TOOLCHAINS: Dict[str, List[str]] = {
    "rust": ["rustc", "cargo"],
    "auto": ["rustc", "cargo"],
    "c": ["cc"],
    "cpp": ["c++"],
    "fortran": ["gfortran"],
    "python": ["python3"],
    "node": ["node"],
}

DEFAULT_PYTHON_PACKAGES: Dict[str, str] = {
    "numpy": "numpy",
}

_PYTHON_PACKAGE_INSTALL_HINTS: Dict[str, str] = {
    "numpy": "pip install numpy",
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
        """Return required binaries for *language* that are absent from PATH."""
        required = SYSTEM_TOOLCHAINS.get(language, [])
        return [binary for binary in required if shutil.which(binary) is None]

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
        return sorted(t for t in self.required_tools() if shutil.which(t) is None)

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

    @classmethod
    def verify_language(cls, language: str) -> None:
        """Check only the toolchain binaries for a single language."""
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
