"""Workspace script inspector: detect runnable commands and scaffold native tooling."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

from aero_forge.bundle_repo import scaffold_native_crate


def _has_make_target(makefile: Path, target: str) -> bool:
    """Return True when ``target`` appears as a top-level target in ``Makefile``."""
    try:
        text = makefile.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    # Match a line that starts the target definition, ignoring recipes and variables.
    pattern = re.compile(rf"^(?:\s*\.PHONY:\s*)?{re.escape(target)}\s*:", re.MULTILINE)
    return bool(pattern.search(text))


def _cargo_targets(cargo_toml: Path) -> List[str]:
    """Return a best-effort list of runnable cargo commands for a ``Cargo.toml``."""
    commands = ["cargo build", "cargo test"]
    try:
        text = cargo_toml.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return commands
    if re.search(r"\[\[bin\]\]", text):
        commands.append("cargo run")
    return commands


def inspect_workspace(workspace_dir: Path) -> List[Dict[str, str]]:
    """Scan ``workspace_dir`` for common project descriptors and return runnable commands.

    Returned dictionaries contain ``source`` (the file that triggered the command),
    ``label`` (a short human-readable label), and ``cmd`` (the command string).
    """
    workspace_dir = Path(workspace_dir).resolve()
    commands: List[Dict[str, str]] = []

    if (workspace_dir / "main.py").is_file():
        commands.append({"source": "main.py", "label": "Run main.py", "cmd": "python main.py"})

    if (workspace_dir / "requirements.txt").is_file():
        commands.append(
            {
                "source": "requirements.txt",
                "label": "Install requirements",
                "cmd": "pip install -r requirements.txt",
            }
        )

    if (workspace_dir / "setup.py").is_file():
        commands.append(
            {
                "source": "setup.py",
                "label": "Install package",
                "cmd": "pip install .",
            }
        )

    pyproject = workspace_dir / "pyproject.toml"
    if pyproject.is_file():
        try:
            pyproject_text = pyproject.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            pyproject_text = ""

        has_pytest = (
            "[tool.pytest" in pyproject_text
            or (workspace_dir / "tests").is_dir()
            or (workspace_dir / "test_").exists()
        )
        if has_pytest:
            commands.append(
                {
                    "source": "pyproject.toml",
                    "label": "Run pytest",
                    "cmd": "pytest",
                }
            )

        if "[tool.maturin]" in pyproject_text:
            commands.append(
                {
                    "source": "pyproject.toml",
                    "label": "Build native extension",
                    "cmd": "maturin develop",
                }
            )
        else:
            commands.append(
                {
                    "source": "pyproject.toml",
                    "label": "Install project",
                    "cmd": "pip install .",
                }
            )

    cargo = workspace_dir / "Cargo.toml"
    if cargo.is_file():
        for cmd in _cargo_targets(cargo):
            commands.append({"source": "Cargo.toml", "label": cmd, "cmd": cmd})

    crate_cargo = workspace_dir / "crates" / "native_core" / "Cargo.toml"
    if crate_cargo.is_file():
        commands.append(
            {
                "source": "crates/native_core/Cargo.toml",
                "label": "Test native core",
                "cmd": "cargo test --manifest-path crates/native_core/Cargo.toml",
            }
        )

    package_json = workspace_dir / "package.json"
    if package_json.is_file():
        commands.append(
            {
                "source": "package.json",
                "label": "Install npm packages",
                "cmd": "npm install",
            }
        )
        try:
            with package_json.open("r", encoding="utf-8") as f:
                pkg = json.load(f)
        except (OSError, json.JSONDecodeError):
            pkg = {}
        scripts = pkg.get("scripts", {})
        for name in scripts:
            commands.append(
                {
                    "source": "package.json",
                    "label": f"npm run {name}",
                    "cmd": f"npm run {name}",
                }
            )

    makefile = workspace_dir / "Makefile"
    if makefile.is_file():
        if _has_make_target(makefile, "test"):
            commands.append({"source": "Makefile", "label": "make test", "cmd": "make test"})
        if _has_make_target(makefile, "build"):
            commands.append({"source": "Makefile", "label": "make build", "cmd": "make build"})
        commands.append({"source": "Makefile", "label": "make", "cmd": "make"})

    return commands


def scaffold_pyo3_workspace(
    workspace_dir: Path,
    project_name: str = "generated-native",
) -> None:
    """Inject a PyO3 native acceleration crate into ``workspace_dir`` if absent."""
    scaffold_native_crate(workspace_dir, project_name=project_name)
