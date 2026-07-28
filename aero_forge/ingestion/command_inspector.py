"""Command inspector for discovering runnable project commands after ingestion."""

from __future__ import annotations

import ast
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import tomllib  # type: ignore[import]
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]

logger = logging.getLogger("aero_forge.ingestion.command_inspector")


def _cargo_bins(workspace: Path) -> List[str]:
    """Return executable target names declared in Cargo.toml [[bin]] sections."""
    cargo_toml = workspace / "Cargo.toml"
    if not cargo_toml.is_file():
        return []
    try:
        with cargo_toml.open("rb") as fh:
            data = tomllib.load(fh) or {}
    except Exception:
        return []

    bins: List[str] = []
    for section in data.get("bin", []):
        if isinstance(section, dict) and section.get("name"):
            bins.append(str(section["name"]))
    # If no explicit bin but src/main.rs exists, the package name is the default bin.
    if not bins and (workspace / "src" / "main.rs").is_file() and data.get("package", {}).get("name"):
        bins.append(str(data["package"]["name"]))
    return bins


def _python_entrypoints(workspace: Path) -> List[Dict[str, str]]:
    """Return Python scripts callable as `python <path>` from the workspace."""
    pyproject = workspace / "pyproject.toml"
    if pyproject.is_file():
        try:
            with pyproject.open("rb") as fh:
                data = tomllib.load(fh) or {}
            scripts = data.get("project", {}).get("scripts", {})
            console = data.get("project", {}).get("entry-points", {}).get("console_scripts", {})
            results = []
            for name in {*scripts.keys(), *console.keys()}:
                module_path = (scripts.get(name) or console.get(name) or "").split(":")[0].replace(".", "/")
                results.append({"label": name, "script": module_path or name})
            return results
        except Exception:
            pass
    # Fallback: files with `if __name__ == "__main__":`
    results: List[Dict[str, str]] = []
    for path in sorted(workspace.rglob("*.py")):
        if "/tests/" in path.as_posix() or path.name.startswith("test_"):
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except Exception:
            continue
        if re.search(r'if\s+__name__\s*==\s*["\']__main__["\']\s*:', source):
            rel = path.relative_to(workspace).as_posix()
            results.append({"label": f"Run {path.stem}", "script": rel})
    return results


def _has_pytest(workspace: Path) -> bool:
    """Return True if the workspace contains pytest-style test files."""
    for path in workspace.rglob("*.py"):
        if path.name.startswith("test_") or path.name.endswith("_test.py"):
            return True
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and any(
                d.id == "pytest" for d in node.decorator_list if isinstance(d, ast.Name)
            ):
                return True
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                return True
    return False


def _cmake_targets(workspace: Path) -> List[str]:
    """Return add_executable target names from CMakeLists.txt."""
    cmake = workspace / "CMakeLists.txt"
    if not cmake.is_file():
        return []
    text = cmake.read_text(encoding="utf-8")
    return re.findall(r'add_executable\s*\(\s*([A-Za-z_]\w*)', text, re.IGNORECASE)


def _make_targets(workspace: Path) -> List[str]:
    """Return common executable targets from a Makefile."""
    makefile = workspace / "Makefile"
    if not makefile.is_file():
        return []
    text = makefile.read_text(encoding="utf-8")
    targets: List[str] = []
    for line in text.splitlines():
        if ":" in line and not line.startswith("\t") and not line.startswith("#"):
            target = line.split(":", 1)[0].strip()
            if target and target not in (".", "clean", "all", "test"):
                targets.append(target)
    return targets


def detect_runnable_commands(workspace: Path) -> List[Dict[str, Any]]:
    """Inspect *workspace* and return a list of runnable command descriptors."""
    workspace = Path(workspace).resolve()
    commands: List[Dict[str, Any]] = []

    # Rust
    bins = _cargo_bins(workspace)
    if bins:
        commands.append({
            "label": f"Run {bins[0]}",
            "cmd": f"cargo run --bin {bins[0]}",
            "type": "primary",
        })
        for name in bins[1:]:
            commands.append({
                "label": f"Run {name}",
                "cmd": f"cargo run --bin {name}",
                "type": "secondary",
            })
    if (workspace / "Cargo.toml").is_file() or list(workspace.rglob("*.rs")):
        commands.append({"label": "Run Tests", "cmd": "cargo test", "type": "secondary"})

    # Python
    entrypoints = _python_entrypoints(workspace)
    if entrypoints:
        primary = entrypoints[0]
        commands.append({
            "label": primary["label"],
            "cmd": f"python {primary['script']}",
            "type": "primary",
        })
        for ep in entrypoints[1:]:
            commands.append({
                "label": ep["label"],
                "cmd": f"python {ep['script']}",
                "type": "secondary",
            })
    if _has_pytest(workspace):
        commands.append({"label": "Run Tests", "cmd": "pytest", "type": "secondary"})

    # C/C++
    cmake_targets = _cmake_targets(workspace)
    for target in cmake_targets[:1]:
        commands.append({
            "label": f"Build & Run {target}",
            "cmd": f"cmake --build build --target {target} && ./build/{target}",
            "type": "secondary",
        })
    make_targets = _make_targets(workspace)
    if make_targets:
        commands.append({
            "label": f"Make {make_targets[0]}",
            "cmd": f"make {make_targets[0]}",
            "type": "secondary",
        })

    # Aero blueprint execution spec
    blueprint = workspace / "blueprint.aero"
    if blueprint.is_file():
        try:
            import yaml

            data = yaml.safe_load(blueprint.read_text(encoding="utf-8")) or {}
            exec_strategy = data.get("execution_strategy", {})
            entrypoint = exec_strategy.get("primary_entrypoint") or data.get("metadata", {}).get("project_name")
            if entrypoint:
                commands.append({
                    "label": "Run Blueprint Entrypoint",
                    "cmd": str(entrypoint),
                    "type": "secondary",
                })
        except Exception:
            pass

    return commands
