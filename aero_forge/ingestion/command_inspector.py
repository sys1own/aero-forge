"""Command inspector for discovering runnable project commands after ingestion."""

from __future__ import annotations

import ast
import logging
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import tomllib  # type: ignore[import]
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

logger = logging.getLogger("aero_forge.ingestion.command_inspector")


def _sanitize_id(text: str) -> str:
    """Convert a display string into a URL-safe command id."""
    text = re.sub(r"[^0-9a-zA-Z]+", "-", text).strip("-")
    return text or "cmd"


def _command(
    name: str,
    cmd: str,
    category: str,
    primary: bool = False,
) -> Dict[str, Any]:
    """Return a standardized command descriptor."""
    return {
        "id": _sanitize_id(name),
        "name": name,
        "cmd": cmd,
        "category": category,
        "primary": primary,
    }


_TEMPLATE_DIR_NAMES = frozenset({"templates", "template", ".templates"})
_TEMPLATE_EXTENSIONS = frozenset({".jinja", ".jinja2", ".j2", ".tmpl"})
_JINJA_PATTERN = re.compile(r"\{\{|\{%")


def _is_jinja_or_template_path(path: Path) -> bool:
    """Return True when *path* lives in a template directory or has a template extension."""
    lower_parts = {part.lower() for part in path.parts}
    if lower_parts & _TEMPLATE_DIR_NAMES:
        return True
    if path.suffix.lower() in _TEMPLATE_EXTENSIONS:
        return True
    return False


def _is_jinja_file(path: Path) -> bool:
    """Return True for Jinja/Template files or files containing mustache syntax."""
    if _is_jinja_or_template_path(path):
        return True
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        if _JINJA_PATTERN.search(text):
            return True
    except Exception:
        pass
    return False


def _unwrap_single_root(workspace: Path) -> Path:
    """Recursively flatten a single nested wrapper directory into *workspace*.

    If *workspace* contains exactly one visible directory (and nothing else),
    move its contents up and remove the wrapper. Repeat until the condition no
    longer holds. Hidden files and aero-forge metadata are ignored when deciding
    whether to unwrap. Common project folders such as ``src`` or ``tests`` are
    never treated as archive wrappers.
    """
    workspace = Path(workspace).resolve()
    ignored = {".aero_forge", ".aero", ".git", "__pycache__", ".pytest_cache"}
    common_folders = {
        "src", "lib", "libs", "tests", "test", "app", "bin", "docs",
        "examples", "scripts", "pkg", "package", "include", "includes",
        "crates", "py_src", "source", "sources",
    }
    while True:
        visible = [
            e for e in workspace.iterdir()
            if e.name not in ignored and not e.name.startswith(".")
        ]
        if len(visible) == 1 and visible[0].is_dir():
            root = visible[0]
            if root.name.lower() in common_folders:
                break
            for child in root.iterdir():
                dest = workspace / child.name
                if dest.exists():
                    dest = workspace / f"{root.name}_{child.name}"
                shutil.move(str(child), str(dest))
            root.rmdir()
        else:
            break
    return workspace


def _find_cargo_manifests(workspace: Path) -> List[Path]:
    """Return all ``Cargo.toml`` paths under *workspace*, sorted and deduplicated.

    Common build directories such as ``target`` and ``.cargo`` are excluded, as
    are Jinja/template manifests.
    """
    manifests: set[Path] = set()
    exclude = {"target", ".cargo"}
    for path in workspace.rglob("Cargo.toml"):
        if any(part in exclude for part in path.parts):
            continue
        if _is_jinja_file(path):
            continue
        manifests.add(path.resolve().parent)
    return sorted(manifests, key=lambda p: str(p))


def _manifest_opt(manifest_dir: Path, workspace: Path) -> str:
    """Return a ``--manifest-path`` option when the manifest is not at the root."""
    if manifest_dir == workspace:
        return ""
    rel = manifest_dir.relative_to(workspace).as_posix()
    return f" --manifest-path {rel}/Cargo.toml"


def _cargo_bins(manifest_dir: Path) -> List[str]:
    """Return executable target names declared in a Cargo.toml [[bin]] section."""
    cargo_toml = manifest_dir / "Cargo.toml"
    if not cargo_toml.is_file() or _is_jinja_file(cargo_toml):
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
    if not bins and (manifest_dir / "src" / "main.rs").is_file():
        package = data.get("package", {})
        if package.get("name"):
            bins.append(str(package["name"]))
    return bins


def _cargo_examples(manifest_dir: Path) -> List[str]:
    """Return example names from examples/*.rs or [[example]] declarations."""
    examples_dir = manifest_dir / "examples"
    names: List[str] = []
    if examples_dir.is_dir():
        for path in sorted(examples_dir.glob("*.rs")):
            if _is_jinja_or_template_path(path):
                continue
            names.append(path.stem)
    cargo_toml = manifest_dir / "Cargo.toml"
    if cargo_toml.is_file() and not _is_jinja_file(cargo_toml):
        try:
            with cargo_toml.open("rb") as fh:
                data = tomllib.load(fh) or {}
            for section in data.get("example", []):
                if isinstance(section, dict) and section.get("name"):
                    names.append(str(section["name"]))
        except Exception:
            pass
    return names


def _pyproject_scripts(workspace: Path) -> List[Dict[str, str]]:
    """Return console scripts defined in pyproject.toml."""
    pyproject = workspace / "pyproject.toml"
    if not pyproject.is_file():
        return []
    try:
        with pyproject.open("rb") as fh:
            data = tomllib.load(fh) or {}
    except Exception:
        return []

    scripts: Dict[str, Any] = {}
    scripts.update(data.get("project", {}).get("scripts", {}) or {})
    scripts.update(
        data.get("project", {}).get("entry-points", {}).get("console_scripts", {}) or {}
    )
    results: List[Dict[str, str]] = []
    for name, spec in scripts.items():
        if not isinstance(spec, str):
            continue
        module = spec.split(":", 1)[0].strip().replace("/", ".")
        if module:
            results.append({"name": name, "module": module})
    return results


def _python_entrypoints(workspace: Path) -> List[Dict[str, str]]:
    """Return Python scripts callable as ``python <path>`` from the workspace."""
    scripts = _pyproject_scripts(workspace)
    results: List[Dict[str, str]] = []
    for s in scripts:
        results.append({"label": s["name"], "script": s["module"]})

    # Named primary entry points
    for candidate in ("main.py", "app.py", "manage.py"):
        path = workspace / candidate
        if path.is_file():
            results.append({"label": f"Run {path.stem}", "script": str(path.relative_to(workspace))})

    # Any script with a `if __name__ == '__main__':` guard.
    for path in sorted(workspace.rglob("*.py")):
        if "/tests/" in path.as_posix() or path.name.startswith("test_"):
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except Exception:
            continue
        if re.search(r'if\s+__name__\s*==\s*["\']__main__["\']\s*:', source):
            rel = path.relative_to(workspace).as_posix()
            label = f"Run {path.stem}"
            if not any(r["script"] == rel for r in results):
                results.append({"label": label, "script": rel})
    return results


def _has_pytest(workspace: Path) -> bool:
    """Return True if the workspace contains pytest-style test files or config."""
    pyproject = workspace / "pyproject.toml"
    if pyproject.is_file():
        try:
            text = pyproject.read_text(encoding="utf-8")
            if "[tool.pytest" in text:
                return True
        except Exception:
            pass
    for path in workspace.rglob("*.py"):
        if path.name.startswith("test_") or path.name.endswith("_test.py"):
            return True
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if any(
                    isinstance(d, ast.Name) and d.id == "pytest"
                    for d in node.decorator_list
                ):
                    return True
                if node.name.startswith("test_"):
                    return True
    return False


def _cmake_targets(workspace: Path) -> List[str]:
    """Return add_executable target names from CMakeLists.txt."""
    cmake = workspace / "CMakeLists.txt"
    if not cmake.is_file():
        return []
    try:
        text = cmake.read_text(encoding="utf-8")
    except Exception:
        return []
    return re.findall(r'add_executable\s*\(\s*([A-Za-z_]\w*)', text, re.IGNORECASE)


def _make_targets(workspace: Path) -> List[str]:
    """Return common targets from a Makefile."""
    makefile = workspace / "Makefile"
    if not makefile.is_file():
        return []
    text = makefile.read_text(encoding="utf-8")
    targets: List[str] = []
    for line in text.splitlines():
        if ":" in line and not line.startswith("\t") and not line.startswith("#"):
            target = line.split(":", 1)[0].strip()
            if target and target not in (".", "..", "clean", "all"):
                targets.append(target)
    return targets


def _blueprint_commands(workspace: Path) -> List[Dict[str, Any]]:
    """Discover commands declared in blueprint files."""
    commands: List[Dict[str, Any]] = []
    candidates = [
        workspace / "blueprint.aero",
        workspace / "workspace_blueprint.yaml",
        workspace / "workspace_blueprint.yml",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            raw = path.read_text(encoding="utf-8")
            if not yaml:
                continue
            data = yaml.safe_load(raw) or {}
        except Exception:
            continue

        # v3 execution strategy
        exec_strategy = data.get("execution_strategy", {})
        if exec_strategy:
            entrypoint = exec_strategy.get("primary_entrypoint") or data.get("metadata", {}).get("project_name")
            if isinstance(entrypoint, dict):
                entrypoint = entrypoint.get("path")
            if entrypoint:
                commands.append(_command(
                    name="Run Blueprint Entrypoint",
                    cmd=str(entrypoint),
                    category="run",
                ))

        verification_nodes = data.get("verification_nodes") or []
        if not verification_nodes:
            verification_nodes = exec_strategy.get("verification_nodes") or []
        for node in verification_nodes:
            if isinstance(node, dict):
                cmd = node.get("command") or node.get("execution_cmd")
                test_id = node.get("test_id") or node.get("node_id") or "test"
                if cmd:
                    commands.append(_command(
                        name=f"Test {test_id}",
                        cmd=str(cmd),
                        category="test",
                    ))

        # Module graph test files
        module_graph = data.get("module_graph") or []
        for entry in module_graph:
            if not isinstance(entry, dict):
                continue
            entry_path = entry.get("path") or ""
            if str(entry_path).startswith("tests/"):
                lang = (entry.get("lang") or "").lower()
                if lang == "rust":
                    commands.append(_command(
                        name=f"Run {Path(entry_path).stem}",
                        cmd=f"cargo test {Path(entry_path).stem}",
                        category="test",
                    ))
                elif lang == "python":
                    commands.append(_command(
                        name=f"Run {Path(entry_path).stem}",
                        cmd=f"python {entry_path}",
                        category="test",
                    ))
        break  # Only read the first existing candidate
    return commands


def detect_runnable_commands(workspace: Path) -> List[Dict[str, Any]]:
    """Inspect *workspace* and return a list of runnable command descriptors.

    Each descriptor follows the UI contract:
    ``{id, name, cmd, category, primary}`` where ``category`` is one of
    ``run``, ``test``, ``build`` and ``primary`` marks the most important
    command in each category.
    """
    workspace = Path(workspace).resolve()
    if workspace.is_dir():
        _unwrap_single_root(workspace)

    commands: List[Dict[str, Any]] = []
    primary_run = False
    primary_test = False
    primary_build = False

    # Rust
    cargo_manifests = _find_cargo_manifests(workspace)
    if not cargo_manifests and list(workspace.rglob("*.rs")):
        # No manifest discovered but Rust source exists; offer a plain cargo test.
        cargo_manifests = [workspace]
    for manifest_dir in cargo_manifests:
        manifest_opt = _manifest_opt(manifest_dir, workspace)
        for i, name in enumerate(_cargo_bins(manifest_dir)):
            commands.append(_command(
                name=f"Run {name}",
                cmd=f"cargo run{manifest_opt} --bin {name}",
                category="run",
                primary=(i == 0 and not primary_run),
            ))
            if i == 0:
                primary_run = True
        for example in _cargo_examples(manifest_dir):
            commands.append(_command(
                name=f"Run example {example}",
                cmd=f"cargo run{manifest_opt} --example {example}",
                category="run",
            ))
        if (manifest_dir / "Cargo.toml").is_file():
            commands.append(_command(
                name=f"Cargo test{(' (' + manifest_dir.relative_to(workspace).as_posix() + ')') if manifest_dir != workspace else ''}".strip(),
                cmd=f"cargo test{manifest_opt}",
                category="test",
                primary=not primary_test,
            ))
            primary_test = True
            commands.append(_command(
                name=f"Cargo build{(' (' + manifest_dir.relative_to(workspace).as_posix() + ')') if manifest_dir != workspace else ''}".strip(),
                cmd=f"cargo build{manifest_opt}",
                category="build",
                primary=not primary_build,
            ))
            primary_build = True

    # Python
    py_entrypoints = _python_entrypoints(workspace)
    if py_entrypoints:
        for i, ep in enumerate(py_entrypoints):
            script = ep["script"]
            if script.endswith(".py"):
                cmd = f"python {script}"
            else:
                cmd = f"python -m {script} --help"
            commands.append(_command(
                name=ep["label"],
                cmd=cmd,
                category="run",
                primary=(i == 0 and not primary_run),
            ))
            if i == 0:
                primary_run = True
    if _has_pytest(workspace):
        commands.append(_command(
            name="Run pytest",
            cmd="pytest",
            category="test",
            primary=not primary_test,
        ))
        primary_test = True
    if (workspace / "pyproject.toml").is_file():
        commands.append(_command(
            name="Install project",
            cmd="pip install -e .",
            category="build",
            primary=not primary_build,
        ))
        if not primary_build:
            primary_build = True
    elif (workspace / "setup.py").is_file():
        commands.append(_command(
            name="Install package",
            cmd="pip install .",
            category="build",
            primary=not primary_build,
        ))
        if not primary_build:
            primary_build = True

    # C/C++
    cmake_targets = _cmake_targets(workspace)
    for i, target in enumerate(cmake_targets):
        commands.append(_command(
            name=f"Build & run {target}",
            cmd=f"cmake -S . -B build && cmake --build build --target {target} && ./build/{target}",
            category="run" if i == 0 else "build",
            primary=(i == 0 and not primary_run),
        ))
        if i == 0:
            primary_run = True
    make_targets = _make_targets(workspace)
    for target in make_targets:
        if target in {"test", "tests"}:
            commands.append(_command(
                name="Make test",
                cmd="make test",
                category="test",
                primary=not primary_test,
            ))
            if not primary_test:
                primary_test = True
        elif target in {"build", "compile"}:
            commands.append(_command(
                name=f"Make {target}",
                cmd=f"make {target}",
                category="build",
                primary=not primary_build,
            ))
            if not primary_build:
                primary_build = True
        else:
            commands.append(_command(
                name=f"Make {target}",
                cmd=f"make {target}",
                category="run",
                primary=(not primary_run),
            ))
            if not primary_run:
                primary_run = True

    # Polyglot / aero blueprints
    blueprint_commands = _blueprint_commands(workspace)
    for bc in blueprint_commands:
        if bc["category"] == "run" and not primary_run:
            bc["primary"] = True
            primary_run = True
        elif bc["category"] == "test" and not primary_test:
            bc["primary"] = True
            primary_test = True
        commands.append(bc)

    # Limit primary badges to at most one per category, preferring earlier rules.
    seen_primary: Dict[str, bool] = {}
    for cmd in commands:
        if cmd.get("primary"):
            if seen_primary.get(cmd["category"]):
                cmd["primary"] = False
            else:
                seen_primary[cmd["category"]] = True

    return commands
