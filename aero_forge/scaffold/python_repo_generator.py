"""Native Python out-of-tree workspace generation for aero-forge."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

_STDLIB_ROOTS = {
    "abc", "aifc", "argparse", "array", "ast", "asyncio", "base64", "binascii",
    "bisect", "builtins", "calendar", "collections", "compileall", "concurrent",
    "contextlib", "copy", "csv", "dataclasses", "datetime", "decimal", "difflib",
    "email", "enum", "functools", "gc", "getpass", "glob", "gzip", "hashlib",
    "heapq", "html", "http", "importlib", "inspect", "io", "itertools", "json",
    "logging", "math", "mimetypes", "multiprocessing", "numbers", "operator",
    "os", "pathlib", "pickle", "platform", "pprint", "py_compile", "queue",
    "random", "re", "shutil", "signal", "socket", "sqlite3", "statistics",
    "string", "struct", "subprocess", "sys", "tempfile", "textwrap", "threading",
    "time", "traceback", "types", "typing", "unittest", "urllib", "uuid",
    "warnings", "weakref", "xml", "zipfile",
}


@dataclass
class PythonRepoSpec:
    """Everything needed to materialise a standalone Python project."""

    name: str
    entry_filename: str
    source: str
    version: str = "0.1.0"
    description: str = ""
    dependencies: List[str] = field(default_factory=list)
    module_name: str = "engine"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "entry_filename": self.entry_filename,
            "module_name": self.module_name,
            "version": self.version,
            "description": self.description,
            "dependencies": list(self.dependencies),
            "language": "python",
        }


@dataclass
class PythonGeneratedRepo:
    """Files written for a standalone Python project."""

    root: Path
    files: List[str] = field(default_factory=list)
    spec: Optional[PythonRepoSpec] = None

    def to_dict(self) -> dict:
        return {
            "root": str(self.root),
            "files": list(self.files),
            "spec": self.spec.to_dict() if self.spec else None,
            "language": "python",
        }


def sanitize_project_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", name.strip()).strip("-").lower()
    return cleaned or "python-app"


_GENERIC_NAMES = {"main", "run", "solve", "helper", "generated", "app", "test"}


_PYTHON_KEYWORDS = {
    "False", "None", "True", "and", "as", "assert", "async", "await", "break",
    "class", "continue", "def", "del", "elif", "else", "except", "finally",
    "for", "from", "global", "if", "import", "in", "is", "lambda", "nonlocal",
    "not", "or", "pass", "raise", "return", "try", "while", "with", "yield",
}


def _sanitize_module_name(name: str) -> str:
    """Convert *name* into a valid Python module identifier."""
    name = re.sub(r"[^A-Za-z0-9]+", "_", name)
    name = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
    name = re.sub(r"_+", "_", name).strip("_")
    if not name or name[0].isdigit() or name in _PYTHON_KEYWORDS or name in _GENERIC_NAMES:
        name = "engine"
    return name[:40]


def _detect_public_names(source: str) -> List[str]:
    """Return public top-level function and class names from *source*."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    return [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not node.name.startswith("_")
    ]


def _derive_module_name_from_source(source: str, project_name: str) -> str:
    """Pick a domain-specific module name from the generated source."""
    for name in _detect_public_names(source):
        if name not in _GENERIC_NAMES:
            return _sanitize_module_name(name)
    return _sanitize_module_name(project_name)


def choose_entry_filename(original_name: str, project_name: str, source: str) -> str:
    """Keep the original filename when idiomatic; otherwise pick a sensible default."""
    if original_name.endswith(".py") and original_name not in ("__init__.py",):
        return original_name
    module_name = _derive_module_name_from_source(source, project_name)
    return f"src/{module_name}.py"


def infer_import_dependencies(source: str, overrides: Optional[List[str]] = None) -> List[str]:
    """Collect third-party import roots from Python source for requirements.txt."""
    found: Set[str] = set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        for match in re.finditer(r"^\s*(?:from|import)\s+([A-Za-z_]\w*)", source, re.MULTILINE):
            found.add(match.group(1))
    else:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    found.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    found.add(node.module.split(".")[0])

    third_party = sorted(
        mod for mod in found
        if mod not in _STDLIB_ROOTS and not mod.startswith("_")
    )
    if overrides:
        merged = list(dict.fromkeys(list(overrides) + third_party))
        return merged
    return third_party


def build_python_spec(
    name: str,
    source: str,
    entry_filename: str,
    dependencies: Optional[List[str]] = None,
    description: str = "",
) -> PythonRepoSpec:
    project_name = sanitize_project_name(name)
    module_name = _derive_module_name_from_source(source, project_name)
    deps = infer_import_dependencies(source, dependencies)
    return PythonRepoSpec(
        name=project_name,
        entry_filename=choose_entry_filename(entry_filename, project_name, source),
        source=source,
        module_name=module_name,
        description=description or f"Standalone Python project generated by aero-forge for '{project_name}'.",
        dependencies=deps,
    )


def render_python_gitignore() -> str:
    return (
        "# Byte-compiled / cache\n"
        "__pycache__/\n"
        "*.py[cod]\n"
        "*$py.class\n\n"
        "# Virtual environments\n"
        ".venv/\n"
        "venv/\n"
        "env/\n\n"
        "# Packaging / distribution\n"
        "*.egg-info/\n"
        ".eggs/\n"
        "dist/\n"
        "build/\n\n"
        "# Local project cache streams\n"
        ".pytest_cache/\n"
        ".mypy_cache/\n"
        ".ruff_cache/\n"
        ".aero/\n"
        "build_artifacts/\n"
    )


def render_pyproject(spec: PythonRepoSpec) -> str:
    deps_block = ""
    if spec.dependencies:
        lines = ",\n".join(f'    "{dep}"' for dep in spec.dependencies)
        deps_block = f"\ndependencies = [\n{lines},\n]\n"
    return (
        "[build-system]\n"
        'requires = ["setuptools>=61"]\n'
        'build-backend = "setuptools.build_meta"\n\n'
        "[project]\n"
        f'name = "{spec.name}"\n'
        f'version = "{spec.version}"\n'
        f'description = "{spec.description}"\n'
        'readme = "README.md"\n'
        'requires-python = ">=3.9"\n'
        f"{deps_block}\n"
        "[tool.setuptools]\n"
        'package-dir = {"" = "src"}\n'
        f'packages = ["{spec.module_name}"]\n'
    )


def render_requirements(spec: PythonRepoSpec) -> str:
    if not spec.dependencies:
        return "# No third-party imports detected.\n"
    return "\n".join(spec.dependencies) + "\n"


def render_python_readme(spec: PythonRepoSpec) -> str:
    deps = ", ".join(f"`{d}`" for d in spec.dependencies) or "(none detected)"
    return (
        f"# {spec.name}\n\n"
        f"{spec.description}\n\n"
        "> Generated by **aero-forge** as a standalone, out-of-tree Python workspace.\n\n"
        "## Entry point\n\n"
        f"Run the project with:\n\n"
        f"```bash\npython {spec.entry_filename}\n```\n\n"
        "## Dependencies\n\n"
        f"{deps}\n\n"
        "## Validate\n\n"
        "```bash\npython -m compileall .\n```\n"
    )


def _render_python_tests(spec: PythonRepoSpec) -> str:
    """Generate a minimal pytest file that imports the generated module."""
    names = _detect_public_names(spec.source)
    module = spec.module_name
    if not names:
        return f"import {module}\n\ndef test_imports():\n    assert {module}\n"
    imports = f"from {module} import {', '.join(names)}\n\n"
    body = "\n\n".join(
        f"def test_{_sanitize_module_name(name)}_imports():\n    assert {name} is not None"
        for name in names
    )
    return imports + body + "\n"


def generate_python_repo(spec: PythonRepoSpec, dest_dir: Path) -> PythonGeneratedRepo:
    """Write the full standalone Python project layout under ``dest_dir``."""
    root = Path(dest_dir)
    root.mkdir(parents=True, exist_ok=True)
    written: List[str] = []

    def _write(relative: str, content: str) -> None:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        written.append(relative)

    entry = spec.entry_filename
    source = spec.source if spec.source.endswith("\n") else spec.source + "\n"
    module = spec.module_name
    _write(entry, source)
    _write("pyproject.toml", render_pyproject(spec))
    _write("requirements.txt", render_requirements(spec))
    _write(".gitignore", render_python_gitignore())
    _write("README.md", render_python_readme(spec))
    _write(f"tests/test_{module}.py", _render_python_tests(spec))

    # Expose public functions/classes from the generated module at package root.
    names = _detect_public_names(spec.source)
    if names:
        init_lines = [f"from .{module} import {', '.join(names)}", ""]
        init_lines.append("__all__ = [" + ", ".join(f'"{n}"' for n in names) + "]")
    else:
        init_lines = ["# Generated Aero-Forge module"]
    _write("src/__init__.py", "\n".join(init_lines) + "\n")

    return PythonGeneratedRepo(root=root, files=written, spec=spec)
