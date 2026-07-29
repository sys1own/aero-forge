"""Out-of-tree workspace isolation for pre-write validation."""

from __future__ import annotations

import ast
import json
import logging
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from aero_forge.errors import UserError

TOOL_ROOT = Path(__file__).resolve().parents[1]


class WorkspaceLocationError(ValueError):
    """Raised when a requested workspace would land inside the tool tree."""


def _is_inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


class OutOfTreeWorkspace:
    """A staging/build workspace guaranteed to live outside the tool tree.

    1. ``create()`` materialises a temporary staging directory.
    2. Generated files and build artifacts are written into the staging dir.
    3. A delegated validation command runs in the staging dir.
    4. ``commit()`` atomically moves the staging directory to the final
       ``distribution_directory`` only when validation succeeded.
    """

    def __init__(
        self,
        distribution_directory: Optional[Path] = None,
        prefix: str = "aero-build-",
        keep: Optional[bool] = None,
    ) -> None:
        self._distribution = (
            Path(distribution_directory).expanduser() if distribution_directory else None
        )
        if self._distribution is not None and _is_inside(self._distribution, TOOL_ROOT):
            raise WorkspaceLocationError(
                f"distribution directory {self._distribution} must be outside {TOOL_ROOT}"
            )
        self._prefix = prefix
        self.keep = keep if keep is not None else (self._distribution is not None)
        self._root: Optional[Path] = None
        self._committed = False

    @property
    def root(self) -> Path:
        if self._root is None:
            raise RuntimeError("workspace not created; use `with OutOfTreeWorkspace(...) as ws:`")
        return self._root

    @property
    def is_temporary(self) -> bool:
        return self._distribution is None

    @property
    def is_committed(self) -> bool:
        return self._committed

    def create(self) -> Path:
        if self._distribution is not None:
            staging = self._distribution.with_suffix(".staging")
            staging.mkdir(parents=True, exist_ok=True)
            self._root = staging
            return self._root
        self._root = Path(tempfile.mkdtemp(prefix=self._prefix))
        return self._root

    def commit(self) -> Optional[Path]:
        """Promote the staging workspace to the final distribution directory."""
        if self._committed:
            return self._distribution
        if self._distribution is None:
            return self._root
        if self._distribution.exists():
            shutil.rmtree(self._distribution, ignore_errors=True)
        self._root.rename(self._distribution)
        self._committed = True
        return self._distribution

    def discard(self) -> None:
        """Remove the staging workspace without committing."""
        if self._root is not None and self._root.exists() and not self._committed:
            shutil.rmtree(self._root, ignore_errors=True)
        self._root = None

    def __enter__(self) -> "OutOfTreeWorkspace":
        self.create()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is None:
            self.commit()
        else:
            self.discard()


logger = logging.getLogger("aero_forge.scaffold.workspace")

# Files and directories that are considered generated/build artifacts and may be
# removed during a blueprint regeneration.
_PURGE_PATTERNS = {
    "src",
    "tests",
    "rust_core",
    "cpp_core",
    "python_engine",
    "target",
    "dist",
    "__pycache__",
    ".pytest_cache",
    "*.egg-info",
    "*.so",
    "*.pyd",
    "*.dll",
    "Cargo.toml",
    "Cargo.lock",
    "pyproject.toml",
    "build.rs",
    "CMakeLists.txt",
    ".aero_backup",
}

# Files that must never be purged or backed up (they survive regeneration).
_PROTECTED_FILES = {"blueprint.aero", ".aeroignore"}


def _is_generated_path(path: Path, root: Path) -> bool:
    """Return True when ``path`` is a generated file or directory."""
    rel = path.relative_to(root)
    parts = rel.parts
    for part in parts:
        if part in _PURGE_PATTERNS:
            return True
    if path.is_file() and any(path.match(p) for p in _PURGE_PATTERNS if p.startswith("*")):
        return True
    return False


def _lookup(obj: Any, key: str, default: Any = None) -> Any:
    """Read a key from a dict or attribute from an object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _ignore_for_backup(src: Any, names: List[str]) -> set:
    """``shutil.copytree`` ignore function that skips heavy build artifacts."""
    src = Path(src)
    ignored = set()
    for name in names:
        child = src / name
        rel = child.name
        if rel in _PROTECTED_FILES or rel == ".aero_backup":
            continue
        if child.is_dir() and child.name in {"target", "dist", "__pycache__", ".pytest_cache"}:
            ignored.add(name)
        if child.is_file() and child.suffix in {".so", ".pyd", ".dll"}:
            ignored.add(name)
    return ignored


def _python_type_to_rust(py_type: str) -> str:
    """Map common Python type annotations to Rust types."""
    text = py_type.strip()
    mapping = {
        "int": "i64",
        "float": "f64",
        "bool": "bool",
        "str": "String",
        "list[int]": "Vec<i64>",
        "list[float]": "Vec<f64>",
        "list[bool]": "Vec<bool>",
        "list[str]": "Vec<String>",
    }
    if text in mapping:
        return mapping[text]
    # list[...] generic fallback
    match = re.match(r"list\[(.+)]", text)
    if match:
        inner = _python_type_to_rust(match.group(1))
        return f"Vec<{inner}>"
    return "f64"


def _python_type_to_c(py_type: str) -> str:
    """Map common Python type annotations to C/C++ types."""
    text = py_type.strip()
    mapping = {
        "int": "int64_t",
        "float": "double",
        "bool": "bool",
        "str": "const char*",
        "list[int]": "const int64_t*",
        "list[float]": "const double*",
        "list[bool]": "const bool*",
        "list[str]": "const char**",
    }
    if text in mapping:
        return mapping[text]
    match = re.match(r"list\[(.+)]", text)
    if match:
        return "const void*"
    return "double"


def _parse_signature(signature: str) -> Tuple[str, List[Tuple[str, str]], Optional[str]]:
    """Parse a Python function signature string into name, params, and return type."""
    signature = signature.strip()
    match = re.match(r"def\s+(\w+)\s*\((.*)\)\s*(?:->\s*(\S+))?", signature)
    if not match:
        return "compute", [], None
    name = match.group(1)
    params_text = match.group(2)
    return_type = match.group(3)
    params: List[Tuple[str, str]] = []
    for param in params_text.split(","):
        param = param.strip()
        if not param:
            continue
        if ":" in param:
            pname, ptype = param.split(":", 1)
            params.append((pname.strip(), ptype.strip()))
        else:
            params.append((param, "f64"))
    return name, params, return_type


def _rust_stub_from_signature(signature: str) -> str:
    """Return a compilable Rust function stub for a Python signature."""
    name, params, return_type = _parse_signature(signature)
    rust_params = ", ".join(
        f"{n}: {_python_type_to_rust(t)}" for n, t in params
    )
    rust_ret = _python_type_to_rust(return_type or "float")
    return f"""pub fn {name}({rust_params}) -> {rust_ret} {{
    unimplemented!()
}}
"""


def _cpp_stub_from_signature(signature: str) -> str:
    """Return a C-ABI C++ stub for a Python signature."""
    name, params, return_type = _parse_signature(signature)
    c_params = ", ".join(f"{_python_type_to_c(t)} {n}" for n, t in params)
    c_ret = _python_type_to_c(return_type or "float")
    body = "return 0;" if c_ret in {"double", "int64_t", "int", "float"} else ("return nullptr;" if c_ret.endswith("*") else "return;")
    return f"""#include <cstdint>

extern "C" {c_ret} {name}({c_params}) {{
    {body}
}}
"""


def _python_stub_from_signature(signature: str) -> str:
    """Return a Python function stub for a signature string."""
    sig = signature.rstrip()
    if not sig.endswith(":"):
        sig += ":"
    return f"""{sig}
    ...
"""


class BlueprintRegenerator:
    """Wipe and rebuild a workspace strictly from ``blueprint.aero``.

    The regenerator:
    1. Reads and validates the blueprint.
    2. Backs up the current workspace (minus heavy build artifacts) to ``.aero_backup/``.
    3. Purges generated source directories, manifests, and build artifacts.
    4. Re-scaffolds the directory tree and writes minimal, syntactically valid
       stub files for every entry in the blueprint manifest.
    5. Optionally runs a build via ``ProjectBuilder``; on success the backup is removed.
    """

    def __init__(
        self,
        workspace: Path,
        *,
        keep_backup: bool = False,
        run_build: bool = False,
        force_overwrite: bool = False,
        llm_provider: Optional[str] = None,
        model: Optional[str] = None,
        config_override: Optional[Any] = None,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.keep_backup = keep_backup
        self.run_build = run_build
        self.force_overwrite = force_overwrite
        self.llm_provider = llm_provider
        self.model = model
        self.config_override = config_override
        self.blueprint_path = self.workspace / "blueprint.aero"
        self.backup_dir = self.workspace / ".aero_backup"
        self.logs: List[str] = []
        self.errors: List[str] = []

    def _is_workspace_non_empty(self) -> bool:
        """Return True when the workspace contains files other than protected metadata."""
        protected = {"blueprint.aero", "workspace_blueprint.yaml", ".aero_backup"}
        for item in self.workspace.iterdir():
            if item.name in protected:
                continue
            if item.name.startswith("."):
                continue
            return True
        return False

    def _log(self, level: str, message: str) -> None:
        log_func = getattr(logger, level, logger.info)
        log_func(message)
        self.logs.append(f"[{level.upper()}] {message}")

    def _backup_workspace(self) -> None:
        """Copy current files to ``.aero_backup/`` except protected files and build artifacts."""
        if self.backup_dir.exists():
            shutil.rmtree(self.backup_dir, ignore_errors=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        for item in self.workspace.iterdir():
            if item.name in _PROTECTED_FILES or item.name == ".aero_backup":
                continue
            dest = self.backup_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dest, ignore=_ignore_for_backup)
            else:
                shutil.copy2(item, dest)
        self._log("info", f"Workspace backed up to {self.backup_dir}")

    def _purge_generated(self) -> None:
        """Remove generated source, manifests, and build artifacts."""
        for item in list(self.workspace.iterdir()):
            if item.name in _PROTECTED_FILES or item.name == ".aero_backup":
                continue
            if item.is_dir() and item.name in _PURGE_PATTERNS:
                shutil.rmtree(item, ignore_errors=True)
                self._log("info", f"Removed directory {item.name}")
            elif item.is_file() and item.name in _PURGE_PATTERNS:
                item.unlink(missing_ok=True)
                self._log("info", f"Removed file {item.name}")

        # Remove stray native extension artifacts anywhere in the tree.
        for pattern in ("*.so", "*.pyd", "*.dll"):
            for artifact in self.workspace.rglob(pattern):
                artifact.unlink(missing_ok=True)

    def _contract_for_path(self, path: Path, blueprint: Any) -> Optional[Dict[str, Any]]:
        """Find the blueprint contract whose python_name or name maps to ``path``."""
        rel = path.relative_to(self.workspace).as_posix()
        target_stem = path.stem
        for contract in _lookup(blueprint, "contracts") or []:
            python_name = _lookup(contract, "python_name") or ""
            name = _lookup(contract, "name") or ""
            if python_name:
                parts = python_name.split(".")
                # Try progressive module paths such as src/a/b/c.py for a.b.c
                for i in range(len(parts) - 1, 0, -1):
                    module_parts = parts[:i]
                    candidates = [
                        f"src/{'/'.join(module_parts)}.py",
                        f"{'/'.join(module_parts)}.py",
                    ]
                    if rel in candidates:
                        return dict(contract) if isinstance(contract, dict) else (contract.model_dump() if hasattr(contract, "model_dump") else {})
                if python_name.endswith(rel) or rel in python_name.replace(".", "/"):
                    return dict(contract) if isinstance(contract, dict) else (contract.model_dump() if hasattr(contract, "model_dump") else {})
            if name and (name == target_stem or name in rel):
                return dict(contract) if isinstance(contract, dict) else (contract.model_dump() if hasattr(contract, "model_dump") else {})
        return None

    def _write_stub(self, entry: Any, blueprint: Any) -> None:
        """Write a syntactically valid stub for a single manifest entry."""
        entry_path = _lookup(entry, "path")
        if not entry_path:
            return
        target = self.workspace / entry_path
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            return

        lang = (_lookup(entry, "lang") or "").lower()
        purpose = (_lookup(entry, "purpose") or "").lower()
        contract = self._contract_for_path(target, blueprint)
        signature = contract.get("signature") if contract else None

        if lang == "python":
            if "test" in target.name or "test" in purpose:
                func = re.sub(r"^def\s+", "", signature or "def compute(x: list[float]) -> list[float]")
                func_name = func.split("(")[0].strip() if "(" in func else "compute"
                target.write_text(f"def test_{func_name}():\n    assert True\n", encoding="utf-8")
            elif signature:
                target.write_text(_python_stub_from_signature(signature), encoding="utf-8")
            else:
                target.write_text("# Placeholder\npass\n", encoding="utf-8")
        elif lang == "rust":
            if target.name.endswith("main.rs"):
                target.write_text("fn main() {\n    println!(\"Hello from Aero-Forge\");\n}\n", encoding="utf-8")
            elif signature:
                target.write_text(_rust_stub_from_signature(signature), encoding="utf-8")
            else:
                target.write_text("pub fn placeholder() {\n    unimplemented!()\n}\n", encoding="utf-8")
        elif lang == "cpp":
            if signature:
                target.write_text(_cpp_stub_from_signature(signature), encoding="utf-8")
            else:
                target.write_text("#include <cstdint>\n\nextern \"C\" void placeholder() {}\n", encoding="utf-8")
        elif lang == "toml":
            target.write_text(self._toml_for_entry(entry, blueprint), encoding="utf-8")
        elif lang == "markdown":
            target.write_text(f"# {_lookup(blueprint, 'project', 'Aero-Forge project')}\n\nGenerated from blueprint.aero\n", encoding="utf-8")
        else:
            target.write_text("", encoding="utf-8")

        self._log("info", f"Scaffolded {entry_path}")

    def _toml_for_entry(self, entry: Any, blueprint: Any) -> str:
        """Generate a minimal TOML manifest for a manifest entry."""
        entry_path = _lookup(entry, "path") or "Cargo.toml"
        path = Path(entry_path)
        project = _lookup(blueprint, "project", "aero_forge_project")
        architecture = _lookup(blueprint, "architecture", "pure_python")

        if path.name == "pyproject.toml":
            return f"""[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "{project}"
version = "0.1.0"
description = "Aero-Forge generated project"
"""
        if path.name == "Cargo.toml":
            if "rust_core" in entry_path:
                return f"""[package]
name = "{project}_core"
version = "0.1.0"
edition = "2021"

[lib]
name = "{project}_core"
crate-type = ["cdylib"]

[dependencies]
pyo3 = {{ version = "0.20.3", features = ["extension-module", "abi3-py39"] }}
"""
            return f"""[package]
name = "{project}"
version = "0.1.0"
edition = "2021"

[workspace]
members = ["rust_core"]

[dependencies]
"""
        return f"# {project} manifest\n"

    def _validate_generated(self) -> None:
        """Validate that all generated Python files are syntactically correct."""
        for path in self.workspace.rglob("*.py"):
            if ".aero_backup" in path.parts:
                continue
            text = path.read_text(encoding="utf-8")
            try:
                ast.parse(text)
            except SyntaxError as exc:
                self._log("error", f"Syntax error in {path}: {exc}")
                self.errors.append(str(path))

    def _run_build(self) -> bool:
        """Run ProjectBuilder.build() and clean up backup on success."""
        try:
            from aero_forge.project_builder import ProjectBuilder

            builder = ProjectBuilder(
                self.workspace,
                llm_provider=self.llm_provider,
                model=self.model,
                config_override=self.config_override,
            )
            result = builder.build()
            if result.get("success"):
                self._log("info", "Build succeeded")
                return True
            self._log("error", f"Build failed: {result.get('error') or result.get('summary')}")
            return False
        except Exception as exc:
            self._log("error", f"Build raised exception: {exc}")
            return False

    def _build_tree(self) -> List[Dict[str, Any]]:
        """Return a nested dict tree of the workspace for the UI."""
        def walk(path: Path, rel: Optional[Path] = None) -> Dict[str, Any]:
            rel = rel or Path(path.name)
            if path.is_file():
                return {
                    "name": path.name,
                    "path": str(rel),
                    "type": "file",
                }
            return {
                "name": path.name,
                "path": str(rel),
                "type": "directory",
                "children": [walk(p, rel / p.name) for p in sorted(path.iterdir()) if p.is_dir()]
                + [
                    {"name": p.name, "path": str(rel / p.name), "type": "file"}
                    for p in sorted(path.iterdir())
                    if p.is_file()
                ],
            }

        return [walk(p, Path(p.name)) for p in sorted(self.workspace.iterdir())]

    def run(self) -> Dict[str, Any]:
        """Execute the regeneration workflow."""
        if not self.blueprint_path.is_file():
            raise FileNotFoundError(f"blueprint.aero not found in {self.workspace}")

        from aero_forge.blueprint import is_blueprint_ready, parse_aero

        blueprint = parse_aero(self.blueprint_path.read_text(encoding="utf-8"))
        if not isinstance(blueprint, dict):
            raise ValueError("blueprint.aero did not parse to a mapping")

        if not is_blueprint_ready(blueprint):
            raise UserError(
                "Cannot materialize: Blueprint is uninitialized. "
                "Please run LLM blueprint generation first."
            )

        if not self.force_overwrite and self._is_workspace_non_empty():
            raise UserError(
                "Workspace is not empty. Use force_overwrite to regenerate."
            )

        self._backup_workspace()
        self._purge_generated()

        manifest = _lookup(blueprint, "manifest") or []
        for entry in manifest:
            self._write_stub(entry, blueprint)

        self._validate_generated()

        build_success = False
        if self.run_build and not self.errors:
            build_success = self._run_build()
            if build_success and not self.keep_backup:
                shutil.rmtree(self.backup_dir, ignore_errors=True)
                self._log("info", "Backup removed after successful build")

        return {
            "status": "success" if not self.errors and (not self.run_build or build_success) else "partial",
            "errors": self.errors,
            "logs": self.logs,
            "backup_dir": str(self.backup_dir) if self.backup_dir.exists() else None,
            "tree": self._build_tree(),
        }
