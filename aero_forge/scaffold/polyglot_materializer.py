"""Physical file materialization for polyglot blueprints.

The materializer takes a ``Blueprint`` that declares a Rust core, Python
orchestrator, tests, and entry points, and writes every missing file to disk.
Rust source is generated through the existing PyO3 transpiler, while Python
files are synthesised from the contract signatures.
"""

from __future__ import annotations

import ast
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from aero_forge.blueprint import (
    Blueprint,
    ContractEntry,
    FunctionSpec,
    ManifestEntry,
    write_blueprint,
)
from aero_forge.scaffold.cargo_manifest import sanitize_crate_name
from aero_forge.scaffold.engine import Engine
from aero_forge.scaffold.python_repo_generator import _sanitize_module_name
from aero_forge.translator import TargetMode, UASTToHINTranslator, python_source_to_uast

logger = logging.getLogger("aero_forge.scaffold.polyglot")


_DEFAULT_CONTRACTS = [
    ContractEntry(
        name="fast_vector_transform",
        signature="def fast_vector_transform(v: list[float], scalar: float) -> list[float]",
        language="python/rust",
        python_name="fast_vector_transform",
        purpose="Vector transformation exposed via PyO3",
    ),
    ContractEntry(
        name="get_engine_status",
        signature="def get_engine_status() -> dict[str, str]",
        language="python/rust",
        python_name="get_engine_status",
        purpose="Engine health/status metadata",
    ),
]


def _annotation_to_str(node: Optional[ast.AST]) -> str:
    if node is None:
        return "None"
    try:
        return ast.unparse(node)
    except Exception:
        return "Any"


def _parse_signature(signature: str) -> Tuple[str, List[Tuple[str, str]], str]:
    """Parse ``signature`` into (function_name, [(arg, type)], return_type)."""
    source = signature.strip()
    if not source.endswith(":"):
        source = source + ":\n    pass"
    else:
        source = source + "\n    pass"
    tree = ast.parse(source)
    func = tree.body[0]
    if not isinstance(func, ast.FunctionDef):
        raise ValueError(f"Invalid signature: {signature!r}")
    args = [(arg.arg, _annotation_to_str(arg.annotation)) for arg in func.args.args]
    return_type = _annotation_to_str(func.returns)
    return func.name, args, return_type


def _generate_stub_body(name: str, args: List[Tuple[str, str]], return_type: str) -> str:
    """Return a body for a stub Python implementation of *name*."""
    rt = return_type.lower()

    if name == "fast_vector_transform":
        list_arg = next((a for a in args if "list" in a[1].lower()), None)
        scalar_arg = next(
            (a for a in args if a[1].lower() in ("float", "int", "f64", "i64")),
            None,
        )
        if list_arg and scalar_arg:
            return f"    return [x * {scalar_arg[0]} for x in {list_arg[0]}]"

    if name == "get_engine_status" or ("dict" in rt and "status" in name.lower()):
        return '    return {"status": "ok", "engine": "polyglot"}'

    if "list" in rt:
        list_arg = next((a for a in args if "list" in a[1].lower()), None)
        if list_arg:
            return f"    return list({list_arg[0]})"
        return "    return []"

    if "dict" in rt:
        return '    return {"status": "ok"}'

    if rt in ("int", "i64", "i32"):
        return "    return 0"

    if rt in ("float", "f64", "f32"):
        return "    return 0.0"

    if rt == "bool":
        return "    return True"

    if rt == "str":
        return '    return "ok"'

    if rt in ("none", "nonetype"):
        return "    return None"

    return "    return None"


def _synthesize_python_source(contracts: List[ContractEntry]) -> str:
    """Build a stub Python module from contract signatures."""
    lines: List[str] = ["from __future__ import annotations", ""]
    names: List[str] = []
    for contract in contracts:
        if not contract.signature:
            continue
        try:
            name, args, return_type = _parse_signature(contract.signature)
        except Exception:
            logger.warning("Could not parse contract signature: %s", contract.signature)
            continue
        names.append(name)
        arg_str = ", ".join(f"{arg}: {typ}" for arg, typ in args)
        lines.append(f"def {name}({arg_str}) -> {return_type}:")
        lines.append(_generate_stub_body(name, args, return_type))
        lines.append("")

    if names:
        lines.append("__all__ = [" + ", ".join(f'"{n}"' for n in names) + "]")
    else:
        lines.append('__all__: list[str] = []')

    return "\n".join(lines) + "\n"


def _native_loader_source(crate_names: List[str]) -> str:
    """Return module-level code that searches for and loads a compiled .so."""
    lines = [
        "import importlib.util",
        "import pathlib",
        "from typing import Any, Optional",
        "",
        "_SO_CANDIDATES = [",
        '    pathlib.Path(__file__).parent.parent / "rust_core" / "target" / "release",',
        '    pathlib.Path(__file__).parent.parent / "target" / "release",',
        '    pathlib.Path(__file__).parent.parent / "dist",',
        '    pathlib.Path(__file__).parent,',
        "]",
        "",
        f"_PREFERRED_MODULE_NAMES = {crate_names!r}",
        "",
        "",
        "def _load_native() -> Optional[Any]:",
        "    for directory in _SO_CANDIDATES:",
        "        if not directory.is_dir():",
        "            continue",
        '        for so in sorted(directory.glob("*.so")):',
        "            stem = so.stem",
        '            if stem.startswith("lib"):',
        '                stem = stem[3:]',
        "            for preferred in _PREFERRED_MODULE_NAMES:",
        "                if preferred in stem:",
        "                    stem = preferred",
        "                    break",
        "            try:",
        "                spec = importlib.util.spec_from_file_location(stem, so)",
        "                mod = importlib.util.module_from_spec(spec)",
        "                spec.loader.exec_module(mod)",
        "            except Exception:",
        "                continue",
        "            return mod",
        "    return None",
        "",
        "",
        "_NATIVE: Optional[Any] = _load_native()",
    ]
    return "\n".join(lines) + "\n"


def _function_impl(name: str, args: List[Tuple[str, str]], return_type: str) -> str:
    """Return a single Python function that delegates to ``_NATIVE`` or falls back."""
    arg_sig = ", ".join(f"{a}: {t}" for a, t in args)
    arg_call = ", ".join(a for a, _ in args)
    fallback = _generate_stub_body(name, args, return_type).strip()
    return (
        f"def {name}({arg_sig}) -> {return_type}:\n"
        f"    if _NATIVE is not None and hasattr(_NATIVE, \"{name}\"):\n"
        f"        return _NATIVE.{name}({arg_call})\n"
        f"{fallback}\n"
    )


def _render_python_module(contracts: List[ContractEntry], module_name: str) -> str:
    """Render a Python module that delegates to a compiled Rust extension."""
    lines: List[str] = [
        "from __future__ import annotations",
        "",
        _native_loader_source([module_name]),
    ]
    for contract in contracts:
        if not contract.signature:
            continue
        try:
            name, args, return_type = _parse_signature(contract.signature)
        except Exception:
            continue
        lines.append(_function_impl(name, args, return_type))
        lines.append("")
    names = [c.name for c in contracts if c.signature]
    if names:
        lines.append("__all__ = [" + ", ".join(f'"{n}"' for n in names) + "]")
    else:
        lines.append('__all__: list[str] = []')
    return "\n".join(lines) + "\n"


def _render_orchestrator(contracts: List[ContractEntry], module_name: str) -> str:
    """Render ``aero_polyglot_runner/orchestrator.py`` with ``PolyglotEngine``."""
    lines: List[str] = [
        '"""Polyglot runner that loads the Rust extension with a pure-Python fallback."""',
        "",
        "from __future__ import annotations",
        "",
        "import importlib.util",
        "import pathlib",
        "from typing import Any, Dict, List, Optional",
        "",
        "_SO_CANDIDATES = [",
        '    pathlib.Path(__file__).parent.parent / "rust_core" / "target" / "release",',
        '    pathlib.Path(__file__).parent.parent / "target" / "release",',
        '    pathlib.Path(__file__).parent.parent / "dist",',
        '    pathlib.Path(__file__).parent,',
        "]",
        "",
        f"_PREFERRED_MODULES = {[module_name]!r}",
        "",
        "",
        "class PolyglotEngine:",
        '    """Loads the compiled Rust extension or falls back to pure Python."""',
        "",
        "    def __init__(self) -> None:",
        "        self._native: Optional[Any] = self._load_native()",
        "",
        "    @property",
        '    def backend(self) -> str:',
        '        return "rust" if self._native is not None else "python"',
        "",
        "    def _load_native(self) -> Optional[Any]:",
        "        for directory in _SO_CANDIDATES:",
        "            if not directory.is_dir():",
        "                continue",
        '            for so in sorted(directory.glob("*.so")):',
        "                stem = so.stem",
        '                if stem.startswith("lib"):',
        '                    stem = stem[3:]',
        "                for preferred in _PREFERRED_MODULES:",
        "                    if preferred in stem:",
        "                        stem = preferred",
        "                        break",
        "                try:",
        "                    spec = importlib.util.spec_from_file_location(stem, so)",
        "                    mod = importlib.util.module_from_spec(spec)",
        "                    spec.loader.exec_module(mod)",
        "                except Exception:",
        "                    continue",
        "                return mod",
        "        return None",
    ]

    for contract in contracts:
        if not contract.signature:
            continue
        try:
            name, args, return_type = _parse_signature(contract.signature)
        except Exception:
            continue
        typed_args = ", ".join(f"{a}: {t}" for a, t in args)
        arg_sig = f"self, {typed_args}" if typed_args else "self"
        arg_call = ", ".join(a for a, _ in args)
        fallback = _generate_stub_body(name, args, return_type).strip()
        lines.append("")
        lines.append(f"    def {name}({arg_sig}) -> {return_type}:")
        lines.append(f"        if self._native is not None and hasattr(self._native, \"{name}\"):")
        lines.append(f"            return self._native.{name}({arg_call})")
        lines.append(f"        {fallback}")

    return "\n".join(lines) + "\n"


def _render_init(module: str, exports: List[str]) -> str:
    if not exports:
        return "# Generated by aero-forge\n"
    lines = [f"from .{module} import {', '.join(exports)}"]
    lines.append("__all__ = [" + ", ".join(f'"{e}"' for e in exports) + "]")
    return "\n".join(lines) + "\n"


def _render_demo(exports_module: str, contracts: List[ContractEntry]) -> str:
    names = [c.name for c in contracts if c.signature]
    if not names:
        return "if __name__ == \"__main__\":\n    pass\n"
    if exports_module == "aero_polyglot_runner.orchestrator":
        lines = [
            "from aero_polyglot_runner.orchestrator import PolyglotEngine",
            "",
            "def main() -> None:",
            "    engine = PolyglotEngine()",
        ]
        for name in names:
            if name == "fast_vector_transform":
                lines.append(f"    print(engine.{name}([1.0, 2.0, 3.0], 2.0))")
            elif name == "get_engine_status":
                lines.append(f"    print(engine.{name}())")
            else:
                lines.append(f"    print(engine.{name}())")
        lines.extend(["", 'if __name__ == "__main__":', "    main()"])
        return "\n".join(lines) + "\n"

    lines = [
        f"from {exports_module} import {', '.join(names)}",
        "",
        "def main() -> None:",
    ]
    for name in names:
        if name == "fast_vector_transform":
            lines.append(f"    print({name}([1.0, 2.0, 3.0], 2.0))")
        elif name == "get_engine_status":
            lines.append(f"    print({name}())")
        else:
            lines.append(f"    print({name}())")
    lines.extend(["", 'if __name__ == "__main__":', "    main()"])
    return "\n".join(lines) + "\n"


def _render_tests(exports_module: str, contracts: List[ContractEntry]) -> str:
    names = [c.name for c in contracts if c.signature]
    if not names:
        return "def test_placeholder():\n    pass\n"

    if exports_module == "aero_polyglot_runner.orchestrator":
        lines = [
            "from aero_polyglot_runner.orchestrator import PolyglotEngine",
            "",
            "def test_engine_instantiates():",
            "    assert PolyglotEngine() is not None",
            "",
        ]
    else:
        lines = [
            f"from {exports_module} import {', '.join(names)}",
            "",
        ]

    for contract in contracts:
        if not contract.signature:
            continue
        try:
            name, args, return_type = _parse_signature(contract.signature)
        except Exception:
            continue

        uses_engine = exports_module == "aero_polyglot_runner.orchestrator"
        prefix = "engine." if uses_engine else ""
        setup = "engine = PolyglotEngine()" if uses_engine else ""

        if name == "fast_vector_transform":
            lines.extend([
                f"def test_{name}():",
                f"    {setup}",
                f"    result = {prefix}{name}([1.0, 2.0, 3.0], 2.0)",
                "    assert isinstance(result, list)",
                "    assert result == [2.0, 4.0, 6.0]",
                "",
            ])
        elif name == "get_engine_status":
            lines.extend([
                f"def test_{name}():",
                f"    {setup}",
                f"    status = {prefix}{name}()",
                "    assert isinstance(status, dict)",
                '    assert status.get("status") == "ok"',
                "",
            ])
        else:
            arg_values = []
            for _, t in args:
                if "list" in t.lower():
                    arg_values.append("[1.0, 2.0]")
                elif t.lower() in ("float", "f64"):
                    arg_values.append("1.0")
                elif t.lower() in ("int", "i64"):
                    arg_values.append("1")
                elif t.lower() == "str":
                    arg_values.append('"x"')
                else:
                    arg_values.append("None")
            lines.extend([
                f"def test_{name}():",
                f"    {setup}",
                f"    result = {prefix}{name}({', '.join(arg_values)})",
                "    assert result is not None",
                "",
            ])

    return "\n".join(lines) + "\n"


def _render_pyproject(pkg_name: str, package_dir: str = ".") -> str:
    lines = [
        "[build-system]",
        'requires = ["setuptools>=61"]',
        'build-backend = "setuptools.build_meta"',
        "",
        "[project]",
        f'name = "{pkg_name}"',
        'version = "0.1.0"',
        'requires-python = ">=3.9"',
        "",
        "[tool.setuptools]",
        f'package-dir = {{"" = "{package_dir}"}}',
        f'packages = ["{pkg_name}"]',
    ]
    return "\n".join(lines) + "\n"


def _render_readme(project: str) -> str:
    lines = [
        f"# {project}",
        "",
        "Aero-Forge generated Python-Rust polyglot workspace.",
        "",
        "## Build",
        "",
        "```bash",
        "cargo build --release",
        "python -m pytest tests -q",
        "```",
    ]
    return "\n".join(lines) + "\n"


def _find_manifest_entries(blueprint: Blueprint) -> Dict[str, ManifestEntry]:
    """Index manifest entries by lowercased basename for quick lookup."""
    return {Path(e.path).name.lower(): e for e in blueprint.manifest}


def _sibling_module_for_init(init_entry: ManifestEntry, blueprint: Blueprint) -> Optional[str]:
    """Return a sibling .py module name that the package ``__init__.py`` can re-export."""
    init_dir = Path(init_entry.path).parent
    for e in blueprint.manifest:
        p = Path(e.path)
        if (
            p.parent == init_dir
            and p.suffix == ".py"
            and p.name != "__init__.py"
            and not p.name.startswith("test_")
        ):
            return p.stem
    return None


class PolyglotMaterializer:
    """Write the files declared by a polyglot ``Blueprint`` to disk."""

    def __init__(self, workspace_root: Path) -> None:
        self.workspace = Path(workspace_root)
        self.workspace.mkdir(parents=True, exist_ok=True)

    def materialize(
        self,
        blueprint: Blueprint,
        *,
        build: bool = False,
    ) -> Blueprint:
        """Create every missing file declared in *blueprint* and return the updated blueprint."""
        project = blueprint.project or "polyglot_project"
        crate_name = sanitize_crate_name(project)
        pkg_name = _sanitize_module_name(project)

        contracts = list(blueprint.contracts) if blueprint.contracts else list(_DEFAULT_CONTRACTS)
        source = _synthesize_python_source(contracts)

        # 1. Write Python / packaging files from the manifest first so the Rust
        #    generator's manifest validation sees a complete workspace.
        self._write_python_files(blueprint, project, crate_name, pkg_name, contracts)

        # 2. Persist the blueprint so Engine.generate can find it.
        blueprint_path = self.workspace / "blueprint.aero"
        write_blueprint(blueprint, blueprint_path)

        # 3. Generate the Rust crate via the PyO3 transpiler.  Engine.generate
        #    copies Cargo.toml/src/lib.rs into the directories declared by the
        #    manifest and validates every declared file.
        uast = python_source_to_uast(source)
        graph = UASTToHINTranslator().translate(uast)
        graph.traits_by_name = {}
        graph.traits = {}
        Engine().generate(
            graph,
            self.workspace / "dist",
            workspace_root=self.workspace,
            module_name=crate_name,
            function_names=[c.name for c in contracts],
            source=source,
            target_mode=TargetMode.PYO3,
        )

        # 4. Compile if requested.
        if build:
            self._build_crates()

        # 5. Populate blueprint.functions with concrete file references.
        updated = self._blueprint_with_functions(blueprint, contracts, pkg_name)
        write_blueprint(updated, blueprint_path)
        return updated

    def _write_python_files(
        self,
        blueprint: Blueprint,
        project: str,
        crate_name: str,
        pkg_name: str,
        contracts: List[ContractEntry],
    ) -> None:
        """Generate missing Python, TOML, and README files declared in the manifest."""
        for entry in blueprint.manifest:
            path = self.workspace / entry.path
            if path.exists():
                continue

            content: Optional[str] = None
            rel = Path(entry.path)

            if entry.lang == "python" and path.name == "orchestrator.py":
                content = _render_orchestrator(contracts, crate_name)
            elif entry.lang == "python" and path.name == "__init__.py":
                if rel.parts[0] == "aero_polyglot_runner":
                    content = _render_init("orchestrator", ["PolyglotEngine"])
                else:
                    module = _sibling_module_for_init(entry, blueprint) or pkg_name
                    content = _render_init(module, [c.name for c in contracts])
            elif entry.lang == "python" and path.name.endswith(".py"):
                if "test" in path.name:
                    if "aero_polyglot_runner" in rel.parts or rel.parts[0] == "tests":
                        content = _render_tests("aero_polyglot_runner.orchestrator", contracts)
                    else:
                        content = _render_tests(pkg_name, contracts)
                elif path.name == "run_demo.py":
                    if "aero_polyglot_runner" in rel.parts:
                        content = _render_demo("aero_polyglot_runner.orchestrator", contracts)
                    else:
                        content = _render_demo(pkg_name, contracts)
                elif path.name in ("service.py", "bench.py"):
                    content = f"# {path.name} placeholder generated by aero-forge\n"
                else:
                    content = _render_python_module(contracts, crate_name)
            elif entry.lang == "toml" and path.name == "pyproject.toml":
                if rel.parts[0] == "python_engine":
                    content = _render_pyproject(pkg_name, package_dir="src")
                else:
                    content = _render_pyproject(pkg_name, package_dir=".")
            elif entry.lang == "toml" and path.name == "Cargo.toml":
                continue
            elif entry.lang == "markdown":
                content = _render_readme(project)

            if content is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
                logger.info("Synthesised %s", path.relative_to(self.workspace))

    def _build_crates(self) -> None:
        """Run ``cargo build --release`` and copy ``.so`` artefacts to ``dist/``."""
        cargo_tomls = [
            self.workspace / e.path
            for e in self._read_blueprint().manifest
            if Path(e.path).name == "Cargo.toml"
        ]
        if not cargo_tomls:
            return

        root_cargo = next((p for p in cargo_tomls if p.parent == self.workspace), None)
        build_dirs = [root_cargo.parent] if root_cargo else [p.parent for p in cargo_tomls]

        for build_dir in build_dirs:
            if not (build_dir / "src" / "lib.rs").is_file() and not any(
                (build_dir / member / "Cargo.toml").is_file() for member in ("rust_core",)
            ):
                continue
            logger.info("Building Rust crate in %s", build_dir)
            subprocess.run(
                ["cargo", "build", "--release"],
                cwd=build_dir,
                check=False,
                capture_output=True,
            )

        dist = self.workspace / "dist"
        dist.mkdir(parents=True, exist_ok=True)

        target_dirs = set()
        if (self.workspace / "target" / "release").is_dir():
            target_dirs.add(self.workspace / "target" / "release")
        for cargo_toml in cargo_tomls:
            td = cargo_toml.parent / "target" / "release"
            if td.is_dir():
                target_dirs.add(td)

        for td in target_dirs:
            for so in td.glob("*.so"):
                shutil.copy(so, dist / so.name)

    def _read_blueprint(self) -> Blueprint:
        """Read the workspace blueprint if it exists, else return an empty one."""
        from aero_forge.blueprint import parse_blueprint

        path = self.workspace / "blueprint.aero"
        if path.is_file():
            return parse_blueprint(path)
        return Blueprint()

    def _blueprint_with_functions(
        self,
        blueprint: Blueprint,
        contracts: List[ContractEntry],
        pkg_name: str,
    ) -> Blueprint:
        """Return a blueprint whose ``functions`` list references materialised files."""
        by_name = _find_manifest_entries(blueprint)

        lib_entry = by_name.get("lib.rs")
        test_entry = next(
            (e for e in blueprint.manifest if e.path.endswith(".py") and "test" in e.path),
            None,
        )
        orchestrator_entry = next(
            (e for e in blueprint.manifest if "orchestrator" in e.path and e.path.endswith(".py")),
            None,
        )

        functions: List[FunctionSpec] = []

        if lib_entry:
            for contract in contracts:
                tests: List[Path] = []
                if test_entry:
                    tests.append(self.workspace / test_entry.path)
                functions.append(
                    FunctionSpec(
                        file=self.workspace / lib_entry.path,
                        name=contract.name,
                        tests=tests,
                        skip_build=True,
                    )
                )

        if orchestrator_entry:
            functions.append(
                FunctionSpec(
                    file=self.workspace / orchestrator_entry.path,
                    name="PolyglotEngine",
                    tests=[self.workspace / test_entry.path] if test_entry else [],
                    skip_build=True,
                )
            )

        return blueprint.model_copy(update={"functions": functions})
