"""Blueprint parser and validation for multi-function builds."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator

from aero_forge.orchestrator.router import (
    BUILD_INTENT_HYBRID_RUST_PYTHON,
    BUILD_INTENT_PURE_RUST,
    classify_build_intent,
    toolchains_for_intent,
)

logger = logging.getLogger("aero_forge.blueprint")


class FunctionSpec(BaseModel):
    """A single function or a wildcard entry that compiles every public function in ``file``."""

    file: Path
    name: Optional[str] = None
    compile_all: bool = False
    tests: List[Path] = Field(default_factory=list)
    output_name: Optional[str] = None
    compiler_flags: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _resolve(self) -> "FunctionSpec":
        if self.name == "*":
            self.compile_all = True
        if self.compile_all:
            self.name = self.name or "*"
        if not self.compile_all and not self.name:
            raise ValueError(
                "FunctionSpec requires 'name' unless 'compile_all' is true"
            )
        if self.output_name is None:
            self.output_name = self.name
        return self


class LLMConfig(BaseModel):
    """LLM settings inside a blueprint."""

    provider: str = "none"
    model: Optional[str] = None


class ManifestEntry(BaseModel):
    """A file emitted by the generator as part of a workspace."""

    path: str
    lang: str
    purpose: str = ""


class ContractEntry(BaseModel):
    """An exported symbol contract between Rust and Python."""

    name: str
    signature: str = ""
    language: str = "python"
    python_name: str = ""
    purpose: str = ""


class Blueprint(BaseModel):
    """Normalized build blueprint with optional workspace planning metadata."""

    project: str = "aero_forge_project"
    architecture: str = "pure_python"
    toolchains: List[str] = Field(default_factory=list)
    manifest: List[ManifestEntry] = Field(default_factory=list)
    contracts: List[ContractEntry] = Field(default_factory=list)
    functions: List[FunctionSpec] = Field(default_factory=list)
    compiler_flags: List[str] = Field(default_factory=list)
    output_dir: Path = Path("./dist")
    llm: LLMConfig = Field(default_factory=LLMConfig)
    prompt: Optional[str] = None
    constraints: Optional[str] = None

    @model_validator(mode="after")
    def _validate_files(self) -> "Blueprint":
        missing: List[str] = []
        for func in self.functions:
            if not func.file.is_file():
                missing.append(str(func.file))
            for test in func.tests:
                if not test.is_file():
                    missing.append(str(test))
        if missing:
            raise ValueError(
                f"Blueprint references missing file(s): {', '.join(missing)}"
            )
        return self


def _is_yaml_content(text: str) -> bool:
    """Heuristic: if the first non-empty character is one of YAML structural markers."""
    first = ""
    for char in text.lstrip():
        if char and not char.isspace():
            first = char
            break
    return first in {"-", "[", "{", "p", "f", "c", "o", "l", "#"}


def parse_aero(text: str) -> Dict[str, Any]:
    """Parse a ``.aero`` blueprint.

    Aero-forge ``.aero`` files are YAML-compatible for the build command. If
    parsing fails, fall back to the legacy INI-style parser for compatibility
    with older accelerator blueprints.
    """
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        data = None

    if data is not None:
        return data

    # Legacy INI/TOML-like fallback.
    data = {}
    current: Optional[str] = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1].strip()
            data.setdefault(current, {})
            continue
        if current is None or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        data[current][key] = _parse_ini_value(value)
    return data


def _parse_ini_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    if value.startswith("[") and value.endswith("]"):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    try:
        if "." in value or "e" in value.lower():
            return float(value)
        return int(value)
    except ValueError:
        pass
    return value


def parse_blueprint(path: Path) -> Blueprint:
    """Parse ``.aero`` or ``.yaml`` blueprint into a normalized model."""
    path = Path(path).resolve()
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()

    if suffix in {".yaml", ".yml"}:
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ValueError(f"Failed to parse YAML blueprint {path}: {exc}") from exc
    elif suffix == ".aero":
        data = parse_aero(text)
    else:
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError:
            data = None
        if data is None:
            data = parse_aero(text)

    if not isinstance(data, dict):
        raise ValueError(f"Blueprint {path} did not parse to a mapping")

    # Normalize functions to absolute paths relative to the blueprint directory.
    base = path.parent
    for func in data.get("functions", []):
        if not isinstance(func, dict):
            continue
        func["file"] = str(base / Path(func["file"]))
        if "tests" in func:
            func["tests"] = [str(base / Path(t)) for t in func["tests"]]

    if "output_dir" in data:
        data["output_dir"] = str(base / Path(data["output_dir"]))

    try:
        return Blueprint.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"Invalid blueprint {path}: {exc}") from exc


def discover_functions(path: Path) -> List[FunctionSpec]:
    """Discover all top-level public functions in a Python file.

    Falls back to token-based discovery when the source has syntax errors so
    that the forge loop can still attempt to heal the file.
    """
    import ast
    import io
    import re
    import tokenize

    source_path = Path(path)
    if not source_path.is_file():
        raise ValueError(f"Source file not found: {source_path}")

    source = source_path.read_text(encoding="utf-8")
    names: List[str] = []
    try:
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                names.append(node.name)
    except SyntaxError:
        # Parse failed (likely a broken function the LLM needs to repair). Use
        # tokenization so we can still collect function names; fall back to a
        # simple regex if tokenization also fails.
        try:
            tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
            i = 0
            while i < len(tokens):
                tok = tokens[i]
                if tok.type == tokenize.NAME and tok.string == "def":
                    j = i + 1
                    while j < len(tokens) and tokens[j].type in (
                        tokenize.NL,
                        tokenize.NEWLINE,
                        tokenize.INDENT,
                        tokenize.DEDENT,
                        tokenize.COMMENT,
                    ):
                        j += 1
                    if j < len(tokens) and tokens[j].type == tokenize.NAME:
                        names.append(tokens[j].string)
                        i = j
                i += 1
        except Exception:
            names = re.findall(r"^\s*def\s+([A-Za-z_]\w*)", source, re.MULTILINE)
        names = [n for n in names if not n.startswith("_")]

    functions: List[FunctionSpec] = []
    for name in names:
        test_candidates = [
            source_path.parent / f"test_{name}.py",
            source_path.parent / f"test_{source_path.stem}.py",
        ]
        tests = [t for t in test_candidates if t.is_file()]
        functions.append(
            FunctionSpec(
                file=source_path,
                name=name,
                tests=tests,
            )
        )
    return functions


def generate_blueprint(
    project: str,
    functions: List[FunctionSpec],
    output_dir: Path = Path("./dist"),
    compiler_flags: Optional[List[str]] = None,
    prompt: Optional[str] = None,
    constraints: Optional[str] = None,
) -> Blueprint:
    """Create a Blueprint from discovered or supplied function specs.

    If ``prompt`` is provided, the architecture, toolchains, and manifest are
    inferred from prompt keywords (``rust``, ``pyo3``, ``ffi``, ``polyglot``,
    ``c++``) so multi-language requests are not silently downgraded to
    ``pure_python``.
    """
    intent = classify_build_intent(prompt or "")
    toolchains = toolchains_for_intent(intent)
    # For a single generated project, emit a minimal Rust/PyO3 crate manifest.
    # The full monorepo layout (rust_core/, python_engine/) is added later by
    # the monorepo packager.
    if intent in (BUILD_INTENT_HYBRID_RUST_PYTHON, BUILD_INTENT_PURE_RUST):
        manifest_entries = [
            ManifestEntry(path="Cargo.toml", lang="toml", purpose="Rust crate manifest"),
            ManifestEntry(path="src/lib.rs", lang="rust", purpose="Rust core library"),
        ]
    else:
        manifest_entries = []
    return Blueprint(
        project=project,
        architecture=intent,
        toolchains=toolchains,
        manifest=manifest_entries,
        functions=functions,
        output_dir=output_dir,
        compiler_flags=compiler_flags or [],
        llm=LLMConfig(provider="none"),
        prompt=prompt,
        constraints=constraints,
    )


def discover_project(
    root: Path,
    *,
    src_dirs: Optional[List[str]] = None,
    ignore_patterns: Optional[List[str]] = None,
) -> List[FunctionSpec]:
    """Discover all public Python functions in a project.

    Searches ``src/`` (if it exists) and the project root for ``.py`` files,
    skipping anything matched by ``ignore_patterns``.  For each source file,
    associated tests are looked up in ``tests/`` or ``test_*.py`` next to the
    file.
    """
    from aero_forge.ignore import is_ignored, parse_aeroignore

    root = Path(root).resolve()
    default_ignores = ["tests/", "test_*.py", "__pycache__/", "*.pyc", "dist/", ".git/"]
    if ignore_patterns is None:
        ignore_patterns = parse_aeroignore(root / ".aeroignore")
    patterns = default_ignores + list(ignore_patterns or [])
    candidates: List[Path] = []
    search_dirs = [root]
    src = root / "src"
    if src.is_dir():
        search_dirs.append(src)
    if src_dirs:
        for d in src_dirs:
            p = Path(d)
            if p.is_dir():
                search_dirs.append(p)

    for directory in search_dirs:
        for path in directory.rglob("*.py"):
            if path.name.startswith("_") or path.name == "setup.py":
                continue
            if is_ignored(path, patterns, root):
                continue
            candidates.append(path)

    # Prefer src/ files; if both root and src contain the same relative path,
    # keep the src/ one.
    seen: set = set()
    unique: List[Path] = []
    for path in candidates:
        rel = path.relative_to(root)
        if rel not in seen:
            seen.add(rel)
            unique.append(path)

    def _find_tests(func_name: str, source_path: Path) -> List[Path]:
        candidates = [
            root / "tests" / f"test_{func_name}.py",
            root / "tests" / f"test_{source_path.stem}.py",
            source_path.parent / f"test_{func_name}.py",
            source_path.parent / f"test_{source_path.stem}.py",
        ]
        return [c for c in candidates if c.is_file()]

    functions: List[FunctionSpec] = []
    for source_path in unique:
        for func in discover_functions(source_path):
            func.tests = _find_tests(func.name, source_path)
            functions.append(func)
    return functions


def write_blueprint(blueprint: Blueprint, path: Path) -> None:
    """Serialize a Blueprint to a YAML ``.aero`` file."""
    path.write_text(
        yaml.safe_dump(
            blueprint.model_dump(mode="json"),
            sort_keys=False,
            default_flow_style=False,
        ),
        encoding="utf-8",
    )
