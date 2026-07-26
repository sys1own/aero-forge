"""Blueprint parser and validation for multi-function builds."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from aero_forge.orchestrator.stack_classifier import (
    INTENT_HYBRID_CPP_PYTHON,
    INTENT_HYBRID_CPP_RUST,
    INTENT_HYBRID_RUST_PYTHON,
    INTENT_PURE_PYTHON,
    INTENT_PURE_RUST,
    INTENT_TRI_POLYGLOT_RUST_CPP_PYTHON,
    classify_stack,
    default_manifest_for_architecture,
    suggested_blueprint_template,
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
    skip_build: bool = False

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


class CLIContractFlag(BaseModel):
    """A single CLI flag in the executable contract."""

    name: str
    short: str = ""
    type: str = "string"
    required: bool = False
    default: Any = None
    choices: List[str] = Field(default_factory=list)
    help: str = ""
    dest_var: str = ""

    @field_validator("type")
    @classmethod
    def _valid_type(cls, value: str) -> str:
        allowed = {"string", "int", "bool", "float"}
        if value not in allowed:
            raise ValueError(f"CLIContractFlag.type must be one of {allowed}, got {value!r}")
        return value


class CLIContract(BaseModel):
    """CLI contract describing the command-line interface."""

    parser_type: str = "argparse"
    flags: List[CLIContractFlag] = Field(default_factory=list)


class ExecutionStrategy(BaseModel):
    """Execution plan for the generated project."""

    primary_entrypoint: Dict[str, Any] = Field(default_factory=dict)
    cli_contract: CLIContract = Field(default_factory=CLIContract)
    run_spec: Dict[str, Any] = Field(default_factory=dict)


class ABIContract(BaseModel):
    """Native ABI contract for cross-language symbol binding."""

    contract_id: str
    target_language: str
    binding_framework: str
    export_symbol: str
    c_symbol_alias: str = ""
    header_path: str = ""
    memory_model: str
    signature: Dict[str, List[Dict[str, str]]] = Field(default_factory=dict)

    @field_validator("target_language")
    @classmethod
    def _valid_target_language(cls, value: str) -> str:
        allowed = {"cpp", "rust", "python"}
        if value not in allowed:
            raise ValueError(f"target_language must be one of {allowed}, got {value!r}")
        return value

    @field_validator("binding_framework")
    @classmethod
    def _valid_binding_framework(cls, value: str) -> str:
        allowed = {"c_abi", "pyo3", "ctypes"}
        if value not in allowed:
            raise ValueError(f"binding_framework must be one of {allowed}, got {value!r}")
        return value

    @field_validator("memory_model")
    @classmethod
    def _valid_memory_model(cls, value: str) -> str:
        allowed = {"callee_allocates", "caller_allocates", "shared_pyo3"}
        if value not in allowed:
            raise ValueError(f"memory_model must be one of {allowed}, got {value!r}")
        return value


class BlueprintSchemaV2(BaseModel):
    """Schema v2.0.0 blueprint: an executable task and contract graph."""

    metadata: Dict[str, str] = Field(
        default_factory=lambda: {"schema_version": "2.0.0"}
    )
    execution_strategy: ExecutionStrategy = Field(default_factory=ExecutionStrategy)
    abi_contracts: List[ABIContract] = Field(default_factory=list)
    module_graph: List[Dict[str, Any]] = Field(default_factory=list)
    verification_nodes: List[Dict[str, Any]] = Field(default_factory=list)


class BlueprintValidator:
    """Validate a Schema v2.0.0 blueprint, with v1.x fallback/upgrade."""

    SUPPORTED_TYPES = {
        "u32",
        "i32",
        "usize",
        "f64",
        "double*",
        "int32_t",
        "*const u32",
        "*mut f64",
        "float*",
        "int*",
    }

    def __init__(self, blueprint_path_or_dict: Any):
        data: Dict[str, Any]
        if isinstance(blueprint_path_or_dict, dict):
            data = blueprint_path_or_dict
        else:
            path = Path(blueprint_path_or_dict)
            text = path.read_text(encoding="utf-8")
            suffix = path.suffix.lower()
            if suffix == ".json":
                data = json.loads(text)
            else:
                data = yaml.safe_load(text) or {}

        if not isinstance(data, dict):
            raise ValueError("Blueprint must be a mapping")

        # Upgrade v1.x blueprints to v2.0.0 when the new metadata block is absent.
        if self._is_v1_schema(data):
            data = self._upgrade_v1(data)

        self._raw = data
        self.blueprint = BlueprintSchemaV2.model_validate(data)

    @staticmethod
    def _is_v1_schema(data: Dict[str, Any]) -> bool:
        """Heuristic: v1 blueprints carry 'project' and 'architecture' without schema metadata."""
        return (
            "project" in data
            or "architecture" in data
            or "functions" in data
        ) and ("metadata" not in data or data.get("metadata", {}).get("schema_version") is None)

    @staticmethod
    def _upgrade_v1(data: Dict[str, Any]) -> Dict[str, Any]:
        """Build a minimal v2.0.0 blueprint from a v1.x blueprint."""
        return {
            "metadata": {
                "schema_version": "2.0.0",
                "project_name": data.get("project", "aero_forge_project"),
                "domain_target": data.get("architecture", "pure_python"),
            },
            "execution_strategy": {
                "primary_entrypoint": {},
                "cli_contract": {"parser_type": "argparse", "flags": []},
                "run_spec": {},
            },
            "abi_contracts": [],
            "module_graph": [],
            "verification_nodes": [],
        }

    def validate_abi_integrity(self) -> bool:
        """Ensure every ABI input/output type is in the supported set."""
        for contract in self.blueprint.abi_contracts:
            for direction in ("inputs", "outputs"):
                entries = contract.signature.get(direction, [])
                for idx, entry in enumerate(entries):
                    if not isinstance(entry, dict):
                        raise TypeError(
                            f"ABI {contract.contract_id} {direction}[{idx}] must be a name/type mapping"
                        )
                    t = entry.get("type", "")
                    if t not in self.SUPPORTED_TYPES:
                        raise TypeError(
                            f"Unsupported ABI type {t!r} in {contract.contract_id} {direction}[{idx}]"
                        )
        return True

    def validate_cli_contract(self) -> bool:
        """Ensure CLI flag names and dest_var values are valid Python identifiers."""
        for flag in self.blueprint.execution_strategy.cli_contract.flags:
            if not flag.name.isidentifier():
                raise ValueError(f"CLI flag name {flag.name!r} is not a valid Python identifier")
            if flag.dest_var and not flag.dest_var.isidentifier():
                raise ValueError(f"CLI dest_var {flag.dest_var!r} is not a valid Python identifier")
        return True


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
    languages: List[str] = Field(default_factory=list)
    features: List[str] = Field(default_factory=list)

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
    classification = classify_stack(prompt or "")
    intent = classification.architecture
    toolchains = classification.toolchains or ["python"]
    # For a single generated project, emit a minimal Rust/PyO3 crate manifest.
    # The full monorepo layout (rust_core/, python_engine/) is added later by
    # the monorepo packager / plan_workspace.
    if intent in (INTENT_HYBRID_RUST_PYTHON, INTENT_PURE_RUST):
        manifest_entries = [
            ManifestEntry(path="Cargo.toml", lang="toml", purpose="Rust crate manifest"),
            ManifestEntry(path="src/lib.rs", lang="rust", purpose="Rust core library"),
        ]
    elif intent == INTENT_HYBRID_CPP_RUST:
        manifest_entries = [
            ManifestEntry(path="Cargo.toml", lang="toml", purpose="Rust package manifest"),
            ManifestEntry(path="build.rs", lang="rust", purpose="C++ build and link script"),
            ManifestEntry(path="src/main.rs", lang="rust", purpose="Rust CLI binary"),
            ManifestEntry(path="src/cpp_core/native.cpp", lang="cpp", purpose="C-ABI math source"),
            ManifestEntry(path="tests/test_hybrid_cpp_rust.rs", lang="rust", purpose="Rust integration test"),
            ManifestEntry(path="README.md", lang="markdown", purpose="Project README"),
        ]
    elif intent == INTENT_TRI_POLYGLOT_RUST_CPP_PYTHON:
        manifest_entries = [
            ManifestEntry(path="Cargo.toml", lang="toml", purpose="Rust workspace manifest"),
            ManifestEntry(path="rust_core/Cargo.toml", lang="toml", purpose="PyO3 crate manifest"),
            ManifestEntry(path="rust_core/src/lib.rs", lang="rust", purpose="Rust native core"),
            ManifestEntry(path="cpp_core/native.cpp", lang="cpp", purpose="C-ABI dynamic shared library source"),
            ManifestEntry(path="pyproject.toml", lang="toml", purpose="Python package manifest"),
            ManifestEntry(path="src/python/__init__.py", lang="python", purpose="Python driver package"),
            ManifestEntry(path="src/python/main.py", lang="python", purpose="Python CLI / REPL entrypoint"),
            ManifestEntry(path="tests/test_tri.py", lang="python", purpose="pytest tests"),
            ManifestEntry(path="README.md", lang="markdown", purpose="Project README"),
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
        languages=classification.languages,
        features=classification.features,
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


def _parse_cargo_toml(path: Path) -> Dict[str, Any]:
    """Parse a Cargo.toml manifest using the available TOML parser."""
    try:
        import tomllib

        with path.open("rb") as fh:
            return tomllib.load(fh) or {}
    except ImportError:  # pragma: no cover
        try:
            import tomli

            with path.open("rb") as fh:
                return tomli.load(fh) or {}
        except ImportError:
            text = path.read_text(encoding="utf-8")
            # Last-ditch YAML-style fallback for simple manifests.
            try:
                return yaml.safe_load(text) or {}
            except Exception:
                return {}
    except (OSError, ValueError):
        return {}


def _is_path_inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def generate_blueprint_from_uploaded_repo(repo_path: Path) -> Path:
    """Synthesize a ``blueprint.aero`` for an uploaded project if none exists.

    Inspects ``Cargo.toml``, Python modules, and C/C++ headers to infer the
    project architecture, toolchains, manifest, and entry-point functions.
    """
    repo_path = Path(repo_path).resolve()
    blueprint_path = repo_path / "blueprint.aero"
    existing_md = repo_path / "BLUEPRINT.md"
    if blueprint_path.is_file():
        return blueprint_path
    if existing_md.is_file():
        return existing_md

    cargo_toml = repo_path / "Cargo.toml"
    pyproject = repo_path / "pyproject.toml"
    setup_py = repo_path / "setup.py"

    has_rust = cargo_toml.is_file()
    has_python = pyproject.is_file() or setup_py.is_file() or bool(
        list(repo_path.rglob("*.py"))
    )
    cpp_sources = [p for p in repo_path.rglob("*") if p.suffix in {".cpp", ".c", ".h", ".hpp"}]
    has_cpp = bool(cpp_sources)

    members: List[str] = []
    if has_rust:
        cargo_data = _parse_cargo_toml(cargo_toml)
        members = cargo_data.get("workspace", {}).get("members", [])
        if not members and "package" in cargo_data:
            members = [cargo_data["package"].get("name", "rust_core")]

    # Classify architecture and toolchains from the detected artifacts.
    if has_rust and has_python:
        architecture = INTENT_HYBRID_RUST_PYTHON
    elif has_rust:
        architecture = INTENT_PURE_RUST
    elif has_cpp and has_python:
        architecture = INTENT_HYBRID_CPP_PYTHON
    elif has_cpp:
        architecture = INTENT_HYBRID_CPP_PYTHON
    else:
        architecture = INTENT_PURE_PYTHON

    classification = classify_stack(architecture)
    toolchains = classification.toolchains or ["python"]
    languages = classification.languages or ["python"]
    features = classification.features or []

    manifest: List[ManifestEntry] = []
    if has_rust:
        manifest.append(ManifestEntry(path="Cargo.toml", lang="toml", purpose="Rust workspace manifest"))
        for member in members:
            manifest.append(ManifestEntry(path=f"{member}/Cargo.toml", lang="toml", purpose="Workspace member crate"))
            manifest.append(ManifestEntry(path=f"{member}/src/lib.rs", lang="rust", purpose="Rust crate source"))
    if pyproject.is_file():
        manifest.append(ManifestEntry(path="pyproject.toml", lang="toml", purpose="Python package metadata"))
    elif setup_py.is_file():
        manifest.append(ManifestEntry(path="setup.py", lang="python", purpose="Python package setup"))
    for src in sorted(repo_path.rglob("*.py")):
        if src.name.startswith("test_") or "/tests/" in str(src.relative_to(repo_path)).replace("\\", "/"):
            continue
        if _is_path_inside(src, repo_path):
            manifest.append(ManifestEntry(path=str(src.relative_to(repo_path)), lang="python", purpose="Python source"))
    for hdr in sorted(cpp_sources):
        if _is_path_inside(hdr, repo_path):
            manifest.append(ManifestEntry(path=str(hdr.relative_to(repo_path)), lang=hdr.suffix.lstrip("."), purpose="C/C++ source"))

    # Discover Python entry points.
    functions: List[FunctionSpec] = []
    try:
        functions = discover_project(repo_path)
    except Exception as exc:
        logger.warning("Could not discover Python functions in %s: %s", repo_path, exc)

    # Build contracts for cross-language entry points.
    contracts: List[ContractEntry] = []
    for func in functions:
        if func.name:
            contracts.append(
                ContractEntry(
                    name=func.name,
                    python_name=func.name,
                    language="python",
                    purpose=f"Callable exported from {func.file}",
                )
            )

    blueprint = Blueprint(
        project=repo_path.name or "uploaded_project",
        architecture=architecture,
        toolchains=toolchains,
        languages=languages,
        features=features,
        manifest=manifest,
        contracts=contracts,
        functions=functions,
        output_dir=repo_path / "dist",
    )
    write_blueprint(blueprint, blueprint_path)
    return blueprint_path
