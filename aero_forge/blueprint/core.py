"""Blueprint parser and validation for multi-function builds."""

from __future__ import annotations

import ast
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from aero_forge.blueprint_templates import list_templates, load_template
from aero_forge.ingestion.zip_parser import generate_draft_v3_blueprint
from aero_forge.orchestrator.stack_classifier import (
    INTENT_HYBRID_CPP_PYTHON,
    INTENT_HYBRID_CPP_RUST,
    INTENT_HYBRID_RUST_PYTHON,
    INTENT_PURE_PYTHON,
    INTENT_PURE_RUST,
    INTENT_TRI_POLYGLOT_RUST_CPP_PYTHON,
    classify_stack,
    default_manifest_for_architecture,
    extract_source_directories,
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


class FunctionalIntent(BaseModel):
    """A structured functional requirement extracted from the user prompt."""

    symbol_name: str
    type: str = "function"
    requirement_level: str = "required"


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
            raise ValueError(
                f"CLIContractFlag.type must be one of {allowed}, got {value!r}"
            )
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
    engine_backend: str = ""
    wavefront_parallelism: int = 0
    precision_shield_mode: str = ""
    hin_jit_opt_level: int = 0


class ABIContract(BaseModel):
    """Native ABI contract for cross-language symbol binding."""

    contract_id: str
    target_language: str
    binding_framework: str
    export_symbol: str
    c_symbol_alias: str = ""
    header_path: Optional[str] = None
    memory_model: str
    signature: Dict[str, List[Dict[str, str]]] = Field(default_factory=dict)

    @field_validator("target_language", mode="before")
    @classmethod
    def _normalize_target_language(cls, value: Any) -> str:
        if value is None:
            return "cpp"
        value = str(value).lower().strip().replace(" ", "_").replace("-", "_")
        synonyms = {
            "c++": "cpp",
            "cxx": "cpp",
            "c": "cpp",
            "c_abi": "cpp",
            "cabi": "cpp",
            "py": "python",
            "py3": "python",
            "zig": "zig",
            "go": "go",
            "golang": "go",
            "java": "java",
            "csharp": "csharp",
            "c#": "csharp",
            "cs": "csharp",
            "mojo": "mojo",
            "nim": "nim",
            "d": "d",
            "f90": "fortran",
            "fortran": "fortran",
            "js": "javascript",
            "javascript": "javascript",
            "ts": "typescript",
            "typescript": "typescript",
        }
        value = synonyms.get(value, value)
        return value

    @field_validator("binding_framework", mode="before")
    @classmethod
    def _normalize_binding_framework(cls, value: Any) -> str:
        if value is None:
            return "c_abi"
        value = str(value).lower().strip().replace("-", "_").replace(" ", "_")
        synonyms = {
            "pyo3": "pyo3",
            "ctypes": "ctypes",
            "c": "c_abi",
            "cabi": "c_abi",
            "c-abi": "c_abi",
            "raw_c": "c_abi",
            "native_c": "c_abi",
            "native_bridge": "c_abi",
            "c_abi_bridge": "c_abi",
            "extern_c": "c_abi",
            "c_api": "c_abi",
            "c_ffi": "c_abi",
            # WebAssembly and pybind11 are normalized to c_abi because the engine
            # exposes them through a C-ABI boundary (cargo --target wasm32-* or
            # pybind11 extern "C" exports).
            "wasm_bindgen": "c_abi",
            "wasm": "c_abi",
            "wasi": "c_abi",
            "wasm32": "c_abi",
            "pybind11": "c_abi",
            "pybind": "c_abi",
            "numpy": "c_abi",
            "cython": "c_abi",
            "cffi": "ctypes",
            "swig": "c_abi",
            "boost": "c_abi",
            "boost_python": "c_abi",
        }
        value = synonyms.get(value, value)
        if value not in {"c_abi", "pyo3", "ctypes"}:
            if "pyo3" in value:
                value = "pyo3"
            elif "ctypes" in value:
                value = "ctypes"
            elif any(
                k in value
                for k in (
                    "c_abi",
                    "cabi",
                    "native",
                    "extern",
                    "c_api",
                    "c_ffi",
                    "wasm",
                    "bindgen",
                    "pybind",
                )
            ):
                value = "c_abi"
        allowed = {"c_abi", "pyo3", "ctypes"}
        if value not in allowed:
            # Unknown binding frameworks are treated as C-ABI rather than
            # failing schema validation; the materializer can later reject
            # unsupported combinations with a clear diagnostic.
            value = "c_abi"
        return value

    @field_validator("memory_model", mode="before")
    @classmethod
    def _normalize_memory_model(cls, value: Any) -> str:
        if value is None:
            return "caller_allocates"
        value = str(value).lower().strip().replace("-", "_")
        synonyms = {
            "borrowed": "shared_pyo3",
            "shared": "shared_pyo3",
            "owned": "callee_allocates",
            "callee": "callee_allocates",
            "caller": "caller_allocates",
            "caller_allocated": "caller_allocates",
            "callee_allocated": "callee_allocates",
            "c_allocated_by_caller": "caller_allocates",
            "allocated_by_caller": "caller_allocates",
            "allocated_by_callee": "callee_allocates",
            "by_caller": "caller_allocates",
            "by_callee": "callee_allocates",
            "c_memory": "caller_allocates",
            "c_alloc": "caller_allocates",
            "callee_owned": "callee_allocates",
            "caller_owned": "caller_allocates",
        }
        value = synonyms.get(value, value)
        if value not in {"callee_allocates", "caller_allocates", "shared_pyo3"}:
            if "pyo3" in value or "borrow" in value or "shared" in value:
                value = "shared_pyo3"
            elif (
                any(k in value for k in ("callee", "owned", "returned", "rust"))
                and "caller" not in value
            ):
                value = "callee_allocates"
            elif any(
                k in value
                for k in (
                    "caller",
                    "c_memory",
                    "c_alloc",
                    "pointer",
                    "buffer",
                    "array",
                    "by_c",
                )
            ):
                value = "caller_allocates"
        allowed = {"callee_allocates", "caller_allocates", "shared_pyo3"}
        if value not in allowed:
            raise ValueError(f"memory_model must be one of {allowed}, got {value!r}")
        return value

    @field_validator("header_path", mode="before")
    @classmethod
    def _normalize_header_path(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        value = str(value).strip()
        return value if value else None


class BlueprintSchemaV2(BaseModel):
    """Schema v2.0.0 blueprint: an executable task and contract graph."""

    metadata: Dict[str, str] = Field(
        default_factory=lambda: {"schema_version": "2.0.0"}
    )
    execution_strategy: ExecutionStrategy = Field(default_factory=ExecutionStrategy)
    abi_contracts: List[ABIContract] = Field(default_factory=list)
    functional_intent: List[FunctionalIntent] = Field(default_factory=list)
    module_graph: List[Dict[str, Any]] = Field(default_factory=list)
    verification_nodes: List[Dict[str, Any]] = Field(default_factory=list)
    cargo_dependencies: Dict[str, Any] = Field(default_factory=dict)


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
        "bool",
        "const char*",
        "void",
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
            "project" in data or "architecture" in data or "functions" in data
        ) and (
            "metadata" not in data
            or data.get("metadata", {}).get("schema_version") is None
        )

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
                raise ValueError(
                    f"CLI flag name {flag.name!r} is not a valid Python identifier"
                )
            if flag.dest_var and not flag.dest_var.isidentifier():
                raise ValueError(
                    f"CLI dest_var {flag.dest_var!r} is not a valid Python identifier"
                )
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
    execution_strategy: Optional[ExecutionStrategy] = None
    abi_contracts: List[ABIContract] = Field(default_factory=list)
    functional_intent: List[FunctionalIntent] = Field(default_factory=list)
    verification_nodes: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, str] = Field(
        default_factory=lambda: {"schema_version": "2.0.0"}
    )
    module_graph: List[Dict[str, Any]] = Field(default_factory=list)
    cargo_dependencies: Dict[str, Any] = Field(default_factory=dict)
    modification_plan: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _sync_manifest_and_module_graph(self) -> "Blueprint":
        """Merge ``module_graph`` into ``manifest`` and remove duplicate paths."""
        existing_paths = {e.path for e in self.manifest}
        for node in self.module_graph:
            if not isinstance(node, dict):
                continue
            path = node.get("path")
            if not path or path in existing_paths:
                continue
            existing_paths.add(path)
            self.manifest.append(
                ManifestEntry(
                    path=path,
                    lang=node.get("lang") or node.get("language") or "python",
                    purpose=node.get("purpose", ""),
                )
            )
        seen: set = set()
        unique: List[ManifestEntry] = []
        for entry in self.manifest:
            if entry.path not in seen:
                seen.add(entry.path)
                unique.append(entry)
        self.manifest = unique

        seen_graph: set = set()
        unique_graph: List[Dict[str, Any]] = []
        for node in self.module_graph:
            path = node.get("path") if isinstance(node, dict) else None
            if path and path in seen_graph:
                continue
            if path:
                seen_graph.add(path)
            unique_graph.append(node)
        self.module_graph = unique_graph
        return self

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

    def _derive_functional_intent(self) -> List[FunctionalIntent]:
        """Derive a minimal functional_intent from contracts/functions/manifest."""
        derived: List[FunctionalIntent] = []
        seen: set = set()
        for contract in self.contracts or []:
            name = getattr(contract, "name", None) or ""
            if name and name not in seen:
                derived.append(FunctionalIntent(symbol_name=name, type="function"))
                seen.add(name)
        for func in self.functions or []:
            name = getattr(func, "name", None) or ""
            if name and name not in seen:
                derived.append(FunctionalIntent(symbol_name=name, type="function"))
                seen.add(name)
        for entry in self.manifest or []:
            symbol = Path(entry.path).stem
            if symbol and symbol not in seen:
                derived.append(FunctionalIntent(symbol_name=symbol, type="function"))
                seen.add(symbol)
        for node in self.module_graph or []:
            if not isinstance(node, dict):
                continue
            node_id = node.get("node_id") or ""
            if node_id and node_id not in seen:
                derived.append(FunctionalIntent(symbol_name=node_id, type="function"))
                seen.add(node_id)
            for sym in node.get("exports") or []:
                if sym and sym not in seen:
                    derived.append(
                        FunctionalIntent(symbol_name=str(sym), type="function")
                    )
                    seen.add(sym)
        return derived

    @model_validator(mode="after")
    def _enforce_architecture_contracts(self) -> "Blueprint":
        """Non-pure-Python architectures must declare contracts and functional_intent."""
        if self.architecture == "pure_python":
            return self
        if not self.contracts:
            raise ValueError(
                f"Architecture {self.architecture!r} requires non-empty 'contracts'. "
                "Add at least one cross-language or exported symbol contract."
            )
        if not self.functional_intent:
            derived = self._derive_functional_intent()
            if derived:
                self.functional_intent = derived
        if not self.functional_intent:
            raise ValueError(
                f"Architecture {self.architecture!r} requires non-empty 'functional_intent'. "
                "Translate every functional requirement from the prompt into a structured list."
            )
        return self

    @model_validator(mode="after")
    def _enforce_toolchain_manifest_alignment(self) -> "Blueprint":
        """Manifest file extensions must be supported by the declared toolchains."""
        ext_toolchains: Dict[str, set] = {
            ".rs": {"cargo", "rustc"},
            ".cpp": {"clang", "gcc", "g++", "clang++", "cmake"},
            ".cc": {"clang", "gcc", "g++", "clang++", "cmake"},
            ".cxx": {"clang", "gcc", "g++", "clang++", "cmake"},
            ".c": {"clang", "gcc", "cmake"},
            ".py": {"python"},
            ".go": {"go"},
            ".java": {"javac"},
            ".cs": {"dotnet", "csc"},
            ".zig": {"zig"},
            ".mojo": {"mojo"},
            ".nim": {"nim", "nimble"},
            ".d": {"dmd", "ldc", "gdc"},
            ".f90": {"gfortran", "ifort"},
        }
        toolchains = {t.lower() for t in self.toolchains}
        if "cpp" in toolchains or "c++" in toolchains:
            toolchains.update({"cmake", "clang", "gcc", "g++", "clang++"})
        if "rust" in toolchains:
            toolchains.add("cargo")
            toolchains.add("rustc")
        if "c" in toolchains:
            toolchains.update({"cmake", "clang", "gcc"})
        if "cpython" in toolchains or "python3" in toolchains or "py" in toolchains:
            toolchains.add("python")
        for entry in self.manifest:
            ext = Path(entry.path).suffix.lower()
            allowed = ext_toolchains.get(ext)
            if not allowed:
                continue
            if not toolchains.intersection(allowed):
                raise ValueError(
                    f"Manifest file {entry.path!r} (extension {ext!r}) requires "
                    f"one of toolchains {sorted(allowed)}; toolchains={sorted(self.toolchains)}."
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

    base = path.parent

    # Convert Blueprint v3.0.0 to the v2 runner blueprint.
    if str(data.get("metadata", {}).get("schema_version")) == "3.0.0":
        from aero_forge.blueprint.schema import BlueprintV3

        v3 = BlueprintV3.model_validate(data)
        blueprint = v3.to_runner_blueprint(base)
        # Normalize any function paths the conversion produced.
        for func in data.get("functions", []):
            if not isinstance(func, dict):
                continue
            func["file"] = str(base / Path(func["file"]))
            if "tests" in func:
                func["tests"] = [str(base / Path(t)) for t in func["tests"]]
        return blueprint

    # Normalize functions to absolute paths relative to the blueprint directory.
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
    dirs = extract_source_directories(prompt or "")
    # For a single generated project, emit a minimal Rust/PyO3 crate manifest.
    # The full monorepo layout (rust_core/, python_engine/) is added later by
    # the monorepo packager / plan_workspace.
    if intent in (INTENT_HYBRID_RUST_PYTHON, INTENT_PURE_RUST):
        rust_dir = dirs["rust_crate_dir"] or ""
        rust_cargo = f"{rust_dir}/Cargo.toml" if rust_dir else "Cargo.toml"
        rust_lib = f"{rust_dir}/src/lib.rs" if rust_dir else "src/lib.rs"
        manifest_entries = [
            ManifestEntry(path=rust_cargo, lang="toml", purpose="Rust crate manifest"),
            ManifestEntry(path=rust_lib, lang="rust", purpose="Rust core library"),
        ]
    elif intent == INTENT_HYBRID_CPP_RUST:
        cpp_entry = dirs["cpp_source"] or "src/cpp_core/native.cpp"
        manifest_entries = [
            ManifestEntry(
                path="Cargo.toml", lang="toml", purpose="Rust package manifest"
            ),
            ManifestEntry(
                path="build.rs", lang="rust", purpose="C++ build and link script"
            ),
            ManifestEntry(path="src/main.rs", lang="rust", purpose="Rust CLI binary"),
            ManifestEntry(path=cpp_entry, lang="cpp", purpose="C-ABI math source"),
            ManifestEntry(
                path="tests/test_hybrid_cpp_rust.rs",
                lang="rust",
                purpose="Rust integration test",
            ),
            ManifestEntry(path="README.md", lang="markdown", purpose="Project README"),
        ]
    elif intent == INTENT_TRI_POLYGLOT_RUST_CPP_PYTHON:
        manifest_entries = [
            ManifestEntry(path=e["path"], lang=e["lang"], purpose=e["purpose"])
            for e in default_manifest_for_architecture(
                INTENT_TRI_POLYGLOT_RUST_CPP_PYTHON, project, prompt=prompt or ""
            )
        ]
    else:
        manifest_entries = []

    # Derive functional_intent and contracts from supplied function specs so that
    # non-pure-Python blueprints pass the structural integrity gate.
    functional_intent = [
        FunctionalIntent(
            symbol_name=f.name, type="function", requirement_level="required"
        )
        for f in functions
        if f.name
    ]
    contracts = [
        ContractEntry(name=f.name, signature="", language="python")
        for f in functions
        if f.name
    ]

    return Blueprint(
        project=project,
        architecture=intent,
        toolchains=toolchains,
        manifest=manifest_entries,
        contracts=contracts,
        functions=functions,
        functional_intent=functional_intent,
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


def _annotation_to_str(node: Optional[ast.AST]) -> str:
    """Convert an AST annotation back to a Python type string."""
    if node is None:
        return "None"
    try:
        return ast.unparse(node)
    except Exception:
        return "Any"


def _parse_signature_local(signature: str) -> Tuple[str, List[Tuple[str, str]], str]:
    """Parse a ``def name(...) -> T`` style signature into (name, args, return_type)."""
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
    return func.name, args, _annotation_to_str(func.returns)


def _py_type_to_abi(py_type: str) -> str:
    """Map a Python type hint to a C-ABI scalar or pointer type (empty if unsupported)."""
    t = (py_type or "").strip().lower().replace(" ", "")
    if not t or t in ("none", "void"):
        return "void"
    if t in ("float", "f64", "double"):
        return "f64"
    if t in ("int", "i64", "i32", "int64_t", "int32_t"):
        return "i32"
    if t == "bool":
        return "bool"
    if t in ("str", "string"):
        return "const char*"
    if t == "list" or (t.startswith("list[") and t.endswith("]")):
        inner = t[5:-1].strip() if t.startswith("list[") else "float"
        inner_abi = _py_type_to_abi(inner)
        if inner_abi == "i32":
            return "int*"
        if inner_abi == "f64":
            return "double*"
        if inner_abi == "bool":
            return "bool*"
        if inner_abi:
            return "double*"
    return ""


def _is_c_abi_compatible(py_type: str) -> bool:
    """Return True when *py_type* can be expressed as a simple C-ABI type."""
    return _py_type_to_abi(py_type) != ""


def _contracts_to_abi_contracts(
    contracts: List[ContractEntry],
    manifest: List[ManifestEntry],
) -> List[ABIContract]:
    """Synthesise ``ABIContract`` entries from legacy ``ContractEntry`` definitions."""
    abi_contracts: List[ABIContract] = []
    header_candidates = [
        e.path for e in manifest if Path(e.path).suffix in (".h", ".hpp")
    ]
    for contract in contracts:
        if not contract.signature:
            continue
        try:
            name, args, return_type = _parse_signature_local(contract.signature)
        except Exception:
            continue
        if not all(
            _is_c_abi_compatible(t) for _, t in args
        ) or not _is_c_abi_compatible(return_type):
            continue

        lang = (contract.language or "").lower()
        if "cpp" in lang or "c++" in lang:
            target_language = "cpp"
            binding_framework = "c_abi"
            memory_model = "callee_allocates"
        elif "rust" in lang:
            target_language = "rust"
            binding_framework = "pyo3"
            memory_model = "shared_pyo3"
        else:
            target_language = "python"
            binding_framework = "ctypes"
            memory_model = "caller_allocates"

        header_path = ""
        base = re.sub(r"\.(cpp|cc|cxx)$", ".h", name)
        for candidate in header_candidates:
            if Path(candidate).stem == name or Path(candidate).name == f"{base}.h":
                header_path = candidate
                break
        if not header_path and header_candidates:
            # Reuse the first declared header for the whole C-ABI surface.
            header_path = header_candidates[0]
        if not header_path and any(
            Path(e.path).suffix in (".cpp", ".cc", ".cxx") for e in manifest
        ):
            header_path = f"include/{name}.h"

        abi_contracts.append(
            ABIContract(
                contract_id=name,
                target_language=target_language,
                binding_framework=binding_framework,
                export_symbol=name,
                c_symbol_alias=name,
                header_path=header_path or None,
                memory_model=memory_model,
                signature={
                    "inputs": [
                        {"name": a, "type": _py_type_to_abi(t)} for a, t in args
                    ],
                    "outputs": [
                        {"name": "return", "type": _py_type_to_abi(return_type)}
                    ],
                },
            )
        )
    return abi_contracts


def _infer_primary_entrypoint(manifest: List[ManifestEntry]) -> Dict[str, Any]:
    """Select the most likely executable entrypoint from the manifest."""
    candidates = [
        ("run_shell.py", "python3"),
        ("main.py", "python3"),
        ("src/main.rs", "rust"),
        ("src/main.py", "python3"),
    ]
    for path, runtime in candidates:
        if any(e.path == path for e in manifest):
            return {"path": path, "runtime": runtime, "wrapper_generation": True}
    for entry in manifest:
        if entry.lang == "python" and entry.path.endswith(".py"):
            return {
                "path": entry.path,
                "runtime": "python3",
                "wrapper_generation": True,
            }
    return {"path": "main.py", "runtime": "python3", "wrapper_generation": True}


def _module_graph_from_manifest(manifest: List[ManifestEntry]) -> List[Dict[str, Any]]:
    """Build a module graph from the manifest when the LLM does not provide one."""
    return [
        {
            "path": e.path,
            "lang": e.lang,
            "purpose": e.purpose,
        }
        for e in manifest
    ]


def _default_verification_nodes(
    primary_entrypoint: Dict[str, Any], project_name: str
) -> List[Dict[str, Any]]:
    """Create minimal verification nodes for a generated project."""
    entry = primary_entrypoint.get("path", "main.py")
    runtime = primary_entrypoint.get("runtime", "python3")
    return [
        {
            "test_id": f"{project_name}_cli_parses",
            "execution_cmd": (
                f"{runtime} {entry} --help"
                if runtime == "python3"
                else f"cargo run -- --help"
            ),
            "expected_exit_code": 0,
            "stdout_match_patterns": ["usage"],
            "stderr_prohibited_patterns": ["Traceback"],
        },
        {
            "test_id": f"{project_name}_runs",
            "execution_cmd": (
                f"{runtime} {entry}" if runtime == "python3" else f"cargo run"
            ),
            "expected_exit_code": 0,
            "stdout_match_patterns": ["ok", "success"],
            "stderr_prohibited_patterns": ["error", "Traceback"],
        },
    ]


def write_blueprint(blueprint: Blueprint, path: Path) -> None:
    """Serialize a Blueprint to a YAML ``.aero`` file using v2 schema defaults."""
    from aero_forge.scaffold.pre_write_validator import deduplicate_manifest_entries

    v2_defaults = BlueprintSchemaV2().model_dump(mode="json")
    metadata = {
        **v2_defaults.get("metadata", {}),
        **(blueprint.metadata or {}),
        "schema_version": "2.0.0",
        "project_name": blueprint.project or "aero_forge_project",
        "domain_target": blueprint.architecture or "pure_python",
    }

    # A blueprint written after a successful planning/build pass must be visible as
    # finalized.  Only keep it as draft when llm_initialized is explicitly false or
    # the blueprint is empty (no prompt/manifest).
    status = metadata.get("status")
    llm_initialized_raw = metadata.get("llm_initialized")
    llm_initialized = (
        str(llm_initialized_raw).strip().lower() in ("true", "1", "yes")
        if llm_initialized_raw is not None
        else False
    )
    if status != "finalized" and (
        llm_initialized or (blueprint.prompt and blueprint.manifest)
    ):
        metadata["status"] = "finalized"
        metadata["llm_initialized"] = "true"

    # Synthesise v2 fields from legacy v1 data when the LLM/planner did not emit them.
    manifest = list(blueprint.manifest) if blueprint.manifest else []
    execution_strategy = blueprint.execution_strategy
    if execution_strategy is None or not execution_strategy.primary_entrypoint:
        primary = _infer_primary_entrypoint(manifest)
        cli_contract = (
            execution_strategy.cli_contract if execution_strategy else CLIContract()
        )
        run_spec = execution_strategy.run_spec if execution_strategy else {}
        execution_strategy = ExecutionStrategy(
            primary_entrypoint=primary,
            cli_contract=cli_contract,
            run_spec=run_spec,
        )

    abi_contracts = blueprint.abi_contracts or _contracts_to_abi_contracts(
        list(blueprint.contracts), manifest
    )
    module_graph = blueprint.module_graph or _module_graph_from_manifest(manifest)
    verification_nodes = blueprint.verification_nodes or _default_verification_nodes(
        execution_strategy.primary_entrypoint, blueprint.project or "aero_forge_project"
    )

    v2 = BlueprintSchemaV2(
        metadata=metadata,
        execution_strategy=execution_strategy,
        abi_contracts=abi_contracts,
        module_graph=module_graph,
        verification_nodes=verification_nodes,
    )
    data = v2.model_dump(mode="json")

    # Preserve v1 fields that are not part of the v2 schema.
    for key, value in blueprint.model_dump(mode="json").items():
        if key not in data:
            data[key] = value

    # Manifest and module graph are kept in sync by ``_sync_manifest_and_module_graph``,
    # but deduplicate once more here before serialising.
    data["manifest"] = deduplicate_manifest_entries(data.get("manifest", []))
    data["module_graph"] = deduplicate_manifest_entries(data.get("module_graph", []))

    text = yaml.safe_dump(data, sort_keys=False, default_flow_style=False)
    try:
        yaml.safe_load(text)
    except Exception as exc:
        raise ValueError(f"blueprint.aero produced invalid YAML: {exc}") from exc
    path.write_text(text, encoding="utf-8")


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
    has_python = (
        pyproject.is_file() or setup_py.is_file() or bool(list(repo_path.rglob("*.py")))
    )
    cpp_sources = [
        p for p in repo_path.rglob("*") if p.suffix in {".cpp", ".c", ".h", ".hpp"}
    ]
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
        manifest.append(
            ManifestEntry(
                path="Cargo.toml", lang="toml", purpose="Rust workspace manifest"
            )
        )
        for member in members:
            manifest.append(
                ManifestEntry(
                    path=f"{member}/Cargo.toml",
                    lang="toml",
                    purpose="Workspace member crate",
                )
            )
            manifest.append(
                ManifestEntry(
                    path=f"{member}/src/lib.rs",
                    lang="rust",
                    purpose="Rust crate source",
                )
            )
    if pyproject.is_file():
        manifest.append(
            ManifestEntry(
                path="pyproject.toml", lang="toml", purpose="Python package metadata"
            )
        )
    elif setup_py.is_file():
        manifest.append(
            ManifestEntry(
                path="setup.py", lang="python", purpose="Python package setup"
            )
        )
    for src in sorted(repo_path.rglob("*.py")):
        if src.name.startswith("test_") or "/tests/" in str(
            src.relative_to(repo_path)
        ).replace("\\", "/"):
            continue
        if _is_path_inside(src, repo_path):
            manifest.append(
                ManifestEntry(
                    path=str(src.relative_to(repo_path)),
                    lang="python",
                    purpose="Python source",
                )
            )
    for hdr in sorted(cpp_sources):
        if _is_path_inside(hdr, repo_path):
            manifest.append(
                ManifestEntry(
                    path=str(hdr.relative_to(repo_path)),
                    lang=hdr.suffix.lstrip("."),
                    purpose="C/C++ source",
                )
            )

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

    blueprint = _build_autodetected_blueprint(
        repo_path,
        architecture,
        toolchains,
        languages,
        features,
        manifest,
        contracts,
        functions,
    )
    write_blueprint(blueprint, blueprint_path)
    return blueprint_path


def _build_autodetected_blueprint(
    repo_path: Path,
    architecture: str,
    toolchains: List[str],
    languages: List[str],
    features: List[str],
    manifest: List[ManifestEntry],
    contracts: List[ContractEntry],
    functions: List[FunctionSpec],
) -> Blueprint:
    """Create an in-memory Blueprint from auto-detected workspace artifacts."""
    return Blueprint(
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


def _sanitize_project_name(name: str) -> str:
    """Convert a directory name into a valid Python/Rust package identifier."""
    sanitized = "".join(c if c.isalnum() or c == "_" else "_" for c in name.lower())
    sanitized = sanitized.strip("_")
    if not sanitized or sanitized[0].isdigit():
        sanitized = "engine"
    return sanitized


def _workspace_has_sources(workspace_root: Path) -> bool:
    """Return True if *workspace_root* contains any real source/material files."""
    skip_names = {
        "blueprint.aero",
        "workspace_blueprint.yaml",
        "workspace_blueprint.yml",
        "workspace.aeroc",
    }
    for path in workspace_root.rglob("*"):
        if not path.is_file():
            continue
        if path.name in skip_names or path.name.startswith("."):
            continue
        try:
            if path.stat().st_size > 0:
                return True
        except OSError:
            continue
    return False


def ensure_workspace_blueprint(workspace_root: Path) -> Path:
    """Return ``workspace_root / blueprint.aero``, creating it if missing.

    For workspaces that already contain source code, a Blueprint v3.0.0 is
    inferred from the discovered artifacts. For truly empty workspaces, a
    minimal draft blueprint is synthesized from the closest standard template
    (``pure_python.aero`` by default) and marked as auto-generated so the
    synthesizer knows it still needs LLM synthesis.
    """
    workspace_root = Path(workspace_root).resolve()
    blueprint_path = workspace_root / "blueprint.aero"
    if blueprint_path.is_file():
        return blueprint_path

    from aero_forge.blueprint.schema import (
        ArtifactType,
        BlueprintStatus,
        BlueprintV3,
        BuildArtifact,
        ContextState,
        ExecutionStrategyV3,
        GenerationMethod,
        LLMContext,
        Metadata,
        ToolchainSpec,
        write_v3_blueprint,
    )

    # If the workspace already has source files, derive the blueprint from them
    # instead of overwriting with an empty template.
    if _workspace_has_sources(workspace_root):
        blueprint = generate_draft_v3_blueprint(workspace_root)
        # Source-derived blueprints are still heuristic sketches; they are not
        # LLM-initialized and must not be treated as finalized.
        blueprint.metadata.status = BlueprintStatus.draft
        blueprint.metadata.auto_generated = True
        blueprint.metadata.llm_initialized = False
        blueprint.metadata.transferable = False
        blueprint.metadata.generation_method = GenerationMethod.static_heuristic
        blueprint.llm_context.state = ContextState.raw
        write_v3_blueprint(blueprint, blueprint_path)
        logger.info("Auto-generated source-derived blueprint: %s", blueprint_path)
        return blueprint_path

    has_rust = (workspace_root / "Cargo.toml").is_file()
    has_python = (
        (workspace_root / "pyproject.toml").is_file()
        or (workspace_root / "setup.py").is_file()
        or bool(list(workspace_root.rglob("*.py")))
    )
    cpp_sources = [
        p for p in workspace_root.rglob("*") if p.suffix in {".cpp", ".c", ".h", ".hpp"}
    ]
    has_cpp = bool(cpp_sources)

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

    template_name = suggested_blueprint_template(architecture)
    template_text = ""
    try:
        template_text = load_template(template_name)
    except KeyError:
        logger.warning(
            "No blueprint template found for %s; using defaults", architecture
        )

    project_name = _sanitize_project_name(workspace_root.name)

    # Pull the human-readable intent from the template to seed the LLM context.
    description = f"Auto-generated {architecture} blueprint for empty workspace."
    if template_text:
        prompt_match = re.search(
            r'^prompt:\s*["\']?(.*?)["\']?$', template_text, re.MULTILINE
        )
        constraints_match = re.search(
            r'^constraints:\s*["\']?(.*?)["\']?$', template_text, re.MULTILINE
        )
        parts = []
        if prompt_match:
            parts.append(prompt_match.group(1).strip("\"'"))
        if constraints_match:
            parts.append(constraints_match.group(1).strip("\"'"))
        if parts:
            description = " ".join(parts)

    classification = classify_stack(architecture)
    toolchains = [
        ToolchainSpec(name=t) for t in classification.toolchains or ["python"]
    ]

    if architecture == INTENT_PURE_RUST:
        source_files = ["src/lib.rs"]
        artifact_type = ArtifactType.cargo_cdylib
    elif architecture == INTENT_HYBRID_RUST_PYTHON:
        source_files = ["src/lib.rs"]
        artifact_type = ArtifactType.cargo_cdylib
    elif has_cpp:
        source_files = [
            str(p.relative_to(workspace_root)) for p in cpp_sources[:1]
        ] or ["native/native.cpp"]
        artifact_type = ArtifactType.shared_library
    else:
        source_files = [f"src/{project_name}/core.py"]
        artifact_type = ArtifactType.python_extension

    primary_entrypoint = source_files[0] if source_files else ""

    blueprint = BlueprintV3(
        metadata=Metadata(
            schema_version="3.0.0",
            project_name=project_name,
            status=BlueprintStatus.draft,
            generation_method=GenerationMethod.static_heuristic,
            transferable=False,
            llm_initialized=False,
            auto_generated=True,
            description=description,
        ),
        llm_context=LLMContext(
            state=ContextState.raw,
            repository_summary=description,
            dependency_graph={},
            compute_hotspots=[],
        ),
        toolchains=toolchains,
        build_pipeline=[
            BuildArtifact(
                id=f"{project_name}_core",
                type=artifact_type,
                source_files=source_files,
                output_path=f"dist/{project_name}",
                description=description,
            )
        ],
        execution_strategy=ExecutionStrategyV3(
            primary_entrypoint=primary_entrypoint,
            runtime="python3",
        ),
    )

    write_v3_blueprint(blueprint, blueprint_path)
    logger.info(
        "Auto-generated minimal blueprint for empty workspace: %s", blueprint_path
    )
    return blueprint_path


class BlueprintCore:
    """Static workspace blueprint introspection helpers."""

    @staticmethod
    def autodetect(workspace_path: Path) -> Dict[str, Any]:
        """Return an in-memory blueprint schema for *workspace_path*.

        If no ``blueprint.aero`` exists, infer architecture, manifest, and
        exported functions by scanning the directory for Python, Rust, and C/C++
        artifacts.  The result is a JSON-serializable dictionary.
        """
        repo_path = Path(workspace_path).resolve()
        blueprint_path = repo_path / "blueprint.aero"
        existing_md = repo_path / "BLUEPRINT.md"
        if blueprint_path.is_file():
            return parse_blueprint(blueprint_path).model_dump(mode="json")
        if existing_md.is_file():
            return BlueprintValidator(existing_md).blueprint.model_dump(mode="json")

        cargo_toml = repo_path / "Cargo.toml"
        pyproject = repo_path / "pyproject.toml"
        setup_py = repo_path / "setup.py"

        has_rust = cargo_toml.is_file()
        has_python = (
            pyproject.is_file()
            or setup_py.is_file()
            or bool(list(repo_path.rglob("*.py")))
        )
        cpp_sources = [
            p
            for p in repo_path.rglob("*")
            if p.suffix in {".cpp", ".c", ".h", ".hpp", ".cc", ".cxx"}
        ]
        has_cpp = bool(cpp_sources)

        members: List[str] = []
        if has_rust:
            cargo_data = _parse_cargo_toml(cargo_toml)
            members = cargo_data.get("workspace", {}).get("members", [])
            if not members and "package" in cargo_data:
                members = [cargo_data["package"].get("name", "rust_core")]

        if has_rust and has_python and has_cpp:
            architecture = INTENT_TRI_POLYGLOT_RUST_CPP_PYTHON
        elif has_rust and has_python:
            architecture = INTENT_HYBRID_RUST_PYTHON
        elif has_python and has_cpp:
            architecture = INTENT_HYBRID_CPP_PYTHON
        elif has_rust and has_cpp:
            architecture = INTENT_HYBRID_CPP_RUST
        elif has_rust:
            architecture = INTENT_PURE_RUST
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
            manifest.append(
                ManifestEntry(
                    path="Cargo.toml", lang="toml", purpose="Rust workspace manifest"
                )
            )
            for member in members:
                manifest.append(
                    ManifestEntry(
                        path=f"{member}/Cargo.toml",
                        lang="toml",
                        purpose="Workspace member crate",
                    )
                )
                manifest.append(
                    ManifestEntry(
                        path=f"{member}/src/lib.rs",
                        lang="rust",
                        purpose="Rust crate source",
                    )
                )
        if pyproject.is_file():
            manifest.append(
                ManifestEntry(
                    path="pyproject.toml",
                    lang="toml",
                    purpose="Python package metadata",
                )
            )
        elif setup_py.is_file():
            manifest.append(
                ManifestEntry(
                    path="setup.py", lang="python", purpose="Python package setup"
                )
            )
        for src in sorted(repo_path.rglob("*.py")):
            rel = str(src.relative_to(repo_path)).replace("\\", "/")
            if src.name.startswith("test_") or "/tests/" in rel:
                continue
            if _is_path_inside(src, repo_path):
                manifest.append(
                    ManifestEntry(path=rel, lang="python", purpose="Python source")
                )
        for hdr in sorted(cpp_sources):
            if _is_path_inside(hdr, repo_path):
                manifest.append(
                    ManifestEntry(
                        path=str(hdr.relative_to(repo_path)).replace("\\", "/"),
                        lang=hdr.suffix.lstrip("."),
                        purpose="C/C++ source",
                    )
                )

        functions: List[FunctionSpec] = []
        try:
            functions = discover_project(repo_path)
        except Exception as exc:
            logger.warning(
                "Could not discover Python functions in %s: %s", repo_path, exc
            )

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

        blueprint = _build_autodetected_blueprint(
            repo_path,
            architecture,
            toolchains,
            languages,
            features,
            manifest,
            contracts,
            functions,
        )
        return blueprint.model_dump(mode="json")
