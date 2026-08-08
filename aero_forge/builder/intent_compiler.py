"""Layer 0 intent compiler: natural-language prompt -> validated blueprint v2.0.0."""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from jsonschema import Draft7Validator, ValidationError as JsonSchemaValidationError

from aero_forge.blueprint import (
    ABIContract,
    Blueprint,
    BlueprintSchemaV2,
    ContractEntry,
    LLMConfig,
    ManifestEntry,
    write_blueprint,
)
from aero_forge.blueprint.schema import (
    BoundaryEdgeSpec,
    BoundaryContractType,
    PolyglotGraphBlueprint,
    PolyglotNodeSpec,
)
from aero_forge.config import Tier
from aero_forge.llm.clients import get_llm_client

logger = logging.getLogger("aero_forge.intent")


def _accel_log(level: str, message: str) -> None:
    """Append a structured line to the per-session accelerator log if set."""
    log_path = os.environ.get("AERO_FORGE_ACCEL_LOG")
    if not log_path:
        return
    try:
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f'{timestamp} [{level}] {message}\n')
    except Exception:
        pass


class IntentCompilerError(Exception):
    """Raised when the intent compiler cannot produce a valid blueprint."""


_SYSTEM_PROMPT = """YOU ARE THE AERO-FORGE INTENT COMPILER (SYSTEM LAYER 0).
YOUR SOLE PURPOSE IS TO CONVERT UNSTRUCTURED USER DOMAIN PROMPTS INTO A VALID, HIGH-PRECISION 'blueprint.aero' SPECIFICATION IN STRICT JSON FORMAT.

STRICT OPERATIONAL RULES:
1. OUTPUT ONLY VALID JSON. DO NOT INCLUDE PREFACES, FOOTERS, MARKDOWN EXPLANATIONS, OR CODE BLOCK TEXT OUTSIDE THE JSON OBJECT.
2. DO NOT GENERATE PLACEHOLDER STUBS ("// TODO", "pass"). EVERY MODULE AND CONTRACT MUST BE FULLY SPECIFIED.
3. MAP ALL CLI FLAGS EXPLICITLY IN 'execution_strategy.cli_contract'. YOU MUST INFER ARGUMENT TYPES, SHORT FLAGS, DEFAULT VALUES, AND DESTINATION VARIABLES.
4. MAP ALL CROSS-LANGUAGE FUNCTION CALLS TO EXPLICIT 'abi_contracts'. DEFINE INPUT/OUTPUT DATA TYPES USING C-ABI PRESERVED KEYWORDS (u32, i32, usize, f64, double*, int32_t).
5. CREATE AT LEAST TWO 'verification_nodes' THAT TEST CLI ARGUMENT PARSING AND NUMERICAL OUTPUT METRICS.
6. FILE BOUNDARY CONSTRAINT: ONLY list files in 'module_graph' that are explicitly required by the prompt or are minimal build config files (e.g. 'Cargo.toml', 'pyproject.toml', 'CMakeLists.txt'). DO NOT rewrite, regenerate, or reference unrelated source files, CLI files, tests, or documentation unless the user explicitly asks for them.
7. ARTIFACT HYGIENE: NEVER stage, commit, or list generated binary targets ('*.so', '*.pyd', '*.dll', '*.dylib', '*.wasm', '*.whl'), virtual environments ('.venv/', 'venv/', 'pyvenv.cfg'), distribution metadata ('*.egg-info/', 'dist/', 'build/'), or package archives ('*.aeroc', '*.aerozip', '*.zip', '*.tar*') as project deliverables.

The JSON must conform to BlueprintSchemaV2.0.0 with these top-level keys:
- metadata: {schema_version: "2.0.0", project_name: "...", domain_target: "..."}
- execution_strategy: {primary_entrypoint: {path, runtime, wrapper_generation}, cli_contract: {parser_type, flags}, run_spec: {working_dir, env_vars, timeout_seconds}}
- abi_contracts: list of {contract_id, target_language, binding_framework, export_symbol, c_symbol_alias, header_path, memory_model, signature: {inputs: [{name, type}], outputs: [{name, type}]}}. For PyO3 bridge functions, use explicit Rust signatures such as "&PyArray2<f64>", "Python", "usize", and "PyResult<...>".
- module_graph: list of {path, lang, purpose, rust_signature?}. When a hybrid Rust/Python extension is requested, list concrete submodule files under "src/" (e.g. "src/ops.rs", "src/array.rs"), the main "src/lib.rs", Python wrapper files, and test files under "tests/".
- verification_nodes: list of {test_id, execution_cmd, expected_exit_code, stdout_match_patterns, stderr_prohibited_patterns, numerical_assertions}
- cargo_dependencies: object mapping crate name to version spec or {version, features}. Always include "pyo3" for PyO3 bindings, "numpy" for numpy-rust array types, and "rayon" when parallel iterators or sliding-window logic are requested. Example: {"pyo3": "0.20.3", "numpy": "0.21", "rayon": "1.10"}.
"""


_GRAPH_SYSTEM_PROMPT = """YOU ARE THE AERO-FORGE INTENT COMPILER (SYSTEM LAYER 0).
YOUR SOLE PURPOSE IS TO CONVERT UNSTRUCTURED USER DOMAIN PROMPTS INTO A VALID `PolyglotGraphBlueprint` JSON SPECIFICATION.

STRICT OPERATIONAL RULES:
1. OUTPUT ONLY VALID JSON. DO NOT INCLUDE PREFACES, FOOTERS, MARKDOWN EXPLANATIONS, OR CODE BLOCK TEXT OUTSIDE THE JSON OBJECT.
2. `architecture` MUST be "graph_polyglot" unless the request is explicitly a single-language project. Never fall back to legacy target strings like "tri_polyglot_rust_cpp_python" or generic stub names like "sliding_window_dtw" / "orchestrate".
3. `nodes` MUST be a list of objects with: `node_id` (unique), `lang` (language name such as python, rust, cpp, go, csharp, java, zig, mojo, nim, or any other target language), `toolchain` (one of: gcc, clang, clang++, cargo, go, nvcc, zig, dotnet, maturin, python, javac, cmake, or a language-specific toolchain), `source_files` (list of relative paths, optional), `compiler_flags` (list, optional), `exports` (list, optional). If the requested language is not in the built-in set (python, rust, cpp, go, csharp, java), the EmitterRegistry will JIT-synthesize a `PolyglotEmitterPlugin` for it.
4. `edges` MUST be a list of cross-language FFI boundary objects with: `source`, `target` (both matching `node_id`s), `boundary_type` (one of: C_ABI, PYO3_MATURIN, WASM_WASI, JNI, CGO, PINVOKE, CUDA_HIP_C), `symbol`, `args` (list of primitive type names: int32, int64, float32, float64, pointer), `return_type` (primitive type name or ""), `is_zero_copy` (boolean).
5. `primary_entrypoint` MUST be the user's requested entrypoint path (e.g. "python_interface/main.py") instead of any default like "run_shell.py".
6. `build_script` MUST be the user's requested root build script path (e.g. "build.sh") when specified.
7. Enforce a DAG: no cycles among `edges`. The `source` node must be an earlier stage than the `target` node.
8. Enforce zero-copy memory layout compatibility: scalars pass by value; vectors/tensors pass as raw pointer + length + capacity triples (`data_ptr`, `length`, `capacity`).
9. DO NOT generate placeholder stubs ("// TODO", "pass", "todo!()"). Every node and contract must be fully specified.
10. FILE BOUNDARY CONSTRAINT: only list `source_files` explicitly required by the prompt. Include custom build files such as `build.sh` and `cpp_engine/CMakeLists.txt` exactly as requested.
11. ARTIFACT HYGIENE: NEVER stage, commit, or list generated binary targets, virtual environments, distribution metadata, or package archives as deliverables.
12. TOOLCHAIN FLAG HYGIENE: For `cargo` and `maturin` nodes, do NOT include `--release` in `compiler_flags`; the dispatcher always adds it. Only include extra Cargo flags (e.g. `--features <feature>`) when the feature is declared in the node's manifest. Do not include `-C` rustc flags in `compiler_flags`; pass those via the `RUSTFLAGS` environment if necessary.

Example JSON shape:
{
  "project": "audio_synth",
  "architecture": "graph_polyglot",
  "primary_entrypoint": "python_interface/main.py",
  "build_script": "build.sh",
  "nodes": [
    {"node_id": "cpp_core", "lang": "cpp", "toolchain": "cmake", "source_files": ["cpp_engine/src/synth.cpp", "cpp_engine/CMakeLists.txt"], "exports": ["synth_render"]},
    {"node_id": "go_server", "lang": "go", "toolchain": "go", "source_files": ["go_server/main.go"], "exports": ["start_server"]}
  ],
  "edges": [
    {"source": "cpp_core", "target": "go_server", "boundary_type": "CGO", "symbol": "synth_render", "args": ["pointer", "int64"], "return_type": "int64", "is_zero_copy": true}
  ],
  "output_dir": "./dist",
  "metadata": {}
}
"""


def _strip_markdown_fences(text: str) -> str:
    """Remove optional JSON/YAML code fences from an LLM response.

    Tolerates language/path labels (e.g. `` ```json:plan.json ``) and surrounding
    prose by only stripping the first and last fence tokens.
    """
    text = text.strip()
    # Strip opening fence with optional language/path label.
    text = re.sub(r"^```\s*(?:\w+)?\s*(?::\s*[^\n\r]*)?\s*\r?\n", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*(?:\w+)?\s*(?::\s*[^\n\r]*)?\s*$", "", text, flags=re.IGNORECASE)
    # Strip trailing fence.
    if text.endswith("```"):
        text = text[:-3].strip()
    return text.strip()


def _find_balanced_json(text: str) -> Optional[str]:
    """Return the first balanced JSON object literal found in *text*."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for i, ch in enumerate(text[start:], start=start):
            if in_string:
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        break
    return None


def _extract_json(raw: str) -> Any:
    """Parse a JSON object from a raw LLM response, tolerating surrounding text."""
    cleaned = _strip_markdown_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Try to locate the first balanced JSON object.
    balanced = _find_balanced_json(cleaned)
    if balanced:
        try:
            return json.loads(balanced)
        except json.JSONDecodeError:
            pass
    # YAML is more tolerant of trailing commas and unquoted strings.
    try:
        data = yaml.safe_load(cleaned)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    raise IntentCompilerError("No JSON object found in LLM response")


def _normalize_cli_type(value: Any) -> str:
    if value is None:
        return "string"
    lowered = str(value).lower().strip()
    synonyms = {
        "str": "string",
        "string": "string",
        "json_string": "string",
        "json": "string",
        "json_array": "string",
        "json_array_f64": "string",
        "json_array_2d_f64": "string",
        "json_array_string": "string",
        "integer": "int",
        "int": "int",
        "boolean": "bool",
        "bool": "bool",
        "store_true": "bool",
        "store_false": "bool",
        "float": "float",
        "double": "float",
    }
    normalized = synonyms.get(lowered, lowered)
    if normalized in ("string", "int", "bool", "float"):
        return normalized
    return "string"


def _normalize_flag_name(value: Any) -> str:
    if value is None:
        return ""
    name = str(value).strip().lstrip("-")
    return name.replace("-", "_")


def _normalize_v2_data(data: Any) -> Any:
    """Coerce common LLM output variants into the strict BlueprintSchemaV2 shape."""
    if not isinstance(data, dict):
        return data

    # metadata
    metadata = data.get("metadata") or {}
    if isinstance(metadata, dict):
        if metadata.get("domain_target"):
            metadata["domain_target"] = str(metadata["domain_target"]).lower().replace(" ", "_")
        data["metadata"] = metadata

    # execution_strategy
    exec_strategy = data.get("execution_strategy") or {}
    if isinstance(exec_strategy, dict):
        primary = exec_strategy.get("primary_entrypoint") or {}
        if isinstance(primary, dict) and primary.get("runtime"):
            runtime = str(primary["runtime"]).lower()
            if "python" in runtime:
                primary["runtime"] = "python3"
            primary["wrapper_generation"] = bool(primary.get("wrapper_generation", True))
            exec_strategy["primary_entrypoint"] = primary

        cli_contract = exec_strategy.get("cli_contract") or {}
        if isinstance(cli_contract, dict):
            flags = cli_contract.get("flags") or []
            normalized_flags = []
            for flag in flags:
                if not isinstance(flag, dict):
                    continue
                normalized: Dict[str, Any] = {
                    "name": _normalize_flag_name(
                        flag.get("name") or flag.get("long") or flag.get("long_flag") or flag.get("dest") or flag.get("dest_var", "")
                    ),
                    "short": str(flag.get("short") or flag.get("short_flag") or "").strip().lstrip("-"),
                    "type": _normalize_cli_type(flag.get("type") or "string"),
                    "required": bool(flag.get("required", False)),
                    "default": flag.get("default", None),
                    "choices": list(flag.get("choices", [])) if flag.get("choices") else [],
                    "help": str(flag.get("help", "")),
                    "dest_var": str(flag.get("dest_var") or flag.get("dest") or ""),
                }
                normalized_flags.append(normalized)
            cli_contract["flags"] = normalized_flags
            exec_strategy["cli_contract"] = cli_contract

        run_spec = exec_strategy.get("run_spec") or {}
        if isinstance(run_spec, dict):
            if "timeout_seconds" in run_spec:
                try:
                    run_spec["timeout_seconds"] = int(run_spec["timeout_seconds"])
                except (TypeError, ValueError):
                    run_spec["timeout_seconds"] = 30
            exec_strategy["run_spec"] = run_spec
        data["execution_strategy"] = exec_strategy

    # abi_contracts
    abi_contracts = data.get("abi_contracts") or []
    normalized_abis = []
    for abi in abi_contracts:
        if not isinstance(abi, dict):
            continue
        normalized_abi: Dict[str, Any] = dict(abi)
        target = str(normalized_abi.get("target_language") or "").lower().strip()
        target_synonyms = {"c": "cpp", "c++": "cpp", "py": "python"}
        normalized_abi["target_language"] = target_synonyms.get(target, target) or "cpp"

        binding = str(normalized_abi.get("binding_framework") or "").lower().strip().replace("-", "_")
        binding_synonyms = {
            "pyo3": "pyo3",
            "ctypes": "ctypes",
            "cabi": "c_abi",
            "c_abi": "c_abi",
            "cpython_api": "pyo3",
            "python_api": "pyo3",
            "python_capi": "pyo3",
            "c_api": "c_abi",
            "python_native": "c_abi",
            "native_python": "c_abi",
            "direct": "c_abi",
            "ffi": "c_abi",
            "cffi": "ctypes",
            "python_cffi": "ctypes",
            "cython": "c_abi",
            "python_cython": "c_abi",
        }
        normalized_abi["binding_framework"] = binding_synonyms.get(binding, binding) or "c_abi"
        c_alias = normalized_abi.get("c_symbol_alias")
        if c_alias is None:
            normalized_abi["c_symbol_alias"] = ""

        memory = re.sub(r"[^a-z0-9_]", "_", str(normalized_abi.get("memory_model") or "").lower().strip())
        memory_synonyms = {
            "owned": "callee_allocates",
            "owned_results": "callee_allocates",
            "borrowed": "shared_pyo3",
            "shared": "shared_pyo3",
            "caller": "caller_allocates",
            "callee": "callee_allocates",
            "manual": "caller_allocates",
            "auto": "callee_allocates",
            "automatic": "callee_allocates",
            "stack": "callee_allocates",
            "stack_automatic": "callee_allocates",
            "stack_automatic_conversion": "callee_allocates",
            "python_gc": "callee_allocates",
            "gc": "callee_allocates",
            "ptr_with_length": "caller_allocates",
            "shared_ptr": "shared_pyo3",
            "c": "caller_allocates",
            "c_owned": "callee_allocates",
            "rust_owned": "callee_allocates",
        }
        memory = memory_synonyms.get(memory, memory) or "caller_allocates"
        if memory not in {"callee_allocates", "caller_allocates", "shared_pyo3"}:
            memory = "caller_allocates"
        normalized_abi["memory_model"] = memory
        normalized_abis.append(normalized_abi)
    data["abi_contracts"] = normalized_abis

    # module_graph
    module_graph = data.get("module_graph") or []
    normalized_graph = []
    for node in module_graph:
        if not isinstance(node, dict):
            continue
        normalized_node = dict(node)
        lang = str(normalized_node.get("lang") or normalized_node.get("language") or "python").lower()
        if "python" in lang:
            lang = "python"
        elif lang in {"rust", "cpp", "c++"}:
            lang = {"c++": "cpp"}.get(lang, lang)
        else:
            lang = "python"
        normalized_node["lang"] = lang
        normalized_graph.append(normalized_node)
    data["module_graph"] = normalized_graph

    # cargo_dependencies
    cargo_deps = data.get("cargo_dependencies") or {}
    if isinstance(cargo_deps, dict):
        data["cargo_dependencies"] = cargo_deps
    else:
        data["cargo_dependencies"] = {}

    return data


def _abi_type_to_py(c_type: str) -> str:
    """Map a C ABI type to a Python type annotation."""
    t = (c_type or "").strip()
    lowered = t.lower()
    scalar_ints = {"u32", "i32", "usize", "int32_t", "i64", "u64", "int"}
    scalar_floats = {"f64", "f32", "double", "float"}
    if lowered in scalar_ints:
        return "int"
    if lowered in scalar_floats:
        return "float"
    if lowered in {"bool"}:
        return "bool"
    if lowered.endswith("*"):
        inner = lowered.rstrip("*").strip()
        if inner in {"float", "f32", "f64", "double"}:
            return "list[float]"
        if inner in {"int", "i32", "i64", "u32", "usize", "int32_t"}:
            return "list[int]"
        return "list"
    if lowered.startswith("*const ") or lowered.startswith("*mut "):
        inner = lowered.split(None, 1)[1]
        if inner in {"float", "f32", "f64", "double"}:
            return "list[float]"
        if inner in {"int", "i32", "i64", "u32", "usize", "int32_t"}:
            return "list[int]"
        return "list"
    return "Any"


def _abi_contract_to_contract_entry(abi: ABIContract) -> Optional[ContractEntry]:
    """Convert an ABIContract into a Python-style ContractEntry signature."""
    sig = abi.signature
    inputs = sig.get("inputs", []) if isinstance(sig, dict) else []
    outputs = sig.get("outputs", []) if isinstance(sig, dict) else []
    if not inputs:
        return None
    args = [(entry["name"], _abi_type_to_py(entry["type"])) for entry in inputs]
    if not outputs:
        return_type = "None"
    elif len(outputs) == 1:
        return_type = _abi_type_to_py(outputs[0]["type"])
    else:
        return_type = f"tuple[{', '.join(_abi_type_to_py(o['type']) for o in outputs)}]"
    arg_str = ", ".join(f"{name}: {typ}" for name, typ in args)
    signature = f"def {abi.export_symbol}({arg_str}) -> {return_type}"
    return ContractEntry(
        name=abi.export_symbol,
        signature=signature,
        language=abi.target_language,
        python_name=abi.export_symbol,
        purpose=f"ABI contract for {abi.contract_id}",
    )


def _contracts_from_abi(abi_contracts: List[ABIContract]) -> List[ContractEntry]:
    """Convert ABI contracts into synthesisable Python-style contract entries."""
    entries: List[ContractEntry] = []
    for abi in abi_contracts:
        entry = _abi_contract_to_contract_entry(abi)
        if entry:
            entries.append(entry)
    return entries


def _infer_architecture(languages: set) -> str:
    """Map a set of language tags to the closest aero-forge architecture string."""
    from aero_forge.orchestrator.router import (
        BUILD_INTENT_HYBRID_CPP_PYTHON,
        BUILD_INTENT_HYBRID_CPP_RUST,
        BUILD_INTENT_HYBRID_RUST_PYTHON,
        BUILD_INTENT_PURE_PYTHON,
        BUILD_INTENT_PURE_RUST,
        BUILD_INTENT_TRI_POLYGLOT_RUST_CPP_PYTHON,
    )

    has_python = "python" in languages
    has_rust = "rust" in languages
    has_cpp = "cpp" in languages or "c++" in languages

    if has_python and has_rust and has_cpp:
        return BUILD_INTENT_TRI_POLYGLOT_RUST_CPP_PYTHON
    if has_rust and has_cpp:
        return BUILD_INTENT_HYBRID_CPP_RUST
    if has_python and has_cpp:
        return BUILD_INTENT_HYBRID_CPP_PYTHON
    if has_python and has_rust:
        return BUILD_INTENT_HYBRID_RUST_PYTHON
    if has_rust:
        return BUILD_INTENT_PURE_RUST
    return BUILD_INTENT_PURE_PYTHON


class IntentCompiler:
    """Compile an unstructured user prompt into a validated Blueprint v2.0.0."""

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        max_schema_retries: int = 3,
        llm_client: Optional[Any] = None,
        config_override: Optional[Any] = None,
    ):
        self.provider = provider or "deepseek"
        self.model = model
        self.api_key = api_key
        self.max_retries = max(1, max_retries)
        self.max_schema_retries = max(1, max_schema_retries)
        self._llm_client = llm_client
        self.config_override = config_override

    def compile_prompt(
        self,
        prompt_text: str,
        output_dir: Optional[str | Path] = None,
        project_name: Optional[str] = None,
    ) -> Blueprint:
        """Convert *prompt_text* into a validated ``Blueprint`` and write ``blueprint.aero``.

        Returns the constructed ``Blueprint``. The schema validation/repair loop
        re-queries the LLM with concrete JSON Schema errors up to
        ``max_schema_retries`` times.
        """
        client = self._llm_client
        if client is None:
            client = get_llm_client(
                self.provider,
                model=self.model,
                max_retries=self.max_retries,
                api_key=self.api_key,
                config_override=self.config_override,
                raise_on_error=True,
                tier=Tier.REASONING,
            )

        schema = BlueprintSchemaV2.model_json_schema()
        validator = Draft7Validator(schema)

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ]

        last_error: Optional[Exception] = None
        for attempt in range(self.max_schema_retries):
            raw = client.generate(messages, temperature=0.2, max_tokens=4096)
            if not raw:
                _accel_log("warning", f"intent_compiler.compile_prompt attempt {attempt + 1}: LLM returned an empty response")
                last_error = IntentCompilerError("LLM returned an empty response")
                messages.append({"role": "assistant", "content": ""})
                messages.append(
                    {"role": "user", "content": f"Attempt {attempt + 1} returned empty. Please return valid JSON."}
                )
                continue

            try:
                data = _normalize_v2_data(_extract_json(raw))
                validator.validate(data)
                v2 = BlueprintSchemaV2.model_validate(data)
            except (JsonSchemaValidationError, Exception) as exc:
                last_error = exc
                _accel_log(
                    "warning",
                    f"intent_compiler.compile_prompt attempt {attempt + 1}: schema validation failed; "
                    f"raw preview: {raw[:800]!r}; error: {exc}",
                )
                logger.warning("Intent JSON schema validation failed (attempt %d): %s", attempt + 1, exc)
                messages.append({"role": "assistant", "content": raw})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Schema validation failed: {exc}. "
                            "Return corrected JSON only, with no markdown or explanatory text."
                        ),
                    }
                )
                continue

            blueprint = self._v2_to_blueprint(
                v2,
                prompt_text,
                output_dir,
                project_name,
            )
            if output_dir:
                out_path = Path(output_dir)
                out_path.mkdir(parents=True, exist_ok=True)
                write_blueprint(blueprint, out_path / "blueprint.aero")
            return blueprint

        raise IntentCompilerError(
            f"Failed to compile intent after {self.max_schema_retries} schema validation attempts: {last_error}"
        )

    def _v2_to_blueprint(
        self,
        v2: BlueprintSchemaV2,
        prompt_text: str,
        output_dir: Optional[str | Path],
        project_name: Optional[str],
    ) -> Blueprint:
        """Convert a validated ``BlueprintSchemaV2`` into a pipeline ``Blueprint``."""
        from aero_forge.orchestrator.router import (
            default_manifest_for_architecture,
            toolchains_for_intent,
        )

        project = project_name or v2.metadata.get("project_name") or "aero_forge_project"
        languages: set = set()
        manifest_entries: List[ManifestEntry] = []

        for node in v2.module_graph:
            path = node.get("path", "")
            lang = (node.get("lang") or node.get("language") or "").lower()
            purpose = node.get("purpose", "")
            if lang:
                languages.add(lang)
            manifest_entries.append(
                ManifestEntry(path=path, lang=lang or "python", purpose=purpose)
            )

        for abi in v2.abi_contracts:
            target = (abi.target_language or "").lower()
            if target:
                languages.add(target)
            if abi.binding_framework == "pyo3":
                languages.add("rust")
                languages.add("python")
            elif abi.binding_framework == "c_abi":
                languages.add("cpp")

        architecture = _infer_architecture(languages)
        if not manifest_entries:
            manifest_entries = [
                ManifestEntry(path=e["path"], lang=e["lang"], purpose=e["purpose"])
                for e in default_manifest_for_architecture(architecture, project)
            ]

        resolved_output_dir = Path(output_dir) / "dist" if output_dir else Path("./dist")

        modification_plan = self._build_modification_plan(v2, output_dir)

        metadata = dict(v2.metadata)
        metadata.setdefault("schema_version", "2.0.0")
        metadata["llm_initialized"] = "true"
        metadata["auto_generated"] = "true"
        metadata["generation_method"] = "llm_synthesized"

        return Blueprint(
            project=project,
            architecture=architecture,
            toolchains=toolchains_for_intent(architecture),
            manifest=manifest_entries,
            contracts=_contracts_from_abi(v2.abi_contracts),
            functions=[],
            output_dir=resolved_output_dir,
            llm=LLMConfig(provider=self.provider, model=self.model or ""),
            prompt=prompt_text,
            constraints="",
            languages=sorted(languages) if languages else ["python"],
            features=[],
            execution_strategy=v2.execution_strategy,
            abi_contracts=v2.abi_contracts,
            verification_nodes=v2.verification_nodes,
            metadata=metadata,
            module_graph=v2.module_graph,
            cargo_dependencies=v2.cargo_dependencies,
            modification_plan=modification_plan,
        )

    def _build_modification_plan(
        self,
        v2: BlueprintSchemaV2,
        output_dir: Optional[str | Path],
    ) -> Dict[str, Any]:
        """Classify every requested file as CREATE or MODIFY based on workspace state."""
        workspace = Path(output_dir) if output_dir else None
        actions: List[Dict[str, str]] = []
        for node in v2.module_graph:
            path = node.get("path", "")
            exists = workspace is not None and (workspace / path).is_file() if path else False
            actions.append({
                "path": path,
                "action": "MODIFY" if exists else "CREATE",
                "lang": node.get("lang") or node.get("language") or "python",
            })
        return {"intent": "incremental_update", "actions": actions}

    def compile_prompt_to_graph(
        self,
        prompt_text: str,
        output_dir: Optional[str | Path] = None,
        project_name: Optional[str] = None,
    ) -> PolyglotGraphBlueprint:
        """Compile *prompt_text* into a validated ``PolyglotGraphBlueprint``.

        The LLM is asked directly for a `graph_polyglot` JSON blueprint. If the
        model returns a legacy v2 blueprint, it is lowered into the graph
        representation used by ``GraphPolyglotMaterializer``.
        """
        client = self._llm_client
        if client is None:
            client = get_llm_client(
                self.provider,
                model=self.model,
                max_retries=self.max_retries,
                api_key=self.api_key,
                config_override=self.config_override,
                raise_on_error=True,
                tier=Tier.REASONING,
            )

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": _GRAPH_SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ]

        last_error: Optional[Exception] = None
        for attempt in range(self.max_schema_retries):
            raw = client.generate(messages, temperature=0.2, max_tokens=4096)
            if not raw:
                _accel_log("warning", f"intent_compiler.compile_prompt_to_graph attempt {attempt + 1}: LLM returned an empty response")
                last_error = IntentCompilerError("LLM returned an empty response")
                messages.append({"role": "assistant", "content": ""})
                messages.append(
                    {"role": "user", "content": f"Attempt {attempt + 1} returned empty. Please return valid JSON."}
                )
                continue

            try:
                data = _extract_json(raw)
            except Exception as exc:
                last_error = exc
                _accel_log(
                    "warning",
                    f"intent_compiler.compile_prompt_to_graph attempt {attempt + 1}: JSON extraction failed; "
                    f"raw preview: {raw[:800]!r}; error: {exc}",
                )
                logger.warning("Intent JSON extraction failed (attempt %d): %s", attempt + 1, exc)
                messages.append({"role": "assistant", "content": raw})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Could not extract JSON: {exc}. "
                            "Return corrected JSON only, with no markdown or explanatory text."
                        ),
                    }
                )
                continue

            # Primary path: a native `graph_polyglot` blueprint.
            try:
                graph = PolyglotGraphBlueprint.model_validate(data)
                graph.metadata["prompt"] = prompt_text
                return graph
            except Exception as graph_exc:
                logger.debug("Graph blueprint validation failed (attempt %d): %s", attempt + 1, graph_exc)

            # Fallback path: legacy v2 blueprint lowered to graph.
            try:
                normalized = _normalize_v2_data(data)
                v2 = BlueprintSchemaV2.model_validate(normalized)
            except Exception as exc:
                last_error = exc
                _accel_log(
                    "warning",
                    f"intent_compiler.compile_prompt_to_graph attempt {attempt + 1}: schema validation failed; "
                    f"raw preview: {raw[:800]!r}; error: {exc}",
                )
                logger.warning("Intent schema validation failed (attempt %d): %s", attempt + 1, exc)
                messages.append({"role": "assistant", "content": raw})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Schema validation failed: {exc}. "
                            "Return corrected JSON only, with no markdown or explanatory text."
                        ),
                    }
                )
                continue

            return self._v2_to_graph_blueprint(
                v2,
                prompt_text,
                output_dir,
                project_name,
            )

        raise IntentCompilerError(
            f"Failed to compile intent after {self.max_schema_retries} schema validation attempts: {last_error}"
        )

    def _v2_to_graph_blueprint(
        self,
        v2: BlueprintSchemaV2,
        prompt_text: str,
        output_dir: Optional[str | Path],
        project_name: Optional[str],
    ) -> PolyglotGraphBlueprint:
        """Lower a validated ``BlueprintSchemaV2`` into a ``PolyglotGraphBlueprint``."""
        project = project_name or v2.metadata.get("project_name") or "aero_forge_project"

        nodes: List[PolyglotNodeSpec] = []
        node_ids: set = set()
        for node in v2.module_graph:
            node_id = Path(node.get("path", "module")).stem or "module"
            if node_id in node_ids:
                base = node_id
                suffix = 2
                while f"{base}_{suffix}" in node_ids:
                    suffix += 1
                node_id = f"{base}_{suffix}"
            node_ids.add(node_id)
            nodes.append(
                PolyglotNodeSpec(
                    node_id=node_id,
                    lang=node.get("lang") or node.get("language") or "python",
                    source_files=[node.get("path", "")],
                    extra={"purpose": node.get("purpose", "")},
                )
            )

        edges: List[BoundaryEdgeSpec] = []
        for abi in v2.abi_contracts:
            binding = str(abi.binding_framework or "c_abi")
            try:
                boundary_type = BoundaryContractType(binding)
            except ValueError:
                boundary_type = BoundaryContractType.C_ABI

            inputs = abi.signature.get("inputs", []) if abi.signature else []
            outputs = abi.signature.get("outputs", []) if abi.signature else []

            source_lang = {
                "pyo3": "rust",
                "ctypes": "python",
                "c_abi": "cpp",
                "raw_c": "c",
                "c": "c",
                "cffi": "c",
                "cxx": "cpp",
            }.get(binding, "cpp")
            target_lang = (abi.target_language or "rust").lower()

            for lang in (source_lang, target_lang):
                if lang not in node_ids:
                    node_ids.add(lang)
                    nodes.append(
                        PolyglotNodeSpec(
                            node_id=lang,
                            lang=lang,
                            extra={"role": "language_endpoint"},
                        )
                    )

            edges.append(
                BoundaryEdgeSpec(
                    source=source_lang,
                    target=target_lang,
                    boundary_type=boundary_type,
                    symbol=abi.export_symbol or abi.contract_id or "ffi_symbol",
                    args=[inp.get("type", "int64") for inp in inputs],
                    return_type=(outputs[0].get("type", "") if outputs else ""),
                )
            )

        resolved_output_dir = str(Path(output_dir) / "dist") if output_dir else "./dist"

        primary_entrypoint = ""
        if v2.execution_strategy:
            primary_entrypoint = v2.execution_strategy.primary_entrypoint.get("path", "")

        build_script = v2.execution_strategy.run_spec.get("command", "") if v2.execution_strategy else ""

        return PolyglotGraphBlueprint(
            project=project,
            nodes=nodes,
            edges=edges,
            output_dir=resolved_output_dir,
            primary_entrypoint=primary_entrypoint or "run_shell.py",
            build_script=build_script or None,
            metadata={
                "prompt": prompt_text,
                **v2.metadata,
            },
        )
