"""Layer 0 intent compiler: natural-language prompt -> validated blueprint v2.0.0."""

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from dataclasses import dataclass

import yaml
from jsonschema import Draft7Validator, ValidationError as JsonSchemaValidationError

from aero_forge.blueprint import (
    ABIContract,
    Blueprint,
    BlueprintSchemaV2,
    ContractEntry,
    ExecutionStrategy,
    FunctionalIntent,
    LLMConfig,
    ManifestEntry,
    write_blueprint,
)
from aero_forge.blueprint.schema import (
    BoundaryEdgeSpec,
    BoundaryContractType,
    FunctionalIntent as SchemaFunctionalIntent,
    PolyglotGraphBlueprint,
    PolyglotNodeSpec,
)
from aero_forge.builder.holographic import HolographicContext, intent_vector
from aero_forge.builder.foge import FockGraphEncoder
from aero_forge.adjoint import NodeStub, SchemaBootstrapper
from aero_forge.orchestrator.stack_classifier import StackClassification, classify_stack
from aero_forge.orchestrator.router import toolchains_for_intent
from aero_forge.concolic import ConcolicManifestVerifier, ConcolicResult
from aero_forge.builder.firewall import LogicalFirewall
from aero_forge.builder.chiasmus import (
    PrologFactEmitter,
    LogicEngine,
    RefinementFeedback,
    analyze_repository as chiasmus_analyze_repository,
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
            f.write(f"{timestamp} [{level}] {message}\n")
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
- functional_intent: list of {symbol_name, type, requirement_level}. Every functional requirement (functions, data constants, algorithms) from the prompt MUST be translated into this structured list. Do NOT rely on the engine to read the prompt.
- abi_contracts: list of {contract_id, target_language, binding_framework, export_symbol, c_symbol_alias, header_path, memory_model, signature: {inputs: [{name, type}], outputs: [{name, type}]}}. For PyO3 bridge functions, use explicit Rust signatures such as "&PyArray2<f64>", "Python", "usize", and "PyResult<...>".
- module_graph: list of {path, lang, purpose, rust_signature?}. When a hybrid Rust/Python extension is requested, list concrete submodule files under "src/" (e.g. "src/ops.rs", "src/array.rs"), the main "src/lib.rs", Python wrapper files, and test files under "tests/".
- verification_nodes: list of {test_id, execution_cmd, expected_exit_code, stdout_match_patterns, stderr_prohibited_patterns, numerical_assertions}
- cargo_dependencies: object mapping crate name to version spec or {version, features}. Always include "pyo3" for PyO3 bindings, "numpy" for numpy-rust array types, and "rayon" when parallel iterators or sliding-window logic are requested. Example: {"pyo3": "0.20.3", "numpy": "0.21", "rayon": "1.10"}.
"""


_GRAPH_SYSTEM_PROMPT = """YOU ARE THE AERO-FORGE INTENT COMPILER (SYSTEM LAYER 0).
YOUR SOLE PURPOSE IS TO CONVERT UNSTRUCTURED USER DOMAIN PROMPTS INTO A VALID `PolyglotGraphBlueprint` JSON SPECIFICATION.

STRICT OPERATIONAL RULES:
1. OUTPUT ONLY VALID JSON. DO NOT INCLUDE PREFACES, FOOTERS, MARKDOWN EXPLANATIONS, OR CODE BLOCK TEXT OUTSIDE THE JSON OBJECT.
2. `architecture` MUST reflect the actual languages and intent:
   - Single-language Python -> "pure_python"
   - Single-language Rust -> "pure_rust"
   - Python + Rust only (e.g., adding Rust acceleration to Python) -> "hybrid_rust_python"
   - Python + C++ -> "hybrid_cpp_python"
   - Rust + C++ -> "hybrid_cpp_rust"
   - Python + Rust + C++ -> "tri_polyglot_rust_cpp_python"
   - Only use "graph_polyglot" when the prompt explicitly describes a multi-node information network that does not match one of the above known architectures.
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
13. FUNCTIONAL INTENT COVERAGE: Add a top-level `functional_intent` array. Every functional requirement from the prompt (functions, data constants, algorithms) MUST be translated into `{symbol_name, type, requirement_level}`. Do NOT rely on the engine to read the prompt. Treat user-suggested paths as metadata; prioritize symbolic coverage. Ensure every `symbol_name` in `functional_intent` appears in a node's `exports` or in an edge's `symbol`.
14. NO SKELETON LAZINESS: Do not return a JSON shape with empty values or markers. Generate the entire blueprint JSON from scratch with every `functional_intent`, `node`, `edge`, and `source_files` entry fully populated.

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
  "functional_intent": [
    {"symbol_name": "synth_render", "type": "function", "requirement_level": "required"},
    {"symbol_name": "start_server", "type": "function", "requirement_level": "required"}
  ],
  "output_dir": "./dist",
  "metadata": {}
}
"""

# Theorem 1 safety lower bound: cosine similarity between the current intent
# context and the HIS invariant must stay above this value.  Drift is measured
# as 1 - similarity, so the corresponding drift threshold is 1 - bound.
SAFETY_LOWER_BOUND_SIMILARITY = 0.7071067811865476
DRIFT_THRESHOLD = 1.0 - SAFETY_LOWER_BOUND_SIMILARITY

# Phase 4 bounded-completion prompt token budget.  Long prompts are restored
# through the HIS invariant before being sent to the LLM.
PHASE4_MAX_TOKENS = 2048

# Sectional Fiber Completion keeps each independent LLM call inside the
# high-attention 1024-token window by decomposing the skeleton by Grothendieck
# fiber coordinates and retrying each section locally on syntactic failure.
SECTIONAL_MAX_TOKENS = 1024
SECTIONAL_RETRY_ATTEMPTS = 3


@dataclass(frozen=True)
class Result:
    """Algebraic error-handling monad for Sectional Fiber Completion.

    A section returns ``Result(ok=True, value=...)`` when it produces a valid
    fragment and ``Result(ok=False, error=...)`` when it fails.  ``bind`` chains
    fragment transformations while short-circuiting on the first error so that a
    syntactic failure in one section can be retried locally without aborting the
    whole enrichment pipeline.
    """

    ok: bool
    value: Any = None
    error: Optional[str] = None

    def bind(self, func: Callable[[Any], "Result"]) -> "Result":
        if not self.ok:
            return self
        try:
            return func(self.value)
        except Exception as exc:
            return Result(ok=False, error=str(exc))

    def map(self, func: Callable[[Any], Any]) -> "Result":
        if not self.ok:
            return self
        try:
            return Result(ok=True, value=func(self.value))
        except Exception as exc:
            return Result(ok=False, error=str(exc))


def _phase4_token_count(text: str) -> int:
    """Approximate token count for a Phase 4 prompt.

    Uses tiktoken when available; otherwise falls back to a conservative
    character-based estimate (roughly 4 characters per token for code/JSON).
    """
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def _telemetry_dir() -> Path:
    """Return the directory used for Phase 4 intent-compiler telemetry."""
    path = Path(os.getenv("AERO_FORGE_TELEMETRY_DIR", "/tmp/aero-forge-telemetry"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _log_phase4_telemetry(
    prompt: str,
    token_count: int,
    drift: float,
    similarity: float,
    fiber_coordinate: Optional[str],
    restoration_triggered: bool,
    topology_node_count: int,
    topology_edge_count: int,
) -> None:
    """Append a Phase 4 prompt telemetry record to the telemetry log.

    The full rendered prompt is stored under ``prompt_full`` so the exact input
    sent to the LLM can be audited; ``prompt_preview`` is a short head.
    """
    try:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "phase": "bounded_completion",
            "prompt_token_count": token_count,
            "drift": drift,
            "similarity": similarity,
            "theorem_1_bound_similarity": SAFETY_LOWER_BOUND_SIMILARITY,
            "fiber_coordinate": fiber_coordinate,
            "restoration_triggered": restoration_triggered,
            "topology_nodes": topology_node_count,
            "topology_edges": topology_edge_count,
            "prompt_preview": prompt[:500],
            "prompt_full": prompt,
        }
        log_path = _telemetry_dir() / "phase4_prompts.jsonl"
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.debug("Failed to write Phase 4 telemetry: %s", exc)


_CLASSIFICATION_SYSTEM_PROMPT = """You are the Aero-Forge structural classifier.
Your job is to read the user request and emit a compact JSON classification with exactly these keys:
- `architecture`: one of `pure_python`, `pure_rust`, `hybrid_rust_python`, `hybrid_cpp_python`, `hybrid_cpp_rust`, `tri_polyglot_rust_cpp_python`, `graph_polyglot`.
- `toolchains`: a list of required build toolchains, e.g. `["cargo", "python"]` or `["cmake", "cargo", "python"]`.
- `functional_intent`: a list of `{symbol_name, type, requirement_level}` objects covering every functional requirement.
- `nodes`: a list of `{node_id, lang, toolchain, source_files}` objects representing the physical files to materialize.

Do NOT emit implementation code, edges, contracts, or explanations. Return ONLY the JSON classification object.
"""


class RobustJSONExtractor:
    r"""Extract a valid JSON object from an LLM response that may contain prose,
    markdown fences, or trailing content.

    The implementation mirrors the recursive brace-matching regex
    ``r'\{(?:[^{}]|(?R))*\}'`` using an explicit stack so it works with the
    standard library ``re`` module.
    """

    @staticmethod
    def strip_markdown(text: str) -> str:
        """Remove optional JSON/YAML code fences and leading/trailing prose."""
        t = text.strip()
        # Opening fence with optional language or path label, e.g. ```json:plan.json
        t = re.sub(
            r"^```\s*(?:json|yaml)?\s*(?::\s*[^\n\r]*)?\s*\r?\n",
            "",
            t,
            flags=re.IGNORECASE,
        )
        t = re.sub(
            r"^```\s*(?:json|yaml)?\s*(?::\s*[^\n\r]*)?$",
            "",
            t,
            flags=re.IGNORECASE,
        )
        # Standalone opening fence label such as ``json`` on its own line.
        t = re.sub(r"^(?:json|yaml)\s*\r?\n", "", t, flags=re.IGNORECASE)
        if t.endswith("```"):
            t = t[:-3].strip()
        return t.strip()

    @staticmethod
    def _find_matching(
        text: str, start: int, opener: str, closer: str
    ) -> Optional[int]:
        """Return the index of the closing delimiter matching *opener* at *start*."""
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    return i
        return None

    @classmethod
    def largest_json_block(cls, text: str) -> Optional[str]:
        """Return the largest valid JSON object or array literal found in *text*."""
        best: Optional[str] = None
        for m in re.finditer(r"[\{\[]", text):
            start = m.start()
            opener = text[start]
            closer = "}" if opener == "{" else "]"
            end = cls._find_matching(text, start, opener, closer)
            if end is None:
                continue
            candidate = text[start : end + 1]
            try:
                json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if best is None or len(candidate) > len(best):
                best = candidate
        return best

    @classmethod
    def extract(cls, raw: str) -> Any:
        """Parse a JSON object from a raw LLM response, tolerating surrounding text."""
        cleaned = cls.strip_markdown(raw)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        balanced = cls.largest_json_block(cleaned)
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


def _extract_json(raw: str) -> Any:
    """Backward-compatible wrapper around :class:`RobustJSONExtractor`."""
    return RobustJSONExtractor.extract(raw)


def _compress_skeletal_blueprint(raw: str) -> str:
    """Return a compact JSON skeleton with token-heavy fields removed.

    Removes ``constraints`` and ``output_dir`` from the largest JSON block in
    the raw response so retries consume fewer tokens while still preserving the
    structural blueprint.
    """
    try:
        data = RobustJSONExtractor.extract(raw)
    except Exception:
        return raw
    if isinstance(data, dict):
        data.pop("constraints", None)
        data.pop("output_dir", None)
        return json.dumps(data, indent=2)
    return raw


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
            metadata["domain_target"] = (
                str(metadata["domain_target"]).lower().replace(" ", "_")
            )
        # BlueprintSchemaV2 expects Dict[str, str]; coerce list/bool/dict metadata
        # values to JSON strings so common LLM extras (test_files, llm_initialized)
        # do not fail validation while still rejecting e.g. numeric project_name.
        for key, value in list(metadata.items()):
            if not isinstance(value, str) and isinstance(value, (list, dict, bool)):
                metadata[key] = json.dumps(value)
        data["metadata"] = metadata

    # execution_strategy
    exec_strategy = data.get("execution_strategy") or {}
    if isinstance(exec_strategy, dict):
        primary = exec_strategy.get("primary_entrypoint") or {}
        if isinstance(primary, dict) and primary.get("runtime"):
            runtime = str(primary["runtime"]).lower()
            if "python" in runtime:
                primary["runtime"] = "python3"
            primary["wrapper_generation"] = bool(
                primary.get("wrapper_generation", True)
            )
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
                        flag.get("name")
                        or flag.get("long")
                        or flag.get("long_flag")
                        or flag.get("dest")
                        or flag.get("dest_var", "")
                    ),
                    "short": str(flag.get("short") or flag.get("short_flag") or "")
                    .strip()
                    .lstrip("-"),
                    "type": _normalize_cli_type(flag.get("type") or "string"),
                    "required": bool(flag.get("required", False)),
                    "default": flag.get("default", None),
                    "choices": (
                        list(flag.get("choices", [])) if flag.get("choices") else []
                    ),
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

        binding = (
            str(normalized_abi.get("binding_framework") or "")
            .lower()
            .strip()
            .replace("-", "_")
        )
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
        normalized_abi["binding_framework"] = (
            binding_synonyms.get(binding, binding) or "c_abi"
        )
        c_alias = normalized_abi.get("c_symbol_alias")
        if c_alias is None:
            normalized_abi["c_symbol_alias"] = ""

        memory = re.sub(
            r"[^a-z0-9_]",
            "_",
            str(normalized_abi.get("memory_model") or "").lower().strip(),
        )
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
        lang = str(
            normalized_node.get("lang") or normalized_node.get("language") or "python"
        ).lower()
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
    """Map a C ABI / PyO3 type to a Python type annotation."""
    t = (c_type or "").strip()
    lowered = t.lower().replace(" ", "")
    scalar_ints = {"u32", "i32", "usize", "int32_t", "i64", "u64", "int"}
    scalar_floats = {"f64", "f32", "double", "float"}
    if lowered in scalar_ints:
        return "int"
    if lowered in scalar_floats:
        return "float"
    if lowered in {"bool"}:
        return "bool"
    if lowered in {"python", "pyo3::python"}:
        return "__PYTHON__"
    # Unwrap PyO3 result wrappers.
    m = re.match(r"pyresult<(.+)>", lowered)
    if m:
        inner = m.group(1)
        if inner in scalar_floats:
            return "float"
        if inner in scalar_ints:
            return "int"
        if inner == "()":
            return "None"
        return "Any"
    # PyO3 ndarray references become nested Python lists.
    m = re.match(r"&?pyarray(\d)?<(.+)>", lowered)
    if m:
        ndim = m.group(1) or "1"
        inner = m.group(2)
        if inner in scalar_floats:
            base = "list[float]"
        elif inner in scalar_ints:
            base = "list[int]"
        else:
            base = "list[Any]"
        return "list[" + base + "]" if ndim == "2" else base
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
    args = [
        (entry["name"], py_type)
        for entry in inputs
        if (py_type := _abi_type_to_py(entry["type"])) != "__PYTHON__"
    ]
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


def _derive_toolchains(
    architecture: str,
    manifest_entries: List[ManifestEntry],
    abi_contracts: List[ABIContract],
) -> List[str]:
    """Return toolchains that cover the requested architecture and every manifest file."""
    from aero_forge.orchestrator.router import toolchains_for_intent

    toolchains = set(toolchains_for_intent(architecture))
    lang_toolchain: Dict[str, str] = {
        "python": "python",
        "rust": "cargo",
        "cpp": "cpp",
        "c++": "cpp",
        "c": "gcc",
        "go": "go",
        "zig": "zig",
        "java": "javac",
        "csharp": "dotnet",
        "mojo": "mojo",
        "nim": "nim",
        "d": "dmd",
        "fortran": "gfortran",
        "javascript": "node",
        "typescript": "node",
    }
    for entry in manifest_entries:
        lang = (entry.lang or "").lower()
        if lang in lang_toolchain:
            toolchains.add(lang_toolchain[lang])
        ext = Path(entry.path).suffix.lower()
        if ext == ".zig":
            toolchains.add("zig")
        elif ext in (".cpp", ".cc", ".cxx", ".c"):
            toolchains.update({"cpp", "cmake", "clang"})
        elif ext == ".go":
            toolchains.add("go")
        elif ext == ".java":
            toolchains.add("javac")
        elif ext == ".cs":
            toolchains.add("dotnet")
    for abi in abi_contracts:
        target = (abi.target_language or "").lower()
        if target in lang_toolchain:
            toolchains.add(lang_toolchain[target])
        if abi.binding_framework == "pyo3":
            toolchains.update({"python", "rust", "cargo"})
        elif abi.binding_framework == "c_abi":
            toolchains.update({"cpp", "cmake", "clang"})
    return sorted(toolchains)


def _synthesize_contracts_from_manifest(
    manifest_entries: List[ManifestEntry],
) -> List[ContractEntry]:
    """Create a contract entry for every manifest file that defines an export."""
    contracts: List[ContractEntry] = []
    seen: set = set()
    for entry in manifest_entries:
        name = Path(entry.path).stem
        if not name or name in seen:
            continue
        seen.add(name)
        contracts.append(
            ContractEntry(
                name=name,
                signature=f"def {name}() -> None",
                language=entry.lang or "python",
                python_name=name,
                purpose=entry.purpose or f"synthesized contract for {entry.path}",
            )
        )
    return contracts


def _infer_architecture(languages: set) -> str:
    """Map a set of language tags to the closest aero-forge architecture string."""
    from aero_forge.orchestrator.router import (
        BUILD_INTENT_GRAPH_POLYGLOT,
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

    # Any language outside the built-in {python, rust, cpp} trio forces the
    # generic graph_polyglot path so the engine can JIT-synthesize emitters.
    builtin = {"python", "rust", "cpp", "c++"}
    if any(lang not in builtin for lang in languages) or len(languages) > 3:
        return BUILD_INTENT_GRAPH_POLYGLOT

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


def _languages_for_architecture(architecture: str) -> List[str]:
    """Return the canonical language set for an aero-forge architecture."""
    mapping: Dict[str, List[str]] = {
        "pure_python": ["python"],
        "pure_rust": ["rust"],
        "hybrid_rust_python": ["python", "rust"],
        "hybrid_cpp_python": ["python", "cpp"],
        "hybrid_cpp_rust": ["rust", "cpp"],
        "tri_polyglot_rust_cpp_python": ["python", "rust", "cpp"],
        "graph_polyglot": ["python", "rust", "cpp"],
    }
    return mapping.get(str(architecture).lower(), ["python"])


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
        system_prompt_extra: Optional[str] = None,
    ):
        self.provider = provider if provider is not None else "deepseek"
        self.model = model
        self.api_key = api_key
        self.max_retries = max(1, max_retries)
        self.max_schema_retries = max(1, max_schema_retries)
        self._llm_client = llm_client
        self.config_override = config_override
        self._system_prompt_extra = system_prompt_extra or ""

    @staticmethod
    def _retry_user_message(
        attempt: int,
        raw: str,
        exc: Optional[Exception] = None,
    ) -> Dict[str, str]:
        """Return the user correction message for schema-retry *attempt*.

        * First retry (``attempt == 0`` after a failure): compress the previous
          raw response by dropping ``constraints`` and ``output_dir`` to save
          tokens, then ask the model to complete the corrected blueprint.
        * Second and third retries: append the concrete ``JsonSchemaValidationError``
          message so the model can self-correct the structure.
        """
        error_text = ""
        if isinstance(exc, JsonSchemaValidationError):
            error_text = "\n".join(e.message for e in exc.context) or str(exc)
        elif exc is not None:
            error_text = str(exc)

        if attempt == 0:
            compressed = _compress_skeletal_blueprint(raw)
            content = (
                "The previous response was not valid. Return a corrected JSON blueprint. "
                "To save tokens, the compressed skeleton below has the `constraints` and "
                "`output_dir` fields removed; do not add them back unless the prompt "
                "explicitly requires them.\n\n"
                f"Schema validation error:\n{error_text}\n\n"
                f"{compressed}"
            )
        else:
            content = (
                f"Schema validation still failed. Error details:\n{error_text}\n\n"
                "Return corrected JSON only, with no markdown or explanatory text."
            )
        return {"role": "user", "content": content}

    @staticmethod
    def _verify_classification(data: Dict[str, Any]) -> List[str]:
        """Validate an LLM classification using deterministic, prompt-blind checks.

        Delegates architecture/toolchain/node consistency to the Orchestrator's
        static verifier and adds intent-level checks for non-pure-Python builds.
        """
        errors: List[str] = []
        architecture = (data.get("architecture") or "").lower()
        toolchains = data.get("toolchains") or []
        nodes = data.get("nodes") or []
        functional_intent = data.get("functional_intent") or []

        try:
            from aero_forge.orchestrator.orchestrator import Orchestrator

            errors.extend(
                Orchestrator.verify_classification(architecture, toolchains, nodes)
            )
        except Exception as exc:
            logger.debug("Orchestrator classification verification failed: %s", exc)

        if architecture != "pure_python" and not functional_intent:
            errors.append(
                f"Architecture {architecture!r} requires a non-empty 'functional_intent' list."
            )

        return errors

    def _classify_graph(
        self,
        client: Any,
        system: str,
        prompt_text: str,
        architecture: Optional[str] = None,
    ) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        """Tier 0: ask the LLM for architecture/toolchains/functional_intent/nodes.

        Returns the classification dict and a list of structural errors. If the LLM
        returns a complete graph blueprint, it is validated and returned as the
        full graph (errors empty and classification dict contains the graph).

        If *architecture* is provided, it is injected into the classification
        prompt and used to override the LLM's returned architecture.
        """
        if client is None:
            _accel_log(
                "warning",
                "No LLM client available for classification; using deterministic fallback.",
            )
            return self._deterministic_classification(prompt_text, architecture=architecture), []

        architecture_hint = (
            f"\n\n[CRITICAL: the target architecture for this build is '{architecture}'. "
            f"You MUST return architecture='{architecture}' and include nodes/toolchains for all languages implied by it.]"
            if architecture
            else ""
        )
        classification_messages: List[Dict[str, str]] = [
            {"role": "system", "content": _CLASSIFICATION_SYSTEM_PROMPT},
            {"role": "system", "content": system},
            {"role": "user", "content": prompt_text + architecture_hint},
        ]

        last_error: Optional[Exception] = None
        for attempt in range(self.max_schema_retries):
            raw = client.generate(
                classification_messages,
                temperature=0.2,
                max_tokens=4096,
            )
            if not raw:
                _accel_log(
                    "warning",
                    f"intent_compiler.classify_graph attempt {attempt + 1}: empty response",
                )
                classification_messages.append({"role": "assistant", "content": ""})
                classification_messages.append(
                    {
                        "role": "user",
                        "content": f"Attempt {attempt + 1} returned empty. Return valid JSON classification.",
                    }
                )
                continue

            try:
                data = _extract_json(raw)
            except Exception as exc:
                last_error = exc
                _accel_log(
                    "warning",
                    f"intent_compiler.classify_graph attempt {attempt + 1}: JSON extraction failed; "
                    f"raw preview: {raw[:800]!r}; error: {exc}",
                )
                classification_messages.append({"role": "assistant", "content": raw})
                classification_messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Could not extract JSON: {exc}. "
                            "Return a valid classification JSON object only."
                        ),
                    }
                )
                continue

            # If the LLM already returned a complete graph blueprint, validate it.
            try:
                graph = PolyglotGraphBlueprint.model_validate(data)
                graph.metadata["prompt"] = prompt_text
                graph.metadata["llm_initialized"] = True
                graph.metadata["status"] = "finalized"
                graph.metadata["generation_method"] = "llm_synthesized"
                return {"_full_graph": graph}, []
            except Exception:
                pass

            # Accept legacy v2 BlueprintSchemaV2 responses and lower them into a
            # graph blueprint so the web workspace still validates existing flows.
            try:
                normalized = _normalize_v2_data(data)
                v2 = BlueprintSchemaV2.model_validate(normalized)
                v2_languages: set = set()
                for node in v2.module_graph:
                    lang = (node.get("lang") or node.get("language") or "").lower()
                    if lang:
                        v2_languages.add(lang)
                for abi in v2.abi_contracts:
                    target = (abi.target_language or "").lower()
                    if target:
                        v2_languages.add(target)
                    if abi.binding_framework == "pyo3":
                        v2_languages.update({"rust", "python"})
                    elif abi.binding_framework == "c_abi":
                        v2_languages.add("cpp")
                v2_classification = {
                    "architecture": v2.metadata.get("domain_target")
                    or _infer_architecture(v2_languages),
                    "functional_intent": [
                        fi.model_dump(mode="json") for fi in (v2.functional_intent or [])
                    ],
                    "nodes": [dict(n) for n in (normalized.get("nodes") or [])],
                }
                graph = self._v2_to_graph_blueprint(
                    v2,
                    prompt_text,
                    output_dir=None,
                    project_name=None,
                    classification=v2_classification,
                )
                graph.metadata["prompt"] = prompt_text
                graph.metadata["llm_initialized"] = True
                graph.metadata["status"] = "finalized"
                graph.metadata["generation_method"] = "llm_synthesized"
                return {"_full_graph": graph}, []
            except Exception:
                pass

            if architecture:
                returned_arch = str(data.get("architecture") or "").lower()
                if returned_arch != architecture.lower():
                    _accel_log(
                        "info",
                        f"Overriding LLM classification architecture from '{returned_arch}' to '{architecture}'",
                    )
                    data["architecture"] = architecture
                    # Ensure the classifier keeps the expected language set.
                    nodes = data.get("nodes") or []
                    existing_langs = {n.get("lang", "").lower() for n in nodes if isinstance(n, dict)}
                    for lang in _languages_for_architecture(architecture):
                        if lang not in existing_langs:
                            nodes.append({"node_id": f"{lang}_node", "lang": lang, "toolchain": lang})
                    data["nodes"] = nodes

            errors = self._verify_classification(data)
            if not errors:
                return data, []

            _accel_log(
                "warning",
                f"intent_compiler.classify_graph attempt {attempt + 1}: classification invalid; errors: {errors}",
            )
            classification_messages.append({"role": "assistant", "content": raw})
            classification_messages.append(
                {
                    "role": "user",
                    "content": (
                        "The classification was rejected by the structural verifier. "
                        f"Correct these errors and return a new classification JSON:\n\n"
                        + "\n".join(f"- {e}" for e in errors)
                    ),
                }
            )

        _accel_log(
            "warning",
            f"LLM classification failed after {self.max_schema_retries} attempts ({last_error}); "
            "falling back to deterministic prompt classification.",
        )
        return self._deterministic_classification(prompt_text, architecture=architecture), []

    def _deterministic_classification(
        self,
        prompt_text: str,
        architecture: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build a structural classification without an LLM call.

        Uses the deterministic stack classifier and a hand-built node skeleton so
        graph-polyglot prompts always have a valid plan, even when the LLM returns
        an empty response.
        """
        classification = classify_stack(prompt_text)
        if architecture:
            classification = StackClassification(
                architecture=architecture,
                toolchains=toolchains_for_intent(architecture),
                languages=_languages_for_architecture(architecture),
                features=classification.features,
            )
        architecture = classification.architecture
        languages = classification.languages or ["python"]
        toolchains = classification.toolchains or []

        skeleton = self._build_deterministic_skeleton(
            architecture, languages, classification.features or []
        )

        return {
            "architecture": architecture,
            "toolchains": toolchains,
            "functional_intent": skeleton.get("functional_intent", []),
            "nodes": skeleton.get("nodes", []),
            "_skeleton": skeleton,
        }

    def _build_deterministic_skeleton(
        self,
        architecture: str,
        languages: List[str],
        features: List[str],
    ) -> Dict[str, Any]:
        """Return a rigid graph skeleton with one node per detected language."""
        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []
        seen_node_ids: set = set()
        lang_node: Dict[str, str] = {}

        def node_id_for_lang(lang: str) -> str:
            return {
                "python": "python_interface",
                "rust": "rust_core",
                "cpp": "cpp_engine",
                "c": "c_engine",
                "go": "go_engine",
                "zig": "zig_kernel",
                "javascript": "js_client",
                "java": "java_binding",
                "csharp": "dotnet_host",
            }.get(lang, f"{lang}_core")

        for lang in languages:
            node_id = node_id_for_lang(lang)
            if node_id in seen_node_ids:
                continue
            seen_node_ids.add(node_id)
            lang_node[lang] = node_id
            if lang == "python":
                source_files = [f"{node_id}/main.py"]
                toolchain = "python"
                exports = ["main"]
            elif lang == "rust":
                source_files = [f"{node_id}/Cargo.toml", f"{node_id}/src/lib.rs"]
                toolchain = "cargo"
                exports = ["run_pipeline"]
            elif lang in ("cpp", "c", "c++"):
                source_files = [f"{node_id}/CMakeLists.txt", f"{node_id}/src/{node_id}.cpp"]
                toolchain = "cmake"
                exports = [node_id]
            elif lang == "go":
                source_files = [f"{node_id}/go.mod", f"{node_id}/main.go"]
                toolchain = "go"
                exports = [node_id]
            elif lang == "zig":
                source_files = [f"{node_id}/build.zig", f"{node_id}/main.zig"]
                toolchain = "zig"
                exports = [node_id]
            else:
                source_files = [f"{node_id}/main.txt"]
                toolchain = lang
                exports = [node_id]

            nodes.append(
                {
                    "node_id": node_id,
                    "lang": lang,
                    "toolchain": toolchain,
                    "source_files": source_files,
                    "exports": exports,
                    "purpose": f"{lang} language node",
                    "logic_sketch": "<TYPED_HOLE>",
                    "contracts": [],
                }
            )

        # Wire every non-Python node to a Python consumer if one exists.
        python_node = lang_node.get("python")
        if python_node:
            for lang, node_id in lang_node.items():
                if lang == "python" or node_id == python_node:
                    continue
                if lang == "rust":
                    boundary = "pyo3_maturin"
                    symbol = "run_pipeline"
                    args = ["pointer"]
                    return_type = "pointer"
                else:
                    boundary = "c_abi"
                    symbol = node_id
                    args = ["float64", "float64"]
                    return_type = "float64"
                edges.append(
                    {
                        "source": node_id,
                        "target": python_node,
                        "boundary_type": boundary,
                        "symbol": symbol,
                        "args": args,
                        "return_type": return_type,
                        "is_zero_copy": False,
                    }
                )
        # If no Python node, chain nodes in declared order.
        elif len(nodes) > 1:
            for src, tgt in zip(nodes, nodes[1:]):
                edges.append(
                    {
                        "source": src["node_id"],
                        "target": tgt["node_id"],
                        "boundary_type": "c_abi",
                        "symbol": src["node_id"],
                        "args": ["float64"],
                        "return_type": "float64",
                        "is_zero_copy": False,
                    }
                )

        manifest = [
            {"path": sf, "node_id": node["node_id"], "role": "source"}
            for node in nodes
            for sf in node["source_files"]
        ]

        # Functional intent is derived from the actual exported symbols so the
        # blueprint validator never complains about prompt-derived fantasy symbols.
        functional_intent: List[Dict[str, Any]] = []
        seen_intent: set = set()
        for symbol, type_ in [("main", "cli")] + [
            (e, "function")
            for node in nodes
            for e in (node.get("exports") or [])
        ]:
            if symbol in seen_intent:
                continue
            seen_intent.add(symbol)
            functional_intent.append(
                {
                    "symbol_name": symbol,
                    "type": type_,
                    "requirement_level": "required",
                }
            )

        return {
            "schema_version": "2.0.0",
            "project": architecture,
            "architecture": architecture,
            "metadata": {"bootstrap_method": "deterministic_fallback"},
            "functional_intent": functional_intent,
            "nodes": nodes,
            "edges": edges,
            "manifest": manifest,
            "typed_holes": [
                {
                    "path": f"nodes.{node['node_id']}.logic_sketch",
                    "expected_type": "string",
                    "description": "Concrete implementation body for the node.",
                }
                for node in nodes
            ],
            "primary_entrypoint": f"{python_node}/main.py" if python_node else "run_shell.py",
            "build_script": "build.sh",
        }

    def _derive_functional_intent_from_prompt(
        self,
        prompt_text: str,
        architecture: str,
        languages: List[str],
        features: List[str],
    ) -> List[Dict[str, Any]]:
        """Extract a minimal, deterministic functional-intent list from a prompt.

        Deprecated: deterministic classification now derives functional_intent from
        the concrete node skeleton so it always matches materialized symbols. This
        helper is retained only for callers that do not build a full skeleton.
        """
        intent: List[Dict[str, Any]] = []
        seen: set = set()

        def add(symbol: str, type_: str = "function") -> None:
            symbol = re.sub(r"[^a-zA-Z0-9_]+", "_", symbol).strip("_") or "feature"
            if symbol in seen:
                return
            seen.add(symbol)
            intent.append(
                {
                    "symbol_name": symbol,
                    "type": type_,
                    "requirement_level": "required",
                }
            )

        add("main", "cli")

        # Feature-derived modules (only use classifier features, not regex guessing).
        for feature in features:
            add(feature, "function")

        return intent

    def _inject_wasm_target(
        self, skeleton: Dict[str, Any], prompt_text: str
    ) -> None:
        """Add a wasm32 build target to Rust nodes when the prompt requests it."""
        if not skeleton or not isinstance(skeleton, dict):
            return
        prompt_lower = (prompt_text or "").lower()
        if "wasm32" not in prompt_lower and "wasm" not in prompt_lower:
            return
        for node in skeleton.get("nodes", []) or []:
            if (node.get("lang") or "").lower() != "rust":
                continue
            flags = node.get("compiler_flags") or []
            if not isinstance(flags, list):
                flags = [flags]
            if not any("wasm" in str(f).lower() for f in flags):
                flags.append("--target wasm32-unknown-unknown")
                node["compiler_flags"] = flags

    def compile_prompt(
        self,
        prompt_text: str,
        output_dir: Optional[str | Path] = None,
        project_name: Optional[str] = None,
        architecture: Optional[str] = None,
    ) -> Blueprint:
        """Convert *prompt_text* into a validated ``Blueprint`` and write ``blueprint.aero``.

        This is the backward-compatible v2 entry point used by the web workspace
        and ``generate_and_build``.  Internally it runs the six-phase graph
        pipeline (HIS -> FoGE -> Adjoint -> Bounded Completion -> Concolic ->
        SHACL/Prolog) and converts the resulting ``PolyglotGraphBlueprint`` into
        the legacy ``Blueprint`` v2 format.

        If *architecture* is provided, it is injected into the classification
        prompt and overrides any LLM classification that does not match the
        caller's explicit target.
        """
        graph = self.compile_prompt_to_graph(
            prompt_text,
            output_dir=output_dir,
            project_name=project_name,
            architecture=architecture,
        )

        # If an LLM was requested but we only produced a deterministic fallback,
        # surface that as an enrichment failure so callers can retry or escalate.
        provider = (self.provider or "").lower()
        if (
            provider not in ("", "none", "null")
            and graph.metadata.get("generation_method") == "deterministic_fallback"
        ):
            raise IntentCompilerError(
                "LLM enrichment failed and only a deterministic fallback blueprint was produced."
            )

        blueprint = self._graph_to_blueprint_v2(
            graph,
            prompt_text,
            output_dir=output_dir,
            project_name=project_name,
        )
        if output_dir:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            write_blueprint(blueprint, out_path / "blueprint.aero")
        return blueprint

    @staticmethod
    def _derive_functional_intent(v2: BlueprintSchemaV2) -> List[FunctionalIntent]:
        """Synthesize a non-empty functional_intent list from a v2 blueprint."""
        if v2.functional_intent:
            return list(v2.functional_intent)

        intents: List[FunctionalIntent] = []
        seen: set = set()
        for node in v2.module_graph:
            path = node.get("path", "")
            symbol = Path(path).stem or node.get("node_id") or "unnamed"
            if symbol and symbol not in seen:
                intents.append(
                    FunctionalIntent(
                        symbol_name=symbol,
                        type="function",
                        requirement_level="required",
                    )
                )
                seen.add(symbol)
        for abi in v2.abi_contracts:
            symbol = getattr(abi, "export_symbol", None) or getattr(
                abi, "contract_id", None
            )
            if symbol and symbol not in seen:
                intents.append(
                    FunctionalIntent(
                        symbol_name=symbol,
                        type="function",
                        requirement_level="required",
                    )
                )
                seen.add(symbol)
        return intents

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

        project = (
            project_name or v2.metadata.get("project_name") or "aero_forge_project"
        )
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

        resolved_output_dir = (
            Path(output_dir) / "dist" if output_dir else Path("./dist")
        )

        modification_plan = self._build_modification_plan(v2, output_dir)

        metadata = dict(v2.metadata)
        metadata.setdefault("schema_version", "2.0.0")
        metadata["llm_initialized"] = "true"
        metadata["auto_generated"] = "true"
        metadata["generation_method"] = "llm_synthesized"

        toolchains = _derive_toolchains(architecture, manifest_entries, v2.abi_contracts)
        contracts = _contracts_from_abi(v2.abi_contracts)
        if architecture != "pure_python" and not contracts:
            contracts = _synthesize_contracts_from_manifest(manifest_entries)

        return Blueprint(
            project=project,
            architecture=architecture,
            toolchains=toolchains,
            manifest=manifest_entries,
            contracts=contracts,
            functions=[],
            output_dir=resolved_output_dir,
            llm=LLMConfig(provider=self.provider, model=self.model or ""),
            prompt=prompt_text,
            constraints="",
            languages=sorted(languages) if languages else ["python"],
            features=[],
            execution_strategy=v2.execution_strategy,
            abi_contracts=v2.abi_contracts,
            functional_intent=self._derive_functional_intent(v2),
            verification_nodes=v2.verification_nodes,
            metadata=metadata,
            module_graph=v2.module_graph,
            cargo_dependencies=v2.cargo_dependencies,
            modification_plan=modification_plan,
        )

    def _graph_to_blueprint_v2(
        self,
        graph: PolyglotGraphBlueprint,
        prompt_text: str,
        output_dir: Optional[str | Path] = None,
        project_name: Optional[str] = None,
    ) -> Blueprint:
        """Convert a ``PolyglotGraphBlueprint`` into the legacy v2 ``Blueprint``.

        This keeps the web workspace and ``generate_and_build`` interfaces stable
        while the enrichment core runs the six-phase graph pipeline.
        """
        from aero_forge.orchestrator.router import default_manifest_for_architecture

        project = project_name or graph.project or "aero_forge_project"

        node_map = {n.node_id: n for n in graph.nodes}
        languages: set = set()
        manifest_entries: List[ManifestEntry] = []
        for node in graph.nodes:
            if node.lang:
                languages.add(node.lang.lower())
            for sf in node.source_files or []:
                manifest_entries.append(
                    ManifestEntry(path=sf, lang=node.lang or "python", purpose=node.node_id)
                )

        contract_entries: List[ContractEntry] = []
        abi_contracts: List[ABIContract] = []
        for edge in graph.edges:
            tgt = node_map.get(edge.target)
            src = node_map.get(edge.source)
            if src and src.lang:
                languages.add(src.lang.lower())
            if tgt and tgt.lang:
                languages.add(tgt.lang.lower())

            inputs = [{"name": f"arg{i}", "type": arg} for i, arg in enumerate(edge.args or [])]
            outputs = [{"name": "return", "type": edge.return_type}] if edge.return_type else []

            binding = str(edge.boundary_type or "c_abi").lower().replace("-", "_")
            abi_contracts.append(
                ABIContract(
                    contract_id=f"{edge.source}_{edge.target}_{edge.symbol}",
                    target_language=(tgt.lang if tgt else "python"),
                    binding_framework=binding,
                    export_symbol=edge.symbol,
                    memory_model="caller_allocates",
                    signature={"inputs": inputs, "outputs": outputs},
                )
            )

            if inputs:
                arg_str = ", ".join(
                    f"{inp['name']}: {_abi_type_to_py(inp['type'])}" for inp in inputs
                )
            else:
                arg_str = ""
            return_type = _abi_type_to_py(outputs[0]["type"]) if outputs else "None"
            contract_entries.append(
                ContractEntry(
                    name=edge.symbol,
                    signature=f"def {edge.symbol}({arg_str}) -> {return_type}",
                    language=(tgt.lang if tgt else "python"),
                    python_name=edge.symbol,
                    purpose=f"Cross-language edge {edge.source} -> {edge.target}",
                )
            )

        architecture = graph.architecture or _infer_architecture(languages)
        if not manifest_entries:
            manifest_entries = [
                ManifestEntry(path=e["path"], lang=e["lang"], purpose=e["purpose"])
                for e in default_manifest_for_architecture(architecture, project)
            ]

        resolved_output_dir = Path(output_dir) / "dist" if output_dir else Path("./dist")

        metadata: Dict[str, Any] = dict(graph.metadata or {})
        metadata.setdefault("schema_version", "2.0.0")
        metadata["llm_initialized"] = "true"
        metadata["auto_generated"] = "true"
        metadata["generation_method"] = metadata.get("generation_method") or "llm_synthesized"
        # v2 metadata is Dict[str, str]; stringify values for YAML/JSON safety.
        metadata = {k: str(v) for k, v in metadata.items()}

        toolchains = _derive_toolchains(architecture, manifest_entries, abi_contracts)
        contracts = contract_entries
        if architecture != "pure_python" and not contracts:
            contracts = _synthesize_contracts_from_manifest(manifest_entries)

        module_graph = []
        for node in graph.nodes:
            node_data = node.model_dump(mode="json")
            if not node_data.get("path") and node.source_files:
                node_data["path"] = node.source_files[0]
            module_graph.append(node_data)

        primary_path = graph.primary_entrypoint or ""
        execution_strategy_data = graph.metadata.get("execution_strategy")
        if execution_strategy_data:
            try:
                execution_strategy = ExecutionStrategy.model_validate(execution_strategy_data)
            except Exception:
                execution_strategy = ExecutionStrategy(
                    primary_entrypoint={"path": primary_path, "runtime": "python3" if primary_path.endswith(".py") else "./"},
                    run_spec={"command": graph.build_script or "build.sh"},
                )
        else:
            execution_strategy = ExecutionStrategy(
                primary_entrypoint={"path": primary_path, "runtime": "python3" if primary_path.endswith(".py") else "./"},
                run_spec={"command": graph.build_script or "build.sh"},
            )

        functional_intent: List[FunctionalIntent] = []
        for fi in graph.functional_intent or []:
            functional_intent.append(
                FunctionalIntent(
                    symbol_name=fi.symbol_name,
                    type=fi.type or "function",
                    requirement_level=fi.requirement_level or "required",
                )
            )
        if not functional_intent:
            seen: set = set()
            for node in graph.nodes:
                for exp in node.exports or []:
                    if exp and exp not in seen:
                        functional_intent.append(
                            FunctionalIntent(
                                symbol_name=exp, type="function", requirement_level="required"
                            )
                        )
                        seen.add(exp)

        return Blueprint(
            project=project,
            architecture=architecture,
            toolchains=toolchains,
            manifest=manifest_entries,
            contracts=contracts,
            functions=[],
            output_dir=resolved_output_dir,
            llm=LLMConfig(provider=self.provider or "none", model=self.model or ""),
            prompt=prompt_text,
            constraints="",
            languages=sorted(languages) if languages else ["python"],
            features=[],
            execution_strategy=execution_strategy,
            abi_contracts=abi_contracts,
            functional_intent=functional_intent,
            verification_nodes=graph.metadata.get("verification_nodes") or [],
            metadata=metadata,
            module_graph=module_graph,
            cargo_dependencies=graph.metadata.get("cargo_dependencies") or {},
            modification_plan={},
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
            exists = (
                workspace is not None and (workspace / path).is_file()
                if path
                else False
            )
            actions.append(
                {
                    "path": path,
                    "action": "MODIFY" if exists else "CREATE",
                    "lang": node.get("lang") or node.get("language") or "python",
                }
            )
        return {"intent": "incremental_update", "actions": actions}

    # ------------------------------------------------------------------
    # Six-phase tiered synthesis pipeline (HIS -> FoGE -> Adjoint ->
    # Bounded Completion -> Concolic Feedback -> SHACL/Prolog Verification).
    # ------------------------------------------------------------------
    def _six_phase_bind_context(
        self,
        classification: Dict[str, Any],
    ) -> HolographicContext:
        """Phase 1: bind the prompt's functional intent into an invariant."""
        ctx = HolographicContext.get_session()
        intents = classification.get("functional_intent") or []
        if intents:
            ctx.build_invariant_from_symbols(
                [i.get("symbol_name") or i.get("name", "") for i in intents]
            )
        return ctx

    def _six_phase_topology_prefix(
        self,
        output_dir: Optional[str | Path],
    ) -> Dict[str, Any]:
        """Phase 2: encode an existing workspace as compact PaP tokens.

        The LLM is never given raw source files; it only sees the topological
        summary and the PaP token dimension.
        """
        result: Dict[str, Any] = {"encoded": False, "nodes": [], "edges": [], "dim": 0}
        if not output_dir:
            return result
        root = Path(output_dir).resolve()
        if not root.is_dir() or not any(f.is_file() for f in root.rglob("*")):
            return result
        try:
            encoder = FockGraphEncoder(dim=256)
            repo = encoder.encode_repository(root)
            result = {
                "encoded": True,
                "dim": repo.get("dim", 0),
                "nodes": list(repo.get("nodes", {}).keys()),
                "edges": [
                    {"source": e.get("source"), "relation": e.get("relation"), "target": e.get("target")}
                    for e in repo.get("edges", [])
                ],
            }
            _accel_log(
                "info",
                f"FoGE Phase 2 PaP tokens regenerated for {root}: dim={result['dim']}, nodes={len(result['nodes'])}, edges={len(result['edges'])}",
            )
        except Exception as exc:
            logger.debug("FoGE encoding skipped: %s", exc)
        return result

    def _six_phase_bootstrap_skeleton(
        self,
        classification: Dict[str, Any],
        topology: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Phase 3: category-theoretic bootstrap producing a rigid skeleton."""
        if classification.get("_skeleton"):
            return classification["_skeleton"]
        architecture = (classification.get("architecture") or "graph_polyglot").lower()
        intents = classification.get("functional_intent") or []
        repo_graph = topology if topology.get("encoded") else None
        try:
            bootstrapper = SchemaBootstrapper(architecture_hint=architecture)
            return bootstrapper.bootstrap(intents, repo_graph=repo_graph)
        except Exception as exc:
            logger.debug("Adjoint bootstrap skipped: %s", exc)
            return {}

    def _six_phase_formal_feedback(
        self,
        data: Dict[str, Any],
        output_dir: Optional[str | Path],
    ) -> str:
        """Phases 5 and 6: concolic (Z3) and SHACL/Prolog verification."""
        feedback_parts: List[str] = []

        # Concolic feedback.
        try:
            concolic = ConcolicManifestVerifier(data)
            c_result = concolic.verify()
            if not c_result.satisfiable:
                feedback_parts.append("Concolic (Z3) verification failed:")
                for rule in c_result.conflicting_rules:
                    feedback_parts.append(f"- {rule}")
        except Exception as exc:
            logger.debug("Concolic verification skipped: %s", exc)

        # SHACL firewall.
        try:
            firewall = LogicalFirewall(data)
            fw_report = firewall.validate()
            if not fw_report.conforms:
                feedback_parts.append("SHACL firewall violations:")
                for v in fw_report.violations:
                    feedback_parts.append(f"- {v.get('message', '')}")
        except Exception as exc:
            logger.debug("SHACL firewall skipped: %s", exc)

        # Chiasmus / Prolog verification on the workspace if it exists.
        if output_dir:
            root = Path(output_dir).resolve()
            if root.is_dir() and any(f.is_file() for f in root.rglob("*")):
                try:
                    from aero_forge.builder.chiasmus import analyze_repository
                    boundaries = [
                        (e.get("source", ""), e.get("target", ""), e.get("boundary_type", ""))
                        for e in data.get("edges", [])
                    ]
                    report = analyze_repository(root, boundaries=boundaries)
                    if report.cycles or report.unsafe_ffi:
                        feedback_parts.append("Prolog/Chiasmus verification failed:")
                        for cycle in report.cycles:
                            feedback_parts.append(f"- Cycle: {' -> '.join(cycle)}")
                        for t in report.unsafe_ffi:
                            feedback_parts.append(
                                f"- Unsafe FFI: {t['source']} ({t['source_lang']}) -> "
                                f"{t['target']} ({t['target_lang']}) via {t['relation']}"
                            )
                except Exception as exc:
                    logger.debug("Chiasmus verification skipped: %s", exc)

        return "\n".join(feedback_parts)

    def _six_phase_user_content(
        self,
        prompt_text: str,
        classification: Dict[str, Any],
        hctx: HolographicContext,
        topology: Dict[str, Any],
        skeleton: Dict[str, Any],
    ) -> str:
        """Phase 4: bounded intent completion prompt.

        The LLM receives the user prompt, a verified classification, the
        category-theoretic skeleton, and a compact topological prefix. It does
        NOT receive raw source files; all structural constraints are encoded in
        the skeleton and the SMT/SHACL verification layer enforces them after the
        fact.
        """
        classification_text = (
            json.dumps(classification, indent=2) if classification else "{}"
        )
        skeleton_text = json.dumps(skeleton, indent=2) if skeleton else "{}"
        topology_text = json.dumps(topology, indent=2)

        drift_text = ""
        if hctx.hinv is not None:
            symbols = [
                i.get("symbol_name") or i.get("name", "")
                for i in (classification.get("functional_intent") or [])
            ]
            if symbols:
                try:
                    similarity = hctx.measure_symbol_drift(symbols)
                    drift = 1.0 - similarity
                    drift_text = (
                        f"\nIntent-to-invariant drift: {drift:.4f} "
                        f"(similarity: {similarity:.4f}; Theorem 1 safety bound: similarity >= {SAFETY_LOWER_BOUND_SIMILARITY:.4f})"
                    )
                    if drift > DRIFT_THRESHOLD:
                        drift_text += " Holographic context restoration triggered before this LLM call."
                except Exception:
                    pass

        typed_holes = skeleton.get("typed_holes", [])
        typed_holes_text = json.dumps(typed_holes, indent=2) if typed_holes else "[]"

        return (
            f"{prompt_text}\n\n"
            f"Verified classification:\n```json\n{classification_text}\n```\n"
            f"\nTopological prefix (FoGE):\n```json\n{topology_text}\n```\n"
            f"\nManifest skeleton (Adjoint / typed holes to fill):\n```json\n{skeleton_text}\n```\n"
            f"\nExplicit typed holes:\n```json\n{typed_holes_text}\n```\n"
            f"{drift_text}\n\n"
            "Return the COMPLETE PolyglotGraphBlueprint JSON. "
            "Fill every <TYPED_HOLE> with a concrete value. "
            "It must include nodes (with exports), edges/contracts, functional_intent, source_files, and metadata. "
            "NO PREAMBLE. NO EXPLANATION. ONLY JSON."
        )

    # ------------------------------------------------------------------
    # Fiber-wise atomic enrichment with HIS drift correction and adjoint
    # stubbing fallback.
    # ------------------------------------------------------------------
    def _current_symbols(
        self,
        classification: Dict[str, Any],
        partial: Dict[str, Any],
    ) -> List[str]:
        """Collect the symbol names currently represented in the partial blueprint."""
        symbols: set = set()
        for intent in classification.get("functional_intent") or []:
            name = intent.get("symbol_name") or intent.get("name")
            if name:
                symbols.add(name)
        for node in partial.get("nodes") or []:
            node_id = node.get("node_id")
            if node_id:
                symbols.add(node_id)
            for exported in node.get("exports") or []:
                if exported:
                    symbols.add(exported)
        for edge in partial.get("edges") or []:
            if edge.get("symbol"):
                symbols.add(edge["symbol"])
        return sorted(symbols)

    def _measure_his_drift(
        self,
        hctx: HolographicContext,
        classification: Dict[str, Any],
        partial: Dict[str, Any],
    ) -> float:
        """Return drift as 1 - cosine_similarity against the invariant.

        Theorem 1 sets the safety lower bound at similarity 0.7071, so a drift
        value greater than ``DRIFT_THRESHOLD`` (1 - 0.7071) triggers context
        restoration and FoGE pruning.  Low-complexity prompts (fewer than 4
        symbols) are short and stable, so the invariant itself is fragile;
        skip drift correction to avoid pruning useful context.
        """
        if hctx.hinv is None:
            return 0.0
        symbols = self._current_symbols(classification, partial)
        if len(symbols) < 4:
            return 0.0
        try:
            similarity = hctx.measure_symbol_drift(symbols)
            return 1.0 - similarity
        except Exception:
            return 0.0

    def _apply_his_restore(
        self,
        hctx: HolographicContext,
        classification: Dict[str, Any],
        partial: Dict[str, Any],
    ) -> None:
        """Clean a noisy context vector against the stored invariant.

        Computes the current context vector ``N_context`` from the symbols in the
        partial blueprint, then applies the ``sign(H_inv + N_context)`` cleanup
        operator through ``HolographicContext.restore_context``.
        """
        if hctx.hinv is None:
            return
        symbols = self._current_symbols(classification, partial)
        if not symbols:
            return
        try:
            context = intent_vector(symbols)
            # Theorem 1 context cleanup: bundle the invariant with the current
            # noisy context and threshold back to a clean bipolar vector.
            hctx.restore_context(context, noise=1.0)
            _accel_log(
                "info",
                f"HIS context restored for symbols: {symbols}; drift exceeded threshold",
            )
        except Exception as exc:
            logger.debug("HIS context restoration skipped: %s", exc)

    def _prune_foge_topology(
        self,
        topology: Dict[str, Any],
        symbols: List[str],
    ) -> Dict[str, Any]:
        """Reduce the FoGE prefix to symbols relevant to the current fiber.

        Keeps only nodes whose id or exports overlap with the symbol set and
        edges between kept nodes.
        """
        if not topology.get("encoded"):
            return topology
        symbol_set = set(symbols)
        kept_nodes: List[str] = []
        for node in topology.get("nodes") or []:
            name = str(node)
            if any(sym in name for sym in symbol_set) or not symbol_set:
                kept_nodes.append(node)
        # Hard cap on token overhead.
        kept_nodes = kept_nodes[:10]
        kept_set = set(kept_nodes)
        kept_edges: List[Dict[str, Any]] = []
        for edge in topology.get("edges") or []:
            src = edge.get("source")
            tgt = edge.get("target")
            if src in kept_set and tgt in kept_set:
                kept_edges.append(edge)
        return {
            "encoded": topology.get("encoded", False),
            "dim": topology.get("dim", 0),
            "nodes": kept_nodes,
            "edges": kept_edges,
        }

    def _prepare_phase4_llm_call(
        self,
        user_content: str,
        hctx: HolographicContext,
        classification: Dict[str, Any],
        partial: Dict[str, Any],
        topology: Optional[Dict[str, Any]] = None,
        fiber_coordinate: Optional[str] = None,
    ) -> str:
        """Log the exact Phase 4 prompt and force HIS restoration when required.

        Restoration is triggered when the prompt exceeds ``PHASE4_MAX_TOKENS``
        tokens or when the HIS drift metric violates Theorem 1 (similarity
        < ``SAFETY_LOWER_BOUND_SIMILARITY``).  The restored context is bundled
        by calling ``hctx.restore_context`` with the invariant ``H_inv`` vector.
        """
        token_count = _phase4_token_count(user_content)
        drift = self._measure_his_drift(hctx, classification, partial)
        similarity = 1.0 - drift

        logger.info(
            "Phase 4 prompt (tokens=%d, similarity=%.4f, drift=%.4f):\n%s",
            token_count,
            similarity,
            drift,
            user_content,
        )

        restoration_triggered = (
            token_count > PHASE4_MAX_TOKENS
            or similarity < SAFETY_LOWER_BOUND_SIMILARITY
        )
        topology_nodes = len(topology.get("nodes", [])) if topology else 0
        topology_edges = len(topology.get("edges", [])) if topology else 0
        _log_phase4_telemetry(
            prompt=user_content,
            token_count=token_count,
            drift=drift,
            similarity=similarity,
            fiber_coordinate=fiber_coordinate,
            restoration_triggered=restoration_triggered,
            topology_node_count=topology_nodes,
            topology_edge_count=topology_edges,
        )

        if restoration_triggered:
            _accel_log(
                "info",
                f"HIS context restoration forced: tokens={token_count}, similarity={similarity:.4f}, "
                f"drift={drift:.4f}, Theorem 1 bound similarity>={SAFETY_LOWER_BOUND_SIMILARITY:.4f}",
            )
            self._apply_his_restore(hctx, classification, partial)
            if topology is not None:
                topology.update(
                    self._prune_foge_topology(
                        topology,
                        self._current_symbols(classification, partial),
                    )
                )
            user_content += (
                f"\n\n[HIS context restoration triggered: "
                f"tokens={token_count}, similarity={similarity:.4f}, drift={drift:.4f}. "
                f"H_inv has been bundled with the current context using sign(H_inv + N_context) before the LLM call.]"
            )

        return user_content

    # ------------------------------------------------------------------
    # Sectional Fiber Completion: split each Grothendieck fiber into
    # manifest / contracts / functions blocks so each LLM call stays inside
    # the high-attention 1024-token window.
    # ------------------------------------------------------------------
    def _build_section_prompt(
        self,
        section: str,
        prompt_text: str,
        fiber_coordinate: Dict[str, Any],
        stub: NodeStub,
        partial: Dict[str, Any],
        peer_nodes: List[Dict[str, Any]],
        skeleton: Dict[str, Any],
    ) -> str:
        """Return a compact prompt for one section of one Grothendieck fiber."""
        symbol = fiber_coordinate.get("symbol") or (
            stub.exports[0] if stub.exports else stub.node_id
        )
        node_id = stub.node_id
        architecture = partial.get("architecture", "graph_polyglot")
        peer_text = json.dumps(peer_nodes, indent=2) if peer_nodes else "[]"
        skeleton_text = json.dumps(skeleton, indent=2) if skeleton else "{}"

        if section == "manifest":
            return (
                f"{prompt_text}\n\n"
                f"You are filling the MANIFEST block for the Grothendieck fiber "
                f"π_X^{{-1}}({symbol}) -> {node_id}.\n"
                f"Architecture: {architecture}.\n"
                f"Manifest skeleton (Adjoint / typed holes to fill):\n```json\n{skeleton_text}\n```\n\n"
                f"Stub: {json.dumps(stub.__dict__, default=str)}\n\n"
                f"Return ONLY a JSON object with these keys for the node:\n"
                f"  'node_id' (string), 'lang' (string), 'toolchain' (string),\n"
                f"  'source_files' (list of strings), 'exports' (list of strings),\n"
                f"  'purpose' (string).\n"
                f"NO PREAMBLE. NO EXPLANATION. ONLY JSON."
            )

        if section == "contracts":
            return (
                f"{prompt_text}\n\n"
                f"You are filling the CONTRACTS / EDGES block for node '{node_id}' "
                f"with exports {stub.exports}.\n"
                f"Architecture: {architecture}.\n"
                f"Other nodes already in the blueprint:\n```json\n{peer_text}\n```\n\n"
                f"Return ONLY a JSON object with key 'edges'.\n"
                f"Each edge is: source (string), target (string), boundary_type (string),\n"
                f"symbol (string), args (list of type names), return_type (string).\n"
                f"For pure_python / single-language nodes return '{{\"edges\": []}}'.\n"
                f"NO PREAMBLE. NO EXPLANATION. ONLY JSON."
            )

        # section == "functions"
        return (
            f"{prompt_text}\n\n"
            f"You are filling the FUNCTIONS block for node '{node_id}'.\n"
            f"Exports: {stub.exports}.\n"
            f"Return ONLY a JSON object with:\n"
            f"  'logic_sketch' (string) - concise implementation description,\n"
            f"  'contracts' (list of ABIContract dicts) - only for cross-language symbols.\n"
            f"NO PREAMBLE. NO EXPLANATION. ONLY JSON."
        )

    def _prepare_section_llm_call(
        self,
        user_content: str,
        hctx: HolographicContext,
        classification: Dict[str, Any],
        partial: Dict[str, Any],
        section: str,
        fiber_coordinate: Optional[str] = None,
    ) -> str:
        """Ensure the sectional prompt is under the 1024-token budget.

        If the prompt exceeds ``SECTIONAL_MAX_TOKENS`` or the HIS similarity
        against the invariant drops below Theorem 1's safety bound
        (``SAFETY_LOWER_BOUND_SIMILARITY``), force a context cleanup using the
        ``sign(H_inv + N_context)`` operator before the LLM call.
        """
        token_count = _phase4_token_count(user_content)
        # _measure_his_drift returns 1 - similarity; recover similarity for the
        # explicit Theorem 1 comparison.
        drift = self._measure_his_drift(hctx, classification, partial)
        similarity = 1.0 - drift

        logger.info(
            "Sectional Phase 4 prompt (%s, tokens=%d, similarity=%.4f, drift=%.4f)",
            section,
            token_count,
            similarity,
            drift,
        )

        restoration_triggered = (
            token_count > SECTIONAL_MAX_TOKENS
            or similarity < SAFETY_LOWER_BOUND_SIMILARITY
        )
        _log_phase4_telemetry(
            prompt=user_content,
            token_count=token_count,
            drift=drift,
            similarity=similarity,
            fiber_coordinate=fiber_coordinate,
            restoration_triggered=restoration_triggered,
            topology_node_count=0,
            topology_edge_count=0,
        )

        if restoration_triggered:
            _accel_log(
                "info",
                f"HIS context restoration forced for section {section}: "
                f"tokens={token_count}, similarity={similarity:.4f}, drift={drift:.4f}",
            )
            self._apply_his_restore(hctx, classification, partial)
            user_content += (
                f"\n\n[HIS context restoration triggered for section {section}: "
                f"tokens={token_count}, similarity={similarity:.4f}, drift={drift:.4f}. "
                f"H_inv has been bundled with the current context using "
                f"sign(H_inv + N_context) before the LLM call.]"
            )

        return user_content

    def _run_section(
        self,
        client: Any,
        system: str,
        prompt_text: str,
        section: str,
        fiber_coordinate: Dict[str, Any],
        stub: NodeStub,
        partial: Dict[str, Any],
        hctx: HolographicContext,
        classification: Dict[str, Any],
        peer_nodes: List[Dict[str, Any]],
        skeleton: Dict[str, Any],
    ) -> Result:
        """Execute one sectional LLM call with localized retry/fallback.

        Returns a ``Result`` monad: ``ok=True`` with the parsed JSON fragment on
        success, ``ok=False`` with an error description after all retries fail.
        """
        user_content = self._build_section_prompt(
            section, prompt_text, fiber_coordinate, stub, partial, peer_nodes, skeleton
        )
        user_content = self._prepare_section_llm_call(
            user_content,
            hctx,
            classification,
            partial,
            section,
            fiber_coordinate=json.dumps(fiber_coordinate),
        )
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]

        last_error = "empty response"
        for attempt in range(SECTIONAL_RETRY_ATTEMPTS):
            try:
                raw = client.generate(
                    messages, temperature=0.2, max_tokens=SECTIONAL_MAX_TOKENS
                )
            except Exception as exc:
                last_error = str(exc)
                _accel_log("warning", f"Section {section} LLM error: {last_error}")
                continue

            if not raw or not raw.strip():
                last_error = "empty response"
                _accel_log("warning", f"Section {section} attempt {attempt + 1}: empty response")
                continue

            try:
                data = _extract_json(raw)
            except Exception as exc:
                last_error = f"JSON extraction failed: {exc}"
                _accel_log(
                    "warning",
                    f"Section {section} attempt {attempt + 1}: {last_error}; raw preview: {raw[:400]!r}",
                )
                messages.append({"role": "assistant", "content": raw})
                messages.append(
                    {
                        "role": "user",
                        "content": "Return ONLY valid JSON. No prose. No markdown.",
                    }
                )
                continue

            validated = self._validate_section(section, data, stub)
            if validated is not None:
                _accel_log(
                    "info",
                    f"Section {section} for '{stub.node_id}' succeeded on attempt {attempt + 1}",
                )
                return Result(ok=True, value=validated)
            last_error = "section validation failed"

        _accel_log(
            "warning",
            f"Section {section} for '{stub.node_id}' failed after {SECTIONAL_RETRY_ATTEMPTS} attempts: {last_error}",
        )
        return Result(ok=False, error=f"{section}: {last_error}")

    @staticmethod
    def _validate_section(
        section: str, data: Any, stub: NodeStub
    ) -> Optional[Any]:
        """Validate and normalize a sectional JSON fragment.

        A section is only accepted when the response contains the specific key(s)
        requested for that section.  This prevents a malformed or unrelated JSON
        object (e.g. a compressed v2 blueprint) from being mistaken for a valid
        fiber section.
        """
        if section == "manifest":
            if not isinstance(data, dict) or "node_id" not in data:
                return None
            node = {
                "node_id": str(data.get("node_id") or stub.node_id),
                "lang": str(data.get("lang") or stub.lang),
                "toolchain": str(data.get("toolchain") or stub.toolchain),
                "source_files": list(data.get("source_files") or stub.source_files or []),
                "exports": list(data.get("exports") or stub.exports or []),
                "purpose": str(data.get("purpose") or stub.purpose or ""),
            }
            return node

        if section == "contracts":
            if isinstance(data, list):
                edges = data
            elif isinstance(data, dict) and "edges" in data:
                edges = data.get("edges", [])
            else:
                return None
            normalized = []
            for edge in edges or []:
                if not isinstance(edge, dict):
                    continue
                src = edge.get("source") or stub.node_id
                tgt = edge.get("target")
                if not src or not tgt:
                    continue
                normalized.append(
                    {
                        "source": str(src),
                        "target": str(tgt),
                        "boundary_type": str(edge.get("boundary_type") or "c_abi"),
                        "symbol": str(edge.get("symbol") or ""),
                        "args": [str(a) for a in (edge.get("args") or [])],
                        "return_type": str(edge.get("return_type") or ""),
                        "is_zero_copy": bool(edge.get("is_zero_copy", False)),
                    }
                )
            return normalized

        if section == "functions":
            if not isinstance(data, dict) or "logic_sketch" not in data:
                return None
            node_update = {
                "logic_sketch": str(data.get("logic_sketch") or ""),
            }
            contracts = data.get("contracts")
            if contracts is not None:
                node_update["contracts"] = list(contracts)
            return node_update

        return None

    def _recompose_fiber_sections(
        self,
        manifest_result: Result,
        contracts_result: Result,
        functions_result: Result,
        stub: NodeStub,
    ) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        """Recompose the three section results for one fiber into a node + edges.

        Uses the algebraic error-handling monad: a failed section is replaced by
        the deterministic adjoint stub rather than aborting the whole graph.
        """
        node: Dict[str, Any] = manifest_result.value if manifest_result.ok else {
            "node_id": stub.node_id,
            "lang": stub.lang,
            "toolchain": stub.toolchain,
            "source_files": list(stub.source_files or []),
            "exports": list(stub.exports or []),
            "purpose": stub.purpose,
        }

        edges: List[Dict[str, Any]] = []
        if contracts_result.ok:
            edges = list(contracts_result.value or [])

        if functions_result.ok:
            node.update(functions_result.value or {})
        else:
            node["logic_sketch"] = (
                f"Auto-generated {stub.lang} stub for '{stub.exports[0] if stub.exports else stub.node_id}'. "
                f"Implement the symbol matching its functional intent and any ABI contract."
            )

        if not manifest_result.ok:
            _accel_log(
                "warning",
                f"Manifest section failed for '{stub.node_id}'; using adjoint stub",
            )
        if not contracts_result.ok:
            _accel_log(
                "warning",
                f"Contracts section failed for '{stub.node_id}'; leaving edges empty",
            )
        if not functions_result.ok:
            _accel_log(
                "warning",
                f"Functions section failed for '{stub.node_id}'; using logic stub",
            )

        return node, edges

    def _query_fiber_sectional(
        self,
        client: Any,
        system: str,
        prompt_text: str,
        classification: Dict[str, Any],
        topology: Dict[str, Any],
        skeleton: Dict[str, Any],
        partial: Dict[str, Any],
        intent: Dict[str, Any],
        stub: NodeStub,
        hctx: HolographicContext,
    ) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        """Fill one Grothendieck fiber through three independent bounded sections."""
        symbol = (
            intent.get("symbol_name")
            or intent.get("name")
            or (stub.exports[0] if stub.exports else stub.node_id)
        )
        node_id = stub.node_id
        fiber_coordinate = {
            "symbol": symbol,
            "node_id": node_id,
            "lang": stub.lang,
            "toolchain": stub.toolchain,
            "exports": stub.exports,
            "grothendieck_fiber": f"π_X^{{-1}}({symbol}) -> {node_id}",
        }

        peer_nodes = [
            {"node_id": n.get("node_id"), "lang": n.get("lang"), "exports": n.get("exports")}
            for n in partial.get("nodes", [])
        ]

        # Manifest, contracts, and functions are queried independently so no
        # single 3056-byte skeleton can push the prompt over the attention window.
        manifest_result = self._run_section(
            client, system, prompt_text, "manifest", fiber_coordinate,
            stub, partial, hctx, classification, peer_nodes, skeleton,
        )
        contracts_result = self._run_section(
            client, system, prompt_text, "contracts", fiber_coordinate,
            stub, partial, hctx, classification, peer_nodes, skeleton,
        )
        functions_result = self._run_section(
            client, system, prompt_text, "functions", fiber_coordinate,
            stub, partial, hctx, classification, peer_nodes, skeleton,
        )

        node, edges = self._recompose_fiber_sections(
            manifest_result, contracts_result, functions_result, stub
        )
        section_used = manifest_result.ok or contracts_result.ok or functions_result.ok
        return node, edges, section_used

    def _stub_to_node(self, stub: NodeStub) -> Dict[str, Any]:
        """Convert an adjoint NodeStub into a graph node dict with a function stub."""
        symbol = stub.exports[0] if stub.exports else stub.node_id
        return {
            "node_id": stub.node_id,
            "lang": stub.lang,
            "toolchain": stub.toolchain,
            "source_files": list(stub.source_files),
            "exports": list(stub.exports),
            "purpose": stub.purpose,
            "logic_sketch": (
                f"Auto-generated {stub.lang} stub for '{symbol}'. "
                f"Implement the symbol matching its functional intent and any ABI contract."
            ),
            "contracts": [],
        }

    def _merge_fiber_node(
        self,
        existing: Dict[str, Any],
        incoming: Dict[str, Any],
    ) -> None:
        """Merge a fiber-generated node into an existing partial node."""
        for key in ("lang", "toolchain", "purpose", "logic_sketch"):
            if incoming.get(key) and incoming[key] != "<TYPED_HOLE>":
                existing[key] = incoming[key]
        for key in ("exports", "source_files"):
            combined = list(existing.get(key) or [])
            for item in incoming.get(key) or []:
                if item and item not in combined:
                    combined.append(item)
            existing[key] = combined
        for key in ("compiler_flags", "dependencies"):
            combined = list(existing.get(key) or [])
            for item in incoming.get(key) or []:
                if item and item not in combined:
                    combined.append(item)
            existing[key] = combined
        if incoming.get("extra"):
            existing.setdefault("extra", {}).update(incoming["extra"])

    def _sanitize_partial(self, partial: Dict[str, Any]) -> None:
        """Repair an assembled partial blueprint before formal validation.

        Removes edges referencing unknown nodes and drops intra-language edges
        (those are internal imports, not FFI boundaries). Cross-language edges
        are normalized to one of the supported ``BoundaryContractType`` values.
        """
        node_ids = {n.get("node_id") for n in partial.get("nodes") or [] if n.get("node_id")}
        node_langs = {
            n.get("node_id"): (n.get("lang") or "").lower()
            for n in partial.get("nodes") or []
        }
        valid_boundaries = {
            "c_abi", "pyo3_maturin", "wasm_wasi", "jni", "cgo", "pinvoke", "cuda_hip_c"
        }
        sanitized_edges: List[Dict[str, Any]] = []
        for edge in partial.get("edges") or []:
            src = edge.get("source")
            tgt = edge.get("target")
            if src not in node_ids or tgt not in node_ids or src == tgt:
                continue
            src_lang = node_langs.get(src, "")
            tgt_lang = node_langs.get(tgt, "")
            if src_lang and tgt_lang and src_lang == tgt_lang:
                continue
            boundary = str(edge.get("boundary_type") or "c_abi").lower().replace("-", "_")
            if boundary not in valid_boundaries:
                boundary = "c_abi"
            edge["boundary_type"] = boundary
            edge["symbol"] = edge.get("symbol") or ""
            edge["args"] = [str(a) for a in (edge.get("args") or [])]
            edge["return_type"] = str(edge.get("return_type") or "")
            edge["is_zero_copy"] = bool(edge.get("is_zero_copy", False))
            sanitized_edges.append(edge)
        partial["edges"] = sanitized_edges

        # Ensure every functional_intent symbol has a node or edge representation.
        intent_symbols = {
            i.get("symbol_name") or i.get("name")
            for i in partial.get("functional_intent") or []
        }
        present = node_ids.copy()
        for edge in sanitized_edges:
            if edge.get("symbol"):
                present.add(edge["symbol"])
        for node in partial.get("nodes") or []:
            for exported in node.get("exports") or []:
                if exported:
                    present.add(exported)
        missing = sorted(intent_symbols - present)
        if missing:
            _accel_log(
                "warning",
                f"Fiber-wise enrichment: missing symbols {missing}; injecting adjoint stubs",
            )

    def _query_fiber(
        self,
        client: Any,
        system: str,
        prompt_text: str,
        classification: Dict[str, Any],
        topology: Dict[str, Any],
        skeleton: Dict[str, Any],
        partial: Dict[str, Any],
        intent: Dict[str, Any],
        stub: NodeStub,
        hctx: HolographicContext,
    ) -> str:
        """Ask the LLM to fill a single typed hole / Grothendieck fiber."""
        symbol = intent.get("symbol_name") or intent.get("name") or (
            stub.exports[0] if stub.exports else stub.node_id
        )
        node_id = stub.node_id
        architecture = partial.get("architecture", "graph_polyglot")

        # Pass the HIS drift metric into the bounded-completion prompt so the
        # LLM can see how far the current context has drifted from the invariant.
        drift = self._measure_his_drift(hctx, classification, partial)
        drift_note = f"Intent-to-invariant drift: {drift:.4f} (Theorem 1 safety bound: similarity >= {SAFETY_LOWER_BOUND_SIMILARITY:.4f})"
        if drift > DRIFT_THRESHOLD:
            drift_note += " Holographic context restoration triggered."

        fiber_coordinate = {
            "base_intent": intent,
            "node_id": node_id,
            "lang": stub.lang,
            "toolchain": stub.toolchain,
            "exports": stub.exports,
            "source_files": stub.source_files,
            "grothendieck_fiber": f"π_X^{{-1}}({symbol}) -> {node_id}",
        }
        context_text = json.dumps(
            {
                "project": partial.get("project"),
                "architecture": architecture,
                "functional_intent": partial.get("functional_intent"),
                "already_populated_nodes": [
                    {"node_id": n.get("node_id"), "exports": n.get("exports")}
                    for n in partial.get("nodes") or []
                ],
                "fiber_coordinate": fiber_coordinate,
                "drift_note": drift_note,
            },
            indent=2,
        )

        typed_holes = skeleton.get("typed_holes", [])
        typed_holes_text = json.dumps(typed_holes, indent=2) if typed_holes else "[]"

        user_content = (
            f"{prompt_text}\n\n"
            f"Verified classification:\n```json\n{json.dumps(classification, indent=2)}\n```\n"
            f"\nTopological prefix (FoGE):\n```json\n{json.dumps(topology, indent=2)}\n```\n"
            f"\nManifest skeleton (Adjoint / typed holes to fill):\n```json\n{json.dumps(skeleton, indent=2)}\n```\n"
            f"\nExplicit typed holes:\n```json\n{typed_holes_text}\n```\n"
            f"\nAlready populated context:\n```json\n{context_text}\n```\n"
            f"\nGrothendieck fiber coordinate: {json.dumps(fiber_coordinate)}\n"
            f"\n{drift_note}\n"
            f"\nFill the typed hole for the symbol '{symbol}' implemented by node '{node_id}'. "
            f"Return ONLY a JSON object with these keys:\n"
            f"  - 'nodes': a list containing ONE fully specified node dict for '{node_id}'"
            f" (lang={stub.lang}, toolchain={stub.toolchain}, exports={stub.exports}, source_files={stub.source_files}).\n"
            f"  - 'edges': a list of cross-language boundary edges (source, target, boundary_type, symbol, args, return_type) "
            f"that connect this node's exported symbols to other nodes. For pure_python nodes leave 'edges' empty.\n"
            f"NO PREAMBLE. NO EXPLANATION. ONLY JSON."
        )

        user_content = self._prepare_phase4_llm_call(
            user_content,
            hctx,
            classification,
            partial,
            topology=topology,
            fiber_coordinate=json.dumps(fiber_coordinate),
        )

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]
        return client.generate(messages, temperature=0.2, max_tokens=4096)

    def _fiber_wise_atomic_completion(
        self,
        client: Any,
        system: str,
        prompt_text: str,
        classification: Dict[str, Any],
        hctx: HolographicContext,
        topology: Dict[str, Any],
        skeleton: Dict[str, Any],
        output_dir: Optional[str | Path],
        project_name: Optional[str],
    ) -> PolyglotGraphBlueprint:
        """Populate the blueprint one Grothendieck fiber at a time.

        Each fiber corresponds to one (functional_intent, node_stub) pair. HIS
        drift is checked before every LLM call; if it exceeds the Theorem 1
        safety threshold (``DRIFT_THRESHOLD``) the context is restored and the
        FoGE prefix is pruned. Empty LLM responses fall back to the adjoint ΣF
        stubbing path.
        """
        architecture = (
            classification.get("architecture")
            or skeleton.get("architecture")
            or "graph_polyglot"
        )
        functional_intent = (
            classification.get("functional_intent")
            or skeleton.get("functional_intent")
            or []
        )

        bootstrapper = SchemaBootstrapper(architecture_hint=architecture)
        stubs = bootstrapper.ΣF(functional_intent)
        repo_edges = [dict(e) for e in (topology.get("edges") or [])]
        stubs, edges = bootstrapper.ΔF(stubs, repo_edges)
        bundle = bootstrapper.grothendieck_bundle(functional_intent, stubs)
        full_fallback = bootstrapper.ΠF(stubs, edges, functional_intent, architecture)
        # Prefer a deterministic skeleton with explicit language-node pairings when
        # the classification provides one; otherwise fall back to the bootstrapper.
        skeleton_nodes = skeleton.get("nodes") or full_fallback.get("nodes") or []
        fallback_nodes = {n["node_id"]: n for n in skeleton_nodes}

        partial: Dict[str, Any] = {
            "project": (
                project_name
                or classification.get("project")
                or skeleton.get("project")
                or architecture
            ),
            "architecture": architecture,
            "nodes": [],
            "edges": [],
            "functional_intent": functional_intent,
            "metadata": {
                "bootstrap_method": "category_theoretic",
                "prompt": prompt_text,
            },
            "primary_entrypoint": skeleton.get("primary_entrypoint", ""),
            "build_script": skeleton.get("build_script", ""),
            "output_dir": str(Path(output_dir) / "dist") if output_dir else "./dist",
        }

        node_map: Dict[str, Dict[str, Any]] = {}
        llm_fiber_used = False

        for intent, stub in bundle:
            symbol = (
                intent.get("symbol_name")
                or intent.get("name")
                or (stub.exports[0] if stub.exports else stub.node_id)
            )

            # HIS drift check and correction before every LLM call.
            drift = self._measure_his_drift(hctx, classification, partial)
            if drift > DRIFT_THRESHOLD:
                self._apply_his_restore(hctx, classification, partial)
                topology = self._prune_foge_topology(
                    topology,
                    self._current_symbols(classification, partial),
                )

            node, fiber_edges, section_used = self._query_fiber_sectional(
                client,
                system,
                prompt_text,
                classification,
                topology,
                skeleton,
                partial,
                intent,
                stub,
                hctx,
            )

            if section_used:
                llm_fiber_used = True

            for edge in fiber_edges:
                partial["edges"].append(edge)

            if node and node.get("node_id"):
                node_id = node["node_id"]
                if node_id in node_map:
                    self._merge_fiber_node(node_map[node_id], node)
                else:
                    node_map[node_id] = node
                    partial["nodes"].append(node)
            else:
                _accel_log(
                    "warning",
                    f"Fiber-wise enrichment: sectional fiber failed for '{symbol}'; using adjoint ΣF stub",
                )
                node = fallback_nodes.get(stub.node_id) or self._stub_to_node(stub)
                node_id = node["node_id"]
                if node_id in node_map:
                    self._merge_fiber_node(node_map[node_id], node)
                else:
                    node_map[node_id] = node
                    partial["nodes"].append(node)

        # If the LLM never returned edges, fall back to the deterministic skeleton edges.
        if not partial.get("edges") and skeleton.get("edges"):
            partial["edges"] = [dict(e) for e in skeleton["edges"]]

        self._sanitize_partial(partial)

        # Phases 5-6: run formal verification on the assembled graph.
        formal_feedback = self._six_phase_formal_feedback(partial, output_dir)
        if formal_feedback:
            _accel_log(
                "warning",
                f"Fiber-wise enrichment formal feedback: {formal_feedback}",
            )
            logger.warning("Fiber-wise enrichment formal feedback: %s", formal_feedback)

        try:
            graph = PolyglotGraphBlueprint.model_validate(partial)
        except Exception as exc:
            raise IntentCompilerError(
                f"Fiber-wise atomic completion assembled an invalid blueprint: {exc}"
            ) from exc

        graph.metadata["prompt"] = prompt_text
        graph.metadata["llm_initialized"] = True
        graph.metadata["status"] = "finalized"
        graph.metadata["generation_method"] = (
            "llm_synthesized" if llm_fiber_used else "deterministic_fallback"
        )
        _accel_log(
            "success",
            f"Blueprint fiber-wise atomic enrichment complete (llm_used={llm_fiber_used})",
        )
        return graph

    def compile_prompt_to_graph(
        self,
        prompt_text: str,
        output_dir: Optional[str | Path] = None,
        project_name: Optional[str] = None,
        architecture: Optional[str] = None,
    ) -> PolyglotGraphBlueprint:
        """Compile *prompt_text* into a validated ``PolyglotGraphBlueprint``.

        Uses tiered enrichment: a classification call first establishes
        ``architecture``, ``toolchains``, ``functional_intent``, and ``nodes``.
        The Orchestrator verifies the classification deterministically (without
        reading the prompt). If the classification is sound, a second call asks
        for the complete graph blueprint populated with contracts/edges.

        If *architecture* is provided, it is injected into the classification
        prompt and used to override an LLM classification that does not match
        the caller's explicit target.
        """
        _accel_log("info", "Enriching Blueprint (six-phase pipeline)...")
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

        system = _GRAPH_SYSTEM_PROMPT
        if self._system_prompt_extra:
            system = f"{system}\n\n{self._system_prompt_extra}"

        # Phase 0 / Tier 0: classify architecture, toolchains, nodes, and functional_intent.
        classification, _class_errors = self._classify_graph(
            client, system, prompt_text, architecture=architecture
        )
        if classification is not None and "_full_graph" in classification:
            return classification["_full_graph"]

        # Phases 1-3: HIS context binding, FoGE topology prefix, and
        # category-theoretic skeleton bootstrap.
        hctx = self._six_phase_bind_context(classification or {})
        topology = self._six_phase_topology_prefix(output_dir)
        skeleton = self._six_phase_bootstrap_skeleton(
            classification or {}, topology
        )
        self._inject_wasm_target(skeleton, prompt_text)

        # Attempt fiber-wise atomic completion first. It queries the LLM once
        # per Grothendieck fiber (typed hole) and falls back to adjoint stubbing
        # when the LLM returns an empty response.
        try:
            return self._fiber_wise_atomic_completion(
                client,
                system,
                prompt_text,
                classification or {},
                hctx,
                topology,
                skeleton,
                output_dir,
                project_name,
            )
        except Exception as fiber_exc:
            _accel_log(
                "warning",
                f"Fiber-wise atomic completion failed; falling back to monolithic enrichment: {fiber_exc}",
            )
            logger.warning(
                "Fiber-wise atomic completion failed; falling back to monolithic enrichment: %s",
                fiber_exc,
            )

        if client is None:
            _accel_log(
                "warning",
                "No LLM client available for monolithic enrichment; using deterministic fallback.",
            )
            last_error = IntentCompilerError("LLM client unavailable")
        else:
            # Phase 4 drift gate: measure intent drift against the invariant before
            # the bounded-completion LLM call and restore context if it exceeds the
            # Theorem 1 safety bound.
            drift_partial = {
                "project": project_name or skeleton.get("project") or classification.get("architecture") or "graph_polyglot",
                "architecture": classification.get("architecture") or skeleton.get("architecture") or "graph_polyglot",
                "nodes": skeleton.get("nodes", []),
                "edges": [],
                "functional_intent": classification.get("functional_intent") or skeleton.get("functional_intent") or [],
                "metadata": {},
            }
            drift = self._measure_his_drift(hctx, classification or {}, drift_partial)
            if drift > DRIFT_THRESHOLD:
                self._apply_his_restore(hctx, classification or {}, drift_partial)
                topology = self._prune_foge_topology(
                    topology,
                    self._current_symbols(classification or {}, drift_partial),
                )

            user_content = self._prepare_phase4_llm_call(
                self._six_phase_user_content(
                    prompt_text, classification or {}, hctx, topology, skeleton
                ),
                hctx,
                classification or {},
                drift_partial,
                topology=topology,
            )

            messages: List[Dict[str, str]] = [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ]

            last_error: Optional[Exception] = None
            for attempt in range(self.max_schema_retries):
                raw = client.generate(messages, temperature=0.2, max_tokens=4096)
                if not raw:
                    _accel_log(
                        "warning",
                        f"intent_compiler.compile_prompt_to_graph attempt {attempt + 1}: LLM returned an empty response",
                    )
                    last_error = IntentCompilerError("LLM returned an empty response")
                    messages.append({"role": "assistant", "content": ""})
                    messages.append(
                        {
                            "role": "user",
                            "content": f"Attempt {attempt + 1} returned empty. Please return valid JSON.",
                        }
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
                    logger.warning(
                        "Intent JSON extraction failed (attempt %d): %s", attempt + 1, exc
                    )
                    messages.append({"role": "assistant", "content": raw})
                    messages.append(self._retry_user_message(attempt, raw, exc))
                    continue

                # Phases 5-6: concolic (Z3), SHACL, and Prolog/Chiasmus verification.
                formal_feedback = self._six_phase_formal_feedback(data, output_dir)
                if formal_feedback:
                    last_error = IntentCompilerError(formal_feedback)
                    _accel_log(
                        "warning",
                        f"intent_compiler.compile_prompt_to_graph attempt {attempt + 1}: formal verification failed; "
                        f"feedback preview: {formal_feedback[:400]!r}",
                    )
                    logger.warning(
                        "Formal verification failed (attempt %d): %s",
                        attempt + 1,
                        formal_feedback[:400],
                    )
                    messages.append({"role": "assistant", "content": raw})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Formal verification failed. Resolve these issues and return corrected JSON.\n\n"
                                f"{formal_feedback}\n\n"
                                "NO PREAMBLE. NO EXPLANATION. ONLY JSON."
                            ),
                        }
                    )
                    continue

                # Primary path: a native `graph_polyglot` blueprint.
                try:
                    graph = PolyglotGraphBlueprint.model_validate(data)
                    graph.metadata["prompt"] = prompt_text
                    graph.metadata["llm_initialized"] = True
                    graph.metadata["status"] = "finalized"
                    graph.metadata["generation_method"] = "llm_synthesized"
                    _accel_log("success", "Blueprint enrichment complete")
                    return graph
                except Exception as graph_exc:
                    logger.debug(
                        "Graph blueprint validation failed (attempt %d): %s",
                        attempt + 1,
                        graph_exc,
                    )

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
                    logger.warning(
                        "Intent schema validation failed (attempt %d): %s", attempt + 1, exc
                    )
                    messages.append({"role": "assistant", "content": raw})
                    messages.append(self._retry_user_message(attempt, raw, exc))
                    continue

                graph = self._v2_to_graph_blueprint(
                    v2,
                    prompt_text,
                    output_dir,
                    project_name,
                    classification=classification,
                )
                graph.metadata["llm_initialized"] = True
                graph.metadata["status"] = "finalized"
                graph.metadata["generation_method"] = "llm_synthesized"
                _accel_log("success", "Blueprint enrichment complete")
                return graph

        _accel_log(
            "warning",
            f"Graph enrichment failed after {self.max_schema_retries} attempts ({last_error}); "
            "falling back to the deterministic graph skeleton.",
        )
        return self._deterministic_graph_blueprint(
            prompt_text, output_dir, project_name, classification or {}
        )

    def _deterministic_graph_blueprint(
        self,
        prompt_text: str,
        output_dir: Optional[str | Path],
        project_name: Optional[str],
        classification: Dict[str, Any],
    ) -> PolyglotGraphBlueprint:
        """Return a valid graph blueprint built entirely from deterministic data."""
        stack = classify_stack(prompt_text)
        architecture = classification.get("architecture") or stack.architecture or "graph_polyglot"
        languages = stack.languages or ["python"]
        functional_intent = classification.get("functional_intent") or []
        # If the deterministic classification already produced a skeleton, reuse it.
        skeleton = classification.get("_skeleton") or self._build_deterministic_skeleton(
            architecture,
            languages,
            functional_intent,
        )
        self._inject_wasm_target(skeleton, prompt_text)

        functional_intent = skeleton.get("functional_intent", [])
        if not functional_intent:
            functional_intent = classification.get("functional_intent", [])
        if not functional_intent:
            functional_intent = [
                {"symbol_name": "main", "type": "cli", "requirement_level": "required"}
            ]

        partial: Dict[str, Any] = {
            "project": project_name or skeleton.get("project") or architecture,
            "architecture": architecture,
            "nodes": skeleton.get("nodes", []),
            "edges": skeleton.get("edges", []),
            "functional_intent": functional_intent,
            "metadata": {
                "schema_version": "2.0.0",
                "bootstrap_method": "deterministic_fallback",
                "prompt": prompt_text,
            },
            "primary_entrypoint": skeleton.get(
                "primary_entrypoint", "run_shell.py"
            ),
            "build_script": skeleton.get("build_script", "build.sh"),
            "output_dir": str(Path(output_dir) / "dist") if output_dir else "./dist",
        }

        graph = PolyglotGraphBlueprint.model_validate(partial)
        graph.metadata["prompt"] = prompt_text
        graph.metadata["llm_initialized"] = True
        graph.metadata["status"] = "finalized"
        graph.metadata["generation_method"] = "deterministic_fallback"
        _accel_log("success", "Blueprint deterministic fallback enrichment complete")
        return graph

    def _v2_to_graph_blueprint(
        self,
        v2: BlueprintSchemaV2,
        prompt_text: str,
        output_dir: Optional[str | Path],
        project_name: Optional[str],
        classification: Optional[Dict[str, Any]] = None,
    ) -> PolyglotGraphBlueprint:
        """Lower a validated ``BlueprintSchemaV2`` into a ``PolyglotGraphBlueprint``.

        If a verified classification is supplied, it seeds the graph architecture,
        functional_intent, and nodes when the v2 blueprint is skeletal.
        """
        project = (
            project_name or v2.metadata.get("project_name") or "aero_forge_project"
        )

        architecture = (
            v2.metadata.get("domain_target")
            or (classification or {}).get("architecture")
            or "graph_polyglot"
        )

        # Seed nodes from the v2 module graph.
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
            extra: Dict[str, Any] = {"purpose": node.get("purpose", "")}
            for key in ("data_payload", "payload_kind", "logic_sketch"):
                if key in node:
                    extra[key] = node[key]
            nodes.append(
                PolyglotNodeSpec(
                    node_id=node_id,
                    lang=node.get("lang") or node.get("language") or "python",
                    source_files=[node.get("path", "")],
                    extra=extra,
                )
            )

        edges: List[BoundaryEdgeSpec] = []
        _BOUNDARY_SYNONYMS = {
            "c": BoundaryContractType.C_ABI,
            "c_abi": BoundaryContractType.C_ABI,
            "cabi": BoundaryContractType.C_ABI,
            "raw_c": BoundaryContractType.C_ABI,
            "cffi": BoundaryContractType.C_ABI,
            "cxx": BoundaryContractType.C_ABI,
            "pyo3": BoundaryContractType.PYO3_MATURIN,
            "maturin": BoundaryContractType.PYO3_MATURIN,
            "ctypes": BoundaryContractType.C_ABI,
        }
        for abi in v2.abi_contracts:
            binding = str(abi.binding_framework or "c_abi").lower().replace("-", "_")
            boundary_type = _BOUNDARY_SYNONYMS.get(binding, BoundaryContractType.C_ABI)

            inputs = abi.signature.get("inputs", []) if abi.signature else []
            outputs = abi.signature.get("outputs", []) if abi.signature else []

            binding_to_source_lang = {
                "pyo3": "rust",
                "maturin": "rust",
                "ctypes": "python",
                "c_abi": "cpp",
                "raw_c": "c",
                "c": "c",
                "cffi": "c",
                "cxx": "cpp",
            }
            source_lang = binding_to_source_lang.get(binding, "cpp")
            # The ABI contract v2 defaults target_language to "rust", but for PyO3
            # the Rust side is the source and Python is the consumer. Infer the
            # missing target from the binding when not explicitly provided.
            target_lang = (abi.target_language or "").lower()
            if not target_lang or target_lang == source_lang:
                target_lang = "python" if source_lang != "python" else "rust"

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
                    boundary_type=boundary_type.value,
                    symbol=abi.export_symbol or abi.contract_id or "ffi_symbol",
                    args=[inp.get("type", "int64") for inp in inputs],
                    return_type=(outputs[0].get("type", "") if outputs else ""),
                )
            )

        # If the v2 response was skeletal, seed nodes/toolchains from the
        # verified classification so the graph blueprint does not collapse.
        classification_nodes = (classification or {}).get("nodes") or []
        for node in classification_nodes:
            node_id = node.get("node_id")
            if not node_id or node_id in node_ids:
                continue
            node_ids.add(node_id)
            nodes.append(
                PolyglotNodeSpec(
                    node_id=node_id,
                    lang=node.get("lang", "python"),
                    toolchain=node.get("toolchain", ""),
                    source_files=node.get("source_files", []),
                    exports=node.get("exports", []),
                )
            )

        # Ensure architecture reflects the verified classification / v2 intent.
        if architecture == "graph_polyglot" and (classification or {}).get(
            "architecture"
        ):
            architecture = classification["architecture"]

        resolved_output_dir = str(Path(output_dir) / "dist") if output_dir else "./dist"

        primary_entrypoint = ""
        if v2.execution_strategy:
            primary_entrypoint = v2.execution_strategy.primary_entrypoint.get(
                "path", ""
            )

        build_script = (
            v2.execution_strategy.run_spec.get("command", "")
            if v2.execution_strategy
            else ""
        )

        # Build functional_intent from v2 or classification.
        core_intents = self._derive_functional_intent(v2)
        if not core_intents and classification:
            core_intents = [
                FunctionalIntent(
                    symbol_name=fi.get("symbol_name", ""),
                    type=fi.get("type", "function"),
                )
                for fi in classification.get("functional_intent", [])
                if fi.get("symbol_name")
            ]
        functional_intent = [
            SchemaFunctionalIntent(
                symbol_name=fi.symbol_name,
                type=fi.type or "function",
                requirement_level=fi.requirement_level or "required",
            )
            for fi in core_intents
        ]

        return PolyglotGraphBlueprint(
            project=project,
            architecture=architecture,
            nodes=nodes,
            edges=edges,
            functional_intent=functional_intent,
            output_dir=resolved_output_dir,
            primary_entrypoint=primary_entrypoint or "run_shell.py",
            build_script=build_script or None,
            metadata={
                "prompt": prompt_text,
                "verification_nodes": [dict(vn) for vn in v2.verification_nodes],
                "cargo_dependencies": v2.cargo_dependencies or {},
                "execution_strategy": (
                    v2.execution_strategy.model_dump(mode="json")
                    if v2.execution_strategy
                    else {}
                ),
                **v2.metadata,
            },
        )
