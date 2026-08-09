"""Pydantic v2 models for ``blueprint.aero`` Schema v3.0.0.

A v3 blueprint is a complete, self-describing build contract: it pins toolchains,
describes the build pipeline as a DAG of artifacts, declares cross-language ABI
contracts, and records the execution strategy and verification nodes needed to
deterministically run the project.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import uuid
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from aero_forge.builder.language_router import _deduplicate_command_args

logger = logging.getLogger("aero_forge.blueprint.schema")


class BlueprintStatus(str, Enum):
    draft = "draft"
    finalized = "finalized"


class GenerationMethod(str, Enum):
    static_heuristic = "static_heuristic"
    llm_synthesized = "llm_synthesized"
    manual = "manual"


class ContextState(str, Enum):
    raw = "raw"
    synthesized = "synthesized"


class ArtifactType(str, Enum):
    shared_library = "shared_library"
    static_library = "static_library"
    binary = "binary"
    cargo_cdylib = "cargo_cdylib"
    python_extension = "python_extension"
    custom_cmd = "custom_cmd"


class MemoryModel(str, Enum):
    caller_allocates = "caller_allocates"
    callee_allocates = "callee_allocates"
    shared_pyo3 = "shared_pyo3"


class BindingFramework(str, Enum):
    ctypes = "ctypes"
    c_abi = "c_abi"
    raw_c = "raw_c"
    c = "c"
    cffi = "cffi"
    pyo3 = "pyo3"
    cxx = "cxx"


class Metadata(BaseModel):
    schema_version: str = "3.0.0"
    project_name: str = "aero_forge_project"
    status: BlueprintStatus = BlueprintStatus.draft
    generation_method: GenerationMethod = GenerationMethod.manual
    transferable: bool = False
    llm_initialized: bool = False
    auto_generated: bool = False
    description: str = ""


class ComputeHotspot(BaseModel):
    name: str
    file: str = ""
    complexity: str = ""
    acceleration_candidate: bool = True
    reason: str = ""


class PolyglotBoundary(BaseModel):
    python_file: str = ""
    native_file: str = ""
    binding: BindingFramework = BindingFramework.c_abi
    shared_struct: str = ""
    memory_model: MemoryModel = MemoryModel.caller_allocates


class LLMContext(BaseModel):
    state: ContextState = ContextState.raw
    repository_summary: str = ""
    dependency_graph: Dict[str, List[str]] = Field(default_factory=dict)
    exported_api_signatures: Dict[str, List[str]] = Field(default_factory=dict)
    polyglot_boundaries: List[PolyglotBoundary] = Field(default_factory=list)
    compute_hotspots: List[ComputeHotspot] = Field(default_factory=list)


class ToolchainSpec(BaseModel):
    name: str
    version: Optional[str] = None
    channel: Optional[str] = None
    env: Dict[str, str] = Field(default_factory=dict)


class BuildArtifact(BaseModel):
    id: str
    type: ArtifactType
    source_files: List[str] = Field(default_factory=list)
    output_path: str = ""
    compiler_flags: List[str] = Field(default_factory=list)
    linker_flags: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    commands: List[str] = Field(default_factory=list)
    description: str = ""


class ABIArgument(BaseModel):
    name: str
    type: str


class ABIContractV3(BaseModel):
    contract_id: str
    symbol: str = ""
    source_language: str = "python"
    target_language: str = "rust"
    binding_framework: BindingFramework = BindingFramework.c_abi
    header_path: Optional[str] = None
    memory_model: MemoryModel = MemoryModel.caller_allocates
    inputs: List[ABIArgument] = Field(default_factory=list)
    outputs: List[ABIArgument] = Field(default_factory=list)
    description: str = ""


class ExecutionStrategyV3(BaseModel):
    primary_entrypoint: str = ""
    runtime: str = "python3"
    args: List[str] = Field(default_factory=list)
    env: Dict[str, str] = Field(default_factory=dict)
    working_dir: str = "${WORKSPACE_ROOT}"
    timeout: float = 60.0
    engine_backend: str = ""
    wavefront_parallelism: int = 0
    precision_shield_mode: str = ""
    hin_jit_opt_level: int = 0


class VerificationMetric(BaseModel):
    name: str
    pattern: str = ""
    expected: Optional[float] = None
    tolerance: float = 1e-9


def _default_node_id() -> str:
    return f"node_{uuid.uuid4().hex[:8]}"


class VerificationNode(BaseModel):
    node_id: str = Field(default_factory=_default_node_id)
    command: str = ""
    expected_exit_code: int = 0
    stdout_match_patterns: List[str] = Field(default_factory=list)
    stderr_prohibited_patterns: List[str] = Field(default_factory=list)
    metrics: List[VerificationMetric] = Field(default_factory=list)
    timeout: float = 60.0


class BlueprintV3(BaseModel):
    metadata: Metadata = Field(default_factory=Metadata)
    llm_context: LLMContext = Field(default_factory=LLMContext)
    toolchains: List[ToolchainSpec] = Field(default_factory=list)
    build_pipeline: List[BuildArtifact] = Field(default_factory=list)
    abi_contracts: List[ABIContractV3] = Field(default_factory=list)
    execution_strategy: ExecutionStrategyV3 = Field(default_factory=ExecutionStrategyV3)
    verification_nodes: List[VerificationNode] = Field(default_factory=list)

    model_config = {"extra": "ignore"}

    @model_validator(mode="after")
    def _enforce_schema_version(self) -> "BlueprintV3":
        if self.metadata.schema_version != "3.0.0":
            raise ValueError(
                f"BlueprintV3 requires schema_version '3.0.0', got {self.metadata.schema_version!r}"
            )
        return self

    def _resolve(self, value: Any, workspace: Path) -> Any:
        """Replace ${WORKSPACE_ROOT} placeholders with *workspace*."""
        if isinstance(value, str):
            return value.replace("${WORKSPACE_ROOT}", str(workspace))
        if isinstance(value, list):
            return [self._resolve(v, workspace) for v in value]
        if isinstance(value, dict):
            return {k: self._resolve(v, workspace) for k, v in value.items()}
        return value

    @staticmethod
    def _looks_like_python_artifact(artifact: BuildArtifact) -> bool:
        return any(Path(f).suffix == ".py" for f in artifact.source_files)

    def to_runner_blueprint(self, workspace: Optional[Path] = None) -> "Blueprint":
        """Convert this v3 blueprint into a v2 ``Blueprint`` model for the legacy runner.

        The conversion is best-effort: Python artifacts become ``FunctionSpec``
        entries whose names are discovered from the source files, and legacy
        fields such as ``architecture`` and ``toolchains`` are inferred from the
        artifact types.
        """
        # Delayed import breaks the import cycle between schema.py and core.py.
        from aero_forge.blueprint.core import Blueprint, FunctionSpec, LLMConfig, ManifestEntry

        workspace = Path(workspace or ".").resolve()

        functions: List[FunctionSpec] = []
        manifest: List[ManifestEntry] = []
        architecture = "pure_python"

        for artifact in self.build_pipeline:
            if not artifact.source_files and not artifact.commands:
                continue

            if artifact.type == ArtifactType.cargo_cdylib:
                architecture = "hybrid_rust_python"
                crate_root = Path(artifact.id)
                manifest.append(
                    ManifestEntry(
                        path=str(crate_root / "Cargo.toml"), lang="toml", purpose="cargo cdylib"
                    )
                )
                for src in artifact.source_files:
                    rel = str(crate_root / src)
                    if not (workspace / rel).is_file():
                        continue
                    manifest.append(ManifestEntry(path=rel, lang="rust", purpose="rust source"))
            elif any(Path(f).suffix in {".cpp", ".c", ".h", ".hpp"} for f in artifact.source_files):
                if "rust" in architecture or architecture == "hybrid_rust_python":
                    architecture = "hybrid_cpp_rust"
                elif architecture == "pure_python":
                    architecture = "hybrid_cpp_python"
                for src in artifact.source_files:
                    if not (workspace / src).is_file():
                        continue
                    manifest.append(
                        ManifestEntry(
                            path=src, lang=Path(src).suffix.lstrip("."), purpose="c/c++ source"
                        )
                    )
            else:
                # Treat as Python / generic binary artifact.
                for src in artifact.source_files:
                    if not (workspace / src).is_file():
                        continue
                    manifest.append(ManifestEntry(path=src, lang="python", purpose="python source"))
                    if self._looks_like_python_artifact(artifact):
                        from aero_forge.blueprint.core import discover_functions as _discover

                        src_path = workspace / src
                        try:
                            for func in _discover(src_path):
                                if not func.name:
                                    continue
                                functions.append(
                                    FunctionSpec(
                                        file=src_path,
                                        name=func.name,
                                        output_name=func.name,
                                        compiler_flags=artifact.compiler_flags,
                                        tests=func.tests,
                                    )
                                )
                        except Exception as exc:
                            logger.warning("Could not discover functions in %s: %s", src_path, exc)

        output_dir = (
            workspace / ".aero" / "sandbox" / "dist"
            if self.metadata.status == BlueprintStatus.draft
            else workspace / "dist"
        )

        metadata_v2: Dict[str, str] = {
            "schema_version": self.metadata.schema_version,
            "project_name": self.metadata.project_name,
            "domain_target": architecture,
            "status": self.metadata.status,
            "generation_method": self.metadata.generation_method,
            "transferable": str(self.metadata.transferable),
            "workspace_root": str(workspace),
        }

        return Blueprint(
            project=self.metadata.project_name,
            architecture=architecture,
            toolchains=[t.name for t in self.toolchains],
            manifest=manifest,
            functions=functions,
            metadata=metadata_v2,
            output_dir=output_dir,
            llm=LLMConfig(provider="none"),
        )

    @classmethod
    def load(cls, path: Path) -> "BlueprintV3":
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            data = json.loads(text)
        else:
            data = yaml.safe_load(text) or {}
        return cls.model_validate(data)

    def execute(
        self,
        workspace: Path,
        env: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Execute the v3 blueprint deterministically inside *workspace*.

        The build pipeline is processed in dependency order. Each artifact is
        compiled/built according to its type, and then the primary entrypoint is
        executed and verified against ``verification_nodes``.
        """
        workspace = Path(workspace).resolve()
        run_env = dict(os.environ)
        if env:
            run_env.update(env)
        run_env.update(
            {k: self._resolve(v, workspace) for k, v in self.execution_strategy.env.items()}
        )

        # Build DAG order.
        order = self._artifact_order()
        built: Dict[str, Path] = {}
        build_results: List[Dict[str, Any]] = []

        for art_id in order:
            artifact = self._artifact_by_id[art_id]
            result = self._build_artifact(artifact, workspace, built, run_env)
            build_results.append(result)
            if not result["success"]:
                return {
                    "status": "failed",
                    "stage": f"build:{art_id}",
                    "error": result.get("error"),
                    "build_results": build_results,
                }
            if artifact.output_path:
                built[art_id] = workspace / self._resolve(artifact.output_path, workspace)

        # Run primary entrypoint.
        entrypoint = self._resolve(self.execution_strategy.primary_entrypoint, workspace)
        runtime = self.execution_strategy.runtime
        args = self._resolve(self.execution_strategy.args, workspace)
        working_dir = self._resolve(self.execution_strategy.working_dir, workspace)

        cmd: List[str] = [runtime]
        if entrypoint:
            if (
                runtime.startswith(("python", "python3"))
                and entrypoint.endswith(".py")
                and "/" in entrypoint
            ):
                # Execute sub-package entrypoints as modules so relative imports
                # inside e.g. ``python_cli/main.py`` work without ImportError.
                module_path = entrypoint[:-3].replace("/", ".").replace("\\", ".").lstrip(".")
                cmd.extend(["-m", module_path])
            else:
                cmd.append(entrypoint)
        cmd.extend(args)

        cwd = Path(working_dir)
        if not cwd.is_absolute():
            cwd = workspace / cwd
        cwd = cwd.resolve()

        logger.info("Executing v3 blueprint: %s in %s", " ".join(cmd), cwd)
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd),
                env=run_env,
                capture_output=True,
                text=True,
                timeout=self.execution_strategy.timeout,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            return {
                "status": "failed",
                "stage": "execution",
                "error": str(exc),
                "build_results": build_results,
            }

        stdout = proc.stdout
        stderr = proc.stderr
        exit_code = proc.returncode

        # Verification nodes.
        verification: List[Dict[str, Any]] = []
        for node in self.verification_nodes:
            vcmd = self._resolve(node.command, workspace)
            vparts = vcmd.split() if vcmd else []
            venv = dict(run_env)
            venv.update(
                {k: self._resolve(v, workspace) for k, v in node.env.items()}
                if hasattr(node, "env")
                else {}
            )
            try:
                if vparts:
                    vproc = subprocess.run(
                        vparts,
                        cwd=str(cwd),
                        env=venv,
                        capture_output=True,
                        text=True,
                        timeout=node.timeout,
                    )
                else:
                    vproc = None
            except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
                verification.append({"node_id": node.node_id, "passed": False, "error": str(exc)})
                continue

            if vproc is None:
                # No explicit command: verify primary run output.
                node_exit = exit_code
                node_stdout = stdout
                node_stderr = stderr
            else:
                node_exit = vproc.returncode
                node_stdout = vproc.stdout
                node_stderr = vproc.stderr

            passed = node_exit == node.expected_exit_code
            if passed and node.stdout_match_patterns:
                passed = all(
                    re.search(p, node_stdout) is not None for p in node.stdout_match_patterns
                )
            if passed and node.stderr_prohibited_patterns:
                passed = all(
                    re.search(p, node_stderr) is None for p in node.stderr_prohibited_patterns
                )
            verification.append(
                {
                    "node_id": node.node_id,
                    "passed": passed,
                    "exit_code": node_exit,
                    "stdout": node_stdout,
                    "stderr": node_stderr,
                }
            )

        all_passed = exit_code == 0 and all(v["passed"] for v in verification)
        return {
            "status": "success" if all_passed else "failed",
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr,
            "build_results": build_results,
            "verification": verification,
        }

    @property
    def _artifact_by_id(self) -> Dict[str, BuildArtifact]:
        return {art.id: art for art in self.build_pipeline}

    def _artifact_order(self) -> List[str]:
        graph: Dict[str, List[str]] = defaultdict(list)
        in_degree: Dict[str, int] = {art.id: 0 for art in self.build_pipeline}
        for art in self.build_pipeline:
            for dep in art.dependencies:
                graph[dep].append(art.id)
                in_degree[art.id] += 1
        queue = [n for n, d in in_degree.items() if d == 0]
        order: List[str] = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for nxt in graph[node]:
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    queue.append(nxt)
        if len(order) != len(self.build_pipeline):
            raise ValueError("build_pipeline contains a dependency cycle")
        return order

    def _build_artifact(
        self,
        artifact: BuildArtifact,
        workspace: Path,
        built: Dict[str, Path],
        env: Dict[str, str],
    ) -> Dict[str, Any]:
        srcs = [workspace / self._resolve(s, workspace) for s in artifact.source_files]
        missing = [str(s) for s in srcs if not s.is_file()]
        if missing:
            return {
                "artifact": artifact.id,
                "success": False,
                "error": f"Missing source files: {missing}",
            }

        output_path = self._resolve(artifact.output_path, workspace)
        resolved_output: Optional[Path] = None
        if output_path:
            resolved_output = workspace / output_path

        if artifact.type == ArtifactType.custom_cmd:
            for cmd in artifact.commands:
                cmd = self._resolve(cmd, workspace)
                cmd_parts = cmd.split()
                if not shutil.which(cmd_parts[0]):
                    return {
                        "artifact": artifact.id,
                        "success": False,
                        "error": f"Command not found: {cmd_parts[0]}",
                    }
                try:
                    subprocess.run(
                        _deduplicate_command_args(cmd_parts),
                        cwd=str(workspace),
                        env=env,
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                except (subprocess.CalledProcessError, FileNotFoundError) as exc:
                    return {"artifact": artifact.id, "success": False, "error": str(exc)}
            return {"artifact": artifact.id, "success": True}

        if artifact.type in (
            ArtifactType.binary,
            ArtifactType.python_extension,
        ) and self._looks_like_python_artifact(artifact):
            # Python artifacts do not require a compilation step for execution.
            if resolved_output:
                resolved_output.parent.mkdir(parents=True, exist_ok=True)
            return {"artifact": artifact.id, "success": True}

        if artifact.type == ArtifactType.cargo_cdylib:
            if not shutil.which("cargo"):
                return {
                    "artifact": artifact.id,
                    "success": False,
                    "error": "cargo is not installed",
                }
            try:
                subprocess.run(
                    _deduplicate_command_args(
                        ["cargo", "build", "--release"] + artifact.compiler_flags
                    ),
                    cwd=str(workspace),
                    env=env,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except (subprocess.CalledProcessError, FileNotFoundError) as exc:
                return {"artifact": artifact.id, "success": False, "error": str(exc)}
            return {"artifact": artifact.id, "success": True}

        if artifact.type in (ArtifactType.shared_library, ArtifactType.static_library):
            # Simple C/C++ build: use the first detected C/C++ compiler.
            compiler = shutil.which("gcc") or shutil.which("clang") or shutil.which("cl")
            if not compiler:
                return {
                    "artifact": artifact.id,
                    "success": False,
                    "error": "No C/C++ compiler found",
                }
            obj_files: List[Path] = []
            build_dir = workspace / "build"
            build_dir.mkdir(parents=True, exist_ok=True)
            for src in srcs:
                try:
                    rel = src.relative_to(workspace)
                except ValueError:
                    rel = src.name
                obj = build_dir / rel.with_suffix(".o")
                obj.parent.mkdir(parents=True, exist_ok=True)
                try:
                    subprocess.run(
                        _deduplicate_command_args(
                            [compiler, "-c", str(src), "-o", str(obj)]
                            + artifact.compiler_flags
                        ),
                        cwd=str(workspace),
                        env=env,
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                except (subprocess.CalledProcessError, FileNotFoundError) as exc:
                    return {"artifact": artifact.id, "success": False, "error": str(exc)}
                obj_files.append(obj)
            if resolved_output and obj_files:
                if artifact.type == ArtifactType.shared_library:
                    link_args = (
                        ["-shared", "-o", str(resolved_output)]
                        + [str(o) for o in obj_files]
                        + artifact.linker_flags
                    )
                else:
                    link_args = ["rcs", str(resolved_output)] + [str(o) for o in obj_files]
                try:
                    subprocess.run(
                        (
                            link_args
                            if artifact.type == ArtifactType.shared_library
                            else ["ar"] + link_args
                        ),
                        cwd=str(workspace),
                        env=env,
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                except (subprocess.CalledProcessError, FileNotFoundError) as exc:
                    return {"artifact": artifact.id, "success": False, "error": str(exc)}
            return {"artifact": artifact.id, "success": True}

        # Unhandled types are treated as no-ops; the entrypoint execution still runs.
        return {"artifact": artifact.id, "success": True}


def write_v3_blueprint(blueprint: BlueprintV3, path: Path) -> None:
    """Serialize a v3 blueprint to YAML (or JSON if the path ends in .json)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = blueprint.model_dump(mode="json")
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    else:
        path.write_text(
            yaml.safe_dump(data, sort_keys=False, default_flow_style=False), encoding="utf-8"
        )


class BoundaryContractType(str, Enum):
    """Cross-language boundary contracts used by the graph polyglot blueprint."""

    C_ABI = "c_abi"
    PYO3_MATURIN = "pyo3_maturin"
    WASM_WASI = "wasm_wasi"
    JNI = "jni"
    CGO = "cgo"
    PINVOKE = "pinvoke"
    CUDA_HIP_C = "cuda_hip_c"


class PolyglotNodeSpec(BaseModel):
    """One language node in a graph-driven polyglot blueprint."""

    node_id: str
    lang: str
    toolchain: str = ""
    source_files: List[str] = Field(default_factory=list)
    compiler_flags: List[str] = Field(default_factory=list)
    exports: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    extra: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _normalize_lang(self) -> "PolyglotNodeSpec":
        self.lang = self.lang.lower().strip()
        if not self.toolchain:
            self.toolchain = self.lang
        return self


class BoundaryEdgeSpec(BaseModel):
    """A directed FFI edge between two polyglot nodes."""

    source: str
    target: str
    boundary_type: BoundaryContractType = BoundaryContractType.C_ABI
    symbol: str
    args: List[str] = Field(default_factory=list)
    return_type: str = ""
    is_zero_copy: bool = False

    @field_validator("boundary_type", mode="before")
    @classmethod
    def _normalize_boundary_type(cls, value: Any) -> str:
        if value is None:
            return "c_abi"
        text = str(value).lower().strip().replace("-", "_")
        synonyms = {
            "c": "c_abi",
            "c_abi": "c_abi",
            "capi": "c_abi",
            "pyo3": "pyo3_maturin",
            "maturin": "pyo3_maturin",
            "wasm": "wasm_wasi",
            "wasi": "wasm_wasi",
            "cgo": "cgo",
            "go": "cgo",
            "pinvoke": "pinvoke",
            "p/invoke": "pinvoke",
            "csharp": "pinvoke",
            "dotnet": "pinvoke",
            "jni": "jni",
            "java": "jni",
            "cuda": "cuda_hip_c",
            "hip": "cuda_hip_c",
            "cuda_hip": "cuda_hip_c",
        }
        return synonyms.get(text, text)


class PolyglotGraphBlueprint(BaseModel):
    """Graph-driven, HIN-style blueprint for the unified polyglot materializer."""

    project: str = "aero_forge_project"
    architecture: str = "graph_polyglot"
    nodes: List[PolyglotNodeSpec] = Field(default_factory=list)
    edges: List[BoundaryEdgeSpec] = Field(default_factory=list)
    output_dir: str = "./dist"
    primary_entrypoint: str = "run_shell.py"
    build_script: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _enforce_dag(self) -> "PolyglotGraphBlueprint":
        """Validate node references and detect cycles."""
        node_ids = {n.node_id for n in self.nodes}
        for edge in self.edges:
            if edge.source not in node_ids:
                raise ValueError(f"edge references unknown source node: {edge.source}")
            if edge.target not in node_ids:
                raise ValueError(f"edge references unknown target node: {edge.target}")

        adj: Dict[str, List[str]] = {n.node_id: [] for n in self.nodes}
        for edge in self.edges:
            adj[edge.source].append(edge.target)

        state: Dict[str, int] = {nid: 0 for nid in node_ids}

        def visit(nid: str) -> None:
            if state[nid] == 1:
                raise ValueError("cycle detected in graph blueprint")
            if state[nid] == 0:
                state[nid] = 1
                for v in adj[nid]:
                    visit(v)
                state[nid] = 2

        for nid in node_ids:
            if state[nid] == 0:
                visit(nid)

        return self
