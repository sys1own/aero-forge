"""High-level polyglot builder for aero-forge engine specs."""

from __future__ import annotations

import json
import logging
import subprocess
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("aero_forge.builder")

from aero_forge.builder.artifact_generator import ArtifactBundle, ArtifactGenerator
from aero_forge.builder.emitters import get_emitter
from aero_forge.builder.language_router import resolve_target_language
from aero_forge.builder.spec import EngineSpec


@dataclass
class BuildOutput:
    """Result of building an engine spec for a target language."""

    language: str
    source: str
    spec: EngineSpec
    artifacts: ArtifactBundle = field(default_factory=ArtifactBundle)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "language": self.language,
            "source": self.source,
            "artifacts": self.artifacts.to_dict(),
            "metadata": dict(self.metadata),
        }


def build_engine(
    spec: EngineSpec,
    target_language: Optional[str] = None,
    *,
    context: Optional[Dict[str, Any]] = None,
    template_names: Optional[List[str]] = None,
    template_dirs: Optional[List[Path]] = None,
    output_paths: Optional[Dict[str, str]] = None,
) -> BuildOutput:
    """Render *spec* to *target_language* source and optional artifacts.

    If *target_language* is not provided, it is resolved from *context* or the
    spec's ``metadata.language`` hint.
    """
    context = context or {}
    language = target_language or resolve_target_language(
        context,
        source_language=spec.metadata.get("language"),
    )
    emitter = get_emitter(language)
    source = emitter.emit(spec)

    artifacts = ArtifactBundle()
    if template_names:
        generator = ArtifactGenerator(template_dirs=template_dirs)
        artifacts = generator.generate(
            spec,
            template_names,
            output_paths=output_paths,
        )

    # Collect any additional files produced by the emitter (e.g. Rust submodules).
    if hasattr(emitter, "emit_artifacts"):
        for artifact in getattr(emitter, "emit_artifacts")().artifacts:
            artifacts.artifacts.append(artifact)

    return BuildOutput(
        language=language,
        source=source,
        spec=spec,
        artifacts=artifacts,
        metadata={"language": language, **spec.metadata},
    )


class ProactivePolyglotBuilder:
    """Build orchestrator that verifies HIN, SMT, and GoI before disk writes."""

    def __init__(self) -> None:
        from aero_forge.builder.fallback_manager import FallbackManager, HeuristicWarning
        from aero_forge.hin_engine import HINEngine
        from aero_forge.precision_shield import SMTASTEngine

        self.hin_engine = HINEngine()
        self.smt_engine = SMTASTEngine()
        self.fallback_manager = FallbackManager()
        self.HeuristicWarning = HeuristicWarning

    def _ingest_hin(self, payload: Dict[str, Any]) -> None:
        """Populate the HIN engine from the payload's nodes and relations."""
        for node in payload.get("nodes", []):
            self.hin_engine.add_ast_node(self._new_hin_node(node))
        for rel in payload.get("relations", []):
            self.hin_engine.add_relation(rel["source"], rel["target"], rel["relation"])

    def _new_hin_node(self, node: Dict[str, Any]) -> "HINNode":
        from aero_forge.hin_engine import HINNode

        return HINNode(
            node_id=node["id"],
            node_type=node["type"],
            language=node["lang"],
            properties=node.get("props") or node.get("properties") or {},
        )

    def _solve_smt_with_healing(self, payload: Dict[str, Any]) -> Dict[str, str]:
        """Solve SMT constraints; attempt tiered fallback remediation on UNSAT."""
        holes = payload.get("holes", [])
        constraints = payload.get("constraints", [])

        try:
            return self.smt_engine.solve_ast_sketch_holes(holes, constraints)
        except ValueError as exc:
            # Level 1/3 fallback: try to remediate the payload in memory and re-check.
            trace = str(exc)
            success, remediated = self.fallback_manager.remediate_smt_unsat(
                payload, trace
            )
            if not success:
                raise exc
            # Update our working payload and re-run SMT with the remediated constraints.
            payload.update(remediated)
            constraints = payload.get("constraints", [])
            try:
                return self.smt_engine.solve_ast_sketch_holes(holes, constraints)
            except ValueError:
                raise exc

    def _hin_graph_json(self) -> str:
        """Serialize the current HIN graph as a simple node-link JSON."""
        import json as _json
        import networkx as nx

        nodes = []
        for nid, data in self.hin_engine.graph.nodes(data=True):
            nodes.append(
                {
                    "id": nid,
                    "type": data.get("node_type", ""),
                    "lang": data.get("language", ""),
                    "props": data.get("properties", {}),
                }
            )
        links = [
            {"source": u, "target": v, "relation": data.get("relation", "")}
            for u, v, data in self.hin_engine.graph.edges(data=True)
        ]
        return _json.dumps({"nodes": nodes, "links": links})

    def _verify_goi(self, payload: Dict[str, Any]) -> bool:
        """Check the GoI proof net for nilpotency/deadlock-freedom."""
        from aero_forge.native_bridge import verify_goi_proof_net

        dimension = payload.get("goi_dim", 2)
        m_data = payload.get("goi_m", [0.0] * (dimension * dimension))
        sigma_data = payload.get("goi_sigma", [0.0] * (dimension * dimension))
        if verify_goi_proof_net(dimension, m_data, sigma_data):
            return True

        # Level 2 fallback: try to prune cyclic edges in the proof net and re-check.
        success, remediated = self.fallback_manager.remediate_goi_non_nilpotent(
            payload
        )
        if not success:
            return False
        payload.update(remediated)
        sigma_data = payload.get("goi_sigma", [0.0] * (dimension * dimension))
        return bool(verify_goi_proof_net(dimension, m_data, sigma_data))

    @staticmethod
    def _llm_initialized_truthy(value: Any) -> bool:
        """Parse ``llm_initialized`` booleans encoded as bool/string."""
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        return str(value).strip().lower() in ("true", "1", "yes")

    @staticmethod
    def _requires_synthesis(payload: Dict[str, Any]) -> bool:
        """Return True when the payload's blueprint is a draft that needs LLM synthesis."""
        blueprint = payload.get("blueprint") or {}
        if not isinstance(blueprint, dict):
            return False
        metadata = blueprint.get("metadata") or {}
        status = str(metadata.get("status", "")).lower()
        auto_generated = bool(metadata.get("auto_generated"))
        llm_initialized = ProactivePolyglotBuilder._llm_initialized_truthy(
            metadata.get("llm_initialized")
        )
        llm_context = blueprint.get("llm_context") or {}
        if isinstance(llm_context, dict):
            state = str(llm_context.get("state", "")).lower()
        else:
            state = ""
        return status == "draft" or state == "raw" or (auto_generated and not llm_initialized)

    @staticmethod
    def _v11_universal_guidance() -> str:
        """Return the v11_universal_architect planning guidance as a string."""
        from aero_forge.prompts import get_template

        template = get_template("v11_universal_architect")
        return (
            "Use the following universal polyglot architect guidance when "
            "designing the blueprint:\n" + template.system_prompt
        )

    @staticmethod
    def _graph_prompt_skeleton(prompt: str) -> str:
        """Return a response skeleton so the LLM starts the blueprint immediately."""
        lower = (prompt or "").lower()
        has_rust = "rust" in lower or "pyo3" in lower or "cargo" in lower
        has_cpp = "c++" in lower or "cpp" in lower or "cmake" in lower or "pybind11" in lower
        has_python = "python" in lower or "pure_python" in lower

        if has_rust and has_python and has_cpp:
            architecture = "tri_polyglot_rust_cpp_python"
            nodes = [
                {"node_id": "rust_core", "lang": "rust", "toolchain": "cargo", "source_files": ["rust_core/src/lib.rs"], "exports": ["butterfly"]},
                {"node_id": "cpp_engine", "lang": "cpp", "toolchain": "cmake", "source_files": ["cpp_engine/src/kernels.cpp"], "exports": ["execute_task"]},
                {"node_id": "python_interface", "lang": "python", "toolchain": "python", "source_files": ["python_interface/main.py"], "exports": ["main"]},
            ]
            edges = [
                {"source": "rust_core", "target": "python_interface", "boundary_type": "PYO3_MATURIN", "symbol": "butterfly", "args": ["pointer", "int64"], "return_type": "int64"},
                {"source": "cpp_engine", "target": "python_interface", "boundary_type": "C_ABI", "symbol": "execute_task", "args": ["pointer"], "return_type": ""},
            ]
        elif has_rust and has_python:
            architecture = "hybrid_rust_python"
            nodes = [
                {"node_id": "rust_core", "lang": "rust", "toolchain": "cargo", "source_files": ["rust_core/src/lib.rs"], "exports": ["butterfly"]},
                {"node_id": "python_interface", "lang": "python", "toolchain": "python", "source_files": ["python_interface/main.py"], "exports": ["main"]},
            ]
            edges = [
                {"source": "rust_core", "target": "python_interface", "boundary_type": "PYO3_MATURIN", "symbol": "butterfly", "args": ["pointer", "int64"], "return_type": "int64"},
            ]
        elif has_cpp and has_python:
            architecture = "hybrid_cpp_python"
            nodes = [
                {"node_id": "cpp_engine", "lang": "cpp", "toolchain": "cmake", "source_files": ["cpp_engine/src/kernels.cpp"], "exports": ["execute_task"]},
                {"node_id": "python_interface", "lang": "python", "toolchain": "python", "source_files": ["python_interface/main.py"], "exports": ["main"]},
            ]
            edges = [
                {"source": "cpp_engine", "target": "python_interface", "boundary_type": "C_ABI", "symbol": "execute_task", "args": ["pointer"], "return_type": ""},
            ]
        elif has_rust and has_cpp:
            architecture = "hybrid_cpp_rust"
            nodes = [
                {"node_id": "cpp_engine", "lang": "cpp", "toolchain": "cmake", "source_files": ["cpp_engine/src/kernels.cpp"], "exports": ["execute_task"]},
                {"node_id": "rust_core", "lang": "rust", "toolchain": "cargo", "source_files": ["rust_core/src/lib.rs"], "exports": ["run_pipeline"]},
            ]
            edges = [
                {"source": "cpp_engine", "target": "rust_core", "boundary_type": "C_ABI", "symbol": "execute_task", "args": ["pointer"], "return_type": ""},
            ]
        elif has_rust:
            architecture = "pure_rust"
            nodes = [
                {"node_id": "rust_core", "lang": "rust", "toolchain": "cargo", "source_files": ["src/main.rs"], "exports": ["main"]},
            ]
            edges = []
        else:
            architecture = "pure_python"
            nodes = [
                {
                    "node_id": "main",
                    "lang": "python",
                    "toolchain": "python",
                    "source_files": ["main.py"],
                    "exports": ["main"],
                }
            ]
            edges = []

        skeleton = {
            "project": "generated_project",
            "architecture": architecture,
            "primary_entrypoint": "python_interface/main.py" if has_python and (has_rust or has_cpp) else (nodes[0]["source_files"][0] if nodes else "main.py"),
            "build_script": "build.sh",
            "nodes": nodes,
            "edges": edges,
            "metadata": {},
        }
        return (
            "\n\nStart your response from this JSON skeleton and complete only the missing values:\n"
            f"{json.dumps(skeleton, indent=2)}\n"
            "\nDo not return prose outside the final JSON."
        )

    def synthesize_and_build(
        self,
        prompt: str,
        output_dir: Path,
        *,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        prompt_template: Optional[str] = None,
        max_retries: int = 3,
        config_override: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Compile a natural-language prompt to a graph blueprint and build it.

        This is the proactive path for project-level prompts that the single-function
        code generator cannot satisfy (e.g. multi-language C-ABI bridges).  It uses the
        graph materializer, which JIT-synthesizes any missing emitter plugins and
        validates boundary contracts before writing source files.
        """
        from aero_forge.builder.intent_compiler import IntentCompiler
        from aero_forge.builder.materializers.graph_materializer import (
            GraphPolyglotMaterializer,
        )
        from aero_forge.config import resolve_llm_provider, _resolve_api_key

        provider = resolve_llm_provider(llm_provider) or "deepseek"
        model = llm_model or "deepseek-chat"

        # Honor an explicitly passed key first, then a request-scoped override,
        # then resolve from the environment so that web/dashboard calls inherit
        # the configured credentials without requiring global env vars.
        resolved_key = llm_api_key
        if not resolved_key and config_override is not None:
            resolved_key = getattr(config_override, "api_key", None)
        if not resolved_key:
            resolved_key = _resolve_api_key(provider)

        compiler = IntentCompiler(
            provider=provider,
            model=model,
            api_key=resolved_key,
            max_retries=max_retries,
            config_override=config_override,
        )

        last_error: Optional[Exception] = None
        graph = None
        # The system prompt already provides a full example; do not inject a
        # hard-coded skeleton that can leak placeholder symbols such as "butterfly".
        skeleton_note = ""
        for attempt, extra in enumerate(
            [
                "",
                (
                    self._v11_universal_guidance()
                    if prompt_template == "v11_universal_architect"
                    else ""
                ),
            ]
        ):
            if attempt > 0:
                logger.warning(
                    "Retrying graph intent synthesis with v11 universal architect guidance"
                )
            try:
                if extra:
                    text = f"{prompt}\n\n{extra}\n{skeleton_note}"
                else:
                    text = f"{prompt}\n\n{skeleton_note}"
                graph = compiler.compile_prompt_to_graph(
                    text, output_dir=output_dir
                )
                break
            except Exception as exc:
                last_error = exc
                if attempt == 0:
                    continue

        if graph is None:
            raise RuntimeError(
                f"ProactivePolyglotBuilder could not compile prompt to graph: {last_error}"
            ) from last_error

        if not ProactivePolyglotBuilder._llm_initialized_truthy(
            graph.metadata.get("llm_initialized")
        ):
            raise RuntimeError(
                "Blueprint Not Enriched: compile_prompt_to_graph did not produce an "
                "llm_initialized blueprint. Run a successful Intent Enrichment pass first."
            )

        # Persist the enriched graph blueprint before any source files are created
        # so the workspace always reflects a finalized state during materialization.
        (output_dir / "blueprint.aero").write_text(
            yaml.safe_dump(
                graph.model_dump(mode="json"), sort_keys=False, default_flow_style=False
            ),
            encoding="utf-8",
        )

        # Build a compacted functional matrix so downstream synthesis is concise
        # and deterministic. The context is attached to the graph metadata so the
        # materializer can inject it into emitter plugin synthesis.
        from aero_forge.builder.smt_engine import SkeletonTypeInjector
        from aero_forge.orchestrator.orchestrator import CompactedContextGenerator

        compacted = CompactedContextGenerator(graph).generate()
        graph.metadata.setdefault("synthesis_context", "")
        graph.metadata["synthesis_context"] = compacted

        # Seed any SMT-inferred native types from the original prompt skeleton if
        # the user provided a Python implementation stub.
        if prompt:
            type_env = SkeletonTypeInjector.infer_type_env(prompt)
            if type_env:
                graph.metadata.setdefault("smt_types", {})
                graph.metadata["smt_types"].update(type_env)

        materializer = GraphPolyglotMaterializer(
            output_dir,
            llm_provider=provider,
            llm_model=model,
            llm_api_key=resolved_key,
            config_override=config_override,
        )
        result = materializer.materialize(
            graph.model_dump(mode="json"), build=True
        )

        build_script = output_dir / (graph.build_script or "build.sh")
        build_success = False
        build_output = ""
        if build_script.is_file():
            proc = subprocess.run(
                ["bash", str(build_script)],
                cwd=str(output_dir),
                capture_output=True,
                text=True,
            )
            build_success = proc.returncode == 0
            build_output = proc.stdout + proc.stderr

        return {
            "success": build_success,
            "source_path": str(output_dir / graph.primary_entrypoint),
            "test_path": "",
            "blueprint_path": str(result["blueprint_path"]),
            "implementation": "",
            "tests": "",
            "explanation": "",
            "build": {
                "success": build_success,
                "passed": 1 if build_success else 0,
                "total": 1,
                "output": build_output,
                "error": "" if build_success else build_output,
            },
        }

    def build_blueprint_proactive(self, payload: Dict[str, Any]) -> bool:
        """Run the full proactive verification pipeline and emit files if valid.

        Steps:
            1. Construct the HIN graph.
            2. Apply DPO FFI string rewrites.
            3. Solve SMT constraints (with one in-memory healing pass on UNSAT).
            4. Verify GoI nilpotency.
            5. Materialize files to disk.

        Returns ``True`` when all verification phases pass and files were
        written, ``False`` otherwise.
        """
        if self._requires_synthesis(payload):
            logger.warning(
                "ProactivePolyglotBuilder: blueprint is a draft/auto-generated sketch; "
                "run LLM synthesis before materialization."
            )
            return False

        self._ingest_hin(payload)
        self.hin_engine.apply_dpo_rewrite_ffi_strings()

        try:
            self._solve_smt_with_healing(payload)
        except ValueError:
            return False

        if not self._verify_goi(payload):
            return False

        # 5. Materialization gate: only pass here when every check succeeded.
        architecture = payload.get("blueprint", {}).get("architecture", "")
        if architecture == "graph_polyglot" or "hin_graph_spec" in payload:
            from aero_forge.builder.materializers.graph_materializer import (
                GraphPolyglotMaterializer,
            )

            workspace = Path(payload.get("workspace", "."))
            graph_spec = payload.get("hin_graph_spec", payload.get("blueprint", {}))
            GraphPolyglotMaterializer(workspace).materialize(graph_spec)
            return True

        from aero_forge.scaffold.polyglot_materializer import _emit_verified_files

        _emit_verified_files(payload)
        return True
