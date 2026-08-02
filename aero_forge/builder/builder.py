"""High-level polyglot builder for aero-forge engine specs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

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
        from aero_forge.builder.fallback_manager import FallbackManager
        from aero_forge.hin_engine import HINEngine
        from aero_forge.precision_shield import SMTASTEngine

        self.hin_engine = HINEngine()
        self.smt_engine = SMTASTEngine()
        self.fallback_manager = FallbackManager()

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
        self._ingest_hin(payload)
        self.hin_engine.apply_dpo_rewrite_ffi_strings()

        try:
            self._solve_smt_with_healing(payload)
        except ValueError:
            return False

        if not self._verify_goi(payload):
            return False

        # 5. Materialization gate: only pass here when every check succeeded.
        from aero_forge.scaffold.polyglot_materializer import _emit_verified_files

        _emit_verified_files(payload)
        return True
