"""Unified, graph-driven polyglot materializer.

The ``GraphPolyglotMaterializer`` consumes a HIN-style graph specification,
computes parallel build wavefronts with the GoI solver, synthesizes FFI bridge
contracts for every cross-language edge, and delegates source/manifest emission
to the language-specific emitter plugins registered in ``EmitterRegistry``.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from aero_forge.builder.emitters.base import (
    BoundaryContract,
    CodeArtifact,
    EmitterRegistry,
    PolyglotEmitterPlugin,
)
from aero_forge.builder.language_router import SystemToolchainRouter
from aero_forge.scheduler.goi_solver import GoIWavefrontSolver, GoiSolverError
from aero_forge.scaffold.contract_synth import (
    DynamicContractSynthesizer,
    FFIBoundaryEdge,
)


class MaterializationError(Exception):
    """Raised when the graph materializer cannot write a valid workspace."""


class GraphPolyglotMaterializer:
    """Materialize a HIN graph into source files and build manifests.

    Args:
        workspace_root: directory in which all project files are written.
        registry: optional ``EmitterRegistry`` instance. If ``None`` the
            singleton is used, ensuring all built-in emitter modules are loaded.
        contract_synth: optional ``DynamicContractSynthesizer`` instance.
    """

    def __init__(
        self,
        workspace_root: Path,
        registry: Optional[EmitterRegistry] = None,
        contract_synth: Optional[DynamicContractSynthesizer] = None,
    ) -> None:
        self.workspace_root = Path(workspace_root)
        self.registry = registry or EmitterRegistry.get_instance()
        self.contract_synth = contract_synth or DynamicContractSynthesizer()
        self._ensure_emitters_loaded()

    @staticmethod
    def _ensure_emitters_loaded() -> None:
        """Import all plugin modules so they auto-register with the registry."""
        from aero_forge.builder.emitters import (  # noqa: F401
            cgo_emitter,
            cpp_emitter,
            cs_emitter,
            jni_emitter,
            python_emitter,
            rust_emitter,
        )

    @staticmethod
    def _build_adjacency_matrix(
        nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, List[str], Dict[str, int]]:
        """Build a dependency matrix M where M[target, source] == 1 for each edge."""
        labels = [n["node_id"] for n in nodes]
        order = {nid: i for i, nid in enumerate(labels)}
        n = len(labels)
        M = np.zeros((n, n), dtype=np.float64)
        for edge in edges:
            src = edge.get("source")
            tgt = edge.get("target")
            if src in order and tgt in order:
                M[order[tgt], order[src]] = 1.0
        return M, labels, order

    @staticmethod
    def _boundary_contracts_for_node(
        node_id: str, edges: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Collect all edges touching *node_id* as plugin contract dicts."""
        contracts: List[Dict[str, Any]] = []
        for edge in edges:
            if edge.get("source") == node_id or edge.get("target") == node_id:
                contracts.append(
                    {
                        "boundary_type": edge.get("boundary_type", "c_abi"),
                        "symbol": edge.get("symbol", ""),
                        "args": edge.get("args", []),
                        "return_type": edge.get("return_type", ""),
                        "is_zero_copy": edge.get("is_zero_copy", False),
                        "source": edge.get("source", ""),
                        "target": edge.get("target", ""),
                    }
                )
        return contracts

    def _write_artifact(self, artifact: CodeArtifact, output_dir: Path) -> Path:
        """Write a single artifact atomically and return its resolved path."""
        output_dir.mkdir(parents=True, exist_ok=True)
        target = (output_dir / artifact.file_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)

        fd, tmp_path = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(artifact.content)
            os.replace(tmp_path, str(target))
        except Exception:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass
            raise
        return target

    def _synthesize_ffi_bridges(
        self, edges: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate ``GeneratedFFIBridge`` artifacts for every cross-language edge."""
        bridges: List[Dict[str, Any]] = []
        for edge in edges:
            boundary = edge.get("boundary_type", "c_abi")
            bridge = self.contract_synth.synthesize_boundary(
                FFIBoundaryEdge(
                    edge_id=edge.get("edge_id") or f"{edge['source']}_{edge['target']}",
                    source_node=edge["source"],
                    source_lang=edge.get("source_lang", ""),
                    target_node=edge["target"],
                    target_lang=edge.get("target_lang", ""),
                    boundary_type=boundary,
                    symbol_name=edge.get("symbol", ""),
                    argument_types=edge.get("args", []),
                    return_type=edge.get("return_type", ""),
                    is_zero_copy=edge.get("is_zero_copy", False),
                )
            )
            bridge_dir = self.workspace_root / "ffi_bridges" / bridge.edge_id
            artifacts: List[CodeArtifact] = []
            if bridge.header:
                ext = ".h" if bridge.boundary_type in ("c_abi", "pinvoke") else ".hpp"
                artifacts.append(
                    CodeArtifact(
                        file_path=f"bridge{ext}",
                        content=bridge.header,
                        language="c",
                        is_header=True,
                    )
                )
            if bridge.source:
                src_ext = ".c"
                if bridge.boundary_type == "pyo3_maturin":
                    src_ext = ".rs"
                elif bridge.boundary_type == "cgo":
                    src_ext = ".go"
                artifacts.append(
                    CodeArtifact(
                        file_path=f"bridge{src_ext}",
                        content=bridge.source,
                        language="c" if src_ext == ".c" else bridge.boundary_type,
                    )
                )
            if bridge.python_loader:
                artifacts.append(
                    CodeArtifact(
                        file_path="loader.py",
                        content=bridge.python_loader,
                        language="python",
                    )
                )
            if bridge.csharp_stub:
                artifacts.append(
                    CodeArtifact(
                        file_path="AeroNative.cs",
                        content=bridge.csharp_stub,
                        language="csharp",
                    )
                )
            for artifact in artifacts:
                self._write_artifact(artifact, bridge_dir)
            bridges.append(
                {
                    "edge_id": bridge.edge_id,
                    "boundary_type": bridge.boundary_type,
                    "directory": str(bridge_dir),
                    "files": [a.file_path for a in artifacts],
                }
            )
        return bridges

    def materialize(
        self,
        hin_graph_spec: Dict[str, Any],
        *,
        build: bool = False,
    ) -> Dict[str, Any]:
        """Materialize *hin_graph_spec* into ``workspace_root``.

        Steps:
            1. Build adjacency/routing matrices and solve wavefront stages.
            2. Synthesize FFI bridges for every edge.
            3. Emit source files and build manifests per wavefront stage.
            4. Optionally dispatch native builds.
        """
        nodes: List[Dict[str, Any]] = hin_graph_spec.get("nodes", [])
        edges: List[Dict[str, Any]] = hin_graph_spec.get("edges", [])
        if not nodes:
            raise MaterializationError("hin_graph_spec must contain at least one node")

        M, labels, order = self._build_adjacency_matrix(nodes, edges)
        U = np.eye(len(labels), dtype=np.float64) * 0.5
        try:
            solver = GoIWavefrontSolver(labels, M, U)
            stages = solver.wavefront_stages()
        except GoiSolverError as exc:
            raise MaterializationError(
                f"GoI wavefront solver rejected the graph: {exc}"
            ) from exc

        bridges = self._synthesize_ffi_bridges(edges)

        node_map = {n["node_id"]: n for n in nodes}
        written_artifacts: List[Dict[str, Any]] = []

        for stage in stages:
            for node_id in stage:
                node_spec = node_map.get(node_id)
                if not node_spec:
                    continue
                lang = node_spec.get("lang", "").lower()
                try:
                    plugin = self.registry.get_plugin(lang)
                except KeyError as exc:
                    raise MaterializationError(
                        f"no emitter plugin registered for language {lang!r}"
                    ) from exc

                node_dir = self.workspace_root / node_id
                contracts = self._boundary_contracts_for_node(node_id, edges)
                source_artifacts = plugin.emit_source_files(node_id, node_spec, contracts)
                for artifact in source_artifacts:
                    self._write_artifact(artifact, node_dir)
                    written_artifacts.append(
                        {
                            "node_id": node_id,
                            "language": lang,
                            "file": artifact.file_path,
                            "path": str(node_dir / artifact.file_path),
                        }
                    )

                manifest = plugin.emit_build_manifest(
                    node_id,
                    node_spec.get("dependencies", []),
                    node_spec.get("compiler_flags", []),
                )
                self._write_artifact(manifest, node_dir)
                written_artifacts.append(
                    {
                        "node_id": node_id,
                        "language": lang,
                        "file": manifest.file_path,
                        "path": str(node_dir / manifest.file_path),
                    }
                )

                if build:
                    try:
                        SystemToolchainRouter.dispatch_node_build(
                            node_id, node_spec, node_dir
                        )
                    except RuntimeError as exc:
                        raise MaterializationError(
                            f"toolchain dispatch failed for {node_id}: {exc}"
                        ) from exc

        result: Dict[str, Any] = {
            "project": hin_graph_spec.get("project", "aero_forge_project"),
            "architecture": "graph_polyglot",
            "workspace": str(self.workspace_root),
            "stages": stages,
            "bridges": bridges,
            "artifacts": written_artifacts,
        }
        return result
