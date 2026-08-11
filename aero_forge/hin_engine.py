"""Python bridge to the Rust HIN (MELL interaction net) engine."""

import json
import os
from typing import Any, Callable, Dict, List, Optional

import networkx as nx

from aero_forge.hin_graph import EDGE_RELATIONS, HINGraph, HINNode

try:
    from aero_forge_native import HinEngine  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - native module is optional at import time
    try:
        from aero_forge._native import HinEngine  # type: ignore[attr-defined]
    except Exception:
        HinEngine = None  # type: ignore[misc,assignment]


# MELL structural type helpers used by the emitters.
class MELLType:
    """MELL-typed wire annotation."""

    def __init__(self, kind: str, left=None, right=None, inner=None):
        self.kind = kind
        self.left = left
        self.right = right
        self.inner = inner

    @staticmethod
    def unit():
        return MELLType("I")

    @staticmethod
    def any_():
        return MELLType("Any")

    @staticmethod
    def bang(inner):
        return MELLType("Bang", inner=inner)

    @staticmethod
    def implication(left, right):
        return MELLType("Implication", left=left, right=right)

    @staticmethod
    def tensor(left, right):
        return MELLType("Tensor", left=left, right=right)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"kind": self.kind}
        if self.left is not None:
            d["left"] = self.left.to_dict() if isinstance(self.left, MELLType) else self.left
        if self.right is not None:
            d["right"] = (
                self.right.to_dict() if isinstance(self.right, MELLType) else self.right
            )
        if self.inner is not None:
            d["inner"] = (
                self.inner.to_dict() if isinstance(self.inner, MELLType) else self.inner
            )
        return d


class HINEngine:
    """Python HIN engine supporting DPO rewrites and ownership propagation."""

    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self._hin_graph = HINGraph()
        # Mirror the HINGraph container onto the raw networkx graph used by tests.
        self._hin_graph.graph = self.graph

    def add_ast_node(self, node: HINNode) -> None:
        """Add a typed HIN node to the engine graph."""
        self.graph.add_node(
            node.node_id,
            node_type=node.node_type,
            language=node.language,
            ownership=node.ownership_level,
            properties=node.properties,
            obj=node,
        )

    def add_relation(
        self, source_id: str, target_id: str, relation_type: str, **attrs
    ) -> None:
        """Add a typed relation edge between two HIN nodes."""
        if relation_type not in EDGE_RELATIONS:
            raise ValueError(f"Unknown relation type: {relation_type}")
        self.graph.add_edge(
            source_id, target_id, key=relation_type, relation=relation_type, **attrs
        )

    def _is_string_ffi(self, u: str, v: str, data: Dict[str, Any]) -> bool:
        """Return True when *u* calls *v* over FFI with a raw string argument."""
        u_data = self.graph.nodes[u]
        v_data = self.graph.nodes[v]
        if v_data.get("node_type") == "FFIBoundary" or u_data.get("node_type") == "FFIBoundary":
            return False
        edge_arg = (data.get("arg_type") or "").lower()
        src_arg = (u_data.get("properties", {}).get("arg_type") or "").lower()
        tgt_arg = (v_data.get("properties", {}).get("arg_type") or "").lower()
        return (
            "str" in edge_arg
            or "string" in edge_arg
            or "&str" in src_arg
            or "str" in src_arg
            or u_data.get("properties", {}).get("ffi_string") is True
            or data.get("ffi_string") is True
        )

    def apply_dpo_rewrite_ffi_strings(self) -> int:
        """Apply DPO rewrite rules to wrap raw string FFI calls with a boundary node.

        For each Rust -> C++ ``CallsFFI`` edge that passes a raw string, inject an
        ``FFIBoundary`` node and redirect the call through it.

        Returns the number of edges rewritten.
        """
        count = 0
        edges = list(self.graph.edges(keys=True, data=True))
        for source, target, key, data in edges:
            if key != "CallsFFI":
                continue
            source_data = self.graph.nodes[source]
            target_data = self.graph.nodes[target]
            if source_data.get("language") != "rust":
                continue
            if target_data.get("language") != "cpp":
                continue
            if not self._is_string_ffi(source, target, data):
                continue

            bridge_id = f"ffi_bridge_{source}_{target}_{count}"
            bridge = HINNode(
                node_id=bridge_id,
                node_type="FFIBoundary",
                language="ffi",
                properties={
                    "abi": "C",
                    "wrapper": "C_String_Wrapper",
                    "source": source,
                    "target": target,
                },
                ownership_level="&",
            )
            self.add_ast_node(bridge)

            # Remove the direct Rust -> C++ edge and route through the bridge.
            self.graph.remove_edge(source, target, key=key)
            self.add_relation(source, bridge_id, "CallsFFI")
            self.add_relation(bridge_id, target, "BindsTo")
            count += 1
        return count

    def propagate_ownership_constraints(self) -> List[str]:
        """Check affine-logic ownership transfers and report static violations.

        The ownership lattice is
        ``⊥ ⊑ & ⊑ &mut ⊑ 1`` and ``!`` is the managed/Python value domain.
        A direct ``TransfersOwnershipTo`` edge from a Rust linear node (``1``)
        to a Python GC node (``!``) without an intermediate managed pointer box
        is flagged as a violation.
        """
        violations: List[str] = []
        for source, target, key, data in self.graph.edges(keys=True, data=True):
            if key != "TransfersOwnershipTo":
                continue
            source_data = self.graph.nodes[source]
            target_data = self.graph.nodes[target]
            source_ownership = source_data.get("ownership", "1")
            target_ownership = target_data.get("ownership", "!")
            source_type = source_data.get("node_type", "")
            target_type = target_data.get("node_type", "")

            if source_ownership == "1" and target_ownership == "!":
                # Allow if there is an intermediate FFIBoundary or borrow node on the path.
                if not self._has_managed_intermediate(source, target):
                    violations.append(
                        f"Ownership violation: linear Rust node {source!r} ({source_type}) "
                        f"directly transfers ownership to Python GC node {target!r} ({target_type}) "
                        f"without an intermediate managed pointer box"
                    )
        return violations

    def _has_managed_intermediate(self, source: str, target: str) -> bool:
        """Return True if a managed pointer box lies between *source* and *target*."""
        # Short-circuit: if there is an FFIBoundary or borrowed node on any
        # simple path between source and target, the transfer is mediated.
        try:
            for path in nx.all_simple_paths(self.graph, source, target, cutoff=3):
                for node in path[1:-1]:
                    node_type = self.graph.nodes[node].get("node_type", "")
                    ownership = self.graph.nodes[node].get("ownership", "")
                    if node_type == "FFIBoundary" or ownership in {"&", "&mut"}:
                        return True
        except nx.NodeNotFound:
            pass
        return False


def reduce_uast(
    uast: Any,
    max_steps: int = 1_000_000,
    timeout_seconds: Optional[float] = None,
    progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Build and reduce a HIN graph from a UAST value.

    Returns ``{"steps": int, "stalled": int, "timed_out": bool, "graph": list}``.
    Falls back to a no-op dictionary when the native module is unavailable so the
    bridge can be imported everywhere.
    """
    if HinEngine is None:
        return {"steps": 0, "stalled": 0, "timed_out": False, "graph": [], "native": False}

    def _emit(event: str, payload: Dict[str, Any]) -> None:
        if progress_callback:
            try:
                progress_callback(event, payload)
            except Exception:
                pass

    engine = HinEngine()
    engine.build_from_json(json.dumps(uast))
    _emit("hin_reduction_steps", {"phase": "build", "nodes": engine.node_count()})
    if timeout_seconds is not None:
        steps, timed_out = engine.reduce_to_completion_with_timeout(
            max_steps, timeout_seconds
        )
    else:
        steps = engine.reduce_to_completion(max_steps)
        timed_out = False
    graph = json.loads(engine.to_json())
    stalled = engine.stalled_pairs()
    _emit(
        "hin_reduction_steps",
        {"phase": "complete", "steps": steps, "stalled": stalled, "timed_out": timed_out, "nodes": len(graph)},
    )
    return {"steps": steps, "stalled": stalled, "timed_out": timed_out, "graph": graph, "native": True}


def native_available() -> bool:
    """Return ``True`` when the Rust HIN engine extension is importable."""
    return HinEngine is not None


# Feature-flag guard for integration with the translator/emitters.
HIN_ENGINE_ENABLED = os.environ.get("AERO_HIN_ENGINE", "1") == "1"
