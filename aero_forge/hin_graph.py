"""Heterogeneous Information Network (HIN) graph data model for Aero-Forge.

Provides the node/edge typing and graph container used by the HIN engine to
model cross-language ASTs and perform DPO rewrites and ownership propagation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import networkx as nx


# Node types T(V) supported by the HIN graph.
NODE_TYPES = {
    "RustStruct",
    "CppClass",
    "PyClass",
    "FFIBoundary",
    "TypeAlias",
    "LifetimeScope",
    "AsyncContext",
    "Hole",
}

# Edge relations R(E) supported by the HIN graph.
EDGE_RELATIONS = {
    "ParentOf",
    "TypeAnnotatedBy",
    "CallsFFI",
    "TransfersOwnershipTo",
    "BorrowsFrom",
    "ImportsSymbol",
    "BindsTo",
}


@dataclass
class HINNode:
    """A typed node in the HIN AST.

    Attributes:
        node_id: Stable identifier used as the networkx node key.
        node_type: One of the HIN node types.
        language: Source language domain ('rust', 'cpp', 'python', 'ffi').
        properties: Free-form metadata carried with the node.
        ownership_level: Affine-logic ownership label.
    """

    node_id: str
    node_type: str
    language: str
    properties: Dict[str, Any] = field(default_factory=dict)
    ownership_level: Optional[str] = None

    def __post_init__(self):
        if self.ownership_level is None:
            self.ownership_level = self._default_ownership()
        if self.node_type == "FFIBoundary":
            self.ownership_level = self.properties.get("ownership", "&")

    def _default_ownership(self) -> str:
        if self.language == "rust":
            return self.properties.get("ownership", "1")
        if self.language == "python":
            return self.properties.get("ownership", "!")
        if self.language == "cpp":
            return self.properties.get("ownership", r"$\bot$")
        if self.language == "ffi":
            return self.properties.get("ownership", "&")
        return self.properties.get("ownership", "1")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "language": self.language,
            "properties": self.properties,
            "ownership": self.ownership_level,
        }


class HINGraph:
    """Networkx-backed container for HIN ASTs."""

    def __init__(self):
        self.graph = nx.MultiDiGraph()

    def add_node(self, node: HINNode) -> None:
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
        if relation_type not in EDGE_RELATIONS:
            raise ValueError(f"Unknown relation type: {relation_type}")
        self.graph.add_edge(
            source_id, target_id, key=relation_type, relation=relation_type, **attrs
        )

    def node_count(self) -> int:
        return self.graph.number_of_nodes()

    def edge_count(self) -> int:
        return self.graph.number_of_edges()
