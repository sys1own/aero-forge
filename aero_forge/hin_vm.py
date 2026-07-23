"""Hierarchical Interaction Net graph model used by the translator."""

from __future__ import annotations

from enum import Enum
from itertools import count
from typing import Dict, List, Optional


class TypeKind(Enum):
    I = "I"
    TENSOR = "Tensor"
    IMPLICATION = "Implication"
    BANG = "Bang"


class MELLType:
    """Light-weight structural type for graph edges."""

    def __init__(
        self,
        kind: TypeKind,
        left: Optional["MELLType"] = None,
        right: Optional["MELLType"] = None,
        wildcard: bool = False,
    ):
        self.kind = kind
        self.left = left
        self.right = right
        self.wildcard = wildcard

    @staticmethod
    def any_() -> "MELLType":
        return MELLType(TypeKind.I, wildcard=True)

    @staticmethod
    def unit() -> "MELLType":
        return MELLType(TypeKind.I)

    @staticmethod
    def tensor(left: "MELLType", right: "MELLType") -> "MELLType":
        return MELLType(TypeKind.TENSOR, left, right)

    @staticmethod
    def implication(left: "MELLType", right: "MELLType") -> "MELLType":
        return MELLType(TypeKind.IMPLICATION, left, right)

    @staticmethod
    def bang(inner: "MELLType") -> "MELLType":
        return MELLType(TypeKind.BANG, inner)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MELLType):
            return NotImplemented
        return (
            self.kind == other.kind
            and self.left == other.left
            and self.right == other.right
        )

    def __hash__(self) -> int:
        return hash((self.kind, self.left, self.right))

    def unifiable(self, other: "MELLType") -> bool:
        if not isinstance(other, MELLType):
            return False
        if self.wildcard or other.wildcard:
            return True
        if self.kind == other.kind:
            if self.kind == TypeKind.I:
                return True
            if self.kind == TypeKind.BANG:
                return _opt_unifiable(self.left, other.left)
            return _opt_unifiable(self.left, other.left) and _opt_unifiable(
                self.right, other.right
            )
        if self.kind == TypeKind.BANG:
            return _opt_unifiable(self.left, other)
        if other.kind == TypeKind.BANG:
            return _opt_unifiable(self, other.left)
        return False

    def __repr__(self) -> str:  # pragma: no cover
        if self.kind == TypeKind.I:
            return "I"
        if self.kind in (TypeKind.TENSOR, TypeKind.IMPLICATION):
            symbol = "*" if self.kind == TypeKind.TENSOR else "->"
            return f"({self.left} {symbol} {self.right})"
        return f"!{self.left}"


def _opt_unifiable(a: Optional[MELLType], b: Optional[MELLType]) -> bool:
    if a is None or b is None:
        return True
    return a.unifiable(b)


class Port:
    """A node's connection point."""

    PRINCIPAL = "p"

    def __init__(self, owner: "Node", name: str, expected_type: MELLType):
        self.owner = owner
        self.name = name
        self.type = expected_type
        self.target: Optional["Port"] = None

    @property
    def is_principal(self) -> bool:
        return self.name == Port.PRINCIPAL

    def connect(self, other: "Port") -> None:
        if not self.type.unifiable(other.type):
            raise TypeError(
                f"non-unifiable port binding: {self.owner.node_id}.{self.name}"
                f":{self.type!r} <-> {other.owner.node_id}.{other.name}"
                f":{other.type!r}"
            )
        self.target = other
        other.target = self


class Node:
    """Base interaction node."""

    symbol = "node"

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.p: Optional[Port] = None
        self.aux: List[Port] = []

    def _set_principal(self, expected_type: MELLType) -> Port:
        self.p = Port(self, Port.PRINCIPAL, expected_type)
        return self.p

    def _add_aux(self, name: str, expected_type: MELLType) -> Port:
        port = Port(self, name, expected_type)
        self.aux.append(port)
        return port

    def ports(self) -> List[Port]:
        ports: List[Port] = []
        if self.p is not None:
            ports.append(self.p)
        ports.extend(self.aux)
        return ports

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{self.symbol} {self.node_id}>"


class ConstructorNode(Node):
    """Function constructor node."""

    symbol = "gamma"

    def __init__(
        self,
        node_id: str,
        arg_type: Optional[MELLType] = None,
        ret_type: Optional[MELLType] = None,
    ):
        super().__init__(node_id)
        arg_type = arg_type or MELLType.unit()
        ret_type = ret_type or MELLType.unit()
        self._set_principal(MELLType.implication(arg_type, ret_type))
        self.a_1 = self._add_aux("a_1", arg_type)
        self.a_2 = self._add_aux("a_2", ret_type)


class DestructorNode(Node):
    """Function application node."""

    symbol = "gamma_inv"

    def __init__(
        self,
        node_id: str,
        arg_type: Optional[MELLType] = None,
        ret_type: Optional[MELLType] = None,
    ):
        super().__init__(node_id)
        arg_type = arg_type or MELLType.unit()
        ret_type = ret_type or MELLType.unit()
        self._set_principal(MELLType.implication(arg_type, ret_type))
        self.a_1 = self._add_aux("a_1", arg_type)
        self.a_2 = self._add_aux("a_2", ret_type)


class DuplicatorNode(Node):
    """Shared-resource duplicator."""

    symbol = "delta"

    def __init__(self, node_id: str, shared_type: Optional[MELLType] = None):
        super().__init__(node_id)
        shared_type = shared_type or MELLType.unit()
        banged = MELLType.bang(shared_type)
        self._set_principal(MELLType.any_())
        self.a_1 = self._add_aux("a_1", banged)
        self.a_2 = self._add_aux("a_2", banged)


class EraserNode(Node):
    """Dead-wire terminator."""

    symbol = "epsilon"

    def __init__(self, node_id: str, erased_type: Optional[MELLType] = None):
        super().__init__(node_id)
        self._set_principal(MELLType.any_())


class ValueNode(Node):
    """Constant value node."""

    symbol = "V"

    def __init__(
        self,
        node_id: str,
        value: object,
        value_type: Optional[MELLType] = None,
    ):
        super().__init__(node_id)
        self.value = value
        self._set_principal(value_type or MELLType.unit())


class SwitchNode(Node):
    """Conditional switch."""

    symbol = "sigma"

    def __init__(self, node_id: str, branch_type: Optional[MELLType] = None):
        super().__init__(node_id)
        branch_type = branch_type or MELLType.unit()
        self._set_principal(MELLType.unit())
        self.a_1 = self._add_aux("a_1", branch_type)
        self.a_2 = self._add_aux("a_2", branch_type)
        self.a_3 = self._add_aux("a_3", branch_type)


class CausalProjectionNode(Node):
    """Stub projection node; kept for interface compatibility."""

    symbol = "P"

    def __init__(self, node_id: str, projected_type: Optional[MELLType] = None):
        super().__init__(node_id)
        projected_type = projected_type or MELLType.unit()
        self._set_principal(MELLType.unit())
        self.a_1 = self._add_aux("a_1", projected_type)


class HINNetwork:
    """A graph of interaction nodes connected by typed ports."""

    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self._gensym = count()

    def register_node(self, node: Node) -> Node:
        self.nodes[node.node_id] = node
        return node

    def fresh_id(self, prefix: str) -> str:
        return f"{prefix}#{next(self._gensym)}"

    def bind(self, port_a: Port, port_b: Port) -> None:
        port_a.connect(port_b)

    def _link(self, port_a: Optional[Port], port_b: Optional[Port]) -> None:
        if port_a is not None:
            port_a.target = port_b
        if port_b is not None:
            port_b.target = port_a


__all__ = [
    "MELLType",
    "Port",
    "Node",
    "ConstructorNode",
    "DestructorNode",
    "DuplicatorNode",
    "EraserNode",
    "ValueNode",
    "SwitchNode",
    "CausalProjectionNode",
    "HINNetwork",
]
