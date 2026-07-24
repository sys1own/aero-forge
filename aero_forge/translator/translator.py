"""UAST -> HIN graph translator.

Walks a normalized UAST and builds a homomorphic interaction-net graph.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from aero_forge.hin_vm import (
    ConstructorNode,
    DestructorNode,
    DuplicatorNode,
    EraserNode,
    HINNetwork,
    MELLType,
    Node,
    Port,
    SwitchNode,
    ValueNode,
)


def _kind(node: dict) -> str:
    return (
        node.get("canonical_kind") or node.get("kind") or node.get("type") or "unknown"
    )


_FUNCTION_KINDS = {
    "function_declaration",
    "function_definition",
    "function",
    "lambda",
}
_BINDING_KINDS = {"binding", "assignment", "let", "variable_declaration"}
_REFERENCE_KINDS = {"reference", "identifier", "name", "var"}
_LITERAL_KINDS = {"literal", "constant", "number", "string", "value"}
_IF_KINDS = {"if", "if_statement", "conditional", "if_else"}
_CALL_KINDS = {
    "call",
    "application",
    "apply",
    "call_expression",
    "user_function_call",
}
_CONTAINER_KINDS = {
    "module",
    "translation_unit",
    "block",
    "lexical_block",
    "body",
    "sequence",
    "program",
}


class BoundaryPortNode(Node):
    """Reified boundary interface port produced by module splitting."""

    symbol = "boundary"

    def __init__(self, node_id: str, contract_id: str, wire_type: MELLType):
        super().__init__(node_id)
        self.contract_id = contract_id
        self.wire_type = wire_type
        self._set_principal(MELLType.any_())


class UASTToHINTranslator:
    """Translate normalized UAST syntax trees into homomorphic HIN networks."""

    def __init__(self, auto_split_threshold: int = 120):
        self.auto_split_threshold = auto_split_threshold
        self.scope_stack: List[Dict[str, Port]] = []
        self._ref_remaining: List[Dict[str, int]] = []

    def translate(self, uast_root: dict) -> HINNetwork:
        return self.translate_uast(uast_root)

    def translate_uast(self, uast: dict) -> HINNetwork:
        net = HINNetwork()
        self.scope_stack = []
        self._ref_remaining = []
        self._push_scope(uast)
        result = self._build_container(uast, net)
        if result is not None and result.target is None:
            self._terminate(net, result)
        self._pop_scope(net)
        return net

    def _traverse_and_build(self, node: dict, net: HINNetwork) -> Optional[Port]:
        if not isinstance(node, dict):
            return None
        kind = _kind(node)

        if kind in _CONTAINER_KINDS:
            return self._build_container(node, net)
        if kind in _FUNCTION_KINDS:
            return self._build_function(node, net)
        if kind in _BINDING_KINDS:
            return self._build_binding(node, net)
        if kind in _REFERENCE_KINDS:
            return self._build_reference(node, net)
        if kind in _LITERAL_KINDS:
            return self._build_literal(node, net)
        if kind in _IF_KINDS:
            return self._build_if(node, net)
        if kind in _CALL_KINDS:
            return self._build_call(node, net)

        return self._build_container(node, net)

    def _build_container(self, node: dict, net: HINNetwork) -> Optional[Port]:
        last: Optional[Port] = None
        for child in self._children(node):
            out = self._traverse_and_build(child, net)
            if last is not None and last.target is None:
                self._terminate(net, last)
            last = out
        return last

    def _build_function(self, node: dict, net: HINNetwork) -> Port:
        ctor = ConstructorNode(net.fresh_id("gamma"))
        net.register_node(ctor)

        body = node.get("body", self._children(node))
        body_node = {"type": "body", "children": self._as_list(body)}

        self._push_scope(body_node)
        param = node.get("param") or node.get("name_param")
        params = node.get("params") or ([param] if param else [])
        if params:
            self.scope_stack[-1][params[0]] = ctor.a_1
        else:
            self._terminate(net, ctor.a_1)

        result = self._build_container(body_node, net)
        if result is None:
            result = ValueNode(net.fresh_id("V"), None).p
            net.register_node(result.owner)
        net._link(result, ctor.a_2)
        self._pop_scope(net)

        name = node.get("name")
        if name and self.scope_stack:
            self.scope_stack[-1][name] = ctor.p
            self._ref_remaining[-1].setdefault(name, self._count_name(node, name))
        return ctor.p

    def _build_binding(self, node: dict, net: HINNetwork) -> None:
        value = node.get("value") or node.get("init") or node.get("expr")
        name = node.get("name") or node.get("target")
        port = self._traverse_and_build(value, net) if value else None
        if port is None:
            port = ValueNode(net.fresh_id("V"), None).p
            net.register_node(port.owner)
        if name:
            self.scope_stack[-1][name] = port
        else:
            self._terminate(net, port)
        return None

    def _build_reference(self, node: dict, net: HINNetwork) -> Port:
        name = node.get("name") or node.get("text") or node.get("value")
        return self._resolve(str(name), net)

    def _resolve(self, name: str, net: HINNetwork) -> Port:
        for idx in range(len(self.scope_stack) - 1, -1, -1):
            frame = self.scope_stack[idx]
            if name not in frame:
                continue
            src = frame[name]
            remaining = self._ref_remaining[idx].get(name, 1) - 1
            self._ref_remaining[idx][name] = remaining
            if remaining <= 0:
                del frame[name]
                return src
            dup = DuplicatorNode(net.fresh_id("delta"))
            net.register_node(dup)
            net._link(src, dup.p)
            frame[name] = dup.a_2
            return dup.a_1

        node = ValueNode(net.fresh_id("V"), None)
        net.register_node(node)
        return node.p

    def _build_literal(self, node: dict, net: HINNetwork) -> Port:
        value = node.get("value", node.get("text"))
        vnode = ValueNode(net.fresh_id("V"), value)
        net.register_node(vnode)
        return vnode.p

    def _build_if(self, node: dict, net: HINNetwork) -> Port:
        switch = SwitchNode(net.fresh_id("switch"))
        net.register_node(switch)

        cond = node.get("condition") or node.get("test") or node.get("cond")
        then_branch = node.get("then") or node.get("consequent")
        else_branch = node.get("else") or node.get("alternate")

        cond_port = self._traverse_and_build(cond, net) if cond else None
        if cond_port is None:
            cond_port = ValueNode(net.fresh_id("V"), False).p
            net.register_node(cond_port.owner)
        net._link(cond_port, switch.p)

        then_port = self._branch_port(then_branch, net)
        else_port = self._branch_port(else_branch, net)
        net._link(then_port, switch.a_1)
        net._link(else_port, switch.a_2)

        return switch.a_3

    def _branch_port(self, branch, net: HINNetwork) -> Port:
        if branch is None:
            node = ValueNode(net.fresh_id("V"), None)
            net.register_node(node)
            return node.p
        port = self._traverse_and_build(branch, net)
        if port is None:
            node = ValueNode(net.fresh_id("V"), None)
            net.register_node(node)
            return node.p
        return port

    def _build_call(self, node: dict, net: HINNetwork) -> Port:
        dtor = DestructorNode(net.fresh_id("app"))
        net.register_node(dtor)

        func = node.get("function") or node.get("callee") or node.get("func")
        arg = node.get("argument") or node.get("arg")
        args = node.get("arguments") or ([arg] if arg is not None else [])

        func_port = self._traverse_and_build(func, net) if func else None
        if func_port is None:
            func_port = ValueNode(net.fresh_id("V"), None).p
            net.register_node(func_port.owner)
        net._link(func_port, dtor.p)

        if args:
            arg_port = self._traverse_and_build(args[0], net)
            if arg_port is None:
                arg_port = ValueNode(net.fresh_id("V"), None).p
                net.register_node(arg_port.owner)
            net._link(arg_port, dtor.a_1)
        else:
            self._terminate(net, dtor.a_1)

        return dtor.a_2

    def _push_scope(self, subtree: dict) -> None:
        self.scope_stack.append({})
        self._ref_remaining.append(self._count_refs(subtree))

    def _pop_scope(self, net: HINNetwork) -> None:
        frame = self.scope_stack.pop()
        self._ref_remaining.pop()
        for port in frame.values():
            if port.target is None:
                self._terminate(net, port)

    def _terminate(self, net: HINNetwork, port: Port) -> None:
        if port is None or port.target is not None:
            return
        eraser = EraserNode(net.fresh_id("erase"))
        net.register_node(eraser)
        net._link(eraser.p, port)

    def _count_refs(self, subtree: dict) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        self._count_into(subtree, counts)
        return counts

    def _count_into(self, node, counts: Dict[str, int]) -> None:
        if isinstance(node, list):
            for item in node:
                self._count_into(item, counts)
            return
        if not isinstance(node, dict):
            return
        if _kind(node) in _REFERENCE_KINDS:
            name = node.get("name") or node.get("text") or node.get("value")
            if name is not None:
                counts[str(name)] = counts.get(str(name), 0) + 1
        for value in node.values():
            if isinstance(value, (dict, list)):
                self._count_into(value, counts)

    def _count_name(self, node, name: str) -> int:
        counts: Dict[str, int] = {}
        self._count_into(node, counts)
        return counts.get(name, 0)

    @staticmethod
    def _children(node: dict) -> List[dict]:
        children = node.get("children") or node.get("body") or []
        return [
            c for c in UASTToHINTranslator._as_list(children) if isinstance(c, dict)
        ]

    @staticmethod
    def _as_list(value) -> list:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]

    # The following helpers are kept for parity but are not used by the
    # accelerate CLI path.

    def evaluate_complexity(self, network: HINNetwork) -> Dict[str, float]:
        nodes = list(network.nodes.values())
        n = len(nodes)
        adj = self._adjacency(network, nodes)
        edge_count = float(np.sum(adj) / 2.0)
        max_edges = n * (n - 1) / 2.0
        density = edge_count / max_edges if max_edges > 0 else 0.0
        avg_degree = (2.0 * edge_count / n) if n > 0 else 0.0
        return {
            "node_count": float(n),
            "edge_count": edge_count,
            "density": density,
            "avg_degree": avg_degree,
            "exceeds_threshold": float(n > self.auto_split_threshold),
        }

    def execute_mitosis(
        self, module: HINNetwork, threshold: Optional[int] = None
    ) -> Tuple[HINNetwork, HINNetwork]:
        limit = self.auto_split_threshold if threshold is None else threshold
        if len(module.nodes) <= limit:
            return module, HINNetwork()
        return self.split_module(module)

    def compute_fiedler_vector(self, adj_matrix: np.ndarray) -> np.ndarray:
        degree_matrix = np.diag(np.sum(adj_matrix, axis=1))
        laplacian = degree_matrix - adj_matrix
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
        sorted_indices = np.argsort(eigenvalues)
        fiedler_index = sorted_indices[1]
        return eigenvectors[:, fiedler_index]

    def split_module(self, net: HINNetwork) -> Tuple[HINNetwork, HINNetwork]:
        node_list = list(net.nodes.values())
        n = len(node_list)
        part_1 = HINNetwork()
        part_2 = HINNetwork()

        if n < 2:
            for node in node_list:
                part_1.register_node(node)
            return part_1, part_2

        node_to_idx = {node.node_id: idx for idx, node in enumerate(node_list)}
        adj = self._adjacency(net, node_list)

        fiedler = self.compute_fiedler_vector(adj)
        side = {node_list[i].node_id: (0 if fiedler[i] >= 0 else 1) for i in range(n)}
        if len({s for s in side.values()}) == 1:
            median = float(np.median(fiedler))
            side = {
                node_list[i].node_id: (0 if fiedler[i] >= median else 1)
                for i in range(n)
            }

        nets = (part_1, part_2)
        contracts: List[dict] = []
        seen: set = set()
        for node in node_list:
            for port in node.ports():
                target = port.target
                if target is None:
                    continue
                other = target.owner
                if other.node_id not in node_to_idx:
                    continue
                key = frozenset((id(port), id(target)))
                if key in seen:
                    continue
                seen.add(key)
                if side[node.node_id] == side[other.node_id]:
                    continue

                contract_id = f"contract#{len(contracts)}"
                wire_type = port.type
                cap_a = BoundaryPortNode(
                    nets[side[node.node_id]].fresh_id("boundary"),
                    contract_id,
                    wire_type,
                )
                cap_b = BoundaryPortNode(
                    nets[side[other.node_id]].fresh_id("boundary"),
                    contract_id,
                    target.type,
                )
                nets[side[node.node_id]].register_node(cap_a)
                nets[side[other.node_id]].register_node(cap_b)
                port.target = cap_a.p
                cap_a.p.target = port
                target.target = cap_b.p
                cap_b.p.target = target
                contracts.append(
                    {
                        "contract_id": contract_id,
                        "side_a": side[node.node_id],
                        "side_b": side[other.node_id],
                        "endpoint_a": (node.node_id, port.name),
                        "endpoint_b": (other.node_id, target.name),
                        "type": repr(wire_type),
                    }
                )

        for node in node_list:
            nets[side[node.node_id]].register_node(node)

        self._rescan_active(part_1)
        self._rescan_active(part_2)
        return part_1, part_2

    @staticmethod
    def _adjacency(net: HINNetwork, node_list: List[Node]) -> np.ndarray:
        n = len(node_list)
        node_to_idx = {node.node_id: idx for idx, node in enumerate(node_list)}
        adj = np.zeros((n, n))
        for node in node_list:
            i = node_to_idx[node.node_id]
            for port in node.ports():
                target = port.target
                if target is None:
                    continue
                j = node_to_idx.get(target.owner.node_id)
                if j is None or j == i:
                    continue
                adj[i][j] += 1.0
        adj = np.maximum(adj, adj.T)
        return adj

    @staticmethod
    def _rescan_active(net: HINNetwork) -> None:
        net.active_pairs = []  # type: ignore[attr-defined]
        seen: set = set()
        for node in net.nodes.values():
            p = node.p
            if p is None or p.target is None or not p.target.is_principal:
                continue
            other = p.target.owner
            if other.node_id not in net.nodes:
                continue
            key = frozenset((node.node_id, other.node_id))
            if key in seen:
                continue
            seen.add(key)


__all__ = ["UASTToHINTranslator", "BoundaryPortNode"]
