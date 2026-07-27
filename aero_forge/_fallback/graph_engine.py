"""Pure-Python graph algorithms used as the fallback reference implementation.

This implementation intentionally operates on string node identifiers to serve
as a straightforward, unoptimized baseline.  The native Rust engine indexes the
same graph and therefore runs the same algorithms much faster.
"""

from aero_forge.accelerator.contracts import GraphEngineABC


class GraphEngine(GraphEngineABC):
    """Topological sort and reachability primitives in straightforward pure Python."""

    def __init__(self, nodes: list[str], edges: dict[str, list[str]]) -> None:
        self.nodes = list(nodes)
        self._edges = {k: list(v) for k, v in edges.items()}
        # Build successors (children) and in-degree from the dependency edges.
        self._successors: dict[str, list[str]] = {n: [] for n in self.nodes}
        for node, deps in self._edges.items():
            if node not in self._successors:
                self._successors[node] = []
            for dep in deps:
                self._successors.setdefault(dep, []).append(node)

    def topological_sort(self) -> list[str]:
        """Return a valid topological ordering using Kahn's algorithm."""
        in_degree: dict[str, int] = {n: 0 for n in self.nodes}
        for node, deps in self._edges.items():
            if node not in in_degree:
                in_degree[node] = 0
            for dep in deps:
                if dep not in in_degree:
                    in_degree[dep] = 0
                in_degree[node] += 1

        queue = [n for n, d in in_degree.items() if d == 0]
        ordered: list[str] = []

        while queue:
            current = queue.pop(0)
            ordered.append(current)
            for succ in self._successors.get(current, []):
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)

        if len(ordered) != len(in_degree):
            raise ValueError("Graph contains a cycle and cannot be topologically sorted")

        return ordered

    def prune_unreachable(self, roots: list[str]) -> list[str]:
        """Return nodes reachable from *roots* along dependency edges, sorted."""
        reachable: set[str] = set()
        queue = list(roots)

        while queue:
            current = queue.pop(0)
            if current in reachable:
                continue
            reachable.add(current)
            for dep in self._edges.get(current, []):
                if dep not in reachable:
                    queue.append(dep)

        return sorted(reachable)
