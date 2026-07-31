"""HINGraph: workspace dependency graph with GoI influence zones for LLM healing."""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger("aero_forge.healing.hin_graph")


def _find_python_files(workspace_root: Path) -> List[Path]:
    """Return all Python files under the workspace, excluding common cache dirs."""
    paths: List[Path] = []
    for p in workspace_root.rglob("*.py"):
        if any(part.startswith(".") or part in {"__pycache__", "target", "node_modules"} for part in p.parts):
            continue
        paths.append(p)
    return paths


def _imports_from_file(path: Path) -> List[str]:
    """Return module names imported by a Python file."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return []
    except OSError:
        return []

    modules: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.append(node.module.split(".")[0])
            elif node.level:
                # relative import; resolve relative to package structure
                modules.append(path.stem)
    return modules


def build_workspace_hingraph(workspace_root: Path) -> Tuple[Dict[str, List[str]], Set[str]]:
    """Build a HIN-style dependency graph of the workspace.

    Nodes are module basenames (e.g. ``foo.py`` -> ``foo``).  Edges point from a
    module to the modules it imports.  The returned ``adj`` dict is suitable for
    the GoI wavefront scheduler and ``precedence_scores``.
    """
    workspace_root = Path(workspace_root).resolve()
    files = _find_python_files(workspace_root)

    nodes: Set[str] = set()
    adj: Dict[str, List[str]] = {}
    for path in files:
        name = path.relative_to(workspace_root).with_suffix("").as_posix().replace("/", ".")
        nodes.add(name)
        adj[name] = []

    for path in files:
        name = path.relative_to(workspace_root).with_suffix("").as_posix().replace("/", ".")
        imports = _imports_from_file(path)
        for imp in imports:
            if imp in nodes and imp != name:
                if imp not in adj[name]:
                    adj[name].append(imp)

    return adj, nodes


def _normalize_symbol(symbol: str, adj: Dict[str, List[str]]) -> Optional[str]:
    """Map a symbol reference to the best matching node in the graph."""
    if symbol in adj:
        return symbol
    # Try stem matches.
    candidates = [n for n in adj if n.endswith(f".{symbol}") or n.split(".")[-1] == symbol]
    return candidates[0] if candidates else None


def influence_zone(
    workspace_root: Path,
    failure_symbols: List[str],
    *,
    radius: int = 2,
) -> Tuple[Set[str], List[List[str]]]:
    """Compute the failure influence zone as a bounded subgraph.

    Returns a tuple ``(affected_nodes, wavefront_schedule)`` where
    ``affected_nodes`` contains the failed symbols plus all ancestors and
    descendants within ``radius`` hops, and ``wavefront_schedule`` is the
    topological wave ordering of that subgraph.
    """
    adj, _ = build_workspace_hingraph(workspace_root)
    if not adj:
        return set(failure_symbols), []

    reverse: Dict[str, List[str]] = {n: [] for n in adj}
    for node, deps in adj.items():
        for dep in deps:
            reverse[dep].append(node)

    seeds = set()
    for sym in failure_symbols:
        mapped = _normalize_symbol(sym, adj)
        if mapped:
            seeds.add(mapped)
        else:
            # If the symbol is not in the graph, add it as an isolated node.
            seeds.add(sym)
            adj.setdefault(sym, [])
            reverse.setdefault(sym, [])

    affected: Set[str] = set(seeds)
    frontier = list(seeds)
    for _ in range(radius):
        next_frontier: Set[str] = set()
        for node in frontier:
            for nbr in list(adj.get(node, [])) + list(reverse.get(node, [])):
                if nbr not in affected:
                    affected.add(nbr)
                    next_frontier.add(nbr)
        frontier = list(next_frontier)
        if not frontier:
            break

    sub_adj = {n: [d for d in adj.get(n, []) if d in affected] for n in affected}
    try:
        from aero_forge.scheduler.wavefront import WavefrontScheduler

        waves = WavefrontScheduler().compute_wavefronts(sub_adj)
    except Exception as exc:
        logger.debug("Could not compute HINGraph wavefront: %s", exc)
        waves = []
    return affected, waves


def delta_m_influence(
    workspace_root: Path,
    changed_files: List[str],
) -> Dict[str, List[str]]:
    """Return a delta-M adjacency diff: nodes that transitively depend on changed files."""
    adj, _ = build_workspace_hingraph(workspace_root)
    if not adj:
        return {}

    reverse: Dict[str, List[str]] = {n: [] for n in adj}
    for node, deps in adj.items():
        for dep in deps:
            reverse[dep].append(node)

    seeds = set()
    for cf in changed_files:
        mapped = _normalize_symbol(cf, adj)
        if mapped:
            seeds.add(mapped)

    impacted: Dict[str, List[str]] = {}
    for seed in seeds:
        reachable: Set[str] = set()
        stack = [seed]
        while stack:
            node = stack.pop()
            for dependent in reverse.get(node, []):
                if dependent not in reachable:
                    reachable.add(dependent)
                    stack.append(dependent)
        impacted[seed] = sorted(reachable)
    return impacted
