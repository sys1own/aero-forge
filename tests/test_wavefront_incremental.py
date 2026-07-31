"""Tests for incremental wavefront schedule repair and GoI precedence."""

import time
from typing import Dict, List

from aero_forge.scheduler.wavefront import GraphMutation, MutationKind, WavefrontScheduler


def _make_large_adj(n: int, branch: int = 2) -> Dict[str, List[str]]:
    """Create a wide DAG with `n` nodes."""
    adj: Dict[str, List[str]] = {}
    for i in range(n):
        deps = []
        if i > 0:
            # Each node depends on the previous node and an earlier branch.
            deps.append(f"node_{i - 1}")
            branch_idx = max(0, i - branch - 1)
            if branch_idx < i - 1:
                deps.append(f"node_{branch_idx}")
        adj[f"node_{i}"] = deps
    return adj


def test_incremental_update_preserves_untouched_levels() -> None:
    """Adding one edge should only recompute the affected subgraph."""
    scheduler = WavefrontScheduler()
    adj = {
        "a": [],
        "b": ["a"],
        "c": ["a"],
        "d": ["b", "c"],
        "e": ["d"],
    }
    old = scheduler.compute_wavefronts(adj)
    # Add an edge from c to e. c already precedes e via d, so no change.
    mutations = [GraphMutation(MutationKind.ADD_EDGE, edge=("c", "e"))]
    new = scheduler.update_schedule(adj, old, mutations)
    assert new == old


def test_incremental_add_node_and_edge() -> None:
    """Adding a new node with an edge should insert it at the correct level."""
    scheduler = WavefrontScheduler()
    adj = {
        "a": [],
        "b": ["a"],
        "c": ["a"],
    }
    old = scheduler.compute_wavefronts(adj)
    adj["d"] = ["b", "c"]
    mutations = [
        GraphMutation(MutationKind.ADD_NODE, node_id="d"),
        GraphMutation(MutationKind.ADD_EDGE, edge=("d", "b")),
        GraphMutation(MutationKind.ADD_EDGE, edge=("d", "c")),
    ]
    new = scheduler.update_schedule(adj, old, mutations)
    assert new == [["a"], ["b", "c"], ["d"]]


def test_incremental_speedup_on_large_graph() -> None:
    """Incremental update should be much faster than a full GoI recompute when
    the influence zone is small relative to the whole graph."""
    scheduler = WavefrontScheduler()
    n = 2000
    adj = {f"node_{i}": [f"node_{i - 1}"] if i > 0 else [] for i in range(n)}

    old = scheduler.compute_wavefronts(adj)

    # Add a shortcut in the chain. The influence zone is the tail after the target.
    adj["node_1500"] = list(adj["node_1500"]) + ["node_1000"]
    mutations = [GraphMutation(MutationKind.ADD_EDGE, edge=("node_1500", "node_1000"))]

    start = time.perf_counter()
    incremental = scheduler.update_schedule(adj, old, mutations)
    incremental_time = time.perf_counter() - start

    # Full recompute with GoI precedence is expensive on large graphs.
    start = time.perf_counter()
    full = scheduler.compute_wavefronts(adj, use_goi=True)
    full_time = time.perf_counter() - start

    assert incremental == full
    assert incremental_time < full_time / 5.0, (
        f"Incremental ({incremental_time:.4f}s) not >=5x faster than full ({full_time:.4f}s)"
    )


def test_goi_precedence_in_wavefront() -> None:
    """GoI ranking should not alter topological levels, only intra-wave order."""
    scheduler = WavefrontScheduler()
    adj = {
        "root": [],
        "left": ["root"],
        "right": ["root"],
        "merge": ["left", "right"],
    }
    normal = scheduler.compute_wavefronts(adj)
    goi = scheduler.compute_wavefronts(adj, use_goi=True)
    assert set(normal[0]) == set(goi[0])
    assert set(normal[1]) == set(goi[1])
    assert set(normal[2]) == set(goi[2])


def test_direct_exec_runs_simple_command() -> None:
    """The scheduler should be able to run a simple command via direct exec."""
    from aero_forge.scheduler.wavefront import Task

    scheduler = WavefrontScheduler()
    tasks = {"echo": Task("echo", "python -c 'print(\"hi\")'")}
    results = scheduler.execute_sync(tasks, {"echo": []})
    assert len(results) == 1
    assert results[0]["returncode"] == 0
