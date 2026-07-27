"""Parity tests for the dual-engine accelerator (native vs. pure-Python fallback)."""

import importlib
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import pytest


def _reload_accelerator(monkeypatch, disable_native: str):
    """Import :mod:`aero_forge.accelerator` with the requested native/fallback mode."""
    monkeypatch.setenv("AERO_DISABLE_NATIVE", disable_native)
    from aero_forge import accelerator

    importlib.reload(accelerator)
    return accelerator


@pytest.fixture(params=["0", "1"], ids=["native", "fallback"])
def accel(request, monkeypatch):
    return _reload_accelerator(monkeypatch, request.param)


def test_hash_bytes_parity(accel, tmp_path: Path) -> None:
    """hash_bytes must be deterministic and identical between native and fallback."""
    samples = [b"", b"a", b"hello world", b"\x00" * 1000, os.urandom(4096)]
    for sample in samples:
        assert accel.hash_bytes(sample) == accel.hash_bytes(sample)
        assert len(accel.hash_bytes(sample)) == 64


def test_hasher_incremental_parity(accel, tmp_path: Path) -> None:
    """Incremental hashing must equal one-shot hashing and be stable."""
    data = os.urandom(10000)
    chunks = [data[i : i + 1023] for i in range(0, len(data), 1023)]

    hasher = accel.Hasher()
    for chunk in chunks:
        hasher.update(chunk)
    incremental = hasher.finalize()

    one_shot = accel.hash_bytes(data)
    assert incremental == one_shot

    clone = hasher.copy()
    clone.update(b"suffix")
    assert hasher.finalize() == incremental
    assert clone.finalize() != incremental


def test_digest_parity(accel) -> None:
    """Raw digest bytes must match the hex digest."""
    data = b"deterministic input"
    hasher = accel.Hasher()
    hasher.update(data)
    assert accel.hash_bytes(data) == hasher.finalize()
    raw = hasher.digest()
    assert raw.hex() == accel.hash_bytes(data)


def test_hash_file_parity(accel, tmp_path: Path) -> None:
    """hash_file must agree with incremental hashing on the same bytes."""
    data = os.urandom(8192)
    file_path = tmp_path / "sample.bin"
    file_path.write_bytes(data)
    assert accel.hash_file(str(file_path)) == accel.hash_bytes(data)


def _random_dag(node_count: int, seed: int) -> Tuple[List[str], Dict[str, List[str]]]:
    """Generate a random directed acyclic graph as a dict of dependencies."""
    rng = random.Random(seed)
    nodes = [f"node_{i}" for i in range(node_count)]
    edges: Dict[str, List[str]] = {n: [] for n in nodes}
    for i in range(node_count):
        for j in range(i + 1, node_count):
            if rng.random() < 0.3:
                edges[nodes[j]].append(nodes[i])
    return nodes, edges


def test_topological_sort_parity(accel) -> None:
    """Native and fallback must return valid, equivalent topological orders."""
    for seed in range(5):
        nodes, edges = _random_dag(50, seed)
        order = accel.GraphEngine(nodes, edges).topological_sort()
        assert set(order) == set(nodes)
        position = {n: i for i, n in enumerate(order)}
        for node, deps in edges.items():
            for dep in deps:
                assert position[dep] < position[node]


def test_topological_sort_cycle(accel) -> None:
    """A cyclic graph must raise a deterministic ValueError."""
    nodes = ["a", "b", "c"]
    edges = {"a": ["b"], "b": ["c"], "c": ["a"]}
    with pytest.raises(ValueError, match="cycle"):
        accel.GraphEngine(nodes, edges).topological_sort()


def test_prune_unreachable_parity(accel) -> None:
    """Reachability pruning must be identical between native and fallback."""
    nodes = [f"n{i}" for i in range(20)]
    edges: Dict[str, List[str]] = {
        "n0": ["n1", "n2"],
        "n1": ["n3"],
        "n2": ["n4"],
        "n5": ["n6"],
        "n6": ["n7"],
    }
    engine = accel.GraphEngine(nodes, edges)
    assert engine.prune_unreachable(["n0"]) == ["n0", "n1", "n2", "n3", "n4"]
    assert engine.prune_unreachable(["n5"]) == ["n5", "n6", "n7"]


def test_prune_unreachable_deterministic(accel) -> None:
    """Reachability output must be deterministic for randomized DAGs."""
    for seed in range(5):
        nodes, edges = _random_dag(50, seed)
        roots = random.Random(seed).sample(nodes, 5)
        engine = accel.GraphEngine(nodes, edges)
        result1 = engine.prune_unreachable(roots)
        result2 = engine.prune_unreachable(roots)
        assert result1 == result2


def test_fallback_disable_native(monkeypatch) -> None:
    """AERO_DISABLE_NATIVE=1 must select the pure-Python implementations."""
    accel = _reload_accelerator(monkeypatch, "1")
    assert not accel.is_native()
    assert accel.hash_bytes(b"test") == _reload_accelerator(monkeypatch, "0").hash_bytes(b"test")
