"""Tests for aero_forge.scheduler.wavefront.WavefrontScheduler."""

import sys

import pytest

from aero_forge.scheduler.wavefront import CycleError, Task, WavefrontScheduler


def test_compute_wavefronts_linear_chain():
    scheduler = WavefrontScheduler()
    # a -> b -> c
    adj = {"a": [], "b": ["a"], "c": ["b"]}
    waves = scheduler.compute_wavefronts(adj)
    assert waves == [["a"], ["b"], ["c"]]


def test_compute_wavefronts_diamond():
    scheduler = WavefrontScheduler()
    # a -> b, a -> c, b -> d, c -> d
    adj = {"a": [], "b": ["a"], "c": ["a"], "d": ["b", "c"]}
    waves = scheduler.compute_wavefronts(adj)
    assert waves[0] == ["a"]
    assert set(waves[1]) == {"b", "c"}
    assert waves[2] == ["d"]


def test_compute_wavefronts_independent_nodes():
    scheduler = WavefrontScheduler()
    adj = {"x": [], "y": [], "z": []}
    waves = scheduler.compute_wavefronts(adj)
    assert waves == [["x", "y", "z"]]


def test_compute_wavefronts_cycle_raises():
    scheduler = WavefrontScheduler()
    adj = {"a": ["c"], "b": ["a"], "c": ["b"]}
    with pytest.raises(CycleError):
        scheduler.compute_wavefronts(adj)


def test_five_node_dag_partitioning():
    scheduler = WavefrontScheduler()
    # W0: 0,1; W1: 2,3; W2: 4
    adj = {"0": [], "1": [], "2": ["0"], "3": ["1"], "4": ["2", "3"]}
    waves = scheduler.compute_wavefronts(adj)
    assert set(waves[0]) == {"0", "1"}
    assert set(waves[1]) == {"2", "3"}
    assert waves[2] == ["4"]


def test_z3_or_heuristic_resource_check():
    scheduler = WavefrontScheduler(thread_limit=2, memory_limit_mb=512)
    tasks = [Task("t1", "echo 1"), Task("t2", "echo 2"), Task("t3", "echo 3")]
    # Three tasks exceed thread limit of 2; heuristic should reject.
    assert scheduler._heuristic_resource_check(tasks) is False
    # Two tasks should be accepted.
    assert scheduler._heuristic_resource_check(tasks[:2]) is True


@pytest.mark.skipif(sys.platform == "win32", reason="shell echo test")
def test_execute_sync_runs_tasks():
    scheduler = WavefrontScheduler()
    tasks = {
        "a": Task("a", "echo A"),
        "b": Task("b", "echo B"),
    }
    results = scheduler.execute_sync(tasks)
    assert len(results) == 2
    assert all(r["returncode"] == 0 for r in results)
    outputs = {r["stdout"].strip() for r in results}
    assert outputs == {"A", "B"}


def test_execute_sync_respects_dependencies():
    scheduler = WavefrontScheduler()
    # a and b are independent; c depends on a and b.
    tasks = {
        "a": Task("a", "echo A"),
        "b": Task("b", "echo B"),
        "c": Task("c", "echo C"),
    }
    adj = {"a": [], "b": [], "c": ["a", "b"]}
    results = scheduler.execute_sync(tasks, adj)
    by_name = {r["name"]: r for r in results}
    assert by_name["a"]["returncode"] == 0
    assert by_name["b"]["returncode"] == 0
    assert by_name["c"]["returncode"] == 0
    assert by_name["c"]["stdout"].strip() == "C"
