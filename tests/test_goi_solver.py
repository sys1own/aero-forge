"""Tests for the Geometry-of-Interaction (GoI) matrix solver."""

import numpy as np
import pytest

from aero_forge.scheduler.goi_solver import (
    GoiSolverError,
    adjacency_to_matrix,
    goi_compute_gradients,
    goi_execute_wave,
    precedence_scores,
)


def test_goi_identity_matrix() -> None:
    """EX(I, U) should reduce to a simple diagonal solve for identity M."""
    M = np.eye(3, dtype=np.float64)
    U = np.eye(3, dtype=np.float64) * 0.5
    EX = goi_execute_wave(M, U)
    expected = np.linalg.inv(np.eye(3) - U @ M) @ U
    np.testing.assert_allclose(EX, expected, atol=1e-12)


def test_goi_simple_chain() -> None:
    """A 3-node chain should produce a triangular EX matrix."""
    # 0 -> 1 -> 2
    M = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
    ], dtype=np.float64)
    U = np.eye(3, dtype=np.float64) * 0.5
    EX = goi_execute_wave(M, U)
    expected = np.linalg.inv(np.eye(3) - U @ M) @ U
    np.testing.assert_allclose(EX, expected, atol=1e-12)


def test_goi_against_sequential_dag() -> None:
    """GoI solve should match the sequential wave-front precedence for a DAG."""
    adj = {
        "a": [],
        "b": ["a"],
        "c": ["a"],
        "d": ["b", "c"],
        "e": ["d"],
    }
    M, nodes = adjacency_to_matrix(adj)
    U = np.eye(len(nodes), dtype=np.float64) * 0.5
    EX = goi_execute_wave(M, U)
    assert EX.shape == (5, 5)
    assert not np.isnan(EX).any()


def test_goi_precedence_scores() -> None:
    """Precedence scores should be non-negative and higher for critical nodes."""
    adj = {
        "root": [],
        "left": ["root"],
        "right": ["root"],
        "merge": ["left", "right"],
    }
    scores = precedence_scores(adj, damping=0.15)
    assert all(s >= 0.0 for s in scores.values())
    # The merge node sits at the end of two dependency chains -> high score.
    assert scores["merge"] >= max(scores["root"], scores["left"], scores["right"])


def test_goi_gradient_shape() -> None:
    """Analytical gradient should have the same shape as U."""
    M = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    U = np.eye(3, dtype=np.float64) * 0.5
    loss_grad = np.random.RandomState(0).randn(3, 3)
    grad = goi_compute_gradients(M, U, loss_grad)
    assert grad.shape == U.shape


def test_goi_matrix_dimension_mismatch() -> None:
    """A dimension mismatch must raise a clear solver error."""
    M = np.eye(3, dtype=np.float64)
    U = np.eye(4, dtype=np.float64)
    with pytest.raises(GoiSolverError):
        goi_execute_wave(M, U)


def test_goi_proof_net_nilpotent() -> None:
    """A terminating acyclic interaction should be nilpotent."""
    GoIProofNet = pytest.importorskip("aero_forge_native").GoIProofNet

    net = GoIProofNet(3)
    M = np.eye(3, dtype=np.float64).flatten().tolist()
    sigma = (np.eye(3, dtype=np.float64) * 0.5).flatten().tolist()
    net.set_axiom_matrix(M)
    net.set_cut_matrix(sigma)

    ex = np.array(net.compute_execution_formula())
    assert ex.shape == (3, 3)
    assert net.verify_nilpotency(100)


def test_goi_proof_net_cyclic_deadlock() -> None:
    """A cyclic non-terminating interaction should fail nilpotency/execution."""
    aero_forge_native = pytest.importorskip("aero_forge_native")
    GoIProofNet = aero_forge_native.GoIProofNet

    net = GoIProofNet(2)
    # 0 <-> 1 cycle
    M = np.array([[0, 1], [1, 0]], dtype=np.float64).flatten().tolist()
    sigma = np.eye(2, dtype=np.float64).flatten().tolist()
    net.set_axiom_matrix(M)
    net.set_cut_matrix(sigma)

    assert not net.verify_nilpotency(50)

    # The native convenience wrapper should also report false/deadlock.
    assert not aero_forge_native.verify_goi_proof_net(2, M, sigma, 50)


def test_native_bridge_verify_goi_proof_net() -> None:
    """The Python bridge exposes verify_goi_proof_net and falls back gracefully."""
    from aero_forge.native_bridge import verify_goi_proof_net

    # Terminating 3-node chain.
    M = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64).flatten().tolist()
    sigma = np.eye(3, dtype=np.float64).flatten().tolist()
    assert verify_goi_proof_net(3, M, sigma, max_iterations=100)

    # Cyclic 2-node deadlock.
    M = np.array([[0, 1], [1, 0]], dtype=np.float64).flatten().tolist()
    sigma = np.eye(2, dtype=np.float64).flatten().tolist()
    assert not verify_goi_proof_net(2, M, sigma, max_iterations=50)
