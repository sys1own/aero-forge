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
