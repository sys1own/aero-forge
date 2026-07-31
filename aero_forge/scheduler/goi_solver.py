"""Geometry of Interaction (GoI) matrix solver for Aero-Forge wavefront scheduling.

Implements Girard's GoI execution formula:

    EX(M, U) = (I - U * M)^(-1) * U

``M`` is the directed dependency adjacency / Hashimoto-style edge matrix.
``U`` is the routing / execution rule matrix.

This module provides a NumPy fallback and an optional JAX/XLA path.  It is
used by ``WavefrontScheduler`` to compute graph precedence scores and can be
used by numeric kernels to solve a whole dependency graph as a single
parallel matrix pass.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class GoiSolverError(Exception):
    """Raised when a GoI matrix solve cannot be completed."""


def _as_array(matrix: Any) -> np.ndarray:
    if isinstance(matrix, np.ndarray):
        return matrix
    return np.asarray(matrix, dtype=np.float64)


def _is_available(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


_JAX_AVAILABLE = _is_available("jax")


def goi_execute_wave(
    M: Any,
    U: Any,
    *,
    use_jax: bool = False,
) -> np.ndarray:
    """Compute EX(M, U) = (I - U*M)^(-1) * U.

    Args:
        M: Dependency matrix (n x n).
        U: Routing rule matrix (n x n).
        use_jax: If True and jax is installed, use jax.numpy for XLA
            compilation; otherwise use NumPy.

    Returns:
        The n x n execution operator EX.
    """
    if use_jax and _JAX_AVAILABLE:
        import jax.numpy as jnp

        M_arr = jnp.asarray(M, dtype=jnp.float64)
        U_arr = jnp.asarray(U, dtype=jnp.float64)
        I = jnp.eye(U_arr.shape[0], dtype=U_arr.dtype)
        inv_term = jnp.linalg.inv(I - jnp.dot(U_arr, M_arr))
        return np.asarray(jnp.dot(inv_term, U_arr))

    M_arr = _as_array(M).astype(np.float64)
    U_arr = _as_array(U).astype(np.float64)
    if M_arr.shape != U_arr.shape or M_arr.ndim != 2 or M_arr.shape[0] != M_arr.shape[1]:
        raise GoiSolverError("M and U must be square matrices of the same shape")

    n = U_arr.shape[0]
    I = np.eye(n, dtype=np.float64)
    inv_term = np.linalg.inv(I - U_arr @ M_arr)
    return inv_term @ U_arr


def goi_compute_gradients(
    M: Any,
    U: Any,
    loss_grad_out: Any,
) -> np.ndarray:
    """Compute the analytical gradient dL/dU of the routing rule matrix.

    Letting X = (I - U*M)^(-1), the derivative is:

        grad_U = X.T * loss_grad_out * (I + M * EX_current).T

    This avoids explicit backpropagation through the recursive solver steps.
    """
    M_arr = _as_array(M).astype(np.float64)
    U_arr = _as_array(U).astype(np.float64)
    loss_arr = _as_array(loss_grad_out).astype(np.float64)

    n = U_arr.shape[0]
    I = np.eye(n, dtype=np.float64)
    X = np.linalg.inv(I - U_arr @ M_arr)
    EX = X @ U_arr

    inv_trans = X.T
    right_factor = I + M_arr.T @ EX.T
    return inv_trans @ (loss_arr @ right_factor)


def adjacency_to_matrix(
    adj_list: Dict[str, List[str]],
    node_order: Optional[List[str]] = None,
    *,
    weighted: bool = False,
    weights: Optional[Dict[Tuple[str, str], float]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Convert an adjacency list into a dense GoI dependency matrix ``M``.

    ``M[i, j]`` is the weight of the edge from ``node_order[j]`` to
    ``node_order[i]`` (i.e. ``j`` must complete before ``i`` can start).
    """
    nodes = node_order or sorted(adj_list.keys())
    index = {name: i for i, name in enumerate(nodes)}
    n = len(nodes)
    M = np.zeros((n, n), dtype=np.float64)
    for target, deps in adj_list.items():
        i = index.get(target)
        if i is None:
            continue
        for dep in deps:
            j = index.get(dep)
            if j is None:
                continue
            if weighted and weights:
                M[i, j] = weights.get((target, dep), 1.0)
            else:
                M[i, j] = 1.0
    return M, nodes


def precedence_scores(
    adj_list: Dict[str, List[str]],
    node_order: Optional[List[str]] = None,
    *,
    damping: float = 0.15,
    weights: Optional[Dict[Tuple[str, str], float]] = None,
    use_jax: bool = False,
) -> Dict[str, float]:
    """Return a GoI-derived precedence score for each node in ``adj_list``.

    The dependency matrix ``M`` is built from ``adj_list``.  The routing
    matrix ``U`` is initialized as a damped diagonal so the GoI solve reflects
    both dependency structure and local execution weight.  Higher scores mean
    the node is on more critical / heavily-routed paths.
    """
    M, nodes = adjacency_to_matrix(adj_list, node_order, weighted=True, weights=weights)
    n = M.shape[0]
    U = np.eye(n, dtype=np.float64) * (1.0 - damping)
    EX = goi_execute_wave(M, U, use_jax=use_jax)
    # Row-norm as a simple scalar precedence for each node.
    scores = np.linalg.norm(EX, axis=1)
    return {node: float(scores[i]) for i, node in enumerate(nodes)}
