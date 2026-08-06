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

import ast
from collections import defaultdict
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


def _native_wavefront_solver() -> Any:
    """Return the native GoI wavefront solver class if the extension is compiled."""
    try:
        from aero_forge._native import GoIWavefrontSolverNative

        return GoIWavefrontSolverNative
    except Exception:
        return None


class GoIWavefrontSolver:
    """Python wrapper for the native GoI wavefront scheduler.

    Falls back to a pure Python implementation when the compiled extension is
    unavailable.
    """

    def __init__(self, labels: List[str], M: Any, U: Any) -> None:
        self.labels = labels
        self._native_cls = _native_wavefront_solver()
        if self._native_cls is not None:
            self._native = self._native_cls(labels, M, U)
        else:
            self._native = None
            self.M = _as_array(M).astype(np.float64)
            self.U = _as_array(U).astype(np.float64)
            if self.M.shape != (len(labels), len(labels)) or self.U.shape != self.M.shape:
                raise GoiSolverError("M and U must be square matrices matching labels length")

    def ex_operator(self) -> Any:
        """Compute EX(M, U) = (I - U * M)^-1 * U."""
        try:
            if self._native is not None:
                return self._native.ex_operator()
        except ArithmeticError as exc:
            raise GoiSolverError(str(exc)) from exc

        I = np.eye(len(self.labels), dtype=np.float64)
        k = I - self.U @ self.M
        det = np.linalg.det(k)
        if abs(det) < 1e-12:
            raise GoiSolverError(
                "Cyclic dependency in G_HIN: (I - U*M) matrix operator is singular."
            )
        k_inv = np.linalg.inv(k)
        return k_inv @ self.U

    def wavefront_stages(self) -> List[List[str]]:
        """Return parallel execution stages by in-degree reduction on M.

        Raises:
            GoiSolverError: if the graph is cyclic (K is singular).
        """
        try:
            if self._native is not None:
                return self._native.wavefront_stages()
        except ArithmeticError as exc:
            raise GoiSolverError(str(exc)) from exc

        n = len(self.labels)
        in_degree = [0] * n
        outgoing: List[List[int]] = [[] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if abs(self.M[i, j]) > 1e-12:
                    # M[i][j] == source j -> target i
                    in_degree[i] += 1
                    outgoing[j].append(i)

        remaining = in_degree.copy()
        seen = [False] * n
        stages: List[List[str]] = []
        while True:
            stage = [self.labels[i] for i in range(n) if not seen[i] and remaining[i] == 0]
            if not stage:
                if all(seen):
                    break
                raise GoiSolverError(
                    "Cyclic dependency in G_HIN: no wavefront stage could be extracted."
                )
            for name in stage:
                seen[self.labels.index(name)] = True
            for i in range(n):
                if seen[i]:
                    for j in outgoing[i]:
                        if not seen[j]:
                            remaining[j] = max(0, remaining[j] - 1)
            stage.sort()
            stages.append(stage)
        return stages


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


def goi_nilpotency_check(M: Any, max_power: Optional[int] = None) -> bool:
    """Return True if ``(σ M)^N = 0`` for some ``N``.

    ``σ M`` is ``M`` with its diagonal removed (no self-loops).  Nilpotency
    of the loop-carried dependency matrix proves that the corresponding
    concurrent loop nest can never deadlock.
    """
    M_arr = _as_array(M).astype(np.float64)
    if M_arr.ndim != 2 or M_arr.shape[0] != M_arr.shape[1]:
        return False
    n = M_arr.shape[0]
    if n == 0:
        return True
    sigma = M_arr.copy()
    np.fill_diagonal(sigma, 0.0)
    limit = max_power or n
    power = sigma.copy()
    for _ in range(1, limit + 1):
        if np.allclose(power, 0.0, atol=1e-12):
            return True
        power = power @ sigma
    return False


def _loop_dependency_matrix(func: ast.FunctionDef) -> Tuple[np.ndarray, List[str]]:
    """Build a per-statement loop-carried dependency matrix.

    An edge ``src -> tgt`` means that a write to ``tgt`` in the loop body reads
    ``src``.  Self-loops are removed because a variable depending on its own
    previous value cannot deadlock.  Cross-statement false dependencies (e.g.
    ``count = count + 1`` creating an edge to ``total``) are avoided by only
    wiring loads to stores that appear in the same statement.
    """
    edges: set = set()
    all_names: set = set()

    def _names(node: ast.AST, ctx_type: type) -> set:
        return {
            child.id
            for child in ast.walk(node)
            if isinstance(child, ast.Name) and isinstance(child.ctx, ctx_type)
        }

    def _process(stmts: List[ast.stmt]) -> None:
        for stmt in stmts:
            if isinstance(stmt, (ast.For, ast.While)):
                if isinstance(stmt, ast.For):
                    iter_loads = _names(stmt.iter, ast.Load)
                    all_names.update(iter_loads)
                    for target in ast.walk(stmt.target):
                        if isinstance(target, ast.Name) and isinstance(
                            target.ctx, ast.Store
                        ):
                            all_names.add(target.id)
                            for src in iter_loads:
                                if src != target.id:
                                    edges.add((src, target.id))
                _process(stmt.body)
                _process(stmt.orelse)
            elif isinstance(stmt, ast.If):
                _process(stmt.body)
                _process(stmt.orelse)
            elif isinstance(stmt, ast.With):
                # Context expressions provide loads; ``as`` targets are stores.
                for item in stmt.items:
                    loads = _names(item.context_expr, ast.Load)
                    if item.optional_vars:
                        stores = _names(item.optional_vars, ast.Store)
                        all_names.update(loads | stores)
                        for s in stores:
                            for l in loads:
                                if l != s:
                                    edges.add((l, s))
                _process(stmt.body)
            else:
                loads = _names(stmt, ast.Load)
                stores = _names(stmt, ast.Store)
                all_names.update(loads | stores)
                for s in stores:
                    for l in loads:
                        if l != s:
                            edges.add((l, s))

    _process(func.body)
    if not all_names:
        return np.zeros((0, 0), dtype=np.float64), []
    nodes = sorted(all_names)
    index = {name: i for i, name in enumerate(nodes)}
    M = np.zeros((len(nodes), len(nodes)), dtype=np.float64)
    for src, tgt in edges:
        M[index[tgt], index[src]] = 1.0
    return M, nodes


def check_python_loops_nilpotent(
    source: str,
    function_name: Optional[str] = None,
) -> Tuple[bool, str]:
    """Check that every loop in ``source`` has a nilpotent dependency matrix.

    Returns ``(True, reason)`` when no loop-carried cycles are detected,
    otherwise ``(False, reason)``.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False, "source could not be parsed"
    funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if function_name:
        funcs = [f for f in funcs if f.name == function_name]
    if not funcs:
        return True, "no function found"
    for func in funcs:
        M, nodes = _loop_dependency_matrix(func)
        if M.size == 0:
            continue
        if not goi_nilpotency_check(M):
            return (
                False,
                f"loop dependency matrix for {func.name} is not nilpotent: "
                f"potential deadlock among {nodes}",
            )
    return True, "all loop dependency matrices nilpotent (deadlock-free)"
