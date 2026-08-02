"""Apply pre-materialization structural healing to an in-memory HIN graph.

This module consumes SMT UNSAT core traces and GoI non-nilpotency failures,
produces in-memory AST rewrite patches, and applies them to the HIN graph
schema *before* any source code files are written to disk.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List


def _extract_target_node_id(trace: str) -> str:
    """Best-effort extraction of the offending node id from a failure trace."""
    for pattern in [
        r'"node_id":"([^"]+)"',
        r'"id":"([^"]+)"',
        r"node_id\s+['\"]?([^\s'\"]+)['\"]?",
        r"id\s+['\"]?([^\s'\"]+)['\"]?",
    ]:
        match = re.search(pattern, trace)
        if match:
            candidate = match.group(1).strip('"')
            if candidate:
                return candidate
    return "node_err_borrow"


def _choose_replacement_type(trace: str) -> str:
    """Select a safe wrapper based on the failure category."""
    lower = trace.lower()
    if "ownership" in lower or "borrow" in lower or "linear" in lower:
        return "Arc<Mutex<T>>"
    if "align" in lower or "ffi" in lower or "layout" in lower:
        return "SerializationBuffer"
    if "nilpot" in lower or "deadlock" in lower or "goi" in lower:
        return "DeadlockFreeChannel"
    return "Arc<Mutex<T>>"


def _needs_patch(trace: str) -> bool:
    lower = trace.lower()
    return (
        "ownership mismatch" in lower
        or "alignment" in lower
        or "ffi layout" in lower
        or "non-nilpotent" in lower
        or "deadlock" in lower
    )


def _build_patches(trace: str) -> List[Dict[str, Any]]:
    """Build a list of rewrite patches from a failure trace."""
    patches: List[Dict[str, Any]] = []
    if _needs_patch(trace):
        patches.append(
            {
                "target_node_id": _extract_target_node_id(trace),
                "replacement_type": _choose_replacement_type(trace),
                "inject_wrapper": True,
            }
        )
    return patches


def _inject_wrapped_type(graph: Any, target_id: str, wrapped_type: str) -> None:
    """Recursively inject ``wrapped_type`` into the node with ``id == target_id``."""
    if isinstance(graph, dict):
        if graph.get("id") == target_id:
            graph["wrapped_type"] = wrapped_type
        for value in graph.values():
            _inject_wrapped_type(value, target_id, wrapped_type)
    elif isinstance(graph, list):
        for item in graph:
            _inject_wrapped_type(item, target_id, wrapped_type)


def _apply_patches_to_graph(graph: Any, patches: List[Dict[str, Any]]) -> None:
    for patch in patches:
        if patch.get("inject_wrapper"):
            _inject_wrapped_type(graph, patch["target_node_id"], patch["replacement_type"])


def apply_pre_materialization_healing(failure_trace: str, graph_json: str) -> str:
    """Apply SMT/GoI-derived healing patches to an in-memory HIN graph JSON.

    Returns the patched graph JSON string. No source files are written.
    """
    try:
        from aero_forge._native import PreWriteHealer

        healer = PreWriteHealer()
        healer.analyze_smt_unsat_core(failure_trace)
        return healer.apply_pre_write_patches(graph_json)
    except Exception:
        pass

    graph = json.loads(graph_json)
    patches = _build_patches(failure_trace)
    _apply_patches_to_graph(graph, patches)
    return json.dumps(graph)


def build_pre_write_patches(failure_trace: str) -> List[Dict[str, Any]]:
    """Return the patches that would be applied for ``failure_trace``."""
    try:
        from aero_forge._native import PreWriteHealer

        healer = PreWriteHealer()
        healer.analyze_smt_unsat_core(failure_trace)
        return [
            {"target_node_id": p[0], "replacement_type": p[1], "inject_wrapper": p[2]}
            for p in healer.patches()
        ]
    except Exception:
        return _build_patches(failure_trace)
