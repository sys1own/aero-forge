"""Blueprint loading and readiness helpers.

A blueprint is considered *ready* (initialized) when it contains explicit
LLM-generated code nodes/definitions and is not marked as ``draft`` or
``uninitialized``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

logger = logging.getLogger("aero_forge.blueprint.blueprint_parser")


def _has_code_nodes(blueprint: Dict[str, Any]) -> bool:
    """Return True if *blueprint* declares concrete code-generation targets."""
    for key in (
        "contracts",
        "abi_contracts",
        "build_pipeline",
        "module_graph",
        "functions",
        "manifest",
        "verification_nodes",
        "nodes",
        "edges",
    ):
        value = blueprint.get(key)
        if value:
            return True
    return False


def load_blueprint(path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Load ``blueprint.aero`` or ``workspace_blueprint.yaml`` as a plain dict."""
    path = Path(path)
    if not path.is_file():
        return None
    if not yaml:
        return None
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return None
    if isinstance(data, dict):
        return data
    return None


def is_blueprint_ready(blueprint: Union[Dict[str, Any], Path, None]) -> bool:
    """Return True when a blueprint is initialized and safe to materialize.

    A blueprint is *not* ready when:
      - it is missing or not a dict,
      - its ``status`` is ``draft`` or ``uninitialized``,
      - it lacks any code-generation nodes (contracts, build_pipeline, etc.).

    It is considered ready when it has explicit LLM-generated code nodes and
    a finalized (non-draft) status.
    """
    if blueprint is None:
        return False
    if isinstance(blueprint, (str, Path)):
        blueprint = load_blueprint(Path(blueprint))
    if not isinstance(blueprint, dict):
        return False

    metadata = blueprint.get("metadata") or {}
    status = str(metadata.get("status", "unknown")).lower()
    if status in {"draft", "uninitialized"}:
        return False

    if not _has_code_nodes(blueprint):
        return False

    # A v3 blueprint with explicit LLM metadata is definitely ready.
    if metadata.get("llm_initialized"):
        return True
    if str(metadata.get("generation_method", "")).lower() in {
        "llm_synthesized",
        "synthesized",
        "llm",
    }:
        return True

    # v2 blueprints and finalized v3 blueprints with concrete code nodes are ready
    # unless their status explicitly says they are not.
    return status not in {"draft", "uninitialized"}
