"""Workspace context builder for follow-up build feedback loops."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from aero_forge.bundle_repo import bundle_workspace, format_context_block
from aero_forge.healing.context_builder import ContextBuilder


BUILD_PROFILE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "debug": {
        "engine_backend": "hin_cpu",
        "wavefront_parallelism": 4,
        "precision_shield_mode": "ieee",
        "hin_jit_opt_level": 0,
    },
    "performance": {
        "engine_backend": "hin_gpu",
        "wavefront_parallelism": 8,
        "precision_shield_mode": "fast_math",
        "hin_jit_opt_level": 2,
    },
    "safety": {
        "engine_backend": "hin_cpu",
        "wavefront_parallelism": 4,
        "precision_shield_mode": "shield",
        "hin_jit_opt_level": 1,
    },
    "wasm": {
        "engine_backend": "hin_wasm",
        "wavefront_parallelism": 1,
        "precision_shield_mode": "ieee",
        "hin_jit_opt_level": 1,
    },
}


def normalize_engine_options(options: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve ``build_profile`` and normalize engine configuration fields.

    Returns a dict with validated ``engine_backend``, ``wavefront_parallelism``,
    ``precision_shield_mode``, and ``hin_jit_opt_level`` keys, applying sensible
    defaults when individual values are missing or invalid.
    """
    profile = str(options.get("build_profile") or "").lower().replace(" ", "_")
    defaults = BUILD_PROFILE_DEFAULTS.get(profile, {})

    raw_backend = options.get("engine_backend") or defaults.get("engine_backend")
    backend = str(raw_backend).lower().replace("-", "_").replace(" ", "_") if raw_backend else None
    if backend == "cpu":
        backend = "hin_cpu"
    elif backend in ("gpu", "cuda", "vulkan"):
        backend = "hin_gpu"
    elif backend in ("wasm", "wasm32"):
        backend = "hin_wasm"
    if backend not in ("hin_cpu", "hin_gpu", "hin_wasm"):
        backend = defaults.get("engine_backend", "hin_cpu")

    raw_par = options.get("wavefront_parallelism", defaults.get("wavefront_parallelism"))
    try:
        parallelism = int(raw_par) if raw_par is not None else defaults.get("wavefront_parallelism", 4)
    except (TypeError, ValueError):
        parallelism = defaults.get("wavefront_parallelism", 4)
    parallelism = max(1, min(16, parallelism))

    raw_shield = options.get("precision_shield_mode") or defaults.get("precision_shield_mode")
    shield = str(raw_shield).lower().replace("-", "_").replace(" ", "_") if raw_shield else None
    if shield == "shield_checks":
        shield = "shield"
    if shield not in ("ieee", "fast_math", "shield"):
        shield = defaults.get("precision_shield_mode", "ieee")

    raw_jit = options.get("hin_jit_opt_level", options.get("jit_optimization_level", defaults.get("hin_jit_opt_level")))
    try:
        jit = int(raw_jit) if raw_jit is not None else defaults.get("hin_jit_opt_level", 0)
    except (TypeError, ValueError):
        jit = defaults.get("hin_jit_opt_level", 0)
    jit = max(0, min(2, jit))

    return {
        "build_profile": profile or "custom",
        "engine_backend": backend,
        "wavefront_parallelism": parallelism,
        "precision_shield_mode": shield,
        "hin_jit_opt_level": jit,
    }


class WorkspaceContext:
    """Provide bundled workspace context for builder feedback and LLM prompts."""

    def __init__(self, workspace_path: Path) -> None:
        self.workspace_path = Path(workspace_path).resolve()

    def bundle(self, max_file_size_kb: int = 100) -> Dict[str, Any]:
        """Return the workspace file bundle."""
        return bundle_workspace(self.workspace_path, max_file_size_kb=max_file_size_kb)

    def format(self, fmt: str = "xml", max_file_size_kb: int = 100) -> str:
        """Return a serialized workspace block suitable for injection into prompts."""
        return format_context_block(self.bundle(max_file_size_kb=max_file_size_kb), fmt=fmt)

    def failure_context(
        self,
        command: str,
        exit_code: int,
        log_text: str,
        diagnosis: Optional[Dict[str, Any]] = None,
        previous_attempts: Optional[list] = None,
    ) -> Dict[str, Any]:
        """Build a structured failure context from the current workspace."""
        return ContextBuilder(self.workspace_path).build_failure_context(
            command, exit_code, log_text, diagnosis, previous_attempts or []
        )


def get_workspace_context(workspace_path: Path) -> WorkspaceContext:
    """Return a context object for *workspace_path*."""
    return WorkspaceContext(workspace_path)
