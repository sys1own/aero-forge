"""Context bundler (Reasoning Tier) that synthesizes a deep `blueprint.aero`.

The Context Bundler wraps `LLMBlueprintSynthesizer` and is invoked whenever the
Builder completes a workspace update or when the Copilot detects that the
workspace context is missing, stale, or not yet LLM-initialized.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from aero_forge.blueprint import BlueprintV3
from aero_forge.blueprint.synthesizer import LLMBlueprintSynthesizer

logger = logging.getLogger("aero_forge.context_bundler")


def get_blueprint_status(workspace: Path) -> Dict[str, Any]:
    """Return existence, freshness, and LLM-initialization status of blueprint.aero."""
    workspace = Path(workspace).resolve()
    blueprint_path = workspace / "blueprint.aero"
    exists = blueprint_path.is_file()
    stale = False
    llm_initialized = False
    source_count = 0
    status = "missing"
    generation_method = None

    if not exists:
        return {
            "exists": False,
            "stale": stale,
            "llm_initialized": llm_initialized,
            "source_count": source_count,
            "status": status,
            "generation_method": generation_method,
        }

    try:
        text = blueprint_path.read_text(encoding="utf-8")
        bp = yaml.safe_load(text) or {}
    except Exception:
        bp = {}

    metadata = bp.get("metadata") or {}
    llm_context = bp.get("llm_context") or {}
    auto_generated = bool(metadata.get("auto_generated"))
    llm_initialized = bool(
        not auto_generated
        and (
            llm_context.get("state") == "synthesized"
            or metadata.get("generation_method") == "llm_synthesized"
            or metadata.get("llm_initialized")
        )
    )
    status = metadata.get("status", "unknown")
    generation_method = metadata.get("generation_method")

    skip_names = {
        "target",
        "dist",
        "build",
        "__pycache__",
        ".pytest_cache",
        ".aero",
        ".venv",
        ".git",
    }
    bp_mtime = blueprint_path.stat().st_mtime
    for path in sorted(workspace.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(workspace)
        if rel.name == "blueprint.aero":
            continue
        if any(part.startswith(".") for part in rel.parts):
            continue
        if any(part in skip_names for part in rel.parts[:1]):
            continue
        try:
            if path.stat().st_mtime > bp_mtime:
                stale = True
        except OSError:
            continue
        source_count += 1

    return {
        "exists": exists,
        "stale": stale,
        "llm_initialized": llm_initialized,
        "auto_generated": auto_generated,
        "source_count": source_count,
        "status": status,
        "generation_method": generation_method,
    }


class ContextBundler:
    """Bundle a workspace and synthesize a rich Blueprint v3 using the Reasoning tier."""

    def __init__(
        self,
        llm_provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        config_override: Optional[Any] = None,
    ) -> None:
        self.synthesizer = LLMBlueprintSynthesizer(
            provider=llm_provider,
            model=model,
            api_key=api_key,
            config_override=config_override,
        )

    def synthesize_blueprint(
        self,
        workspace: Path,
        output_path: Optional[Path] = None,
    ) -> BlueprintV3:
        """Synthesize (or re-synthesize) `blueprint.aero` for *workspace*."""
        workspace = Path(workspace).resolve()
        if output_path is None:
            output_path = workspace / "blueprint.aero"
        return self.synthesizer.synthesize(
            workspace,
            output_path=output_path,
        )

    def synthesize_blueprint_async(
        self,
        workspace: Path,
        output_path: Optional[Path] = None,
    ) -> threading.Thread:
        """Start a non-blocking background thread to synthesize `blueprint.aero`.

        Returns the started thread so callers can join it if they need to wait.
        """

        def _run() -> None:
            try:
                self.synthesize_blueprint(workspace, output_path=output_path)
            except Exception as exc:
                logger.warning(
                    "Async blueprint synthesis failed for %s: %s", workspace, exc
                )

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        return thread
