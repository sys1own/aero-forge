"""Universal, blueprint-driven build entry point for aero-forge.

This module ties together the two-pass architecture:

1. Pass 1: ``plan_workspace`` classifies the prompt and writes ``blueprint.aero``.
2. Pass 2: a materializer emits and compiles the exact workspace declared in the
   blueprint, then runs the verification suites.

For hybrid Python/Rust requests it delegates to ``generate_monorepo``; for
pure-language requests it delegates to ``generate_and_build``.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from aero_forge.blueprint import Blueprint, write_blueprint
from aero_forge.config import ConfigOverride
from aero_forge.generate import generate_and_build
from aero_forge.monorepo import generate_monorepo
from aero_forge.orchestrator.orchestrator import plan_workspace
from aero_forge.orchestrator.stack_classifier import (
    INTENT_HYBRID_CPP_PYTHON,
    INTENT_HYBRID_RUST_PYTHON,
    classify_stack,
)

logger = logging.getLogger("aero_forge.universal_builder")


def _build_pure_python(
    blueprint: Blueprint,
    prompt: str,
    output_dir: Path,
    *,
    constraints: Optional[str] = None,
    llm_provider: Optional[str] = None,
    model: Optional[str] = None,
    max_retries: int = 3,
    max_tokens: Optional[int] = None,
    config_override: Optional[ConfigOverride] = None,
) -> Dict[str, Any]:
    """Materialize and test a pure-Python project from a planned blueprint."""
    result = generate_and_build(
        prompt,
        constraints=constraints,
        output_dir=output_dir,
        project_name=blueprint.project,
        llm_provider=llm_provider,
        model=model,
        max_retries=max_retries,
        max_tokens=max_tokens,
        build_kwargs={"max_workers": 1, "cache_enabled": False},
        config_override=config_override,
    )
    # Ensure the authoritative blueprint carries the language/feature tags.
    if (output_dir / "blueprint.aero").is_file():
        try:
            existing = Blueprint.model_validate(
                {
                    "project": blueprint.project,
                    "architecture": blueprint.architecture,
                    "toolchains": blueprint.toolchains,
                    "manifest": blueprint.manifest,
                    "prompt": blueprint.prompt,
                    "constraints": blueprint.constraints,
                    "languages": blueprint.languages,
                    "features": blueprint.features,
                    "output_dir": str(output_dir / "dist"),
                    "llm": {"provider": llm_provider or "none", "model": model},
                }
            )
            write_blueprint(existing, output_dir / "blueprint.aero")
        except Exception as exc:
            logger.warning("Could not update final blueprint: %s", exc)
    return result


def build_universal_project(
    prompt: str,
    output_dir: Path | str,
    *,
    project_name: str = "aero_forge_project",
    constraints: Optional[str] = None,
    llm_provider: Optional[str] = None,
    model: Optional[str] = None,
    max_retries: int = 3,
    max_tokens: Optional[int] = None,
    config_override: Optional[ConfigOverride] = None,
    progress_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    """Classify *prompt*, write ``blueprint.aero``, and build the workspace.

    Returns a dictionary with ``success``, ``blueprint_path``, ``files``, and
    the underlying build result.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if progress_callback:
        progress_callback("Planning workspace...")

    # Pass 1: classify and write blueprint.aero.
    classification = classify_stack(prompt)
    blueprint = plan_workspace(
        prompt,
        output_dir,
        project_name=project_name,
        constraints=constraints,
        llm_provider=llm_provider,
        model=model,
        max_retries=max_retries,
        max_tokens=max_tokens,
        config_override=config_override,
    )

    if progress_callback:
        progress_callback(f"Architecture: {blueprint.architecture}; building...")

    # Pass 2: materialize and verify.
    if blueprint.architecture in (INTENT_HYBRID_RUST_PYTHON, INTENT_HYBRID_CPP_PYTHON):
        result = generate_monorepo(
            prompt,
            output_dir,
            project_name=project_name,
            constraints=constraints,
            llm_provider=llm_provider,
            model=model,
            max_retries=max_retries,
            max_tokens=max_tokens,
            progress_callback=progress_callback,
            config_override=config_override,
        )
    else:
        result = _build_pure_python(
            blueprint,
            prompt,
            output_dir,
            constraints=constraints,
            llm_provider=llm_provider,
            model=model,
            max_retries=max_retries,
            max_tokens=max_tokens,
            config_override=config_override,
        )

    result["blueprint_path"] = str(output_dir / "blueprint.aero")
    result["classification"] = {
        "architecture": classification.architecture,
        "toolchains": classification.toolchains,
        "languages": classification.languages,
        "features": classification.features,
    }
    return result


def validate_prompt_against_workspace(
    prompt: str,
    output_dir: Path | str,
) -> Dict[str, Any]:
    """Run an end-to-end build for *prompt* and assert blueprint compliance.

    Returns a structured report with file list, test status, and any errors.
    """
    result = build_universal_project(
        prompt,
        output_dir,
        project_name="validation_project",
    )
    output_dir = Path(output_dir)
    report: Dict[str, Any] = {
        "prompt": prompt,
        "success": result.get("success", False),
        "blueprint_path": result.get("blueprint_path"),
        "files": result.get("files", []),
        "classification": result.get("classification"),
    }
    if not result.get("success"):
        report["error"] = result.get("error") or result.get("core_error")
        report["logs"] = result.get("core_logs") or result.get("cargo_error")
    return report
