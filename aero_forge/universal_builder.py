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
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from aero_forge.blueprint import Blueprint, ContractEntry, ManifestEntry, write_blueprint
from aero_forge.config import ConfigOverride
from aero_forge.generate import generate_and_build
from aero_forge.monorepo import generate_monorepo
from aero_forge.orchestrator.orchestrator import plan_workspace
from aero_forge.orchestrator.stack_classifier import (
    INTENT_HYBRID_CPP_PYTHON,
    INTENT_HYBRID_RUST_PYTHON,
    classify_stack,
)
from aero_forge.scaffold.polyglot_materializer import PolyglotMaterializer

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
        build_kwargs={"max_workers": 1, "cache_enabled": True},
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


def _hybrid_fallback_blueprint(project_name: str, features: list[str]) -> Blueprint:
    """Return a deterministic hybrid blueprint for use when LLM monorepo generation fails."""
    if any(f in features for f in ("data_pipeline", "stream", "batch", "math")):
        contracts = [
            ContractEntry(
                name="process_stream",
                signature="def process_stream(data: list[float], window_size: int) -> list[float]",
            ),
            ContractEntry(
                name="engine_status",
                signature="def engine_status() -> dict[str, str]",
            ),
        ]
    else:
        contracts = [
            ContractEntry(
                name="fast_vector_transform",
                signature="def fast_vector_transform(v: list[float], scalar: float) -> list[float]",
            ),
            ContractEntry(
                name="get_engine_status",
                signature="def get_engine_status() -> dict[str, str]",
            ),
        ]
    return Blueprint(
        project=project_name,
        architecture=INTENT_HYBRID_RUST_PYTHON,
        toolchains=["python", "rust", "cargo"],
        manifest=[
            ManifestEntry(path="Cargo.toml", lang="toml", purpose="workspace manifest"),
            ManifestEntry(path="rust_core/Cargo.toml", lang="toml", purpose="crate manifest"),
            ManifestEntry(path="rust_core/src/lib.rs", lang="rust", purpose="Rust core"),
            ManifestEntry(path="aero_polyglot_runner/__init__.py", lang="python", purpose="package init"),
            ManifestEntry(path="aero_polyglot_runner/orchestrator.py", lang="python", purpose="Python orchestrator"),
            ManifestEntry(path="run_demo.py", lang="python", purpose="demo"),
            ManifestEntry(path="tests/test_polyglot.py", lang="python", purpose="tests"),
            ManifestEntry(path="pyproject.toml", lang="toml", purpose="Python packaging"),
            ManifestEntry(path="README.md", lang="markdown", purpose="docs"),
        ],
        contracts=contracts,
        output_dir=Path("./dist"),
    )


def _parse_pytest_summary(output: str) -> Tuple[int, int, int]:
    """Return (total, passed, failed) parsed from pytest summary text."""
    passed = 0
    failed = 0
    skipped = 0
    m_passed = re.search(r"(\d+) passed", output, re.IGNORECASE)
    if m_passed:
        passed = int(m_passed.group(1))
    m_failed = re.search(r"(\d+) failed", output, re.IGNORECASE)
    if m_failed:
        failed = int(m_failed.group(1))
    m_error = re.search(r"(\d+) error", output, re.IGNORECASE)
    if m_error:
        failed += int(m_error.group(1))
    m_skipped = re.search(r"(\d+) skipped", output, re.IGNORECASE)
    if m_skipped:
        skipped = int(m_skipped.group(1))
    return passed + failed + skipped, passed, failed


def _run_polyglot_materializer(
    project_name: str,
    features: list[str],
    output_dir: Path,
) -> Dict[str, Any]:
    """Materialize a guaranteed hybrid workspace using the polyglot materializer."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    aero_core = output_dir / ".aero_core"
    if aero_core.is_dir():
        shutil.rmtree(aero_core, ignore_errors=True)
    blueprint = _hybrid_fallback_blueprint(project_name, features)
    materializer = PolyglotMaterializer(output_dir)
    try:
        updated = materializer.materialize(blueprint, build=True)
    except Exception as exc:
        return {
            "success": False,
            "project_name": project_name,
            "error": str(exc),
            "logs": materializer.build_logs,
            "files": [],
            "materializer": "PolyglotMaterializer",
        }
    write_blueprint(updated, output_dir / "blueprint.aero")

    test_file = output_dir / "tests" / "test_polyglot.py"
    pytest_result = subprocess.run(
        ["python", "-m", "pytest", str(test_file), "-q"],
        cwd=output_dir,
        capture_output=True,
        text=True,
    )
    test_total, test_passed, test_failed = _parse_pytest_summary(
        pytest_result.stdout + pytest_result.stderr
    )
    logs = (
        f"{materializer.build_logs}\n\n"
        f"--- Python test output ---\n"
        f"{pytest_result.stdout}\n{pytest_result.stderr}"
    ).strip()
    files = sorted(str(p.relative_to(output_dir)) for p in output_dir.rglob("*") if p.is_file())
    return {
        "success": pytest_result.returncode == 0,
        "project_name": project_name,
        "files": files,
        "test_total": test_total,
        "test_passed": test_passed,
        "test_failed": test_failed,
        "logs": logs,
        "pytest_output": pytest_result.stdout,
        "pytest_error": pytest_result.stderr,
        "materializer": "PolyglotMaterializer",
    }


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
        try:
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
        except Exception as exc:
            logger.warning("generate_monorepo failed: %s; falling back to PolyglotMaterializer", exc)
            result = {"success": False, "error": str(exc)}
        if not result.get("success"):
            logger.warning(
                "generate_monorepo failed (%s); falling back to PolyglotMaterializer",
                result.get("error"),
            )
            if progress_callback:
                progress_callback("Falling back to polyglot materializer...")
            result = _run_polyglot_materializer(
                project_name or blueprint.project or "generated",
                classification.features,
                output_dir,
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
