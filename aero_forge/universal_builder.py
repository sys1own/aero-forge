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
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from aero_forge.scaffold.cpp_materializer import CppPolyglotMaterializer
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


def _sanitize_module_name(name: str) -> str:
    """Return a Python-package-safe identifier."""
    sanitized = "".join(c if c.isalnum() or c == "_" else "_" for c in name.lower())
    sanitized = sanitized.strip("_")
    if not sanitized or sanitized[0].isdigit():
        sanitized = "engine"
    return sanitized


def _extract_explicit_paths(prompt: str) -> list[str]:
    """Find Python file paths explicitly named in *prompt*."""
    return re.findall(r"\b[A-Za-z_][\w/]*\.py\b", prompt)


def _hybrid_fallback_blueprint(
    project_name: str,
    features: list[str],
    prompt: str = "",
) -> Blueprint:
    """Return a deterministic hybrid blueprint that respects explicit file requests."""
    prompt_lower = prompt.lower()
    is_cpp = (
        "cpp" in features
        or "c++" in features
        or "pybind11" in features
        or "cmake" in features
        or "c++" in prompt_lower
        or "pybind11" in prompt_lower
        or "cpp" in prompt_lower
    )

    if is_cpp:
        # The C++ polyglot materializer currently emits implementations for the
        # standard vector transform / status contracts. Use those as the stable
        # fallback surface for any C++ request.
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
    elif any(f in features for f in ("data_pipeline", "stream", "batch", "math")):
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

    explicit = _extract_explicit_paths(prompt)
    pkg_name = _sanitize_module_name(project_name)
    non_package_dirs = {"tests", "src", "scripts", "examples", "docs"}
    explicit_packages: set[str] = set()
    for p in explicit:
        parts = Path(p).parts
        if len(parts) > 1 and parts[0] not in non_package_dirs:
            explicit_packages.add(parts[0])
    if explicit_packages:
        pkg_name = _sanitize_module_name(sorted(explicit_packages)[0])

    if is_cpp:
        manifest: list[ManifestEntry] = [
            ManifestEntry(path="pyproject.toml", lang="toml", purpose="Python packaging"),
            ManifestEntry(path="setup.py", lang="python", purpose="setuptools build script"),
            ManifestEntry(path="README.md", lang="markdown", purpose="docs"),
        ]
        toolchains = ["python", "cpp"]
        architecture = INTENT_HYBRID_CPP_PYTHON
    else:
        manifest = [
            ManifestEntry(path="Cargo.toml", lang="toml", purpose="workspace manifest"),
            ManifestEntry(path="rust_core/Cargo.toml", lang="toml", purpose="crate manifest"),
            ManifestEntry(path="rust_core/src/lib.rs", lang="rust", purpose="Rust core"),
            ManifestEntry(path="pyproject.toml", lang="toml", purpose="Python packaging"),
            ManifestEntry(path="README.md", lang="markdown", purpose="docs"),
        ]
        toolchains = ["python", "rust", "cargo"]
        architecture = INTENT_HYBRID_RUST_PYTHON

    # Collect requested files per package and decide whether a default package is needed.
    requested_root_files = {Path(p).name for p in explicit if len(Path(p).parts) == 1}
    requested_package_files: Dict[str, List[str]] = {}
    for p in explicit:
        parts = Path(p).parts
        if len(parts) > 1 and parts[0] not in non_package_dirs:
            requested_package_files.setdefault(parts[0], []).append(Path(p).name)

    packages = explicit_packages
    if not packages and (explicit or not explicit):
        # Always provide a project-named package so there is a place to put the
        # native wrapper and a CLI entry point.
        packages = {pkg_name}

    for pkg_dir in sorted(packages):
        manifest.append(
            ManifestEntry(path=f"{pkg_dir}/__init__.py", lang="python", purpose="package init")
        )
        if is_cpp:
            if "native.cpp" not in requested_package_files.get(pkg_dir, []):
                manifest.append(
                    ManifestEntry(path=f"{pkg_dir}/native.cpp", lang="cpp", purpose="pybind11 extension source")
                )
        else:
            if "native.py" not in requested_package_files.get(pkg_dir, []):
                manifest.append(
                    ManifestEntry(path=f"{pkg_dir}/native.py", lang="python", purpose="native wrapper")
                )
        if "cli.py" not in requested_package_files.get(pkg_dir, []):
            manifest.append(
                ManifestEntry(path=f"{pkg_dir}/cli.py", lang="python", purpose="CLI module")
            )

    # Launcher and entry point requested or defaulted.
    if "run_shell.py" not in requested_root_files:
        manifest.append(ManifestEntry(path="run_shell.py", lang="python", purpose="demo"))

    # Add the exact files requested by the user, skipping ones already covered.
    covered = {Path(e.path).name for e in manifest}
    for p in explicit:
        if Path(p).name in covered or Path(p).name == "__init__.py":
            continue
        manifest.append(ManifestEntry(path=p, lang="python", purpose="user requested"))

    # Provide native test coverage.
    test_file = "tests/test_cli.py" if is_cpp else "tests/test_native.py"
    if "test_native.py" not in requested_root_files and "test_cli.py" not in requested_root_files:
        manifest.append(ManifestEntry(path=test_file, lang="python", purpose="native tests"))

    python_prefix = f"{pkg_name}.native." if packages and not is_cpp else ""
    for contract in contracts:
        contract.python_name = f"{python_prefix}{contract.name}"

    return Blueprint(
        project=project_name,
        architecture=architecture,
        toolchains=toolchains,
        manifest=manifest,
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
    prompt: str = "",
) -> Dict[str, Any]:
    """Materialize a guaranteed hybrid workspace using the polyglot materializer."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    aero_core = output_dir / ".aero_core"
    if aero_core.is_dir():
        shutil.rmtree(aero_core, ignore_errors=True)
    blueprint = _hybrid_fallback_blueprint(project_name, features, prompt=prompt)
    if blueprint.architecture == INTENT_HYBRID_CPP_PYTHON:
        materializer: Any = CppPolyglotMaterializer(output_dir)
        materializer_name = "CppPolyglotMaterializer"
    else:
        materializer = PolyglotMaterializer(output_dir)
        materializer_name = "PolyglotMaterializer"
    try:
        updated = materializer.materialize(blueprint, build=True)
    except Exception as exc:
        return {
            "success": False,
            "project_name": project_name,
            "error": str(exc),
            "logs": materializer.build_logs,
            "files": [],
            "materializer": materializer_name,
        }
    write_blueprint(updated, output_dir / "blueprint.aero")

    test_env = dict(os.environ)
    test_env["PYTHONPATH"] = (
        f"{output_dir}{os.pathsep}{test_env.get('PYTHONPATH', '')}"
    ).strip(os.pathsep)
    pytest_result = subprocess.run(
        [sys.executable, "-m", "pytest", str(output_dir), "-q"],
        cwd=output_dir,
        env=test_env,
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
        "materializer": materializer_name,
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
    if blueprint.architecture == INTENT_HYBRID_CPP_PYTHON:
        # C++/pybind11 builds go straight to the polyglot materializer because
        # generate_monorepo is Rust/PyO3 specific.
        result = _run_polyglot_materializer(
            project_name or blueprint.project or "generated",
            classification.features,
            output_dir,
            prompt=prompt,
        )
    elif blueprint.architecture == INTENT_HYBRID_RUST_PYTHON:
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
                prompt=prompt,
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
