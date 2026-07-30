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

from aero_forge.blueprint import Blueprint, ContractEntry, ManifestEntry, parse_blueprint, write_blueprint
from aero_forge.builder.aeroc_compiler import compile_directory_to_aeroc
from aero_forge.builder.executor import ExecutionReport
from aero_forge.config import ConfigOverride
from aero_forge.generate import generate_and_build
from aero_forge.monorepo import generate_monorepo
from aero_forge.orchestrator.orchestrator import plan_workspace
from aero_forge.orchestrator.router import toolchains_for_intent
from aero_forge.orchestrator.stack_classifier import (
    INTENT_HYBRID_CPP_PYTHON,
    INTENT_HYBRID_CPP_RUST,
    INTENT_HYBRID_RUST_PYTHON,
    INTENT_PURE_PYTHON,
    INTENT_PURE_RUST,
    INTENT_TRI_POLYGLOT_RUST_CPP_PYTHON,
    StackClassification,
    classify_stack,
    default_manifest_for_architecture,
)
from aero_forge.scaffold.cpp_materializer import CppPolyglotMaterializer
from aero_forge.scaffold.hybrid_cpp_rust_materializer import HybridCppRustMaterializer
from aero_forge.scaffold.polyglot_materializer import PolyglotMaterializer
from aero_forge.scaffold.tri_polyglot_materializer import TriPolyglotMaterializer

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
    # normalize legacy generate_and_build result to include top-level success
    result.setdefault(
        "success", result.get("build", {}).get("success", False) if isinstance(result.get("build"), dict) else False
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


def _is_explicit_native_rust_update(prompt: str) -> bool:
    """Return True when the prompt explicitly requests a PyO3/NumPy/Rayon native extension."""
    if not prompt:
        return False
    lower = prompt.lower()
    return (
        "fn " in lower
        and ("&pyarray" in lower or "pyarray2" in lower or "numpy" in lower)
        and ("rayon" in lower or "pyo3" in lower)
    )


def _augment_blueprint_with_explicit_paths(
    blueprint: Blueprint,
    prompt: str,
    project_name: str,
    features: list[str],
) -> Blueprint:
    """Add Python file paths explicitly requested in *prompt* to the manifest.

    Filters out bare file names (e.g. ``main.py`` mentioned in a ``python main.py``
    command) when the materializer already emits the same file inside a package.
    """
    explicit = _extract_explicit_paths(prompt)
    if not explicit:
        return blueprint
    existing = {e.path for e in blueprint.manifest}
    existing_names = {Path(e.path).name for e in blueprint.manifest}
    existing_package_names = {
        Path(e.path).parts[0]
        for e in blueprint.manifest
        if len(Path(e.path).parts) > 1 and Path(e.path).parts[0] not in {"tests", "src", "scripts", "examples", "docs"}
    }

    additions: List[ManifestEntry] = []
    for p in explicit:
        if p in existing:
            continue
        parts = Path(p).parts
        # If the user wrote ``python main.py`` but the package already emits
        # ``<pkg>/main.py``, don't add a conflicting top-level main.py.
        if len(parts) == 1 and parts[0] in existing_names and existing_package_names:
            continue
        additions.append(ManifestEntry(path=p, lang="python", purpose="user requested"))

    if additions:
        fallback = _hybrid_fallback_blueprint(project_name, features, prompt=prompt)
        # Also pull in default package scaffolding (e.g. __init__.py, cli.py)
        # if the caller asked for a file inside a package that doesn't exist yet.
        for entry in fallback.manifest:
            if entry.path not in existing:
                additions.append(entry)
        blueprint = blueprint.model_copy(update={"manifest": list(blueprint.manifest) + additions})
    return blueprint


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
    is_rust = (
        "rust" in features
        or "cargo" in features
        or "pyo3" in features
        or "rust" in prompt_lower
        or "cargo" in prompt_lower
        or "pyo3" in prompt_lower
    )
    has_python = "python" in prompt_lower or "python" in features
    is_tri = is_cpp and is_rust and has_python
    is_hybrid_cpp_rust = is_cpp and is_rust and not has_python

    if is_tri:
        contracts = [
            ContractEntry(
                name="fast_vector_transform",
                signature="def fast_vector_transform(v: list[float], scalar: float) -> list[float]",
            ),
            ContractEntry(
                name="validate_token",
                signature="def validate_token(token: str) -> bool",
            ),
            ContractEntry(
                name="get_engine_status",
                signature="def get_engine_status() -> dict[str, str]",
            ),
        ]
    elif is_cpp:
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

    if is_tri:
        manifest = [
            ManifestEntry(path="Cargo.toml", lang="toml", purpose="Rust workspace manifest"),
            ManifestEntry(path="rust_core/Cargo.toml", lang="toml", purpose="PyO3 crate manifest"),
            ManifestEntry(path="rust_core/src/lib.rs", lang="rust", purpose="Rust native core"),
            ManifestEntry(path="cpp_core/native.cpp", lang="cpp", purpose="C-ABI shared library source"),
            ManifestEntry(path="pyproject.toml", lang="toml", purpose="Python package manifest"),
            ManifestEntry(path=f"{pkg_name}/__init__.py", lang="python", purpose="Python driver package"),
            ManifestEntry(path=f"{pkg_name}/main.py", lang="python", purpose="Python CLI / REPL entrypoint"),
            ManifestEntry(path="run_shell.py", lang="python", purpose="Headless launcher"),
            ManifestEntry(path="tests/test_tri.py", lang="python", purpose="pytest tests"),
            ManifestEntry(path="README.md", lang="markdown", purpose="docs"),
        ]
        toolchains = ["python", "rust", "cpp", "cargo"]
        architecture = INTENT_TRI_POLYGLOT_RUST_CPP_PYTHON
    elif is_cpp:
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

    if not is_tri:
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
    if not is_tri:
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
    blueprint: Optional[Blueprint] = None,
) -> Dict[str, Any]:
    """Materialize a guaranteed hybrid workspace using the polyglot materializer."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    aero_core = output_dir / ".aero_core"
    if aero_core.is_dir():
        shutil.rmtree(aero_core, ignore_errors=True)
    if blueprint is None:
        blueprint = _hybrid_fallback_blueprint(project_name, features, prompt=prompt)
    else:
        blueprint = _augment_blueprint_with_explicit_paths(
            blueprint, prompt, project_name, features
        )
    if prompt and not blueprint.prompt:
        blueprint = blueprint.model_copy(update={"prompt": prompt})
    if blueprint.architecture == INTENT_TRI_POLYGLOT_RUST_CPP_PYTHON:
        materializer: Any = TriPolyglotMaterializer(output_dir)
        materializer_name = "TriPolyglotMaterializer"
    elif blueprint.architecture == INTENT_HYBRID_CPP_PYTHON:
        materializer = CppPolyglotMaterializer(output_dir)
        materializer_name = "CppPolyglotMaterializer"
    else:
        materializer = PolyglotMaterializer(output_dir)
        materializer_name = "PolyglotMaterializer"
    try:
        updated = materializer.materialize(blueprint, build=True, force_overwrite=True)
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
    pythonpath_parts = [str(output_dir)]
    if (output_dir / "src").is_dir():
        pythonpath_parts.append(str(output_dir / "src"))
    pythonpath_parts.append(test_env.get("PYTHONPATH", ""))
    test_env["PYTHONPATH"] = os.pathsep.join(p for p in pythonpath_parts if p).strip(os.pathsep)
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
    files = ExecutionReport(output_dir).filter_paths(
        sorted(str(p.relative_to(output_dir)) for p in output_dir.rglob("*") if p.is_file())
    )
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


def _run_hybrid_cpp_rust_materializer(
    project_name: str,
    blueprint: Blueprint,
    output_dir: Path,
) -> Dict[str, Any]:
    """Materialize and build a Rust binary that statically links a C++ C-ABI library."""
    from aero_forge.scaffold.hybrid_cpp_rust_materializer import (
        HybridCppRustMaterializer,
    )

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    materializer = HybridCppRustMaterializer(output_dir)
    try:
        updated = materializer.materialize(blueprint, build=True, force_overwrite=True)
    except Exception as exc:
        return {
            "success": False,
            "project_name": project_name,
            "error": str(exc),
            "logs": materializer.build_logs,
            "files": [],
            "materializer": "HybridCppRustMaterializer",
        }
    write_blueprint(updated, output_dir / "blueprint.aero")
    files = ExecutionReport(output_dir).filter_paths(
        sorted(str(p.relative_to(output_dir)) for p in output_dir.rglob("*") if p.is_file())
    )
    return {
        "success": "BUILD: hybrid C++/Rust binary compiled successfully" in materializer.build_logs
        or (
            (output_dir / "target" / "release" / project_name).is_file()
            or (output_dir / "target" / "release" / project_name.replace("-", "_")).is_file()
        ),
        "project_name": project_name,
        "files": files,
        "logs": materializer.build_logs,
        "materializer": "HybridCppRustMaterializer",
    }


def _classification_for_architecture(
    architecture: str, features: List[str]
) -> StackClassification:
    """Create a StackClassification for an explicitly chosen architecture."""
    languages_map = {
        INTENT_PURE_PYTHON: ["python"],
        INTENT_PURE_RUST: ["rust"],
        INTENT_HYBRID_RUST_PYTHON: ["python", "rust"],
        INTENT_HYBRID_CPP_PYTHON: ["python", "cpp"],
        INTENT_HYBRID_CPP_RUST: ["rust", "cpp"],
        INTENT_TRI_POLYGLOT_RUST_CPP_PYTHON: ["python", "rust", "cpp"],
    }
    return StackClassification(
        architecture=architecture,
        toolchains=toolchains_for_intent(architecture),
        languages=languages_map.get(architecture, []),
        features=features,
    )


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
    architecture: Optional[str] = None,
    acceleration_policy: Optional[str] = None,
    workspace_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Classify *prompt*, write ``blueprint.aero``, and build the workspace.

    If *workspace_path* points to an existing workspace, any contracts from an
    existing ``blueprint.aero`` are merged into the planned blueprint so the
    materializer has concrete functions to accelerate.

    Returns a dictionary with ``success``, ``blueprint_path``, ``files``, and
    the underlying build result.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Seed the planner with contracts from an existing workspace blueprint.
    existing_contracts: List[ContractEntry] = []
    existing_manifest: List[ManifestEntry] = []
    if workspace_path:
        existing_path = Path(workspace_path).resolve() / "blueprint.aero"
        if existing_path.is_file():
            try:
                existing = parse_blueprint(existing_path)
                existing_contracts = list(existing.contracts or [])
                existing_manifest = list(existing.manifest or [])
            except Exception as exc:
                logger.warning("Could not read existing workspace blueprint %s: %s", existing_path, exc)

    if progress_callback:
        progress_callback("Planning workspace...")

    # Pass 1: classify and write blueprint.aero.
    classification = classify_stack(prompt)
    if architecture:
        classification = _classification_for_architecture(
            architecture, classification.features
        )
    effective_constraints = constraints or ""
    if acceleration_policy and acceleration_policy != "selective":
        effective_constraints = (
            f"{effective_constraints}\n\nAcceleration policy: {acceleration_policy}"
        ).strip()
    blueprint = plan_workspace(
        prompt,
        output_dir,
        project_name=project_name,
        constraints=effective_constraints or None,
        llm_provider=llm_provider,
        model=model,
        max_retries=max_retries,
        max_tokens=max_tokens,
        config_override=config_override,
        architecture=architecture,
    )

    # If the planned blueprint has no concrete contracts/manifest but an
    # existing workspace blueprint does, merge them so the materializer has
    # real functions to accelerate.
    if not blueprint.contracts and existing_contracts:
        blueprint.contracts = existing_contracts
    if not blueprint.manifest and existing_manifest:
        blueprint.manifest = existing_manifest

    # Force the manifest to the deterministic default for the chosen architecture
    # so emitted files match the planner's declarations and pre-write validation
    # does not fail on LLM-invented paths.
    deterministic_manifest = [
        ManifestEntry(path=e["path"], lang=e["lang"], purpose=e["purpose"])
        for e in default_manifest_for_architecture(blueprint.architecture, project_name or blueprint.project or "generated")
    ]
    blueprint.manifest = deterministic_manifest
    blueprint.module_graph = []

    if progress_callback:
        progress_callback(f"Architecture: {blueprint.architecture}; building...")

    # Pass 2: materialize and verify.
    if blueprint.architecture == INTENT_TRI_POLYGLOT_RUST_CPP_PYTHON:
        result = _run_polyglot_materializer(
            project_name or blueprint.project or "generated",
            classification.features,
            output_dir,
            prompt=prompt,
            blueprint=blueprint,
        )
    elif blueprint.architecture == INTENT_HYBRID_CPP_RUST:
        result = _run_hybrid_cpp_rust_materializer(
            project_name or blueprint.project or "generated",
            blueprint,
            output_dir,
        )
    elif blueprint.architecture == INTENT_HYBRID_CPP_PYTHON:
        # C++/pybind11 builds go straight to the polyglot materializer because
        # generate_monorepo is Rust/PyO3 specific.
        result = _run_polyglot_materializer(
            project_name or blueprint.project or "generated",
            classification.features,
            output_dir,
            prompt=prompt,
            blueprint=blueprint,
        )
    elif blueprint.architecture == INTENT_HYBRID_RUST_PYTHON:
        # Explicit PyO3/NumPy/Rayon native extension updates are handled directly
        # by the polyglot materializer so the concrete requested function is built
        # instead of a generic monorepo stub.
        if _is_explicit_native_rust_update(prompt):
            logger.info("Explicit native Rust update detected; routing to PolyglotMaterializer")
            result = _run_polyglot_materializer(
                project_name or blueprint.project or "generated",
                classification.features,
                output_dir,
                prompt=prompt,
                blueprint=blueprint,
            )
        else:
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
                    blueprint=blueprint,
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
    if not result.get("files"):
        result["files"] = ExecutionReport(output_dir).filter_paths(
            sorted(
                str(p.relative_to(output_dir))
                for p in output_dir.rglob("*")
                if p.is_file()
            )
        )

    # Produce a portable standalone ``workspace.aeroc`` artifact from the materialized tree.
    try:
        compile_directory_to_aeroc(output_dir, output_dir / "workspace.aeroc")
        result["aeroc_path"] = str(output_dir / "workspace.aeroc")
    except Exception as exc:
        logger.warning("Failed to compile workspace.aeroc: %s", exc)

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
