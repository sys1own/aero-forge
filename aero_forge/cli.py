"""Aero-Forge CLI entrypoint."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import click

from .blueprint import (
    FunctionSpec,
    discover_functions,
    discover_project,
    generate_blueprint,
    parse_blueprint,
    write_blueprint,
)
from .build_runner import BuildRunner
from .build_summary import format_build_summary
from .error_explainer import explain_error
from .errors import UserError
from .examples import create_example, list_examples, run_example
from .orchestrator.orchestrator import ForgeError, Orchestrator


def _setup_logging(verbose: bool, json_output: bool = False) -> None:
    if verbose:
        level = logging.DEBUG
    elif json_output:
        level = logging.CRITICAL
    else:
        level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    logging.getLogger().setLevel(level)


def _resolve_llm_provider(
    llm_provider: str | None, no_llm: bool, llm_fix: bool = False
) -> str | None:
    if no_llm:
        return "none"
    if not llm_provider:
        llm_provider = os.getenv("AERO_FORGE_LLM_PROVIDER")
    if llm_fix and not llm_provider:
        raise UserError(
            "--llm-fix requires an LLM provider. "
            "Set AERO_FORGE_LLM_PROVIDER or use --llm-provider."
        )
    if not llm_provider:
        click.echo(
            "No LLM provider configured (set AERO_FORGE_LLM_PROVIDER or use --llm-provider); "
            "running in router-only mode.",
            err=True,
        )
        return "none"
    return llm_provider


def _emit_json_or_text(
    json_obj: Any, text: str, *, json_output: bool, err: bool = False
) -> None:
    """Print ``json_obj`` as JSON or ``text`` depending on ``json_output``."""
    if json_output:
        click.echo(json.dumps(json_obj, default=str))
    else:
        click.echo(text, err=err)


def _output_generate_json(
    result: dict[str, Any],
    prompt: str,
    elapsed: float,
    summary: str,
) -> None:
    """Emit the final JSON payload for ``aero-forge generate --json``."""
    build = result.get("build") or {}
    output: dict[str, Any] = {
        "success": build.get("success", False) if build else True,
        "prompt": prompt,
        "source_path": result.get("source_path"),
        "test_path": result.get("test_path"),
        "blueprint_path": result.get("blueprint_path"),
        "elapsed_seconds": round(elapsed, 3),
        "summary": summary,
    }
    if build:
        output["build"] = {
            "success": build.get("success", False),
            "passed": build.get("passed", 0),
            "total": build.get("total", 0),
            "output_dir": build.get("output_dir", ""),
        }
    click.echo(json.dumps(output, default=str))


@click.group()
@click.version_option(
    version=__import__("aero_forge").__version__, prog_name="aero-forge"
)
def main() -> None:
    """Aero-Forge: transpile, compile, and heal Python functions."""


@main.command()
@click.argument("file", type=click.Path(exists=True, dir_okay=False, path_type=str))
@click.option(
    "--function",
    "-f",
    required=True,
    help="Name of the function to fix/compile.",
)
@click.option(
    "--test-file",
    "-t",
    type=click.Path(dir_okay=False, path_type=str),
    default=None,
    help="Path to the test file (default: test_<file>.py in the same directory).",
)
@click.option(
    "--llm-provider",
    default=None,
    help="LLM provider: openai, openrouter, deepseek, gemini, or none (default: config/env).",
)
@click.option(
    "--model",
    default=None,
    help="Model name to use (default depends on provider).",
)
@click.option(
    "--max-iterations",
    "-i",
    type=int,
    default=None,
    help="Maximum number of fix iterations.",
)
@click.option(
    "--max-retries",
    type=int,
    default=None,
    help="Retries per LLM call.",
)
@click.option(
    "--no-llm",
    is_flag=True,
    help="Run the accelerator without LLM-based healing.",
)
@click.option(
    "--llm-fix",
    is_flag=True,
    help="Use an LLM to explain and auto-fix failures.",
)
@click.option(
    "--no-cache",
    is_flag=True,
    help="Disable the fix cache.",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Print full compiler/test output and debug logs.",
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    help="Output structured JSON instead of human-readable text.",
)
def fix(
    file: str,
    function: str,
    test_file: str | None,
    llm_provider: str | None,
    model: str | None,
    max_iterations: int | None,
    max_retries: int | None,
    no_llm: bool,
    llm_fix: bool,
    no_cache: bool,
    verbose: bool,
    json_output: bool,
) -> None:
    """Compile and test FILE's FUNCTION, healing failures automatically."""
    _setup_logging(verbose, json_output)

    start = time.perf_counter()
    llm_provider = _resolve_llm_provider(llm_provider, no_llm, llm_fix=llm_fix)

    def _error_json(error: str, error_type: str = "build_error") -> dict[str, Any]:
        return {
            "status": "failure",
            "execution_time_ms": round((time.perf_counter() - start) * 1000, 2),
            "error": {
                "type": error_type,
                "message": error,
                "details": "",
                "suggestion": "Add type hints, simplify the construct, or run with --llm-fix.",
            },
            "files_generated": [],
            "rust_extensions": [],
        }

    try:
        orchestrator = Orchestrator(
            source_path=file,
            function_name=function,
            test_path=test_file,
            max_iterations=max_iterations,
            llm_provider=llm_provider,
            model=model,
            max_retries=max_retries,
            cache_enabled=not no_cache,
        )
    except (UserError, ImportError) as exc:
        payload = _error_json(str(exc), error_type="invalid_input")
        if json_output:
            click.echo(json.dumps(payload))
        else:
            click.echo(str(exc), err=True)
        sys.exit(1)

    try:
        result = orchestrator.run()
    except UserError as exc:
        payload = _error_json(str(exc), error_type="unsupported_construct")
        if json_output:
            click.echo(json.dumps(payload))
        else:
            click.echo(str(exc), err=True)
        sys.exit(1)
    except ForgeError as exc:
        payload = _error_json(str(exc), error_type="build_error")
        if json_output:
            click.echo(json.dumps(payload))
        else:
            click.echo(f"Forge failed: {exc}", err=True)
        sys.exit(1)
    except Exception as exc:
        payload = _error_json(str(exc), error_type="unexpected_error")
        if json_output:
            click.echo(json.dumps(payload))
        else:
            click.echo(f"Unexpected error: {exc}", err=True)
        sys.exit(1)

    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    artifact = result.get("artifact")
    source_file = str(Path(file).resolve())
    extensions = [str(Path(artifact).resolve())] if artifact else []

    if result.get("success"):
        payload: dict[str, Any] = {
            "status": "success",
            "execution_time_ms": elapsed_ms,
            "files_generated": [source_file],
            "rust_extensions": extensions,
            "iterations": result.get("iterations"),
            "error": None,
        }
        if json_output:
            click.echo(json.dumps(payload))
        else:
            if verbose:
                click.echo(result.get("logs", ""))
            click.echo(
                f"Success after {result['iterations']} iteration(s). Native extension built."
            )
    else:
        error_text = result.get("error", "unknown")
        payload = {
            "status": "failure",
            "execution_time_ms": elapsed_ms,
            "files_generated": [source_file],
            "rust_extensions": extensions,
            "iterations": result.get("iterations"),
            "error": {
                "type": "build_error",
                "message": error_text,
                "details": result.get("logs", ""),
                "suggestion": "Run with --llm-fix to let the LLM repair the code.",
            },
        }
        if json_output:
            click.echo(json.dumps(payload))
        else:
            if verbose:
                click.echo(result.get("logs", ""))
            click.echo(
                f"Partial success after {result['iterations']} iteration(s). {error_text}",
                err=True,
            )
            if artifact:
                click.echo(f"Best compiled artifact: {artifact}", err=True)
        sys.exit(1)


@main.command()
@click.argument(
    "blueprint",
    required=False,
    type=click.Path(dir_okay=False, path_type=str),
    default="blueprint.aero",
)
@click.option(
    "--auto",
    "auto_file",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    default=None,
    help="Discover all public functions in FILE and build them.",
)
@click.option(
    "--auto-detect",
    "auto_detect",
    is_flag=True,
    help="Auto-detect project structure (src/ and tests/) and build.",
)
@click.option(
    "--project",
    "project_dir",
    type=click.Path(file_okay=False, exists=True, path_type=str),
    default=None,
    help="Build every public function in a project directory and bundle the result.",
)
@click.option(
    "--upload",
    "upload_zip",
    type=click.Path(dir_okay=False, exists=True, path_type=str),
    default=None,
    help="Extract a zip archive, build the project, and produce an output zip.",
)
@click.option(
    "--output-zip",
    type=click.Path(dir_okay=False, path_type=str),
    default=None,
    help="Path for the resulting bundled zip (used with --project or --upload).",
)
@click.option(
    "--llm-provider",
    default=None,
    help="LLM provider: openai, openrouter, deepseek, gemini, or none.",
)
@click.option("--model", default=None, help="Model name to use.")
@click.option(
    "--max-iterations",
    "-i",
    type=int,
    default=None,
    help="Maximum fix iterations per function.",
)
@click.option(
    "--max-retries",
    type=int,
    default=None,
    help="Retries per LLM call.",
)
@click.option(
    "--no-llm",
    is_flag=True,
    help="Run without LLM-based healing.",
)
@click.option(
    "--llm-fix",
    is_flag=True,
    help="Use an LLM to explain and auto-fix failures.",
)
@click.option(
    "--no-cache",
    is_flag=True,
    help="Disable the build cache.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Ignore the build cache and force a recompile.",
)
@click.option(
    "--cache-dir",
    type=click.Path(file_okay=False, path_type=str),
    default=None,
    envvar="AERO_FORGE_CACHE_DIR",
    help="Directory for the build cache (default: ~/.cache/aero-forge/build_cache).",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(file_okay=False, path_type=str),
    default=None,
    help="Override output directory.",
)
@click.option(
    "--jobs",
    "-j",
    type=int,
    default=None,
    help="Parallel build jobs (default: min(4, functions)).",
)
@click.option(
    "--distribute",
    is_flag=True,
    help="Use process-based parallelism for local distributed builds.",
)
@click.option(
    "--workers",
    type=int,
    default=None,
    help="Number of parallel workers for distributed builds (default: --jobs or min(4, functions)).",
)
@click.option(
    "--write-blueprint",
    "write_blueprint_flag",
    is_flag=True,
    help="When using --auto, write a generated blueprint.aero next to the file.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Preview what would be built without compiling.",
)
@click.option(
    "--target",
    type=click.STRING,
    default="native",
    help="Build target triple (default: native; e.g. wasm32-unknown-unknown, x86_64-unknown-linux-musl).",
)
@click.option(
    "--gpu",
    is_flag=True,
    help="Attempt GPU acceleration for functions annotated with # @accelerate gpu.",
)
@click.option(
    "--prompt-template",
    type=click.Choice(
        [
            "v1_minimal",
            "v2_structured",
            "v3_algorithm",
            "v4_performance",
            "v5_balanced",
            "v6_creative",
            "v7_conservative",
            "v8_iterative",
            "v9_transpiler_friendly",
            "v10_correctness_focused",
        ],
        case_sensitive=False,
    ),
    default="v5_balanced",
    help="System prompt template for generated blueprints (default: v5_balanced).",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Show debug logs and full output.",
)
@click.option(
    "--progress",
    is_flag=True,
    help="Show a real-time progress bar during the build.",
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    help="Output structured JSON instead of human-readable text.",
)
def build(
    blueprint: str,
    auto_file: str | None,
    auto_detect: bool,
    project_dir: str | None,
    upload_zip: str | None,
    output_zip: str | None,
    llm_provider: str | None,
    model: str | None,
    max_iterations: int | None,
    max_retries: int | None,
    no_llm: bool,
    llm_fix: bool,
    no_cache: bool,
    force: bool,
    cache_dir: str | None,
    output_dir: str | None,
    jobs: int | None,
    distribute: bool,
    workers: int | None,
    write_blueprint_flag: bool,
    dry_run: bool,
    target: str,
    gpu: bool,
    verbose: bool,
    prompt_template: str,
    progress: bool,
    json_output: bool,
) -> None:
    """Build all functions described by BLUEPRINT (default: blueprint.aero)."""
    _setup_logging(verbose, json_output)
    start = time.perf_counter()

    if project_dir or upload_zip:
        from .project_builder import ProjectBuilder, build_from_upload

        if upload_zip:
            result = build_from_upload(
                Path(upload_zip),
                output_zip=Path(output_zip) if output_zip else None,
                llm_provider=llm_provider,
                model=model,
                max_workers=workers or jobs or 4,
                cache_enabled=not no_cache,
                target=target,
            )
        else:
            builder = ProjectBuilder(
                Path(project_dir),
                output_zip=Path(output_zip) if output_zip else None,
                output_dir=Path(output_dir) if output_dir else None,
                llm_provider=llm_provider,
                model=model,
                max_workers=workers or jobs or 4,
                cache_enabled=not no_cache,
                target=target,
            )
            result = builder.build()

        elapsed = time.perf_counter() - start
        result["total_time_seconds"] = round(elapsed, 3)

        if json_output:
            click.echo(json.dumps(result, default=str))
        else:
            click.echo(f"Status: {result['status']}")
            click.echo(
                f"Functions compiled: {', '.join(result['functions_compiled']) or 'none'}"
            )
            click.echo(f"Tests passed: {result['passed']}/{result['total']}")
            click.echo(f"Output zip: {result['output_zip']}")
            if result.get("summary"):
                click.echo(f"\n{result['summary']}")

        if not result.get("success"):
            sys.exit(1)
        return

    try:
        if auto_file:
            bp = _blueprint_from_auto(Path(auto_file), output_dir)
            if write_blueprint_flag:
                blueprint_path = Path(auto_file).parent / "blueprint.aero"
                write_blueprint(bp, blueprint_path)
                click.echo(f"Wrote blueprint: {blueprint_path}")
        elif auto_detect:
            root = (
                Path(blueprint).parent
                if blueprint and blueprint != "blueprint.aero"
                else Path(".")
            )
            bp = _blueprint_from_auto_detect(root, output_dir)
            if write_blueprint_flag:
                blueprint_path = root / "blueprint.aero"
                write_blueprint(bp, blueprint_path)
                click.echo(f"Wrote blueprint: {blueprint_path}")
        else:
            bp = parse_blueprint(Path(blueprint))

        if bp.prompt:
            from .generate import generate_project

            if not llm_provider:
                llm_provider = os.getenv("AERO_FORGE_LLM_PROVIDER") or bp.llm.provider
            if not model:
                model = os.getenv("AERO_FORGE_MODEL") or bp.llm.model
            _, _, bp, _, _, _ = generate_project(
                bp.prompt,
                constraints=bp.constraints,
                output_dir=Path(output_dir) if output_dir else bp.output_dir,
                project_name=bp.project,
                llm_provider=llm_provider,
                model=model,
                prompt_template=prompt_template,
            )
    except (UserError, ValueError) as exc:
        _emit_json_or_text(
            {"success": False, "error": f"Invalid blueprint: {exc}"},
            f"Invalid blueprint: {exc}",
            json_output=json_output,
            err=True,
        )
        sys.exit(1)
    except ImportError as exc:
        _emit_json_or_text(
            {"success": False, "error": str(exc)},
            str(exc),
            json_output=json_output,
            err=True,
        )
        sys.exit(1)

    if output_dir:
        bp.output_dir = Path(output_dir)

    try:
        llm_provider = _resolve_llm_provider(llm_provider, no_llm, llm_fix=llm_fix)
    except UserError as exc:
        click.echo(str(exc), err=True)
        sys.exit(1)
    if not model:
        model = os.getenv("AERO_FORGE_MODEL") or bp.llm.model

    runner = BuildRunner(
        blueprint=bp,
        max_workers=workers or jobs or min(4, len(bp.functions) or 1),
        llm_provider=llm_provider,
        model=model,
        max_iterations=max_iterations,
        max_retries=max_retries,
        cache_enabled=not no_cache,
        cache_dir=Path(cache_dir) if cache_dir else None,
        force=force,
        gpu=gpu,
        target=target,
        distributed=distribute,
        dry_run=dry_run,
        progress=progress,
    )

    try:
        result = runner.build()
    except UserError as exc:
        _emit_json_or_text(
            {"success": False, "error": str(exc)},
            str(exc),
            json_output=json_output,
            err=True,
        )
        sys.exit(1)
    except Exception as exc:
        _emit_json_or_text(
            {"success": False, "error": f"Unexpected error: {exc}"},
            f"Unexpected error: {exc}",
            json_output=json_output,
            err=True,
        )
        sys.exit(1)

    if verbose:
        for item in result.get("results", []):
            click.echo(
                f"{', '.join(item['functions'])}: "
                f"{'OK' if item['success'] else 'FAIL'} "
                f"({item['iterations']} iterations)"
            )

    elapsed = time.perf_counter() - start

    if result.get("dry_run"):
        if json_output:
            click.echo(
                json.dumps({"success": True, "dry_run": True, **result}, default=str)
            )
        else:
            click.echo(
                f"Dry-run complete: {result['total']} function(s) would be built "
                f"into {result['output_dir']}"
            )
        return

    if result.get("success"):
        summary = format_build_summary(
            result,
            output_dir=Path(result.get("output_dir", ".")),
            llm_provider=llm_provider,
            model=model,
        )
        if json_output:
            click.echo(
                json.dumps(
                    {
                        "success": True,
                        "summary": summary,
                        "elapsed_seconds": round(elapsed, 3),
                        **result,
                    },
                    default=str,
                )
            )
        else:
            click.echo(f"\n{summary}")
    else:
        if json_output:
            click.echo(json.dumps({"success": False, **result}, default=str))
        else:
            click.echo(
                f"Build complete: {result['passed']}/{result['total']} succeeded. "
                f"Output directory: {result['output_dir']}"
            )
        sys.exit(1)


@main.command("generate")
@click.option(
    "--prompt",
    "-p",
    default=None,
    help="Natural language description of the function to generate.",
)
@click.option(
    "--prompt-file",
    "-P",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    default=None,
    help="Path to a file containing the prompt.",
)
@click.option(
    "--constraints",
    "-c",
    default=None,
    help="Optional constraints for the generated code (e.g. 'iterative only').",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(file_okay=False, writable=True, path_type=str),
    default=".",
    help="Directory where src/generated.py and tests/test_generated.py will be written.",
)
@click.option(
    "--project-dir",
    "--project",
    "project_dir",
    type=click.Path(file_okay=False, exists=True, path_type=str),
    default=None,
    help="Generate code into an existing project directory and rebuild it.",
)
@click.option(
    "--output-zip",
    type=click.Path(dir_okay=False, path_type=str),
    default=None,
    help="Path for the bundled output zip (used with --project).",
)
@click.option(
    "--project-name",
    "project_name",
    default="generated_project",
    help="Project name to use in the generated blueprint.",
)
@click.option(
    "--llm-provider",
    default=None,
    help="LLM provider: openai, openrouter, deepseek, gemini, or none (default: config/env).",
)
@click.option(
    "--model",
    default=None,
    help="Model name to use (default depends on provider).",
)
@click.option(
    "--build",
    "do_build",
    is_flag=True,
    help="Run aero-forge build after generating the code.",
)
@click.option(
    "--optimize",
    is_flag=True,
    help="Run an iterative optimization loop after the initial build.",
)
@click.option(
    "--max-iterations",
    type=int,
    default=5,
    help="Maximum optimization iterations (default: 5).",
)
@click.option(
    "--no-llm",
    is_flag=True,
    help="Skip LLM generation and only write stubs (not useful with --prompt).",
)
@click.option(
    "--prompt-template",
    type=click.Choice(
        [
            "v1_minimal",
            "v2_structured",
            "v3_algorithm",
            "v4_performance",
            "v5_balanced",
            "v6_creative",
            "v7_conservative",
            "v8_iterative",
            "v9_transpiler_friendly",
            "v10_correctness_focused",
        ],
        case_sensitive=False,
    ),
    default="v5_balanced",
    help="System prompt template for generation (default: v5_balanced).",
)
@click.option(
    "--algorithm-library",
    is_flag=True,
    help="Select and adapt from the built-in algorithm library instead of free-form generation.",
)
@click.option(
    "--selected-algorithm",
    default=None,
    help="Use a specific library algorithm by name (requires --algorithm-library or overrides selection).",
)
@click.option(
    "--variants",
    "-v",
    type=int,
    default=1,
    help="Generate N variants, compile each, and select the best (default: 1).",
)
@click.option(
    "--discover",
    is_flag=True,
    help="Allow the LLM to design a new algorithm when no library match exists.",
)
@click.option(
    "--explain",
    is_flag=True,
    help="Request and display an explanation of the chosen algorithm and tradeoffs.",
)
@click.option(
    "--review",
    is_flag=True,
    help="Run an LLM self-review step on the generated code before compilation.",
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    help="Output structured JSON instead of human-readable text.",
)
@click.option(
    "--stream",
    "stream_progress",
    is_flag=True,
    help="Emit NDJSON progress events during generation/build.",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Show debug logs.",
)
def generate(
    prompt: str | None,
    prompt_file: str | None,
    constraints: str | None,
    output_dir: str,
    project_dir: str | None,
    output_zip: str | None,
    project_name: str,
    llm_provider: str | None,
    model: str | None,
    do_build: bool,
    optimize: bool,
    max_iterations: int,
    no_llm: bool,
    verbose: bool,
    prompt_template: str,
    algorithm_library: bool,
    selected_algorithm: str | None,
    variants: int,
    discover: bool,
    explain: bool,
    review: bool,
    json_output: bool,
    stream_progress: bool,
) -> None:
    """Generate Python code and tests from a natural language prompt."""
    _setup_logging(verbose, json_output)
    start = time.perf_counter()

    if no_llm:
        llm_provider = "none"

    if prompt_file:
        prompt = Path(prompt_file).read_text(encoding="utf-8").strip()
    if not prompt:
        click.echo("Error: --prompt or --prompt-file is required.", err=True)
        sys.exit(1)

    def _progress(message: str) -> None:
        if stream_progress or json_output:
            click.echo(
                json.dumps({"type": "progress", "message": message}, default=str)
            )

    if project_dir:
        try:
            from .project_builder import ProjectBuilder

            builder = ProjectBuilder(
                Path(project_dir),
                output_zip=Path(output_zip) if output_zip else None,
                llm_provider=llm_provider,
                model=model,
                max_workers=1,
                cache_enabled=False,
            )
            result = builder.generate_and_build(
                prompt,
                constraints=constraints,
                prompt_template=prompt_template,
                output_name="generated",
            )
            _progress("Done")
            elapsed = time.perf_counter() - start
            result["total_time_seconds"] = round(elapsed, 3)

            if json_output:
                click.echo(json.dumps(result, default=str))
            else:
                click.echo(f"Generated into project: {project_dir}")
                click.echo(f"Status: {result['status']}")
                click.echo(
                    f"Functions compiled: {', '.join(result['functions_compiled']) or 'none'}"
                )
                click.echo(f"Tests passed: {result['passed']}/{result['total']}")
                click.echo(f"Output zip: {result['output_zip']}")
                if result.get("summary"):
                    click.echo(f"\n{result['summary']}")

            if not result.get("success"):
                sys.exit(1)
            return
        except Exception as exc:
            _emit_json_or_text(
                {"success": False, "error": str(exc), "prompt": prompt},
                f"Project generation failed: {exc}",
                json_output=json_output,
            )
            sys.exit(1)

    try:
        from .generate import generate_and_build

        result = generate_and_build(
            prompt,
            constraints=constraints,
            output_dir=Path(output_dir),
            project_name=project_name,
            llm_provider=llm_provider,
            model=model,
            max_iterations=max_iterations,
            optimize=optimize,
            prompt_template=prompt_template,
            algorithm_library=algorithm_library,
            selected_algorithm=selected_algorithm,
            variants=variants,
            discover=discover,
            explain=explain,
            review=review,
            build_kwargs=(
                {"max_workers": 1, "cache_enabled": False} if do_build else None
            ),
            progress_callback=_progress,
        )
    except Exception as exc:
        _emit_json_or_text(
            {"success": False, "error": str(exc), "prompt": prompt},
            f"Generation failed: {exc}",
            json_output=json_output,
        )
        sys.exit(1)
    elapsed = time.perf_counter() - start

    summary = ""
    if result.get("build"):
        build_result = result["build"]
        if build_result.get("success"):
            summary = format_build_summary(
                build_result,
                output_dir=Path(result.get("blueprint_path", output_dir)).parent
                / "dist",
                prompt=prompt,
                llm_provider=llm_provider,
                model=model,
            )

    if json_output:
        _output_generate_json(result, prompt, elapsed, summary)
        return

    click.echo(f"Generated: {result['source_path']}")
    click.echo(f"Tests:     {result['test_path']}")
    click.echo(f"Blueprint: {result['blueprint_path']}")

    if explain and result.get("explanation"):
        click.echo("\nExplanation:\n" + result["explanation"])

    if variants > 1 and "variants" in result:
        click.echo(f"Variants generated: {len(result['variants'])}")
        for vr in result["variants"]:
            build = vr.get("build") or {}
            click.echo(
                f"  variant {vr['variant']}: "
                f"{build.get('passed', 0)}/{build.get('total', 0)} passed "
                f"({vr.get('elapsed_seconds', 0.0):.3f}s)"
            )

    if result.get("build"):
        build_result = result["build"]
        if build_result.get("success"):
            click.echo(f"\n{summary}")
        else:
            click.echo(
                f"Build: {build_result.get('passed', 0)}/{build_result.get('total', 0)} "
                f"failed ({build_result.get('output_dir', '')})"
            )
            sys.exit(1)


def _blueprint_from_auto(
    source: Path,
    output_dir: str | None,
) -> Any:
    from .blueprint import Blueprint

    functions = discover_functions(source)
    if not functions:
        raise ValueError(f"No public functions found in {source}")

    out = Path(output_dir) if output_dir else Path("./dist")
    return generate_blueprint(
        project=source.stem,
        functions=functions,
        output_dir=out,
    )


def _blueprint_from_auto_detect(
    root: Path,
    output_dir: str | None,
) -> Any:
    from aero_forge.ignore import parse_aeroignore

    ignore_file = root / ".aeroignore"
    functions = discover_project(
        root,
        ignore_patterns=parse_aeroignore(ignore_file),
    )
    if not functions:
        raise ValueError(f"No public Python functions found in {root}")

    out = Path(output_dir) if output_dir else root / "dist"
    return generate_blueprint(
        project=root.name or "auto_project",
        functions=functions,
        output_dir=out,
    )


@main.command()
@click.argument("project", required=True, type=str)
@click.option(
    "--path",
    type=click.Path(file_okay=False, writable=True, path_type=str),
    default=".",
    help="Parent directory for the new project (default: current directory).",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["aero", "yaml"], case_sensitive=False),
    default="aero",
    help="Blueprint file format (default: aero).",
)
def init(project: str, path: str, fmt: str) -> None:
    """Create a new Aero-Forge project skeleton with an example function."""
    base = Path(path) / project
    try:
        base.mkdir(parents=True, exist_ok=True)
        src = base / "src"
        tests = base / "tests"
        dist = base / "dist"
        src.mkdir(parents=True, exist_ok=True)
        tests.mkdir(parents=True, exist_ok=True)
        dist.mkdir(parents=True, exist_ok=True)

        example_py = src / "example.py"
        example_py.write_text(
            "def factorial(n):\n"
            "    if n <= 1:\n"
            "        return 1\n"
            "    result = 1\n"
            "    for i in range(2, n + 1):\n"
            "        result *= i\n"
            "    return result\n",
            encoding="utf-8",
        )

        test_py = tests / "test_example.py"
        test_py.write_text(
            "from example import factorial\n\n"
            "def test_factorial():\n"
            "    assert factorial(0) == 1\n"
            "    assert factorial(5) == 120\n",
            encoding="utf-8",
        )

        blueprint = generate_blueprint(
            project=project,
            functions=[
                FunctionSpec.model_validate(
                    {
                        "file": str(example_py),
                        "name": "factorial",
                        "tests": [str(test_py)],
                    }
                )
            ],
            output_dir=dist,
        )

        if fmt.lower() == "yaml":
            blueprint_path = base / "blueprint.yaml"
        else:
            blueprint_path = base / "blueprint.aero"
        write_blueprint(blueprint, blueprint_path)

        click.echo(f"Initialized project at {base}")
        click.echo(f"Blueprint: {blueprint_path}")
    except OSError as exc:
        click.echo(f"Failed to initialize project: {exc}", err=True)
        sys.exit(1)


@main.command()
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(file_okay=False, writable=True, path_type=str),
    default=".",
    help="Directory for generated files and builds.",
)
@click.option(
    "--llm-provider",
    default=None,
    help="LLM provider: openai, openrouter, deepseek, gemini, or none (default: config/env).",
)
@click.option(
    "--model",
    default=None,
    help="Model name to use (default depends on provider).",
)
@click.option(
    "--max-iterations",
    type=int,
    default=5,
    help="Maximum optimization iterations (default: 5).",
)
@click.option(
    "--prompt-template",
    type=click.Choice(
        [
            "v1_minimal",
            "v2_structured",
            "v3_algorithm",
            "v4_performance",
            "v5_balanced",
            "v6_creative",
            "v7_conservative",
            "v8_iterative",
            "v9_transpiler_friendly",
            "v10_correctness_focused",
        ],
        case_sensitive=False,
    ),
    default="v5_balanced",
    help="System prompt template for generation (default: v5_balanced).",
)
@click.option(
    "--session-id",
    default=None,
    help="Resume a previous chat session (default: start a new one).",
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    help="Output chat turns and progress as NDJSON.",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Show debug logs.",
)
def chat(
    output_dir: str,
    llm_provider: str | None,
    model: str | None,
    max_iterations: int,
    verbose: bool,
    prompt_template: str,
    session_id: str | None,
    json_output: bool,
) -> None:
    """Interactive chat session for prompt-driven generation and optimization."""
    _setup_logging(verbose, json_output)
    from .chat import ChatSession

    def _progress(message: str) -> None:
        if json_output:
            click.echo(
                json.dumps({"type": "progress", "message": message}, default=str)
            )
        else:
            click.echo(f"[{message}]")

    session = ChatSession(
        Path(output_dir),
        llm_provider=llm_provider,
        model=model,
        max_iterations=max_iterations,
        prompt_template=prompt_template,
        session_id=session_id,
        progress_callback=_progress,
    )

    if json_output:
        click.echo(
            json.dumps(
                {
                    "type": "welcome",
                    "message": "Aero-Forge chat is ready. What would you like to build?",
                    "session_id": session.session_id,
                },
                default=str,
            )
        )
    else:
        click.echo("Aero-Forge chat is ready. What would you like to build?")
        if session_id:
            click.echo(f"(Resuming session {session.session_id})")
        click.echo("Type 'help' for ideas or 'exit' to leave.")

    while True:
        try:
            user_input = click.prompt("> ", prompt_suffix="")
        except (EOFError, click.exceptions.Abort):
            break
        text = user_input.strip()
        if not text:
            continue
        if text.lower() in {"exit", "quit"}:
            if json_output:
                click.echo(json.dumps({"type": "goodbye"}, default=str))
            else:
                click.secho("Goodbye!", fg="green")
            break

        if json_output:
            click.echo(json.dumps({"type": "user", "message": text}, default=str))

        response = session.process(text)
        _print_chat_response(session, response, json_output=json_output)


def _print_chat_response(
    session: Any, response: str, *, json_output: bool = False
) -> None:
    """Print a chat response with color or as JSON."""
    if json_output:
        build = (session.last_build_result or {}).get("build") or {}
        click.echo(
            json.dumps(
                {
                    "type": "assistant",
                    "message": response,
                    "build_success": build.get("success") if build else None,
                },
                default=str,
            )
        )
        return
    build = (session.last_build_result or {}).get("build") or {}
    lowered = response.lower()
    if not build.get("success") and (
        "oops" in lowered or "snag" in lowered or "error" in lowered
    ):
        click.secho(response, fg="red")
    elif build.get("success") and ("done!" in lowered or "passed" in lowered):
        click.secho(response, fg="green")
    elif "not sure" in lowered or "did you mean" in lowered:
        click.secho(response, fg="yellow")
    else:
        click.echo(response)


@main.command()
@click.argument("source", type=click.Path(exists=True, dir_okay=False, path_type=str))
@click.option(
    "--error-file",
    "-e",
    type=click.Path(dir_okay=False, path_type=str),
    default=None,
    help="File containing the build error log (default: read from stdin).",
)
@click.option(
    "--llm-provider",
    default=None,
    help="LLM provider for the explanation (default: config/env).",
)
@click.option("--model", default=None, help="Model name to use.")
def explain(
    source: str,
    error_file: str | None,
    llm_provider: str | None,
    model: str | None,
) -> None:
    """Explain a build error in plain English and suggest fixes."""
    if error_file:
        error_log = Path(error_file).read_text(encoding="utf-8")
    else:
        if sys.stdin.isatty():
            click.echo(
                "Paste the error log and press Ctrl+D (or use --error-file).",
                err=True,
            )
        error_log = sys.stdin.read()

    source_text = Path(source).read_text(encoding="utf-8")
    result = explain_error(
        error_log,
        source=source_text,
        llm_provider=llm_provider,
        model=model,
    )
    click.echo(result)


@main.group()
def examples() -> None:
    """Curated example projects."""


@examples.command("list")
def examples_list() -> None:
    """List available curated examples."""
    items = list_examples()
    if not items:
        click.echo("No examples found.")
        return
    for item in items:
        click.echo(f"  {item['name']} - {item['description']}")


@examples.command("run")
@click.argument("name")
@click.option(
    "--llm-provider",
    default=None,
    help="LLM provider for auto-fixing failures (default: config/env).",
)
@click.option("--model", default=None, help="Model name to use.")
@click.option(
    "--max-iterations",
    "-i",
    type=int,
    default=None,
    help="Maximum fix iterations per source file.",
)
@click.option("--verbose", is_flag=True, help="Show full build output.")
def examples_run(
    name: str,
    llm_provider: str | None,
    model: str | None,
    max_iterations: int | None,
    verbose: bool,
) -> None:
    """Build and test a curated example by NAME."""
    _setup_logging(verbose)
    try:
        result = run_example(
            name,
            build_kwargs={
                "llm_provider": llm_provider,
                "model": model,
                "max_iterations": max_iterations,
                "cache_enabled": False,
            },
        )
    except ValueError as exc:
        click.echo(str(exc), err=True)
        sys.exit(1)

    if verbose:
        for item in result.get("results", []):
            click.echo(
                f"{', '.join(item['functions'])}: "
                f"{'OK' if item['success'] else 'FAIL'} "
                f"({item['iterations']} iterations)"
            )

    click.echo(
        f"Build complete: {result['passed']}/{result['total']} succeeded. "
        f"Output directory: {result['output_dir']}"
    )
    if not result["success"]:
        sys.exit(1)


@examples.command("create")
@click.argument("name")
@click.option(
    "--prompt",
    "-p",
    required=True,
    help="Natural language description for the new example.",
)
@click.option(
    "--llm-provider",
    default=None,
    help="LLM provider to generate the example (default: config/env).",
)
@click.option("--model", default=None, help="Model name to use.")
@click.option(
    "--prompt-template",
    default="v5_balanced",
    help="System prompt template for generation.",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(file_okay=False, path_type=str),
    default=".",
    help="Directory where the example folder will be created.",
)
def examples_create(
    name: str,
    prompt: str,
    llm_provider: str | None,
    model: str | None,
    prompt_template: str,
    output_dir: str,
) -> None:
    """Create a new example project from a prompt."""
    project_dir = create_example(
        name,
        prompt,
        output_dir=Path(output_dir),
        llm_provider=llm_provider,
        model=model,
        prompt_template=prompt_template,
    )
    click.echo(f"Created example at {project_dir}")


if __name__ == "__main__":
    main()
