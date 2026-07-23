"""Aero-Forge CLI entrypoint."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import click

from .blueprint import (
    FunctionSpec,
    discover_functions,
    generate_blueprint,
    parse_blueprint,
    write_blueprint,
)
from .build_runner import BuildRunner
from .errors import UserError
from .orchestrator.orchestrator import ForgeError, Orchestrator


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )


def _resolve_llm_provider(llm_provider: str | None, no_llm: bool) -> str | None:
    if no_llm:
        return "none"
    if not llm_provider and not os.getenv("AERO_FORGE_LLM_PROVIDER"):
        click.echo(
            "No LLM provider configured (set AERO_FORGE_LLM_PROVIDER or use --llm-provider); "
            "running in router-only mode.",
            err=True,
        )
        return "none"
    return llm_provider


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
    help="LLM provider: openai, openrouter, gemini, or none (default: config/env).",
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
    "--no-cache",
    is_flag=True,
    help="Disable the fix cache.",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Print full compiler/test output and debug logs.",
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
    no_cache: bool,
    verbose: bool,
) -> None:
    """Compile and test FILE's FUNCTION, healing failures automatically."""
    _setup_logging(verbose)

    llm_provider = _resolve_llm_provider(llm_provider, no_llm)

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
    except UserError as exc:
        click.echo(str(exc), err=True)
        sys.exit(1)
    except ImportError as exc:
        click.echo(str(exc), err=True)
        sys.exit(1)

    try:
        result = orchestrator.run()
    except UserError as exc:
        click.echo(str(exc), err=True)
        sys.exit(1)
    except ForgeError as exc:
        click.echo(f"Forge failed: {exc}", err=True)
        sys.exit(1)
    except Exception as exc:
        click.echo(f"Unexpected error: {exc}", err=True)
        sys.exit(1)

    if verbose:
        click.echo(result.get("logs", ""))

    if result.get("success"):
        click.echo(
            f"Success after {result['iterations']} iteration(s). Native extension built."
        )
    elif result.get("partial"):
        click.echo(
            f"Partial success after {result['iterations']} iteration(s). "
            f"{result.get('error', '')}",
            err=True,
        )
        if result.get("artifact"):
            click.echo(f"Best compiled artifact: {result['artifact']}", err=True)
        sys.exit(1)
    else:
        click.echo(f"Forge failed: {result.get('error', 'unknown')}", err=True)
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
    "--llm-provider",
    default=None,
    help="LLM provider: openai, openrouter, gemini, or none.",
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
    "--no-cache",
    is_flag=True,
    help="Disable the build cache.",
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
    "--verbose",
    is_flag=True,
    help="Show debug logs and full output.",
)
def build(
    blueprint: str,
    auto_file: str | None,
    llm_provider: str | None,
    model: str | None,
    max_iterations: int | None,
    max_retries: int | None,
    no_llm: bool,
    no_cache: bool,
    output_dir: str | None,
    jobs: int | None,
    write_blueprint_flag: bool,
    dry_run: bool,
    verbose: bool,
) -> None:
    """Build all functions described by BLUEPRINT (default: blueprint.aero)."""
    _setup_logging(verbose)

    if no_llm:
        llm_provider = "none"

    try:
        if auto_file:
            bp = _blueprint_from_auto(Path(auto_file), output_dir)
            if write_blueprint_flag:
                blueprint_path = Path(auto_file).parent / "blueprint.aero"
                write_blueprint(bp, blueprint_path)
                click.echo(f"Wrote blueprint: {blueprint_path}")
        else:
            bp = parse_blueprint(Path(blueprint))
    except (UserError, ValueError) as exc:
        click.echo(f"Invalid blueprint: {exc}", err=True)
        sys.exit(1)
    except ImportError as exc:
        click.echo(str(exc), err=True)
        sys.exit(1)

    if output_dir:
        bp.output_dir = Path(output_dir)

    if not llm_provider:
        llm_provider = os.getenv("AERO_FORGE_LLM_PROVIDER") or bp.llm.provider
    if not model:
        model = os.getenv("AERO_FORGE_MODEL") or bp.llm.model

    runner = BuildRunner(
        blueprint=bp,
        max_workers=jobs or min(4, len(bp.functions) or 1),
        llm_provider=llm_provider,
        model=model,
        max_iterations=max_iterations,
        max_retries=max_retries,
        cache_enabled=not no_cache,
        dry_run=dry_run,
    )

    try:
        result = runner.build()
    except UserError as exc:
        click.echo(str(exc), err=True)
        sys.exit(1)
    except Exception as exc:
        click.echo(f"Unexpected error: {exc}", err=True)
        sys.exit(1)

    if verbose:
        for item in result.get("results", []):
            click.echo(
                f"{', '.join(item['functions'])}: "
                f"{'OK' if item['success'] else 'FAIL'} "
                f"({item['iterations']} iterations)"
            )

    if result.get("dry_run"):
        click.echo(
            f"Dry-run complete: {result['total']} function(s) would be built "
            f"into {result['output_dir']}"
        )
        return

    click.echo(
        f"Build complete: {result['passed']}/{result['total']} succeeded. "
        f"Output directory: {result['output_dir']}"
    )
    if not result["success"]:
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
                FunctionSpec.model_validate({
                    "file": str(example_py),
                    "name": "factorial",
                    "tests": [str(test_py)],
                })
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


if __name__ == "__main__":
    main()
