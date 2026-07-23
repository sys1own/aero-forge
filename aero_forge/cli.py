"""Aero-Forge CLI entrypoint."""

from __future__ import annotations

import logging
import os
import sys

import click

from .errors import UserError
from .orchestrator.orchestrator import ForgeError, Orchestrator


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )


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
    help="Retries per LLM model before falling back.",
)
@click.option(
    "--model-priority",
    default=None,
    help="Comma-separated list of LLM models in priority order.",
)
@click.option(
    "--fallback-model",
    default=None,
    help="Model to append to the priority list as final fallback.",
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
    "--model",
    default=None,
    help="(Deprecated) Single LLM model; sets the priority list to this model.",
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
    max_iterations: int | None,
    max_retries: int | None,
    model_priority: str | None,
    fallback_model: str | None,
    no_llm: bool,
    no_cache: bool,
    model: str | None,
    verbose: bool,
) -> None:
    """Compile and test FILE's FUNCTION, healing failures automatically."""
    _setup_logging(verbose)

    priority: list[str] | None = None
    if model_priority:
        priority = [m.strip() for m in model_priority.split(",") if m.strip()]

    use_llm = not no_llm
    if use_llm and not _has_any_api_key():
        click.echo(
            "No API key found (OPENAI_API_KEY, OPENROUTER_API_KEY, or GEMINI_API_KEY). "
            "Set one or use --no-llm.",
            err=True,
        )
        sys.exit(1)

    try:
        orchestrator = Orchestrator(
            source_path=file,
            function_name=function,
            test_path=test_file,
            max_iterations=max_iterations,
            use_llm=use_llm,
            model=model,
            model_priority=priority,
            max_retries=max_retries,
            cache_enabled=not no_cache,
            fallback_model=fallback_model,
        )
    except UserError as exc:
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


def _has_any_api_key() -> bool:
    return any(
        os.getenv(name)
        for name in (
            "OPENAI_API_KEY",
            "OPENROUTER_API_KEY",
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
        )
    )


if __name__ == "__main__":
    main()
