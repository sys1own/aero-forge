"""Aero-Forge CLI entrypoint."""

from __future__ import annotations

import os
import sys

import click

from .errors import UserError
from .orchestrator.orchestrator import ForgeError, Orchestrator


@click.group()
@click.version_option(version=__import__("aero_forge").__version__, prog_name="aero-forge")
def main() -> None:
    """Aero-Forge: transpile, compile, and heal Python functions."""


@main.command()
@click.argument("file", type=click.Path(exists=True, dir_okay=False, path_type=str))
@click.option("--function", "-f", required=True, help="Name of the function to fix/compile.")
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
    default=5,
    show_default=True,
    help="Maximum number of fix iterations.",
)
@click.option(
    "--no-llm",
    is_flag=True,
    help="Run the accelerator without LLM-based healing.",
)
@click.option(
    "--model",
    default="gpt-4",
    show_default=True,
    help="OpenAI model to use for code generation.",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Print full compiler/test output.",
)
def fix(
    file: str,
    function: str,
    test_file: str | None,
    max_iterations: int,
    no_llm: bool,
    model: str,
    verbose: bool,
) -> None:
    """Compile and test FILE's FUNCTION, healing failures automatically."""
    if not no_llm and not os.getenv("OPENAI_API_KEY"):
        click.echo(
            "OPENAI_API_KEY is not set. Set it or use --no-llm.", err=True
        )
        sys.exit(1)

    orchestrator = Orchestrator(
        source_path=file,
        function_name=function,
        test_path=test_file,
        max_iterations=max_iterations,
        use_llm=not no_llm,
        model=model,
    )
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
    click.echo(
        f"Success after {result['iterations']} iteration(s). Native extension built."
    )


if __name__ == "__main__":
    main()
