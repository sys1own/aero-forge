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
def build(
    blueprint: str,
    auto_file: str | None,
    llm_provider: str | None,
    model: str | None,
    max_iterations: int | None,
    max_retries: int | None,
    no_llm: bool,
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
    "--project",
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
    "--verbose",
    is_flag=True,
    help="Show debug logs.",
)
def generate(
    prompt: str | None,
    prompt_file: str | None,
    constraints: str | None,
    output_dir: str,
    project: str,
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
) -> None:
    """Generate Python code and tests from a natural language prompt."""
    _setup_logging(verbose)

    if no_llm:
        llm_provider = "none"

    if prompt_file:
        prompt = Path(prompt_file).read_text(encoding="utf-8").strip()
    if not prompt:
        click.echo("Error: --prompt or --prompt-file is required.", err=True)
        sys.exit(1)

    try:
        from .generate import generate_and_build

        result = generate_and_build(
            prompt,
            constraints=constraints,
            output_dir=Path(output_dir),
            project_name=project,
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
        )
    except Exception as exc:
        click.echo(f"Generation failed: {exc}", err=True)
        sys.exit(1)

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
        click.echo(
            f"Build: {build_result.get('passed', 0)}/{build_result.get('total', 0)} "
            f"succeeded ({build_result.get('output_dir', '')})"
        )
        if not build_result.get("success"):
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
        ],
        case_sensitive=False,
    ),
    default="v5_balanced",
    help="System prompt template for generation (default: v5_balanced).",
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
) -> None:
    """Interactive chat session for prompt-driven generation and optimization."""
    _setup_logging(verbose)
    from .chat import ChatSession

    session = ChatSession(
        Path(output_dir),
        llm_provider=llm_provider,
        model=model,
        max_iterations=max_iterations,
        prompt_template=prompt_template,
    )

    click.echo("Aero-Forge chat mode. Type 'exit' or 'quit' to leave.")
    while True:
        try:
            user_input = click.prompt("> ", prompt_suffix="")
        except (EOFError, click.exceptions.Abort):
            break
        text = user_input.strip()
        if not text:
            continue
        if text.lower() in {"exit", "quit"}:
            break

        action_result = session.handle_command(text)
        if action_result:
            build = action_result.get("build")
            if build:
                click.echo(
                    f"Build: {build.get('passed', 0)}/{build.get('total', 0)} succeeded"
                )
            else:
                click.echo("Action completed.")

        response = session.reply(text)
        click.echo(response)


if __name__ == "__main__":
    main()
