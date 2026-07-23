"""Examples gallery: list, run, and create curated Aero-Forge examples."""

from __future__ import annotations

import importlib.resources as pkg_resources
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from aero_forge.blueprint import parse_blueprint
from aero_forge.build_runner import BuildRunner
from aero_forge.generate import generate_and_build

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


def _example_root(name: str) -> Path:
    return EXAMPLES_DIR / name


def list_examples() -> List[Dict[str, str]]:
    """Return the list of available example projects."""
    examples: List[Dict[str, str]] = []
    if not EXAMPLES_DIR.is_dir():
        return examples
    for entry in sorted(EXAMPLES_DIR.iterdir()):
        if entry.is_dir() and (entry / "blueprint.aero").is_file():
            prompt_file = entry / "prompt.txt"
            prompt = (
                prompt_file.read_text(encoding="utf-8").strip()
                if prompt_file.exists()
                else ""
            )
            examples.append({"name": entry.name, "description": prompt})
    return examples


def run_example(
    name: str, *, build_kwargs: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Build an example project by name.

    Returns the ``BuildRunner.build()`` result dict.
    """
    root = _example_root(name)
    if not root.is_dir():
        raise ValueError(f"Unknown example: {name!r}")
    blueprint_path = root / "blueprint.aero"
    if not blueprint_path.is_file():
        raise ValueError(f"Example {name!r} has no blueprint.aero")

    blueprint = parse_blueprint(blueprint_path)
    runner = BuildRunner(blueprint, **(build_kwargs or {}))
    return runner.build()


def create_example(
    name: str,
    prompt: str,
    *,
    output_dir: Optional[Path] = None,
    llm_provider: Optional[str] = None,
    model: Optional[str] = None,
    prompt_template: Optional[str] = None,
) -> Path:
    """Generate a new example directory from a natural-language prompt.

    Uses the same pipeline as ``aero-forge generate``.  Returns the path to the
    generated project directory.
    """
    root = output_dir or Path(".")
    project_dir = root / name
    project_dir.mkdir(parents=True, exist_ok=True)

    generate_and_build(
        prompt,
        output_dir=project_dir,
        project_name=name,
        llm_provider=llm_provider,
        model=model,
        prompt_template=prompt_template,
    )

    (project_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
    return project_dir
