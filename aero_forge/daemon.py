"""Python interface to the native ``aeroc-daemon`` execution engine."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from aero_forge._native import run_aeroc

if TYPE_CHECKING:
    from aero_forge.blueprint.schema import BlueprintV3


def run_aeroc_file(aeroc_path: str | Path, workspace: str | Path, max_workers: int = 4) -> None:
    """Execute a compiled ``workspace.aeroc`` binary inside *workspace*."""
    run_aeroc(str(aeroc_path), str(workspace), max_workers)


def compile_and_run_blueprint(
    blueprint: "BlueprintV3",
    workspace: str | Path,
    output_dir: str | Path | None = None,
    max_workers: int = 4,
) -> Path:
    """Compile *blueprint* to ``workspace.aeroc`` and run it via the daemon."""
    from aero_forge.builder.aeroc_compiler import compile_blueprint_to_aeroc

    workspace = Path(workspace).resolve()
    output_dir = Path(output_dir) if output_dir else workspace
    output_dir.mkdir(parents=True, exist_ok=True)
    aeroc_path = output_dir / "workspace.aeroc"
    compile_blueprint_to_aeroc(blueprint, aeroc_path, workspace=workspace)
    run_aeroc_file(aeroc_path, workspace, max_workers=max_workers)
    return aeroc_path
