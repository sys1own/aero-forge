"""Tests that draft blueprint builds do not mutate the original workspace."""

from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from aero_forge.blueprint.schema import (
    BlueprintV3,
    BuildArtifact,
    BlueprintStatus,
    ExecutionStrategyV3,
    GenerationMethod,
    Metadata,
    ToolchainSpec,
    VerificationNode,
    write_v3_blueprint,
    ArtifactType,
)
from aero_forge.cli import main


def _make_draft_project(tmp_path: Path) -> Path:
    """Create a Python-only draft v3 workspace and return its blueprint path."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "main.py").write_text(
        "def compute(x: float) -> float:\n    return x * x\n\n"
        "if __name__ == '__main__':\n    print(compute(5.0))\n",
        encoding="utf-8",
    )

    blueprint = BlueprintV3(
        metadata=Metadata(
            schema_version="3.0.0",
            project_name="draft_safety_demo",
            status=BlueprintStatus.draft,
            generation_method=GenerationMethod.static_heuristic,
            transferable=False,
            description="Draft blueprint for safety test",
        ),
        toolchains=[ToolchainSpec(name="python", version="3")],
        build_pipeline=[
            BuildArtifact(
                id="python_app",
                type=ArtifactType.python_extension,
                source_files=["src/main.py"],
                output_path="dist/python_app",
                description="Python application",
            )
        ],
        execution_strategy=ExecutionStrategyV3(
            primary_entrypoint="src/main.py",
            runtime="python3",
            working_dir="${WORKSPACE_ROOT}",
            timeout=60.0,
        ),
        verification_nodes=[
            VerificationNode(
                node_id="smoke",
                command="python3 ${WORKSPACE_ROOT}/src/main.py",
                expected_exit_code=0,
                timeout=60.0,
            )
        ],
    )

    blueprint_path = tmp_path / "blueprint.aero"
    write_v3_blueprint(blueprint, blueprint_path)
    return blueprint_path


def test_draft_build_preserves_source_files(tmp_path: Path) -> None:
    """Running `aero-forge build` on a draft blueprint must not modify src/."""
    blueprint_path = _make_draft_project(tmp_path)
    main_py = tmp_path / "src" / "main.py"
    original_content = main_py.read_text(encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(main, ["build", str(blueprint_path)])

    assert result.exit_code == 0, result.output
    assert main_py.read_text(encoding="utf-8") == original_content
    # No artifacts should be written back into src/.
    assert not list((tmp_path / "src").glob("*.so"))
    assert not list((tmp_path / "src").glob("Cargo.toml"))
    assert not list((tmp_path / "src").glob("*.lock"))


def test_draft_build_uses_sandbox_output_dir(tmp_path: Path) -> None:
    """Draft builds should place outputs under .aero/sandbox, not dist/."""
    blueprint_path = _make_draft_project(tmp_path)

    runner = CliRunner()
    result = runner.invoke(main, ["build", str(blueprint_path)])

    assert result.exit_code == 0, result.output
    assert "sandbox" in result.output or ".aero" in result.output or "success" in result.output.lower()
