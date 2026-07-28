"""Tests for Blueprint v3.0.0 schema and validation."""

from pathlib import Path

import pytest
import yaml

from aero_forge.blueprint import (
    BlueprintV3,
    BlueprintV3Validator,
    DraftBlueprintExportError,
    InvalidBlueprintError,
    write_v3_blueprint,
)
from aero_forge.blueprint.schema import ArtifactType, BuildArtifact, Metadata


def test_v3_schema_validates_complete_polyglot_dag(tmp_path: Path) -> None:
    """A complete polyglot DAG blueprint passes v3 validation."""
    blueprint = BlueprintV3(
        metadata=Metadata(
            schema_version="3.0.0",
            project_name="polyglot_demo",
            status="finalized",
            generation_method="llm_synthesized",
            transferable=True,
        ),
        toolchains=[
            {"name": "CPython", "version": "3.11"},
            {"name": "Rust", "channel": "stable"},
            {"name": "GCC"},
        ],
        build_pipeline=[
            BuildArtifact(
                id="rust_core",
                type=ArtifactType.cargo_cdylib,
                source_files=["rust_core/src/lib.rs"],
                output_path="target/release/librust_core.so",
                compiler_flags=["--release"],
            ),
            BuildArtifact(
                id="cpp_native",
                type=ArtifactType.shared_library,
                source_files=["cpp_core/native.cpp"],
                output_path="dist/libnative.so",
                dependencies=["rust_core"],
            ),
            BuildArtifact(
                id="python_app",
                type=ArtifactType.python_extension,
                source_files=["src/main.py"],
                output_path="dist/app",
                dependencies=["cpp_native"],
            ),
        ],
        execution_strategy={
            "primary_entrypoint": "src/main.py",
            "runtime": "python3",
            "working_dir": "${WORKSPACE_ROOT}",
            "timeout": 30.0,
        },
        verification_nodes=[
            {
                "node_id": "run_app",
                "command": "python3 src/main.py",
                "expected_exit_code": 0,
                "stdout_match_patterns": ["ok"],
            }
        ],
    )

    path = tmp_path / "blueprint.aero"
    write_v3_blueprint(blueprint, path)
    validated = BlueprintV3Validator(path, workspace=tmp_path).validate()
    assert validated.metadata.transferable is True
    assert validated.metadata.status == "finalized"
    assert len(validated.build_pipeline) == 3


def test_v3_rejects_absolute_paths(tmp_path: Path) -> None:
    """Any absolute path in the blueprint causes validation to fail."""
    blueprint = BlueprintV3(
        metadata=Metadata(
            schema_version="3.0.0",
            project_name="bad_paths",
            status="finalized",
            generation_method="manual",
            transferable=True,
        ),
        build_pipeline=[
            BuildArtifact(
                id="app",
                type=ArtifactType.binary,
                source_files=["/tmp/main.py"],
                output_path="dist/app",
            ),
        ],
    )
    path = tmp_path / "blueprint.aero"
    write_v3_blueprint(blueprint, path)
    with pytest.raises(InvalidBlueprintError, match="Absolute path not allowed"):
        BlueprintV3Validator(path, workspace=tmp_path).validate()


def test_v3_draft_cannot_be_exported(tmp_path: Path) -> None:
    """Draft blueprints raise DraftBlueprintExportError when checked for export."""
    blueprint = BlueprintV3(
        metadata=Metadata(
            schema_version="3.0.0",
            project_name="draft",
            status="draft",
            generation_method="static_heuristic",
            transferable=False,
        ),
        build_pipeline=[
            BuildArtifact(
                id="app",
                type=ArtifactType.binary,
                source_files=["src/main.py"],
                output_path="dist/app",
            ),
        ],
    )
    path = tmp_path / "blueprint.aero"
    write_v3_blueprint(blueprint, path)

    validator = BlueprintV3Validator(path, workspace=tmp_path)
    # Local validation should allow drafts.
    assert validator.validate().metadata.status == "draft"
    # Export validation must reject drafts.
    with pytest.raises(DraftBlueprintExportError):
        validator.check_exportable()


def test_v3_finalized_requires_build_pipeline(tmp_path: Path) -> None:
    """A finalized transferable blueprint must have a build pipeline."""
    blueprint = BlueprintV3(
        metadata=Metadata(
            schema_version="3.0.0",
            project_name="empty",
            status="finalized",
            generation_method="llm_synthesized",
            transferable=True,
        ),
        build_pipeline=[],
    )
    path = tmp_path / "blueprint.aero"
    write_v3_blueprint(blueprint, path)
    with pytest.raises(InvalidBlueprintError, match="non-empty build_pipeline"):
        BlueprintV3Validator(path, workspace=tmp_path).validate()


def test_v3_load_from_yaml(tmp_path: Path) -> None:
    """BlueprintV3.load parses YAML .aero files."""
    data = {
        "metadata": {
            "schema_version": "3.0.0",
            "project_name": "yaml_test",
            "status": "draft",
            "generation_method": "manual",
            "transferable": False,
        },
        "build_pipeline": [
            {
                "id": "app",
                "type": "binary",
                "source_files": ["main.py"],
                "output_path": "dist/app",
            }
        ],
    }
    path = tmp_path / "blueprint.aero"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    bp = BlueprintV3.load(path)
    assert bp.metadata.project_name == "yaml_test"


def test_v3_dag_order_respects_dependencies(tmp_path: Path) -> None:
    """Artifact order is topologically sorted by dependencies."""
    blueprint = BlueprintV3(
        metadata=Metadata(
            schema_version="3.0.0",
            project_name="dag",
            status="finalized",
            generation_method="manual",
            transferable=True,
        ),
        build_pipeline=[
            BuildArtifact(id="app", type=ArtifactType.binary, source_files=["main.py"], dependencies=["lib"]),
            BuildArtifact(id="lib", type=ArtifactType.shared_library, source_files=["lib.c"]),
        ],
    )
    order = blueprint._artifact_order()
    assert order.index("lib") < order.index("app")
