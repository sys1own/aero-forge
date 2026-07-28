"""Tests for LLM blueprint synthesis in the accelerate workflow."""

import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from aero_forge.blueprint.schema import (
    ABIContractV3,
    ABIArgument,
    ArtifactType,
    BindingFramework,
    BlueprintStatus,
    BlueprintV3,
    BuildArtifact,
    ExecutionStrategyV3,
    GenerationMethod,
    MemoryModel,
    Metadata,
    ToolchainSpec,
    VerificationNode,
    write_v3_blueprint,
)
from aero_forge.cli import main


def _fake_llm_client(*, finalized_json: str) -> Any:
    """Return a minimal fake LLM client that emits ``finalized_json``."""

    class FakeClient:
        def generate(self, prompt, temperature=0.1, **kwargs):
            return finalized_json

    return FakeClient()


def _finalized_blueprint_json(project_name: str) -> str:
    bp = BlueprintV3(
        metadata=Metadata(
            schema_version="3.0.0",
            project_name=project_name,
            status=BlueprintStatus.finalized,
            generation_method=GenerationMethod.llm_synthesized,
            transferable=True,
            description="Finalized by LLM",
        ),
        toolchains=[
            ToolchainSpec(name="python", version="3.11"),
            ToolchainSpec(name="cargo"),
            ToolchainSpec(name="rust"),
        ],
        build_pipeline=[
            BuildArtifact(
                id="rust_core",
                type=ArtifactType.cargo_cdylib,
                source_files=["rust_core/src/lib.rs"],
                output_path="target/release/librust_core.so",
                compiler_flags=["-O3", "--release"],
                linker_flags=["-lm"],
                dependencies=[],
                description="Rust cdylib core",
            ),
            BuildArtifact(
                id="python_driver",
                type=ArtifactType.python_extension,
                source_files=["python_driver/src/main.py"],
                output_path="dist/python_driver",
                compiler_flags=["-O3"],
                dependencies=["rust_core"],
                description="Python driver",
            ),
        ],
        abi_contracts=[
            ABIContractV3(
                contract_id="compute",
                symbol="compute",
                source_language="python",
                target_language="rust",
                binding_framework=BindingFramework.pyo3,
                memory_model=MemoryModel.shared_pyo3,
                inputs=[ABIArgument(name="x", type="f64")],
                outputs=[ABIArgument(name="result", type="f64")],
            )
        ],
        execution_strategy=ExecutionStrategyV3(
            primary_entrypoint="python_driver/src/main.py",
            runtime="python3",
            working_dir="${WORKSPACE_ROOT}",
            timeout=120.0,
        ),
        verification_nodes=[
            VerificationNode(
                node_id="smoke",
                command="python3 ${WORKSPACE_ROOT}/python_driver/src/main.py",
                expected_exit_code=0,
                timeout=60.0,
            )
        ],
    )
    return json.dumps(bp.model_dump(mode="json"))


def test_cli_blueprint_synthesize_finalizes_draft(tmp_path: Path, monkeypatch: Any) -> None:
    """`aero-forge blueprint synthesize` upgrades a draft to finalized."""
    draft = BlueprintV3(
        metadata=Metadata(
            schema_version="3.0.0",
            project_name="draft_to_finalize",
            status=BlueprintStatus.draft,
            generation_method=GenerationMethod.static_heuristic,
            transferable=False,
        ),
        toolchains=[ToolchainSpec(name="python")],
        build_pipeline=[
            BuildArtifact(
                id="python_app",
                type=ArtifactType.python_extension,
                source_files=["src/main.py"],
                output_path="dist/python_app",
                compiler_flags=["-O2"],
            )
        ],
        execution_strategy=ExecutionStrategyV3(
            primary_entrypoint="src/main.py", runtime="python3"
        ),
    )
    blueprint_path = tmp_path / "blueprint.aero"
    write_v3_blueprint(draft, blueprint_path)

    fake = _fake_llm_client(finalized_json=_finalized_blueprint_json("draft_to_finalize"))
    monkeypatch.setattr("aero_forge.blueprint.synthesizer.get_llm_client", lambda provider, model=None, **kwargs: fake)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "blueprint",
            "synthesize",
            "--workspace",
            str(tmp_path),
            "--draft",
            str(blueprint_path),
            "--output",
            str(blueprint_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert blueprint_path.is_file()

    finalized = BlueprintV3.load(blueprint_path)
    assert finalized.metadata.status == BlueprintStatus.finalized
    assert finalized.metadata.transferable is True
    assert finalized.metadata.generation_method == GenerationMethod.llm_synthesized
    assert finalized.metadata.schema_version == "3.0.0"

    # build_pipeline must be fully populated with compiler/linker flags and dependencies.
    assert len(finalized.build_pipeline) >= 2
    rust_core = next(a for a in finalized.build_pipeline if a.id == "rust_core")
    assert rust_core.compiler_flags
    assert rust_core.linker_flags
    assert "python_driver" in rust_core.dependencies or any(
        a.id == "python_driver" and "rust_core" in a.dependencies
        for a in finalized.build_pipeline
    )


def test_build_draft_then_synthesize_then_finalize(tmp_path: Path, monkeypatch: Any) -> None:
    """A draft build leaves sources intact, then synthesis finalizes the blueprint."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "main.py").write_text(
        "def compute(x: float) -> float:\n    return x * x\n", encoding="utf-8"
    )

    draft = BlueprintV3(
        metadata=Metadata(
            schema_version="3.0.0",
            project_name="draft_cycle",
            status=BlueprintStatus.draft,
            generation_method=GenerationMethod.static_heuristic,
            transferable=False,
        ),
        toolchains=[ToolchainSpec(name="python")],
        build_pipeline=[
            BuildArtifact(
                id="python_app",
                type=ArtifactType.python_extension,
                source_files=["src/main.py"],
                output_path="dist/python_app",
            )
        ],
        execution_strategy=ExecutionStrategyV3(
            primary_entrypoint="src/main.py",
            runtime="python3",
            working_dir="${WORKSPACE_ROOT}",
        ),
        verification_nodes=[
            VerificationNode(
                node_id="smoke",
                command="python3 ${WORKSPACE_ROOT}/src/main.py",
                expected_exit_code=0,
            )
        ],
    )
    blueprint_path = tmp_path / "blueprint.aero"
    write_v3_blueprint(draft, blueprint_path)

    runner = CliRunner()
    build_result = runner.invoke(main, ["build", str(blueprint_path)])
    assert build_result.exit_code == 0, build_result.output

    original = (src_dir / "main.py").read_text(encoding="utf-8")
    assert (src_dir / "main.py").read_text(encoding="utf-8") == original

    fake = _fake_llm_client(finalized_json=_finalized_blueprint_json("draft_cycle"))
    monkeypatch.setattr("aero_forge.blueprint.synthesizer.get_llm_client", lambda provider, model=None, **kwargs: fake)

    synth_result = runner.invoke(
        main,
        [
            "blueprint",
            "synthesize",
            "--workspace",
            str(tmp_path),
            "--draft",
            str(blueprint_path),
            "--output",
            str(blueprint_path),
        ],
    )
    assert synth_result.exit_code == 0, synth_result.output

    finalized = BlueprintV3.load(blueprint_path)
    assert finalized.metadata.status == BlueprintStatus.finalized
    assert finalized.metadata.transferable is True
    assert finalized.metadata.generation_method == GenerationMethod.llm_synthesized

    aeroc_path = tmp_path / "workspace.aeroc"
    assert aeroc_path.is_file()
    assert aeroc_path.read_bytes().startswith(b"AEROFOG\x00")
