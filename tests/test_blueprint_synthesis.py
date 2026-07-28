"""Integration tests for Blueprint v3 ingestion, synthesis, and execution."""

import io
import zipfile
from pathlib import Path
from typing import Any, Dict

import pytest

from aero_forge.blueprint import BlueprintV3, BlueprintV3Validator, write_v3_blueprint
from aero_forge.blueprint.synthesizer import LLMBlueprintSynthesizer
from aero_forge.ingestion.zip_parser import ingest_zip_archive


def _make_zip(root: Path) -> bytes:
    """Create a minimal ZIP archive from *root* in memory."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in root.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(root))
    return buf.getvalue()


class _FakeLLMClient:
    """Deterministic LLM client that emits a valid finalized v3 blueprint."""

    def __init__(self, project_name: str = "synthesized_project", omit_node_id: bool = False) -> None:
        self.project_name = project_name
        self.omit_node_id = omit_node_id

    def generate(self, prompt: str, **kwargs: Any) -> str:
        node: Dict[str, Any] = {
            "command": "python3 main.py",
            "expected_exit_code": 0,
            "stdout_match_patterns": ["Hello"],
            "stderr_prohibited_patterns": ["error"],
            "metrics": [],
            "timeout": 30.0,
        }
        if not self.omit_node_id:
            node["node_id"] = "smoke"
        blueprint: Dict[str, Any] = {
            "metadata": {
                "schema_version": "3.0.0",
                "project_name": self.project_name,
                "status": "finalized",
                "generation_method": "llm_synthesized",
                "transferable": True,
                "description": "Synthesized blueprint",
            },
            "toolchains": [{"name": "CPython", "version": "3.x"}],
            "build_pipeline": [
                {
                    "id": "app",
                    "type": "binary",
                    "source_files": ["main.py"],
                    "output_path": "dist/app",
                    "compiler_flags": [],
                    "linker_flags": [],
                    "dependencies": [],
                    "commands": [],
                    "description": "Main application",
                }
            ],
            "abi_contracts": [],
            "execution_strategy": {
                "primary_entrypoint": "main.py",
                "runtime": "python3",
                "args": [],
                "env": {},
                "working_dir": "${WORKSPACE_ROOT}",
                "timeout": 30.0,
            },
            "verification_nodes": [node],
        }
        import json
        return json.dumps(blueprint)


def test_zip_ingestion_creates_draft_v3_blueprint(tmp_path: Path) -> None:
    """Uploading a custom ZIP generates a Blueprint v3 draft."""
    source = tmp_path / "upload"
    source.mkdir()
    (source / "main.py").write_text("print('Hello from main')\n", encoding="utf-8")
    zip_bytes = _make_zip(source)

    workspace = tmp_path / "workspace"
    _, draft = ingest_zip_archive(zip_bytes, workspace)

    assert draft is not None
    assert draft.metadata.schema_version == "3.0.0"
    assert draft.metadata.status == "draft"
    assert draft.metadata.generation_method == "static_heuristic"
    assert draft.metadata.transferable is False
    blueprint_path = workspace / "blueprint.aero"
    assert blueprint_path.is_file()
    # Validator should allow the draft locally but reject export.
    validator = BlueprintV3Validator(blueprint_path, workspace=workspace)
    assert validator.validate().metadata.status == "draft"
    with pytest.raises(Exception):
        validator.check_exportable()


def test_synthesize_upgrades_draft_to_finalized(tmp_path: Path, monkeypatch: Any) -> None:
    """LLM synthesis upgrades a draft blueprint to finalized and transferable."""
    draft = BlueprintV3(
        metadata={
            "schema_version": "3.0.0",
            "project_name": "draft_project",
            "status": "draft",
            "generation_method": "static_heuristic",
            "transferable": False,
        },
        build_pipeline=[
            {
                "id": "app",
                "type": "binary",
                "source_files": ["main.py"],
                "output_path": "dist/app",
            }
        ],
    )
    workspace = tmp_path / "project"
    workspace.mkdir()
    (workspace / "main.py").write_text("print('Hello')\n", encoding="utf-8")
    write_v3_blueprint(draft, workspace / "blueprint.aero")

    # Patch the LLM client factory so no real network call is made.
    import aero_forge.llm.clients as clients

    monkeypatch.setattr(clients, "get_llm_client", lambda provider, model=None: _FakeLLMClient(project_name="draft_project"))

    synthesizer = LLMBlueprintSynthesizer(provider="fake")
    synthesizer._client = lambda: _FakeLLMClient(project_name="draft_project")
    finalized = synthesizer.synthesize(workspace)

    assert finalized.metadata.status == "finalized"
    assert finalized.metadata.generation_method == "llm_synthesized"
    assert finalized.metadata.transferable is True
    BlueprintV3Validator(finalized.model_dump(mode="json"), workspace=workspace).check_exportable()


def test_synthesize_fills_missing_verification_node_id(tmp_path: Path, monkeypatch: Any) -> None:
    """LLM synthesis that omits verification node_id should still produce a valid BlueprintV3."""
    workspace = tmp_path / "project"
    workspace.mkdir()
    (workspace / "main.py").write_text("print('Hello')\n", encoding="utf-8")

    import aero_forge.llm.clients as clients
    monkeypatch.setattr(clients, "get_llm_client", lambda provider, model=None: _FakeLLMClient(omit_node_id=True))

    synthesizer = LLMBlueprintSynthesizer(provider="fake")
    synthesizer._client = lambda: _FakeLLMClient(omit_node_id=True)
    finalized = synthesizer.synthesize(workspace)

    assert finalized.metadata.status == "finalized"
    assert len(finalized.verification_nodes) == 1
    assert finalized.verification_nodes[0].node_id.startswith("node_")
    assert finalized.verification_nodes[0].command == "python3 main.py"
    BlueprintV3Validator(
        finalized.model_dump(mode="json"), workspace=workspace
    ).check_exportable()


def test_synthesized_v3_executes_deterministically(tmp_path: Path, monkeypatch: Any) -> None:
    """A finalized v3 blueprint can be executed and verified deterministically."""
    workspace = tmp_path / "project"
    workspace.mkdir()
    (workspace / "main.py").write_text("print('Hello from blueprint')\n", encoding="utf-8")

    blueprint = BlueprintV3(
        metadata={
            "schema_version": "3.0.0",
            "project_name": "run_test",
            "status": "finalized",
            "generation_method": "llm_synthesized",
            "transferable": True,
        },
        toolchains=[{"name": "CPython"}],
        build_pipeline=[
            {
                "id": "app",
                "type": "binary",
                "source_files": ["main.py"],
                "output_path": "dist/app",
                "dependencies": [],
            }
        ],
        execution_strategy={
            "primary_entrypoint": "main.py",
            "runtime": "python3",
            "args": [],
            "env": {},
            "working_dir": "${WORKSPACE_ROOT}",
            "timeout": 30.0,
        },
        verification_nodes=[
            {
                "node_id": "smoke",
                "command": "",
                "expected_exit_code": 0,
                "stdout_match_patterns": ["Hello from blueprint"],
                "stderr_prohibited_patterns": ["error"],
                "metrics": [],
                "timeout": 30.0,
            }
        ],
    )

    result = blueprint.execute(workspace)
    assert result["status"] == "success"
    assert result["exit_code"] == 0
    assert "Hello from blueprint" in result["stdout"]
    assert all(v["passed"] for v in result["verification"])
