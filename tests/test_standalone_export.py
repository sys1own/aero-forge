"""Tests for the standalone ``.aeroc`` export acceleration system."""

import io
import shutil
import subprocess
import zipfile

import pytest

import json

from aero_forge.errors import ExportVerificationError
from aero_forge.materializer import unpack_aeroc_file
from aero_forge.scaffold.aeroc_export import (
    compile_aeroc,
    compile_hybrid_aeroc,
    export_aeroc_project,
    export_scaffold_zip,
    generate_verification_json,
    package_aeroc,
    verify_workspace_for_export,
)
from aero_forge.scaffold.cargo_config import CARGO_CONFIG_TOML, write_cargo_config
from aero_forge.scaffold.export_options import ExportMode, ExportOptions, export_workspace


def test_write_cargo_config_injects_optimized_flags(tmp_path):
    config = write_cargo_config(tmp_path)
    assert config.is_file()
    text = config.read_text(encoding="utf-8")
    assert "target-cpu=native" in text
    assert 'opt-level = 3' in text
    assert 'lto = "fat"' in text


def test_export_aeroc_project_creates_runnable_layout(tmp_path):
    project = export_aeroc_project(tmp_path, tmp_path / "exported")
    assert (project / "pyproject.toml").is_file()
    assert (project / "aeroc" / "__init__.py").is_file()
    assert (project / "aeroc" / "cli.py").is_file()
    crate = project / "aeroc" / "aero_core"
    assert (crate / "Cargo.toml").is_file()
    assert (crate / "src" / "lib.rs").is_file()
    assert (crate / "src" / "wavefront.rs").is_file()
    assert (crate / "src" / "main.rs").is_file()
    assert (crate / ".cargo" / "config.toml").is_file()


@pytest.mark.skipif(not shutil.which("cargo"), reason="cargo not installed")
def test_compile_aeroc_runs_default_pipeline(tmp_path):
    project = export_aeroc_project(tmp_path, tmp_path / "exported")
    binary = compile_aeroc(project)
    assert binary.is_file()
    proc = subprocess.run([str(binary)], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert "aeroc setup" in proc.stdout
    assert "aeroc compile" in proc.stdout
    assert "aeroc verify" in proc.stdout
    assert "[WAVE] aeroc pipeline completed" in proc.stdout


@pytest.mark.skipif(not shutil.which("cargo"), reason="cargo not installed")
def test_package_aeroc_creates_zip(tmp_path):
    project = export_aeroc_project(tmp_path, tmp_path / "exported")
    compile_aeroc(project)
    archive = package_aeroc(project)
    assert archive.suffix == ".aerozip"
    assert zipfile.is_zipfile(archive)
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        assert "aeroc/aero_core/Cargo.toml" in names
        assert "aeroc/cli.py" in names


def test_export_options_modes():
    opts = ExportOptions()
    assert opts.mode == ExportMode.STRICT
    draft_opts = ExportOptions.from_dict({"mode": "draft", "run_tests": False})
    assert draft_opts.mode == ExportMode.DRAFT
    assert not draft_opts.run_tests


def test_verify_workspace_fails_strict_on_syntax_error(tmp_path):
    (tmp_path / "main.py").write_text("def broken(\n", encoding="utf-8")
    with pytest.raises(ExportVerificationError):
        verify_workspace_for_export(tmp_path, {"mode": "strict"})


def test_verify_workspace_allows_draft_with_syntax_error(tmp_path):
    (tmp_path / "main.py").write_text("def broken(\n", encoding="utf-8")
    verification = verify_workspace_for_export(tmp_path, {"mode": "draft"})
    assert verification["success"] is False
    assert verification["unverified"] is True


def test_generate_verification_json_includes_hashes_and_telemetry():
    file_hashes = {"main.py": "abc123"}
    verification = {
        "timestamp": "2026-01-01T00:00:00+00:00",
        "commit_or_version": "deadbeef",
        "test_summary": {"passed": 2, "failed": 0, "total": 2, "success": True},
        "performance_baseline": {"verification_seconds": 1.23},
        "success": True,
    }
    payload = json.loads(generate_verification_json(verification, file_hashes))
    assert payload["file_hashes"]["main.py"] == "abc123"
    assert payload["test_summary"]["passed"] == 2
    assert payload["unverified"] is False


def test_export_workspace_includes_verification_json(tmp_path):
    (tmp_path / "main.py").write_text("print('hello')\n", encoding="utf-8")
    archive_bytes, filename = export_workspace(
        tmp_path, {"pure_target": True, "run_tests": False, "run_compilation": False}
    )
    assert filename.endswith(".zip")
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zf:
        names = zf.namelist()
        assert "verification.json" in names
        data = json.loads(zf.read("verification.json"))
        assert data["success"] is True
        assert "file_hashes" in data
        assert "main.py" in data["file_hashes"]


def test_export_scaffold_zip_includes_verification_json(tmp_path):
    (tmp_path / "main.py").write_text("print('hello')\n", encoding="utf-8")
    archive = export_scaffold_zip(
        tmp_path,
        output_path=tmp_path / "out.aerozip",
        options={"run_tests": False, "run_compilation": False},
    )
    assert archive.is_file()
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        assert "verification.json" in names
        data = json.loads(zf.read("verification.json"))
        assert "file_hashes" in data
        assert "main.py" in data["file_hashes"]


def test_compile_hybrid_aeroc_includes_lineage_json(tmp_path):
    """A hybrid .aeroc embeds blueprint and healing metadata in lineage.json."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "main.py").write_text("print('hello')\n", encoding="utf-8")
    (workspace / "blueprint.aero").write_text(
        "metadata:\n  schema_version: '3.0.0'\n  description: 'demo prompt'\n",
        encoding="utf-8",
    )
    aero_dir = workspace / ".aero"
    aero_dir.mkdir()
    (aero_dir / "healing_attempts.json").write_text(
        json.dumps([{"strategy": "ast", "success": True}, {"strategy": "llm", "success": False}]),
        encoding="utf-8",
    )

    aeroc = tmp_path / "app.aeroc"
    compile_hybrid_aeroc(workspace, aeroc)

    out = tmp_path / "extracted"
    out.mkdir()
    unpack_aeroc_file(aeroc, out)

    assert (out / "lineage.json").is_file()
    lineage = json.loads((out / "lineage.json").read_text(encoding="utf-8"))
    assert lineage["blueprint"]["metadata"]["description"] == "demo prompt"
    assert lineage["healing"]["ast_attempts"] == 1
    assert lineage["healing"]["llm_attempts"] == 1
    assert lineage["healing"]["successful"] == 1
    assert lineage["healing"]["failed"] == 1


def test_compile_delta_aeroc_against_base_bundle(tmp_path):
    """A delta .aeroc contains only changed/added files and a delta manifest."""
    base_workspace = tmp_path / "base"
    base_workspace.mkdir()
    (base_workspace / "main.py").write_text("print('base')\n", encoding="utf-8")
    (base_workspace / "keep.py").write_text("print('keep')\n", encoding="utf-8")
    (base_workspace / "remove.py").write_text("print('remove')\n", encoding="utf-8")
    base_aeroc = tmp_path / "base.aeroc"
    compile_hybrid_aeroc(base_workspace, base_aeroc)

    new_workspace = tmp_path / "new"
    shutil.copytree(base_workspace, new_workspace)
    (new_workspace / "main.py").write_text("print('updated')\n", encoding="utf-8")
    (new_workspace / "new.py").write_text("print('new')\n", encoding="utf-8")
    (new_workspace / "remove.py").unlink()

    delta_aeroc = tmp_path / "delta.aeroc"
    compile_hybrid_aeroc(new_workspace, delta_aeroc, base_bundle=base_aeroc)

    out = tmp_path / "delta_extract"
    out.mkdir()
    unpack_aeroc_file(delta_aeroc, out)

    delta_manifest = json.loads((out / "delta.json").read_text(encoding="utf-8"))
    assert "main.py" in [op["path"] for op in delta_manifest["operations"] if op["op"] == "modify"]
    assert "new.py" in [op["path"] for op in delta_manifest["operations"] if op["op"] == "add"]
    assert any(op["op"] == "delete" and op["path"] == "remove.py" for op in delta_manifest["operations"])
    assert (out / "main.py").read_text(encoding="utf-8") == "print('updated')\n"
    assert (out / "new.py").is_file()
    assert not (out / "remove.py").exists()
