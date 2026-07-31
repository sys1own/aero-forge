"""Tests for the HIN native bundle (``.hinb``) export, CLI inspection, and runtime."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

from click.testing import CliRunner

from aero_forge.cli import main
from aero_forge.hin_runtime import HINBundle, load_hin_bundle
from aero_forge.scaffold.export_options import export_workspace


def _write_workspace(path: Path) -> None:
    src = path / "src"
    src.mkdir(parents=True)
    (src / "main.py").write_text("def add(a: float, b: float) -> float:\n    return a + b\n")
    (path / "blueprint.aero").write_text("project: test_add\n")
    (path / "environment.lock").write_text(json.dumps({"python": "3.10"}))


def test_hinb_export_contains_manifest_and_graphs(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _write_workspace(workspace)

    archive_bytes, filename = export_workspace(
        workspace,
        {
            "hin_native_bundle": True,
            "pure_target": False,
            "run_tests": False,
            "run_compilation": False,
            "engine_backend": "hin_cpu",
            "precision_mode": "fast_math",
        },
        project_name="test_add",
    )
    assert filename == "test_add.zip"
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as outer:
        hinb_names = [n for n in outer.namelist() if n.endswith(".hinb")]
        assert hinb_names == ["test_add.hinb"]
        with zipfile.ZipFile(io.BytesIO(outer.read(hinb_names[0]))) as hinb:
            assert "manifest.json" in hinb.namelist()
            manifest = json.loads(hinb.read("manifest.json"))
            assert manifest["project"] == "test_add"
            assert manifest["hin_version"] == "1.0"
            assert manifest["precision_mode"] == "fast_math"
            assert manifest["default_backend"] == "hin_cpu"
            assert len(manifest["entrypoints"]) == 1
            assert manifest["entrypoints"][0]["name"] == "add"
            assert [i["name"] for i in manifest["entrypoints"][0]["inputs"]] == ["a", "b"]


def test_hinb_cli_inspect(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _write_workspace(workspace)

    archive_bytes, _ = export_workspace(
        workspace,
        {
            "hin_native_bundle": True,
            "pure_target": False,
            "run_tests": False,
            "run_compilation": False,
        },
        project_name="test_add",
    )
    hinb_path = tmp_path / "test_add.hinb"
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as outer:
        hinb_path.write_bytes(outer.read("test_add.hinb"))

    runner = CliRunner()
    result = runner.invoke(main, ["inspect", str(hinb_path)])
    assert result.exit_code == 0
    output = result.output
    assert "Bundle:" in output
    assert "test_add" in output
    assert "HIN version:" in output
    assert "add(a:float64[], b:float64[])" in output


def test_hinb_runtime_load_and_execute(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _write_workspace(workspace)

    archive_bytes, _ = export_workspace(
        workspace,
        {
            "hin_native_bundle": True,
            "pure_target": False,
            "run_tests": False,
            "run_compilation": False,
        },
        project_name="test_add",
    )
    hinb_path = tmp_path / "test_add.hinb"
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as outer:
        hinb_path.write_bytes(outer.read("test_add.hinb"))

    bundle = load_hin_bundle(hinb_path)
    assert bundle.project == "test_add"
    assert len(bundle.input_schema) == 2
    assert bundle.run("add", 2.0, 3.0) == 5.0
