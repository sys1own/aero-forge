"""Tests for CLI JSON output modes."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from aero_forge.scaffold.aeroc_export import compile_hybrid_aeroc, run_aeroc_archive

CLI = [sys.executable, "-m", "aero_forge.cli"]


@pytest.mark.skipif(
    not shutil.which("cargo") or not shutil.which("rustc"),
    reason="Rust toolchain not installed",
)
def test_fix_json_output_success(tmp_path: Path) -> None:
    src = tmp_path / "myfunc.py"
    src.write_text("def double(x: int) -> int:\n    return x * 2\n", encoding="utf-8")
    test = tmp_path / "test_myfunc.py"
    test.write_text(
        "from myfunc import double\n\ndef test_double():\n    assert double(3) == 6\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        CLI
        + [
            "fix",
            str(src),
            "--function",
            "double",
            "--no-llm",
            "--json",
        ],
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["status"] == "success"
    assert payload["error"] is None
    assert payload["iterations"] == 1
    assert len(payload["rust_extensions"]) == 1


@pytest.mark.skipif(
    not shutil.which("cargo") or not shutil.which("rustc"),
    reason="Rust toolchain not installed",
)
def test_fix_json_output_failure(tmp_path: Path) -> None:
    src = tmp_path / "bad.py"
    src.write_text(
        "def bad(x: int) -> int:\n    return x + 'string'\n", encoding="utf-8"
    )
    result = subprocess.run(
        CLI
        + [
            "fix",
            str(src),
            "--function",
            "bad",
            "--no-llm",
            "--json",
        ],
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["status"] == "failure"
    assert payload["error"]["type"] == "build_error"


@pytest.mark.skipif(
    not shutil.which("cargo") or not shutil.which("rustc"),
    reason="Rust toolchain not installed",
)
def test_build_project_json_output(tmp_path: Path) -> None:
    project = tmp_path / "json_project"
    src = project / "src"
    tests = project / "tests"
    src.mkdir(parents=True)
    tests.mkdir(parents=True)
    (src / "calc.py").write_text(
        "def add(a: int, b: int) -> int:\n    return a + b\n", encoding="utf-8"
    )
    (tests / "test_calc.py").write_text(
        "from calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )
    output_zip = tmp_path / "bundle.zip"
    result = subprocess.run(
        CLI
        + [
            "build",
            "--project",
            str(project),
            "--output-zip",
            str(output_zip),
            "--no-llm",
            "--json",
        ],
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["status"] == "success"
    assert "add" in payload["functions_compiled"]
    assert payload["passed"] == payload["total"] == 1
    assert Path(payload["output_zip"]).is_file()
    assert payload["manifest"]["project"] == "json_project"


def test_run_aeroc_archive_executes_python_entrypoint(tmp_path: Path) -> None:
    """``aero run`` unpacks and executes the Python entrypoint of a .aeroc archive."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "main.py").write_text("print('aero run ok')\n", encoding="utf-8")
    aeroc = tmp_path / "app.aeroc"
    compile_hybrid_aeroc(workspace, aeroc)

    result = run_aeroc_archive(aeroc, args=[], json_output=True, keep_workdir=True)
    assert result["success"] is True
    assert "aero run ok" in result["stdout"]


def test_cli_run_subcommand_exists() -> None:
    """The ``run`` subcommand is registered in the CLI and prints usage."""
    result = subprocess.run(
        CLI + ["run", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "ARCHIVE" in result.stdout
    assert ".aeroc" in result.stdout
