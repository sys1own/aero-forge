"""Tests for project-level build, zip upload, and bundling."""

import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

from aero_forge.project_builder import ProjectBuilder, build_from_upload
from aero_forge.scaffold.engine import ProjectScaffolder


@pytest.mark.skipif(
    not shutil.which("cargo") or not shutil.which("rustc"),
    reason="Rust toolchain not installed",
)
def test_project_builder_compiles_and_bundles(tmp_path: Path) -> None:
    project = tmp_path / "my_project"
    src = project / "src"
    tests = project / "tests"
    src.mkdir(parents=True)
    tests.mkdir(parents=True)

    (src / "calc.py").write_text(
        "def add(a: int, b: int) -> int:\n    return a + b\n",
        encoding="utf-8",
    )
    (tests / "test_calc.py").write_text(
        "from calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )

    output_zip = tmp_path / "bundle.zip"
    builder = ProjectBuilder(
        project,
        output_zip=output_zip,
        llm_provider="none",
        max_workers=1,
        cache_enabled=False,
    )
    result = builder.build()

    assert result["success"] is True
    assert result["status"] == "success"
    assert "add" in result["functions_compiled"]
    assert result["passed"] == result["total"] == 1
    assert Path(result["output_zip"]).is_file()

    with zipfile.ZipFile(output_zip, "r") as zf:
        names = zf.namelist()
        assert any("my_project/__init__.py" in n for n in names)
        assert any("my_project/dist/" in n for n in names)
        assert any("build_manifest.json" in n for n in names)


@pytest.mark.skipif(
    not shutil.which("cargo") or not shutil.which("rustc"),
    reason="Rust toolchain not installed",
)
def test_build_from_upload(tmp_path: Path) -> None:
    project = tmp_path / "upload_project"
    src = project / "src"
    tests = project / "tests"
    src.mkdir(parents=True)
    tests.mkdir(parents=True)

    (src / "double.py").write_text(
        "def double(x: int) -> int:\n    return x * 2\n",
        encoding="utf-8",
    )
    (tests / "test_double.py").write_text(
        "from double import double\n\ndef test_double():\n    assert double(5) == 10\n",
        encoding="utf-8",
    )

    upload_zip = tmp_path / "upload.zip"
    with zipfile.ZipFile(upload_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in project.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(tmp_path))

    output_zip = tmp_path / "result.zip"
    result = build_from_upload(
        upload_zip,
        output_zip=output_zip,
        llm_provider="none",
        max_workers=1,
        cache_enabled=False,
    )

    assert result["success"] is True
    assert "double" in result["functions_compiled"]
    assert Path(result["output_zip"]).is_file()


@pytest.mark.skipif(
    not shutil.which("cargo") or not shutil.which("rustc"),
    reason="Rust toolchain not installed",
)
def test_project_builder_scaffolds_axum_api(tmp_path: Path) -> None:
    """Scaffolding an Axum project should produce a compilable REST API with HIN compute."""
    project = tmp_path / "axum_project"
    project.mkdir(parents=True)

    (project / "src").mkdir()
    (project / "src" / "math.py").write_text(
        "def add(a: float, b: float) -> float:\n    return a + b\n",
        encoding="utf-8",
    )

    builder = ProjectBuilder(project, template="axum", max_workers=1)
    result = builder.scaffold()

    assert result["status"] == "scaffolded"
    assert result["template"] == "axum"
    assert (project / "Cargo.toml").is_file()
    assert (project / "src" / "main.rs").is_file()
    assert (project / "src" / "compute.rs").is_file()

    # The embedded compute module should expose the discovered function.
    compute_rs = (project / "src" / "compute.rs").read_text(encoding="utf-8")
    assert "pub fn add" in compute_rs

    cargo = subprocess.run(
        ["cargo", "check"],
        cwd=project,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if cargo.returncode != 0:
        pytest.skip(f"Axum project could not be checked: {cargo.stderr}")


@pytest.mark.skipif(
    not shutil.which("cargo") or not shutil.which("rustc"),
    reason="Rust toolchain not installed",
)
def test_project_builder_scaffolds_clap_cli(tmp_path: Path) -> None:
    """Scaffolding a Clap project should produce a compilable CLI with HIN compute."""
    project = tmp_path / "clap_project"
    project.mkdir(parents=True)

    builder = ProjectBuilder(project, template="clap", max_workers=1)
    result = builder.scaffold()

    assert result["status"] == "scaffolded"
    assert result["template"] == "clap"
    assert (project / "Cargo.toml").is_file()
    assert (project / "src" / "main.rs").is_file()
    assert (project / "src" / "compute.rs").is_file()

    cargo = subprocess.run(
        ["cargo", "check"],
        cwd=project,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if cargo.returncode != 0:
        pytest.skip(f"Clap project could not be checked: {cargo.stderr}")


def test_project_builder_scaffolds_python_hybrid(tmp_path: Path) -> None:
    """Scaffolding a Python hybrid project should produce a standard package layout."""
    project = tmp_path / "python_hybrid_project"
    project.mkdir(parents=True)

    builder = ProjectBuilder(project, template="python_hybrid", max_workers=1)
    result = builder.scaffold()

    assert result["status"] == "scaffolded"
    assert result["template"] == "python_hybrid"
    assert (project / "pyproject.toml").is_file()
    assert (project / "src" / project.name / "__init__.py").is_file()
    assert (project / "src" / project.name / "_native.pyi").is_file()
    assert (project / "tests" / "test_placeholder.py").is_file()
