"""Tests for the standalone ``.aeroc`` export acceleration system."""

import shutil
import subprocess
import zipfile

import pytest

from aero_forge.scaffold.aeroc_export import (
    compile_aeroc,
    export_aeroc_project,
    package_aeroc,
)
from aero_forge.scaffold.cargo_config import CARGO_CONFIG_TOML, write_cargo_config


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
