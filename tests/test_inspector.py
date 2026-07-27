"""Tests for aero_forge.inspector workspace command detection and PyO3 scaffolding."""

from pathlib import Path

from aero_forge import inspector


def test_inspect_detects_main_py(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("print('ok')\n")
    commands = inspector.inspect_workspace(tmp_path)
    assert any(c["cmd"] == "python main.py" for c in commands)


def test_inspect_detects_cargo_toml(tmp_path: Path) -> None:
    (tmp_path / "Cargo.toml").write_text("[package]\nname = 'demo'\n")
    commands = inspector.inspect_workspace(tmp_path)
    assert any(c["cmd"] == "cargo build" for c in commands)
    assert any(c["cmd"] == "cargo test" for c in commands)


def test_inspect_detects_pyproject_and_pytest(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\ntestpaths = ['tests']\n")
    (tmp_path / "tests").mkdir()
    commands = inspector.inspect_workspace(tmp_path)
    assert any(c["cmd"] == "pytest" for c in commands)


def test_inspect_detects_package_json(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text(
        '{"scripts": {"start": "node index.js", "test": "jest"}}'
    )
    commands = inspector.inspect_workspace(tmp_path)
    assert any(c["cmd"] == "npm install" for c in commands)
    assert any(c["cmd"] == "npm run start" for c in commands)
    assert any(c["cmd"] == "npm run test" for c in commands)


def test_inspect_detects_makefile(tmp_path: Path) -> None:
    (tmp_path / "Makefile").write_text("test:\n\techo ok\nbuild:\n\techo build\n")
    commands = inspector.inspect_workspace(tmp_path)
    assert any(c["cmd"] == "make test" for c in commands)
    assert any(c["cmd"] == "make build" for c in commands)


def test_scaffold_pyo3_writes_crate_and_pyproject(tmp_path: Path) -> None:
    inspector.scaffold_pyo3_workspace(tmp_path, project_name="test-native")
    assert (tmp_path / "crates" / "native_core" / "Cargo.toml").is_file()
    assert (tmp_path / "crates" / "native_core" / "src" / "lib.rs").is_file()
    pyproject = tmp_path / "pyproject.toml"
    assert pyproject.is_file()
    text = pyproject.read_text(encoding="utf-8")
    assert "[tool.maturin]" in text
    assert "crates/native_core/Cargo.toml" in text
