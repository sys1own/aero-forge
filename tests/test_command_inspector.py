"""Tests for the ingestion command inspector."""

from pathlib import Path

from aero_forge.ingestion.command_inspector import detect_runnable_commands


def test_detects_cargo_run_and_test(tmp_path: Path) -> None:
    (tmp_path / "Cargo.toml").write_text(
        '[package]\nname = "demo"\n\n[[bin]]\nname = "server"\n',
        encoding="utf-8",
    )
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.rs").write_text("fn main() {}")
    cmds = detect_runnable_commands(tmp_path)
    assert any(c["cmd"] == "cargo run --bin server" and c["category"] == "run" for c in cmds)
    assert any(c["cmd"] == "cargo test" and c["category"] == "test" for c in cmds)
    assert any(c["cmd"] == "cargo build" and c["category"] == "build" for c in cmds)


def test_detects_python_main_and_pytest(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text(
        'if __name__ == "__main__":\n    print("ok")\n', encoding="utf-8"
    )
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_ok.py").write_text("def test_ok(): pass")
    cmds = detect_runnable_commands(tmp_path)
    assert any(c["cmd"] == "python main.py" and c["category"] == "run" and c["primary"] for c in cmds)
    assert any(c["cmd"] == "pytest" and c["category"] == "test" for c in cmds)


def test_detects_app_and_manage_py(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text('if __name__ == "__main__":\n    pass\n')
    (tmp_path / "manage.py").write_text('if __name__ == "__main__":\n    pass\n')
    cmds = detect_runnable_commands(tmp_path)
    assert any(c["name"] == "Run app" for c in cmds)
    assert any(c["name"] == "Run manage" for c in cmds)


def test_detects_cmake_and_make_targets(tmp_path: Path) -> None:
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.0)\nadd_executable(runner src/main.cpp)\n",
        encoding="utf-8",
    )
    (tmp_path / "Makefile").write_text("run:\n\techo ok\ntest:\n\techo test\nbuild:\n\techo build\n")
    cmds = detect_runnable_commands(tmp_path)
    assert any("runner" in c["name"] and c["category"] == "run" for c in cmds)
    assert any(c["cmd"] == "make test" and c["category"] == "test" for c in cmds)
    assert any(c["cmd"] == "make build" and c["category"] == "build" for c in cmds)


def test_detects_pyproject_scripts_and_pytest(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "demo"\nscripts = { "demo" = "demo.cli:main" }\n'
        '[tool.pytest.ini_options]\ntestpaths = ["tests"]\n'
    )
    cmds = detect_runnable_commands(tmp_path)
    assert any(c["cmd"] == "python -m demo.cli --help" and c["category"] == "run" for c in cmds)
    assert any(c["cmd"] == "pytest" and c["category"] == "test" for c in cmds)
    assert any(c["cmd"] == "pip install -e ." and c["category"] == "build" for c in cmds)


def test_detects_rust_examples(tmp_path: Path) -> None:
    (tmp_path / "Cargo.toml").write_text('[package]\nname = "demo"\n')
    (tmp_path / "examples").mkdir()
    (tmp_path / "examples" / "bench.rs").write_text("fn main() {}")
    cmds = detect_runnable_commands(tmp_path)
    assert any(c["cmd"] == "cargo run --example bench" and c["category"] == "run" for c in cmds)


def test_detects_blueprint_commands(tmp_path: Path) -> None:
    (tmp_path / "blueprint.aero").write_text(
        "metadata:\n  project_name: demo\n"
        "execution_strategy:\n  primary_entrypoint: main.py\n"
        "verification_nodes:\n  - test_id: smoke\n    command: python tests/test_smoke.py\n"
    )
    cmds = detect_runnable_commands(tmp_path)
    assert any(c["cmd"] == "main.py" and c["category"] == "run" for c in cmds)
    assert any("smoke" in c["name"] and c["category"] == "test" for c in cmds)


def test_unwraps_single_nested_zip_root(tmp_path: Path) -> None:
    """Detect commands inside a wrapper directory created by a zip archive."""
    root = tmp_path / "archive_name"
    root.mkdir()
    (root / "main.py").write_text('if __name__ == "__main__":\n    pass\n')
    (root / "tests").mkdir()
    (root / "tests" / "test_ok.py").write_text("def test_ok(): pass")
    cmds = detect_runnable_commands(tmp_path)
    assert any(c["cmd"] == "python main.py" for c in cmds)
    assert any(c["cmd"] == "pytest" for c in cmds)


def test_detects_nested_cargo_manifest(tmp_path: Path) -> None:
    """Rust crates under a subdirectory get commands with --manifest-path."""
    (tmp_path / "README.md").write_text("# project\n")
    crate = tmp_path / "rust_core"
    crate.mkdir()
    (crate / "Cargo.toml").write_text(
        '[package]\nname = "native"\nversion = "0.1.0"\n\n[[bin]]\nname = "native"\n',
        encoding="utf-8",
    )
    (crate / "src").mkdir()
    (crate / "src" / "main.rs").write_text("fn main() {}")
    (crate / "examples").mkdir()
    (crate / "examples" / "bench.rs").write_text("fn main() {}")
    cmds = detect_runnable_commands(tmp_path)
    assert any(
        c["cmd"] == "cargo run --manifest-path rust_core/Cargo.toml --bin native"
        and c["category"] == "run"
        for c in cmds
    )
    assert any(
        c["cmd"] == "cargo test --manifest-path rust_core/Cargo.toml"
        and c["category"] == "test"
        for c in cmds
    )
    assert any(
        c["cmd"] == "cargo build --manifest-path rust_core/Cargo.toml"
        and c["category"] == "build"
        for c in cmds
    )
    assert any(
        c["cmd"] == "cargo run --manifest-path rust_core/Cargo.toml --example bench"
        and c["category"] == "run"
        for c in cmds
    )


def test_ignores_jinja_template_cargo_manifests(tmp_path: Path) -> None:
    """Template Cargo.toml files are not treated as runnable crates."""
    (tmp_path / "Cargo.toml").write_text(
        '[package]\nname = "demo"\nversion = "0.1.0"\n',
        encoding="utf-8",
    )
    (tmp_path / "src" / "main.rs").parent.mkdir(parents=True)
    (tmp_path / "src" / "main.rs").write_text("fn main() {}")

    templates = tmp_path / "src" / "accelerator" / "templates"
    templates.mkdir(parents=True)
    (templates / "Cargo.toml").write_text(
        '[package]\nname = "{{ crate_name }}"\nversion = "{{ version }}"\n',
        encoding="utf-8",
    )
    (templates / "lib.rs").write_text("// {{ template }}")

    cmds = detect_runnable_commands(tmp_path)
    assert any(c["cmd"] == "cargo run --bin demo" and c["category"] == "run" for c in cmds)
    assert not any("templates/Cargo.toml" in c["cmd"] for c in cmds)


def test_ignores_template_directory_layout(tmp_path: Path) -> None:
    """Rust source inside a templates directory does not produce cargo commands."""
    crate = tmp_path / "rust_core"
    crate.mkdir()
    (crate / "Cargo.toml").write_text(
        '[package]\nname = "native"\nversion = "0.1.0"\n',
        encoding="utf-8",
    )
    (crate / "src" / "main.rs").parent.mkdir(parents=True)
    (crate / "src" / "main.rs").write_text("fn main() {}")

    templates = tmp_path / "templates"
    templates.mkdir()
    (templates / "Cargo.toml").write_text('[package]\nname = "tpl"\n')
    (templates / "src" / "main.rs").parent.mkdir(parents=True)
    (templates / "src" / "main.rs").write_text("fn main() {}")

    cmds = detect_runnable_commands(tmp_path)
    assert any("rust_core" in c["cmd"] for c in cmds)
    assert not any("templates" in c["cmd"] for c in cmds)
