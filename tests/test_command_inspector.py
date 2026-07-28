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
    labels = [c["label"] for c in cmds]
    assert any("Run server" in label for label in labels)
    assert any(c["cmd"] == "cargo test" for c in cmds)


def test_detects_python_main_and_pytest(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text(
        'if __name__ == "__main__":\n    print("ok")\n', encoding="utf-8"
    )
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_ok.py").write_text("def test_ok(): pass")
    cmds = detect_runnable_commands(tmp_path)
    labels = [c["label"] for c in cmds]
    assert any("Run main" in label for label in labels)
    assert any(c["cmd"] == "pytest" for c in cmds)


def test_detects_cmake_and_make_targets(tmp_path: Path) -> None:
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.0)\nadd_executable(runner src/main.cpp)\n",
        encoding="utf-8",
    )
    (tmp_path / "Makefile").write_text("run:\n\techo ok\n", encoding="utf-8")
    cmds = detect_runnable_commands(tmp_path)
    assert any("runner" in c["label"] for c in cmds)
    assert any("make run" == c["cmd"] for c in cmds)
