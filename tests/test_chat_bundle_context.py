"""Tests that the chat engine injects a workspace bundle into multi-file prompts."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aero_forge.chat import ChatSession


class FakeLLM:
    def __init__(self, response: str):
        self.response = response

    def generate(self, messages, **kwargs):
        self.messages = messages
        return self.response


def test_multi_file_generate_includes_project_context(tmp_path: Path) -> None:
    (tmp_path / "Cargo.toml").write_text("[workspace]\n")
    (tmp_path / "rust_core" / "src").mkdir(parents=True)
    (tmp_path / "rust_core" / "src" / "lib.rs").write_text("pub fn add() {}\n")
    (tmp_path / "scripts" / "run.py").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "scripts" / "run.py").write_text("print('ok')\n")

    fake_response = (
        "```rust\n// file: rust_core/src/lib.rs\npub fn add() {}\n```\n"
        "```python\n# file: scripts/run.py\nprint('updated')\n```\n"
    )
    fake_llm = FakeLLM(fake_response)

    with patch("aero_forge.chat.get_llm_client", return_value=fake_llm), \
         patch("aero_forge.chat.run_cargo") as mock_run_cargo, \
         patch("aero_forge.chat.subprocess.run") as mock_subprocess_run:
        # Cargo build and pytest should appear to succeed.
        mock_run_cargo.return_value = MagicMock(returncode=0, stdout="", stderr="")
        mock_subprocess_run.return_value = MagicMock(returncode=0, stdout="1 passed", stderr="")

        session = ChatSession(
            tmp_path,
            llm_provider="deepseek",
            model="deepseek-v4-pro",
            progress_callback=lambda _: None,
        )
        result = session.process(
            "Update the Rust core to support subtraction. "
            "This is a hybrid Python and Rust project."
        )

    assert session.last_build_result["success"]
    # The system prompt should contain the workspace bundle.
    system_message = fake_llm.messages[0]["content"]
    assert "CURRENT_PROJECT_CONTEXT" in system_message
    assert '<file path="Cargo.toml">' in system_message
    assert '<file path="rust_core/src/lib.rs">' in system_message
    assert '<file path="scripts/run.py">' in system_message
    # The project context block (after the marker) should not list build artifacts.
    context_block = system_message.split("CURRENT_PROJECT_CONTEXT", 1)[-1]
    assert '<file path="target/' not in context_block
    assert '<file path="dist/' not in context_block
    assert '<file path="__pycache__/' not in context_block

    # Files should have been written from the LLM response.
    assert (tmp_path / "rust_core" / "src" / "lib.rs").read_text(encoding="utf-8") == "pub fn add() {}\n"
    assert (tmp_path / "scripts" / "run.py").read_text(encoding="utf-8") == "print('updated')\n"


def test_bundle_excludes_build_artifacts_from_context(tmp_path: Path) -> None:
    (tmp_path / "src" / "main.py").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "src" / "main.py").write_text("def run(): pass\n")
    (tmp_path / "target" / "release").mkdir(parents=True)
    (tmp_path / "target" / "release" / "libx.so").write_bytes(b"\x00binary")
    (tmp_path / "dist" / "libx.so").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "dist" / "libx.so").write_bytes(b"\x00binary")

    fake_response = "```python\n# file: src/main.py\ndef run(): return 1\n```\n"
    fake_llm = FakeLLM(fake_response)

    with patch("aero_forge.chat.get_llm_client", return_value=fake_llm), \
         patch("aero_forge.chat.run_cargo") as mock_run_cargo, \
         patch("aero_forge.chat.subprocess.run") as mock_subprocess_run:
        mock_run_cargo.return_value = MagicMock(returncode=0, stdout="", stderr="")
        mock_subprocess_run.return_value = MagicMock(returncode=0, stdout="1 passed", stderr="")

        session = ChatSession(
            tmp_path,
            llm_provider="deepseek",
            model="deepseek-v4-pro",
            progress_callback=lambda _: None,
        )
        result = session.process("Update the Python engine. This is a hybrid Python and Rust project.")

    assert session.last_build_result["success"]
    system_message = fake_llm.messages[0]["content"]
    assert "CURRENT_PROJECT_CONTEXT" in system_message
    assert "src/main.py" in system_message
    context_block = system_message.split("CURRENT_PROJECT_CONTEXT", 1)[-1]
    assert "target/release/libx.so" not in context_block
    assert "dist/libx.so" not in context_block


def test_multi_file_generate_self_heals_test_typo(tmp_path: Path) -> None:
    fake_response = (
        "```python\n"
        "# file: tests/test_demo.py\n"
        "def test_demo():\n"
        "    rstats = {}\n"
        "    assert r_stats == {}\n"
        "```\n"
    )
    fake_llm = FakeLLM(fake_response)

    run_count = {"n": 0}
    def fake_subprocess_run(cmd, **kwargs):
        run_count["n"] += 1
        if run_count["n"] == 1:
            return MagicMock(
                returncode=1,
                stdout="",
                stderr="tests/test_demo.py:3: NameError: name 'r_stats' is not defined",
            )
        return MagicMock(returncode=0, stdout="1 passed", stderr="")

    with patch("aero_forge.chat.get_llm_client", return_value=fake_llm), \
         patch("aero_forge.chat.run_cargo") as mock_run_cargo, \
         patch("aero_forge.chat.subprocess.run", side_effect=fake_subprocess_run):
        mock_run_cargo.return_value = MagicMock(returncode=0, stdout="", stderr="")

        session = ChatSession(
            tmp_path,
            llm_provider="deepseek",
            model="deepseek-v4-pro",
            progress_callback=lambda _: None,
        )
        session.process("Build a hybrid Python and Rust demo project with tests.")

    assert session.last_build_result["success"]
    assert run_count["n"] == 2
    fixed_test = (tmp_path / "tests" / "test_demo.py").read_text(encoding="utf-8")
    assert "rstats" in fixed_test
    assert "r_stats" not in fixed_test
