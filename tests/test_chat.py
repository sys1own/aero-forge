"""Tests for the interactive chat session."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock, patch

import pytest

from aero_forge.chat import ChatSession


def test_chat_session_reply(tmp_path):
    """``reply`` calls the LLM and records the conversation."""
    client = MagicMock()
    client.generate.return_value = "```python\ndef cube(n):\n    return n ** 3\n```"
    session = ChatSession(tmp_path)
    with patch("aero_forge.chat.get_llm_client", return_value=client):
        response = session.reply("write a function that cubes a number")
    assert "def cube" in response
    assert len(session.messages) == 3  # system + user + assistant


def test_chat_session_no_provider(tmp_path):
    """``reply`` reports when no LLM provider is configured."""
    session = ChatSession(tmp_path)
    with patch("aero_forge.chat.get_llm_client", return_value=None):
        response = session.reply("hello")
    assert "No LLM provider" in response


def test_chat_session_remembers_context(tmp_path):
    """The session remembers the last prompt and source across turns."""
    session = ChatSession(tmp_path)
    session.last_prompt = "build a fibonacci function"
    session.last_source = "def fib(n):\n    return n"
    response = session.process("show me the code")
    assert "def fib" in response
    assert session.last_prompt == "build a fibonacci function"


def test_chat_session_optimizes_existing_code(tmp_path):
    """'make it faster' optimizes the existing generated source."""
    from unittest.mock import patch

    src = tmp_path / "src" / "generated.py"
    tests = tmp_path / "tests" / "test_generated.py"
    src.parent.mkdir(parents=True, exist_ok=True)
    tests.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("def fib(n):\n    return n\n", encoding="utf-8")
    tests.write_text(
        "from generated import fib\ndef test_fib():\n    assert fib(0) == 0\n",
        encoding="utf-8",
    )

    session = ChatSession(tmp_path)
    session.last_prompt = "build a fibonacci function"

    with (
        patch("aero_forge.chat.optimize_generated_code") as mock_opt,
        patch("aero_forge.chat.generate_and_build") as mock_build,
    ):
        mock_opt.return_value = []
        mock_build.return_value = {
            "build": {
                "success": True,
                "results": [{"function_name": "fib", "success": True}],
            },
        }
        response = session.process("make it faster")

    assert "Done!" in response or "optimized" in response.lower()
    assert session.last_prompt == "build a fibonacci function"


def test_chat_suggest_command(tmp_path):
    """The session suggests similar commands for typos."""
    session = ChatSession(tmp_path)
    assert session.suggest_command("shwo") == "show"
    assert session.suggest_command("opttimize") == "optimize"
    assert session.suggest_command("xyzabc") is None


def test_chat_summary_for_build(tmp_path):
    """A successful build produces a friendly, concise summary."""
    session = ChatSession(tmp_path)
    result = {
        "build": {
            "success": True,
            "results": [
                {"function_name": "fibonacci", "success": True},
            ],
        }
    }
    summary = session._summarize_build(result, "build a fibonacci function")
    assert "Done!" in summary or "fibonacci" in summary
    assert "dist" in summary


def test_chat_summary_for_failed_build(tmp_path):
    """A failed build produces a friendly error message."""
    session = ChatSession(tmp_path)
    result = {"build": {"success": False, "error": "Rust compilation failed"}}
    summary = session._summarize_build(result, "build a broken function")
    assert "Oops" in summary
    assert "explain" in summary


def test_chat_session_save_and_load(tmp_path):
    """Session state can be saved and resumed by ``session_id``."""
    session = ChatSession(tmp_path, session_id="test-session-42")
    session.messages = [{"role": "user", "content": "hello"}]
    session.last_prompt = "build a fibonacci function"
    session._save_session()

    loaded = ChatSession(tmp_path, session_id="test-session-42")
    assert loaded.last_prompt == "build a fibonacci function"
    assert loaded.messages == [{"role": "user", "content": "hello"}]


def test_chat_cli_command(tmp_path):
    """The ``aero-forge chat`` command accepts input and replies."""
    from click.testing import CliRunner
    from aero_forge.cli import main

    client = MagicMock()
    client.generate.return_value = "```python\ndef chat_greet():\n    return 1\n```"
    runner = CliRunner()
    with patch("aero_forge.chat.get_llm_client", return_value=client):
        result = runner.invoke(
            main,
            ["chat", "--output-dir", str(tmp_path), "--llm-provider", "openai"],
            input="hello\nexit\n",
        )

    assert result.exit_code == 0
    assert "Aero-Forge chat is ready" in result.output


def test_chat_help_command(tmp_path):
    """The ``help`` command lists available chat commands."""
    from click.testing import CliRunner
    from aero_forge.cli import main

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["chat", "--output-dir", str(tmp_path), "--llm-provider", "none"],
        input="help\nexit\n",
    )
    assert result.exit_code == 0
    assert "generate" in result.output
    assert "build" in result.output


def test_chat_show_without_code(tmp_path):
    """The ``show`` command reports when no generated code exists yet."""
    from click.testing import CliRunner
    from aero_forge.cli import main

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["chat", "--output-dir", str(tmp_path), "--llm-provider", "none"],
        input="show\nexit\n",
    )
    assert result.exit_code == 0
    assert "No generated code" in result.output


def test_chat_cli_resumes_session(tmp_path):
    """The ``--session-id`` flag resumes a previous session."""
    from click.testing import CliRunner
    from aero_forge.cli import main

    # Prime a session on disk.
    session = ChatSession(tmp_path, session_id="resume-me")
    session.messages = [{"role": "user", "content": "hello"}]
    session._save_session()

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "chat",
            "--output-dir",
            str(tmp_path),
            "--llm-provider",
            "none",
            "--session-id",
            "resume-me",
        ],
        input="exit\n",
    )
    assert result.exit_code == 0
    assert "Resuming session" in result.output
    assert "resume-me" in result.output


def test_chat_copilot_injects_workspace_bundle(tmp_path: Path) -> None:
    """The copilot system prompt includes a workspace bundle from bundle_repo."""
    (tmp_path / "main.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    captured_messages: List[Any] = []

    class FakeClient:
        def generate(self, messages, temperature=0.2):
            captured_messages.extend(messages)
            return '{"reply": "Done", "action": null}'

    session = ChatSession(tmp_path)
    with patch("aero_forge.chat.get_llm_client", return_value=FakeClient()):
        result = session.reply_structured("How can I speed this up?")

    assert result["reply"] == "Done"
    assert result["action"] is None
    assert captured_messages
    system = captured_messages[0]["content"]
    assert "CURRENT_PROJECT_CONTEXT" in system
    assert "main.py" in system
    assert "def add(a, b)" in system


def test_chat_copilot_parses_propose_build_action(tmp_path: Path) -> None:
    """reply_structured extracts a PROPOSE_BUILD action from a JSON reply."""
    fake_response = (
        '{"reply": "Use a Rust core for the hot loop.", '
        '"action": {"type": "PROPOSE_BUILD", "params": '
        '{"prompt": "matrix multiplication in rust", "target": "hybrid_cpp_rust", '
        '"acceleration": "Selective Acceleration (Auto-Detect Heavy Compute)"}}}'
    )

    class FakeClient:
        def generate(self, messages, temperature=0.2, **kwargs):
            return fake_response

    session = ChatSession(tmp_path)
    with patch("aero_forge.chat.get_llm_client", return_value=FakeClient()):
        result = session.reply_structured("Speed up matrix multiplication")

    assert result["reply"] == "Use a Rust core for the hot loop."
    assert result["action"]["type"] == "PROPOSE_BUILD"
    assert result["action"]["params"]["target"] == "hybrid_cpp_rust"


def test_chat_copilot_enforces_json_propose_build_for_fibonacci(tmp_path: Path) -> None:
    """A build-oriented prompt returns a valid PROPOSE_BUILD action payload."""
    fake_response = (
        '{"reply": "I will propose a pure-Python Fibonacci implementation.", '
        '"action": {"type": "PROPOSE_BUILD", "params": '
        '{"prompt": "Build a fast iterative Fibonacci function in Python", '
        '"target": "pure_python", '
        '"acceleration": "Standard Runtime (Bypass Bridge)"}}}'
    )

    class FakeClient:
        def generate(self, messages, temperature=0.2, **kwargs):
            return fake_response

    session = ChatSession(tmp_path)
    with patch("aero_forge.chat.get_llm_client", return_value=FakeClient()):
        result = session.reply_structured(
            "I want to build a fast, iterative Fibonacci function"
        )

    assert result["action"]["type"] == "PROPOSE_BUILD"
    assert "prompt" in result["action"]["params"]
    assert result["action"]["params"]["target"] in {
        "pure_python",
        "hybrid_rust_python",
        "pure_rust",
    }
    assert result["action"]["params"]["acceleration"] in {
        "Selective Acceleration (Auto-Detect Heavy Compute)",
        "Force Native Bridge",
        "Standard Runtime (Bypass Bridge)",
    }


def test_chat_copilot_fallback_builds_action_from_plain_text(tmp_path: Path) -> None:
    """If the LLM returns plain prose for a build request, recover a PROPOSE_BUILD action."""
    class FakeClient:
        def generate(self, messages, temperature=0.2, **kwargs):
            return "I'll build a fast Rust Fibonacci core wrapped in Python."

    session = ChatSession(tmp_path)
    with patch("aero_forge.chat.get_llm_client", return_value=FakeClient()):
        result = session.reply_structured("I want to build a fast Fibonacci function")

    assert result["action"]["type"] == "PROPOSE_BUILD"
    assert result["action"]["params"]["target"] == "hybrid_rust_python"


def test_chat_copilot_falls_back_to_plain_text_for_non_json(tmp_path: Path) -> None:
    """Non-JSON assistant replies are returned as the reply with no action."""
    class FakeClient:
        def generate(self, messages, temperature=0.2):
            return "Just a plain text response."

    session = ChatSession(tmp_path)
    with patch("aero_forge.chat.get_llm_client", return_value=FakeClient()):
        result = session.reply_structured("Hello")

    assert result["reply"] == "Just a plain text response."
    assert result["action"] is None


def test_chat_reply_returns_extracted_markdown_for_json(tmp_path: Path) -> None:
    """The legacy ``reply`` method returns only the Markdown portion of a JSON response."""
    class FakeClient:
        def generate(self, messages, temperature=0.2):
            return '{"reply": "**Hello**", "action": null}'

    session = ChatSession(tmp_path)
    with patch("aero_forge.chat.get_llm_client", return_value=FakeClient()):
        response = session.reply("Hi")

    assert response == "**Hello**"


def test_chat_blank_workspace_plan(tmp_path: Path) -> None:
    """On a blank workspace the copilot returns a plan and a PROPOSE_BUILD action."""
    fake_response = (
        '{"reply": "Starting a blank project with a pure Python core.", '
        '"action": {"type": "PROPOSE_BUILD", "params": '
        '{"prompt": "Build a fast iterative Fibonacci function in Python", '
        '"target": "pure_python", '
        '"acceleration": "Standard Runtime (Bypass Bridge)"}}}'
    )

    class FakeClient:
        def generate(self, messages, temperature=0.2, **kwargs):
            return fake_response

    session = ChatSession(tmp_path)
    with patch("aero_forge.chat.get_llm_client", return_value=FakeClient()):
        result = session.reply_structured(
            "I want to build a fast, iterative Fibonacci function on a blank workspace"
        )

    assert result["reply"]
    assert result["action"]["type"] == "PROPOSE_BUILD"
    assert result["action"]["params"]["target"] in {
        "pure_python",
        "hybrid_rust_python",
        "pure_rust",
    }
    assert "prompt" in result["action"]["params"]


def test_chat_existing_workspace_feature_plan(tmp_path: Path) -> None:
    """On an existing workspace the copilot uses bundle_repo context and proposes a feature."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "math_core.py").write_text(
        "def add(a, b):\n    return a + b\n", encoding="utf-8"
    )
    (tmp_path / "blueprint.aero").write_text(
        "project: math\narchitecture: pure_python\n", encoding="utf-8"
    )

    captured_messages: List[Any] = []

    class FakeClient:
        def generate(self, messages, temperature=0.2, **kwargs):
            captured_messages.extend(messages)
            return (
                '{"reply": "I see an existing pure_python workspace. I will add a multiply function.", '
                '"action": {"type": "PROPOSE_BUILD", "params": '
                '{"prompt": "Add a multiply function to the existing pure_python math_core project", '
                '"target": "pure_python", '
                '"acceleration": "Standard Runtime (Bypass Bridge)"}}}'
            )

    session = ChatSession(tmp_path)
    with patch("aero_forge.chat.get_llm_client", return_value=FakeClient()):
        result = session.reply_structured("Add a multiply function to math_core")

    assert result["reply"]
    assert result["action"]["type"] == "PROPOSE_BUILD"
    assert result["action"]["params"]["target"] == "pure_python"
    system = str([m for m in captured_messages if m.get("role") == "system"])
    assert "src/math_core.py" in system or "def add(a, b)" in system


def test_chat_empty_response_fallback(tmp_path: Path) -> None:
    """An empty LLM response still returns a valid {reply, action} dict."""
    calls: List[Any] = []

    class FakeClient:
        def generate(self, messages, temperature=0.2, **kwargs):
            calls.append(kwargs)
            return ""

    session = ChatSession(tmp_path)
    with patch("aero_forge.chat.get_llm_client", return_value=FakeClient()):
        result = session.reply_structured("build a fibonacci function")

    assert result["reply"]
    assert "reply" in result and "action" in result
    # No response_format was honored, but the result is still valid.
    assert result["action"]["type"] == "PROPOSE_BUILD"


def test_chat_malformed_prose_fallback(tmp_path: Path) -> None:
    """Malformed or prose LLM output is safely wrapped into {reply, action}."""
    class FakeClient:
        def generate(self, messages, temperature=0.2, **kwargs):
            return "Here is a Build prompt: create a Rust core for matrix multiplication."

    session = ChatSession(tmp_path)
    with patch("aero_forge.chat.get_llm_client", return_value=FakeClient()):
        result = session.reply_structured("How do I speed up matrix multiplication?")

    assert result["reply"]
    assert result["action"]["type"] == "PROPOSE_BUILD"
    assert result["action"]["params"]["target"] == "pure_rust"


def test_chat_copilot_parses_markdown_yaml_contract(tmp_path: Path) -> None:
    """A Markdown + yaml blueprint code fence is split into reply and action."""
    fake_response = """\
## Overview
Build an accelerated numeric compute core using Rust with a Python driver.

## Polyglot Boundaries
- Rust handles the hot loop.
- Python provides the CLI and orchestration.

## Build Contract
```yaml blueprint
prompt: Build a fast Fibonacci function in Python using @accelerate and PyO3 Rust backings
target: hybrid_rust_python
acceleration: "Selective Acceleration (Auto-Detect Heavy Compute)"
```
"""

    class FakeClient:
        def generate(self, messages, temperature=0.2, **kwargs):
            return fake_response

    session = ChatSession(tmp_path)
    with patch("aero_forge.chat.get_llm_client", return_value=FakeClient()):
        result = session.reply_structured("I want to build a fast Fibonacci function")

    assert result["reply"]
    assert "## Overview" in result["reply"]
    assert "```yaml blueprint" not in result["reply"]
    assert result["action"]["type"] == "PROPOSE_BUILD"
    assert result["action"]["params"]["target"] == "hybrid_rust_python"
    assert "Fibonacci" in result["action"]["params"]["prompt"]
