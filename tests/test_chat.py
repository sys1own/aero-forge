"""Tests for the interactive chat session."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aero_forge.chat import ChatSession


def test_chat_session_reply(tmp_path):
    from aero_forge.generate import extract_code_blocks

    client = MagicMock()
    client.generate.return_value = "```python\ndef cube(n):\n    return n ** 3\n```"
    session = ChatSession(tmp_path)
    with patch("aero_forge.chat.get_llm_client", return_value=client):
        response = session.reply("write a function that cubes a number")
    assert "def cube" in response
    assert len(session.messages) == 3  # system + user + assistant


def test_chat_session_no_provider(tmp_path):
    session = ChatSession(tmp_path)
    with patch("aero_forge.chat.get_llm_client", return_value=None):
        response = session.reply("hello")
    assert "No LLM provider" in response


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
    assert "Aero-Forge chat mode" in result.output


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
