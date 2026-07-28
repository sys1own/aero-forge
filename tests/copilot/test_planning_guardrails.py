"""Tests for Copilot engine grounding and polyglot build-planning guardrails."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock, patch

import pytest

from aero_forge.chat import ChatSession
from aero_forge.copilot.agent import format_copilot_response
from aero_forge.copilot.prompts import COPILOT_SYSTEM_PROMPT


def test_copilot_system_prompt_contains_engine_identity():
    """The system prompt grounds Copilot in Aero Forge's identity and scope."""
    assert "Aero Forge" in COPILOT_SYSTEM_PROMPT
    assert "high-performance polyglot materialization engine" in COPILOT_SYSTEM_PROMPT
    assert "Python, Rust, C/C++" in COPILOT_SYSTEM_PROMPT


def test_copilot_system_prompt_rejects_unsupported_runtimes():
    """The system prompt explicitly forbids proposing unsupported runtimes."""
    assert "JavaScript" in COPILOT_SYSTEM_PROMPT
    assert "Node.js" in COPILOT_SYSTEM_PROMPT
    assert "NOT supported" in COPILOT_SYSTEM_PROMPT


def test_copilot_system_prompt_mandates_polyglot_architecture():
    """The system prompt requires native polyglot architecture for multi-language builds."""
    assert "native polyglot architecture" in COPILOT_SYSTEM_PROMPT
    assert "PyO3" in COPILOT_SYSTEM_PROMPT
    assert "C-ABI" in COPILOT_SYSTEM_PROMPT
    assert "avoid crude subprocess wrappers" in COPILOT_SYSTEM_PROMPT.lower()


def test_copilot_system_prompt_has_capability_guardrail():
    """The system prompt includes the three-step capability guardrail."""
    assert "Capability Guardrail" in COPILOT_SYSTEM_PROMPT
    assert "Explain that Aero Forge is a high-performance polyglot workspace generator" in COPILOT_SYSTEM_PROMPT
    assert "Offer a viable polyglot design" in COPILOT_SYSTEM_PROMPT


def test_copilot_system_prompt_demands_realistic_build_prompts():
    """The system prompt requires modular repo structure, boundaries, entrypoints, and blueprint integration."""
    assert "Modular repository structure" in COPILOT_SYSTEM_PROMPT
    assert "cross-language boundary contracts" in COPILOT_SYSTEM_PROMPT
    assert "Exact executable entrypoints" in COPILOT_SYSTEM_PROMPT
    assert "Integration with `blueprint.aero`" in COPILOT_SYSTEM_PROMPT


def test_format_copilot_response_preserves_tri_polyglot_plan():
    """A tri-polyglot build response surfaces a native polyglot prompt and target."""
    response = (
        '{"display_text": "Rust scheduler + C++ kernels + Python API.", '
        '"action": {"type": "build", "clean_prompt": "Build a tri_polyglot_rust_cpp_python workspace: '
        'rust_core scheduler, cpp_engine kernels with C-ABI bindings, python_interface API. '
        'Use PyO3 and shared C-ABI buffers. Target: tri_polyglot_rust_cpp_python. '
        'Acceleration: Selective Acceleration (Auto-Detect Heavy Compute).", '
        '"parameters": {"target": "tri_polyglot_rust_cpp_python", '
        '"acceleration": "Selective Acceleration (Auto-Detect Heavy Compute)"}}}'
    )
    display, action = format_copilot_response(response)
    assert action is not None
    assert action["params"]["target"] == "tri_polyglot_rust_cpp_python"
    prompt = action["params"]["prompt"].lower()
    assert "rust" in prompt
    assert "cpp" in prompt or "c++" in prompt
    assert "python" in prompt
    assert "subprocess" not in prompt
    assert "pure_python" not in prompt


def test_format_copilot_response_handles_unsupported_runtime_guardrail():
    """An unsupported-runtime response yields no build action and an explanatory display text."""
    response = (
        '{"display_text": "Aero Forge is a high-performance polyglot workspace generator focused on '
        'Python, Rust, and C/C++. Node.js is not supported because the engine has no JS runtime '
        'or npm toolchain. I can build an equivalent Python/Rust/C++ polyglot API instead.", '
        '"action": null}'
    )
    display, action = format_copilot_response(response)
    assert action is None
    assert "Aero Forge" in display
    assert "Node.js" in display or "JavaScript" in display
    assert "Python, Rust, and C/C++" in display


def test_chat_tri_language_query_uses_polyglot_target(tmp_path: Path) -> None:
    """A tri-language request produces a tri_polyglot action target, not a pure Python fallback."""
    fake_response = (
        '{"display_text": "Tri-language orchestration engine.", '
        '"action": {"type": "build", "clean_prompt": "Build a tri_polyglot_rust_cpp_python workspace '
        'with rust_core scheduler, cpp_engine C-ABI task runner, and python_interface PyO3 API. '
        'Target: tri_polyglot_rust_cpp_python. Acceleration: Selective Acceleration.", '
        '"parameters": {"target": "tri_polyglot_rust_cpp_python", '
        '"acceleration": "Selective Acceleration (Auto-Detect Heavy Compute)"}}}'
    )

    class FakeClient:
        def generate(self, messages: List[Any], temperature: float = 0.2, **kwargs):
            return fake_response

    session = ChatSession(tmp_path)
    with patch("aero_forge.chat.get_llm_client", return_value=FakeClient()):
        result = session.reply_structured("Design a prompt for a tri-language orchestration engine")

    assert result["action"]["params"]["target"] == "tri_polyglot_rust_cpp_python"
    assert "pure_python" not in result["action"]["params"]["prompt"].lower()
    assert "subprocess" not in result["action"]["params"]["prompt"].lower()


def test_chat_unsupported_language_query_returns_guardrail(tmp_path: Path) -> None:
    """A request for an unsupported runtime yields a guardrail explanation and no build action."""
    fake_response = (
        '{"display_text": "Aero Forge focuses on Python, Rust, and C/C++. Go is not supported as a '
        'build target. I can implement an equivalent polyglot service using Python for the API, '
        'Rust for the concurrency core, and C++ for compute kernels.", "action": null}'
    )

    class FakeClient:
        def generate(self, messages: List[Any], temperature: float = 0.2, **kwargs):
            return fake_response

    session = ChatSession(tmp_path)
    with patch("aero_forge.chat.get_llm_client", return_value=FakeClient()):
        result = session.reply_structured("Build a Go microservice with WebSockets")

    assert result["action"] is None
    assert "Aero Forge" in result["reply"]
    assert "Go" in result["reply"]
    assert "Python, Rust, and C/C++" in result["reply"]
