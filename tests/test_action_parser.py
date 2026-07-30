"""Tests for the Co-pilot ActionParser and clean-prompt extraction pipeline."""

from __future__ import annotations

import json

import pytest

from aero_forge.copilot.action_parser import ActionParser, extract_clean_prompt


@pytest.fixture
def parser():
    return ActionParser()


def test_extract_clean_prompt_strips_meta_preamble(parser: ActionParser) -> None:
    """Meta-intros like 'Here is a detailed prompt' are removed from the clean prompt."""
    dirty = """Here's a detailed prompt you can paste into the builder:

Build a hybrid_rust_python matrix kernel with PyO3 bindings.
"""
    clean = parser.extract_clean_prompt(dirty)
    assert clean is not None
    assert "Here's a detailed prompt" not in clean
    assert "paste" not in clean
    assert "hybrid_rust_python" in clean


def test_extract_clean_prompt_strips_yaml_blueprint_headers(parser: ActionParser) -> None:
    """Outer 'Build Contract' / 'yaml blueprint' wrappers are not part of the prompt."""
    dirty = """Build Contract
yaml blueprint
Build a pure_rust Fibonacci core exposed through a C ABI.
"""
    clean = parser.extract_clean_prompt(dirty)
    assert "Build Contract" not in clean
    assert "yaml blueprint" not in clean
    assert "pure_rust" in clean


def test_extract_clean_prompt_strips_outer_quotes_and_escapes(parser: ActionParser) -> None:
    """Escaped newlines and surrounding quotes are normalized."""
    dirty = '"Build a pure_python image processor\\nwith Pillow.\\nTarget: pure_python"'
    clean = parser.extract_clean_prompt(dirty)
    assert clean is not None
    assert clean[0] not in ('"', "'")
    assert "\\n" not in clean
    assert "pure_python" in clean


def test_extract_clean_prompt_json_structured_response(parser: ActionParser) -> None:
    """A JSON response with display_text + action.clean_prompt is parsed cleanly."""
    payload = json.dumps(
        {
            "display_text": "### Plan\nUse a Rust hot loop.",
            "action": {
                "type": "build",
                "clean_prompt": "Build a hybrid_rust_python matmul kernel.",
                "parameters": {
                    "target": "hybrid_rust_python",
                    "acceleration": "Selective Acceleration (Auto-Detect Heavy Compute)",
                },
            },
        }
    )
    parsed = parser.parse(payload)
    assert parsed["display_text"] == "### Plan\nUse a Rust hot loop."
    assert parsed["action"]["type"] == "build"
    assert "matmul" in parsed["action"]["clean_prompt"]
    assert parsed["action"]["parameters"]["target"] == "hybrid_rust_python"


def test_extract_clean_prompt_build_prompt_fence(parser: ActionParser) -> None:
    """The dedicated ```build_prompt fence is extracted without the conversational wrapper."""
    response = """### Architecture Overview
Use a Rust core with PyO3 bindings.

```build_prompt
Build a hybrid_rust_python project with a Rust `matmul` kernel and PyO3 wrapper.
```
"""
    parsed = parser.parse(response)
    assert "Architecture Overview" in parsed["display_text"]
    assert "```build_prompt" not in parsed["display_text"]
    assert parsed["action"]["clean_prompt"] == "Build a hybrid_rust_python project with a Rust `matmul` kernel and PyO3 wrapper."


def test_extract_clean_prompt_builder_prompt_tag(parser: ActionParser) -> None:
    """The new <builder_prompt> delimiter isolates the inner prompt from surrounding chitchat."""
    response = """I'll give you a ready-to-use prompt for an AI assistant.

<builder_prompt>
You are an expert systems builder. Implement a pure Python weighted decision matrix evaluator with the signature: def weighted_decision_matrix(scores, weights, criteria_types) -> list[float].
</builder_prompt>

You can paste this directly into your builder.
"""
    parsed = parser.parse(response)
    clean = parsed["action"]["clean_prompt"]
    assert clean.startswith("You are an expert systems builder")
    assert "paste" not in clean
    assert "<builder_prompt>" not in clean
    assert "weighted_decision_matrix" in clean
    assert parsed["action"]["parameters"]["target"] == "pure_python"


def test_extract_clean_prompt_aggressive_intro_outro_stripping(parser: ActionParser) -> None:
    """Without tags, common meta-preambles and postambles are stripped from the clean prompt."""
    response = """I'll give you a ready-to-use prompt.

You are an expert systems builder. Build a hybrid_rust_python matrix kernel with PyO3 bindings.

You can paste this directly into your builder."""
    clean = parser.extract_clean_prompt(response)
    assert clean is not None
    assert "I'll give you" not in clean
    assert "You can paste this" not in clean
    assert clean.startswith("You are an expert systems builder")
    assert clean.endswith("PyO3 bindings.")


def test_extract_action_returns_structured_packet(parser: ActionParser) -> None:
    """The module-level extract_action helper returns a full display/action packet."""
    from aero_forge.copilot.action_parser import extract_action
    response = """<builder_prompt>Build a wasm image filter.</builder_prompt>"""
    packet = extract_action(response)
    assert packet["display_text"] == ""
    assert packet["action"]["clean_prompt"] == "Build a wasm image filter."
    assert packet["action"]["parameters"]["target"] == "wasm"


def test_extract_clean_prompt_yaml_contract_returns_inner_prompt(parser: ActionParser) -> None:
    """A ```yaml blueprint contract block is parsed and only the prompt field is returned."""
    response = """## Overview
Accelerated numeric compute core.

```yaml blueprint
prompt: Build a fast Fibonacci function in Python using @accelerate and PyO3 Rust backings
target: hybrid_rust_python
acceleration: "Selective Acceleration (Auto-Detect Heavy Compute)"
```
"""
    clean = parser.extract_clean_prompt(response)
    assert "prompt:" not in clean
    assert "yaml blueprint" not in clean
    assert "Fibonacci" in clean
    parsed = parser.parse(response)
    assert parsed["action"]["parameters"]["target"] == "hybrid_rust_python"


def test_extract_clean_prompt_no_build_intent_returns_none(parser: ActionParser) -> None:
    """Purely conversational text with no build intent yields no executable prompt."""
    assert parser.extract_clean_prompt("Hello, how are you today?") is None


def test_sanitize_collapses_whitespace_and_meta_lines(parser: ActionParser) -> None:
    """Sanitize removes headers and collapses blank runs."""
    raw = """  Build Prompt:\n\n\n\n  Build a wasm image filter.  """
    assert parser.sanitize(raw).startswith("Build a wasm image filter")


def test_module_level_extract_clean_prompt() -> None:
    """The module-level helper delegates to ActionParser."""
    text = "```build_prompt\nBuild a pure_python CLI tool.\n```"
    assert "pure_python" in extract_clean_prompt(text)


def test_clean_explanation_text_removes_prompt_duplication() -> None:
    """A conversational explanation must not repeat the extracted build prompt."""
    from aero_forge.copilot.action_parser import clean_explanation_text

    explanation = "I suggest a Rust hot loop.\n\n```build_prompt\nBuild a hybrid_rust_python matmul kernel.\n```\n\nUse this prompt."
    prompt = "Build a hybrid_rust_python matmul kernel."
    cleaned = clean_explanation_text(explanation, prompt)
    assert prompt not in cleaned
    assert "Build a hybrid_rust_python" not in cleaned
    assert "I suggest a Rust hot loop" in cleaned


def test_parse_isolates_display_text_from_prompt(parser: ActionParser) -> None:
    """JSON responses that embed the prompt inside display_text are deduplicated."""
    payload = json.dumps(
        {
            "display_text": "### Rationale\nThis prompt builds a Python CLI: Build a pure_python CLI tool with argparse.",
            "action": {
                "type": "build",
                "clean_prompt": "Build a pure_python CLI tool with argparse.",
                "parameters": {"target": "pure_python", "acceleration": "Standard Runtime (Bypass Bridge)"},
            },
        }
    )
    parsed = parser.parse(payload)
    display = parsed["display_text"]
    assert "Build a pure_python CLI tool" not in display
    assert "Rationale" in display
    assert "argparse" not in display


def test_parse_generates_concise_default_when_no_explanation(parser: ActionParser) -> None:
    """When the model returns only a prompt with no explanation, a short rationale is produced."""
    response = "```build_prompt\nBuild a hybrid_rust_python matrix core.\n```"
    parsed = parser.parse(response)
    assert parsed["action"]["clean_prompt"] == "Build a hybrid_rust_python matrix core."
    # Display text should be empty; the caller will generate a default rationale.
    assert "Build a hybrid_rust_python" not in parsed["display_text"]


def test_sanitize_builder_prompt_strips_preamble_and_extracts_code_block() -> None:
    """Meta-preambles inside the prompt payload are removed and inner code blocks are extracted."""
    from aero_forge.copilot.action_parser import sanitize_builder_prompt

    dirty = "I've crafted a detailed prompt for you: Build a pure_rust CLI tool using clap."
    cleaned = sanitize_builder_prompt(dirty)
    assert cleaned.startswith("Build a pure_rust")
    assert "I've crafted" not in cleaned
    assert "prompt for you" not in cleaned

    fenced = '```build_prompt\nBuild a hybrid_cpp_python extension with pyo3.\n```'
    assert "Build a hybrid_cpp_python extension with pyo3." in sanitize_builder_prompt(fenced)

    no_colon = "Here is a prompt\nBuild a tri_polyglot Rust, C++, and Python CLI."
    assert "Build a tri_polyglot" in sanitize_builder_prompt(no_colon)
    assert "Here is a prompt" not in sanitize_builder_prompt(no_colon)


def test_suggested_prompt_top_level_is_extracted_and_sanitized(parser: ActionParser) -> None:
    """A top-level suggested_prompt field is sanitized and separated from display_text."""
    payload = (
        '{"display_text": "Use this prompt.", '
        '"suggested_prompt": "Here is the prompt: Build a wasm image filter.", '
        '"parameters": {"target": "wasm", "acceleration": "Standard Runtime (Bypass Bridge)"}}'
    )
    parsed = parser.parse(payload)
    assert parsed["action"]["clean_prompt"] == "Build a wasm image filter."
    assert "Here is the prompt" not in parsed["action"]["clean_prompt"]


def test_clean_prompt_strips_trailing_target_and_acceleration(parser: ActionParser) -> None:
    """Trailing Target:/Acceleration: metadata tags are stripped from clean_prompt."""
    payload = json.dumps(
        {
            "display_text": "Use a Rust core.",
            "action": {
                "type": "build",
                "clean_prompt": "Build a hybrid_rust_python matmul kernel. Target: hybrid_rust_python. Acceleration: Force Native Bridge.",
                "parameters": {
                    "target": "hybrid_rust_python",
                    "acceleration": "Force Native Bridge",
                },
            },
        }
    )
    parsed = parser.parse(payload)
    clean = parsed["action"]["clean_prompt"]
    assert "Target:" not in clean
    assert "Acceleration:" not in clean
    assert "matmul" in clean
    # Parameters are still recovered from the raw prompt before stripping.
    assert parsed["action"]["parameters"]["target"] == "hybrid_rust_python"
    assert parsed["action"]["parameters"]["acceleration"] == "Force Native Bridge"


def test_action_trigger_build_block_is_valid_json_and_clean(parser: ActionParser) -> None:
    """An action:trigger_build card parses as valid JSON and yields a clean prompt."""
    block = '''```action:trigger_build
{
  "target_language": "cpp",
  "architecture": "hybrid_cpp_python",
  "target_files": ["src/native.cpp", "src/native_bridge.py", "tests/test_native.py"],
  "builder_prompt": "Build a hybrid_cpp_python C-ABI/ctypes native extension. Implement sliding_window_dtw as an extern \\"C\\" AERO_EXPORT function in src/native.cpp compiled into a shared library. Provide a Python loader in src/native_bridge.py using ctypes.CDLL, and pytest tests in tests/test_native.py comparing native output to a naive Python reference."
}
```'''
    # The block itself is valid JSON.
    import re
    inner = re.search(r"```action:trigger_build\s*\n(.*?)\n```", block, re.DOTALL).group(1)
    data = json.loads(inner)
    assert data["architecture"] == "hybrid_cpp_python"
    assert isinstance(data["target_files"], list)
    # The parser surfaces a clean action with no trailing tags.
    parsed = parser.parse(block)
    assert parsed["action"]["type"] == "trigger_build"
    assert "extern" in parsed["action"]["clean_prompt"]
    assert "Target:" not in parsed["action"]["clean_prompt"]
