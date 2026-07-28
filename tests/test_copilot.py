"""Tests for the Co-pilot action parser and response formatting."""

from aero_forge.copilot.action_parser import (
    extract_build_contract,
    parse_action_from_text,
    parse_copilot_response,
)
from aero_forge.copilot.agent import format_copilot_response


def test_extract_build_contract_from_yaml_fence() -> None:
    response = """
## Overview
Build a fast numeric compute core.

## Build Contract
```yaml blueprint
prompt: Build an accelerated Fibonacci function in Python using @accelerate and PyO3 Rust backings
target: hybrid_rust_python
acceleration: "Selective Acceleration (Auto-Detect Heavy Compute)"
```
"""
    contract = extract_build_contract(response)
    assert contract is not None
    assert contract["prompt"] == "Build an accelerated Fibonacci function in Python using @accelerate and PyO3 Rust backings"
    assert contract["target"] == "hybrid_rust_python"
    assert contract["acceleration"] == "Selective Acceleration (Auto-Detect Heavy Compute)"


def test_extract_build_contract_from_json_fence() -> None:
    response = """
## Overview
Batch processor.

```json blueprint
{
  "prompt": "Build a Python-Rust batch processor",
  "target": "hybrid_rust_python",
  "acceleration": "Standard Runtime (Bypass Bridge)"
}
```
"""
    contract = extract_build_contract(response)
    assert contract is not None
    assert contract["target"] == "hybrid_rust_python"


def test_extract_build_contract_ignores_broad_keys() -> None:
    """Only explicitly tagged blueprint fences are parsed."""
    response = """
architecture_overview: this should not be matched
prompt: oops
"""
    assert extract_build_contract(response) is None


def test_parse_copilot_response_returns_markdown_reply_and_action() -> None:
    response = """## Overview
Use a Rust core for the hot loop.

## Build Contract
```yaml blueprint
prompt: Matrix multiplication in Rust wrapped by Python
target: hybrid_rust_python
acceleration: "Selective Acceleration (Auto-Detect Heavy Compute)"
```"""
    reply, action = parse_copilot_response(response)
    assert "## Overview" in reply
    assert "```yaml blueprint" not in reply
    assert action is not None
    assert action["type"] == "PROPOSE_BUILD"
    assert action["params"]["target"] == "hybrid_rust_python"
    assert "Matrix multiplication" in action["params"]["prompt"]


def test_parse_action_from_text_uses_code_fence() -> None:
    text = """
Some prose.

```yaml blueprint
prompt: Build a C++ extension for Python
target: hybrid_cpp_python
```
"""
    action = parse_action_from_text(text)
    assert action is not None
    assert action["params"]["target"] == "hybrid_cpp_python"


def test_parse_action_from_text_falls_back_to_prose() -> None:
    action = parse_action_from_text("Build a fast Rust Fibonacci core wrapped in Python.")
    assert action is not None
    assert action["type"] == "PROPOSE_BUILD"
    assert action["params"]["target"] == "hybrid_rust_python"


def test_parse_copilot_response_handles_legacy_json() -> None:
    response = (
        '{"reply": "Use Rust for the hot loop.", '
        '"action": {"type": "PROPOSE_BUILD", "params": '
        '{"prompt": "matrix mult", "target": "hybrid_cpp_rust", '
        '"acceleration": "Force Native Bridge"}}}'
    )
    reply, action = parse_copilot_response(response)
    assert reply == "Use Rust for the hot loop."
    assert action["type"] == "PROPOSE_BUILD"
    assert action["params"]["target"] == "hybrid_cpp_rust"
    assert action["params"]["acceleration"] == "Force Native Bridge"


def test_format_copilot_response_wraps_raw_json_in_markdown() -> None:
    """A raw top-level JSON action payload is restructured as Markdown + YAML."""
    response = (
        '{"reply": "Use Rust for the hot loop.", '
        '"action": {"type": "PROPOSE_BUILD", "params": '
        '{"prompt": "matrix mult", "target": "hybrid_cpp_rust", '
        '"acceleration": "Force Native Bridge"}}}'
    )
    reply, action = format_copilot_response(response)
    assert "### Architecture Overview" in reply
    assert "```yaml blueprint" in reply
    assert action["type"] == "PROPOSE_BUILD"
    assert action["params"]["target"] == "hybrid_cpp_rust"
    assert action["params"]["acceleration"] == "Force Native Bridge"


def test_format_copilot_response_wraps_plain_text_build_intent() -> None:
    """Plain prose with build intent is wrapped in Markdown and a YAML build contract."""
    response = "Build a fast Rust Fibonacci core wrapped in Python."
    reply, action = format_copilot_response(response)
    assert "### Architecture Overview" in reply
    assert "```yaml blueprint" in reply
    assert action["type"] == "PROPOSE_BUILD"
    assert action["params"]["target"] == "hybrid_rust_python"


def test_format_copilot_response_preserves_existing_markdown_yaml_contract() -> None:
    """A properly formatted Markdown + YAML response is passed through unchanged."""
    response = """## Overview
Use a Rust core for the hot loop.

## Build Contract
```yaml blueprint
prompt: Matrix multiplication in Rust wrapped by Python
target: hybrid_rust_python
acceleration: "Selective Acceleration (Auto-Detect Heavy Compute)"
```"""
    reply, action = format_copilot_response(response)
    assert "## Overview" in reply
    assert action is not None
    assert action["type"] == "PROPOSE_BUILD"
    assert action["params"]["target"] == "hybrid_rust_python"


def test_format_copilot_response_pretty_prints_non_action_json() -> None:
    """Non-action JSON is presented as a Markdown code block rather than raw text."""
    response = '{"key": "value", "nested": {"x": 1}}'
    reply, action = format_copilot_response(response)
    assert "```json" in reply
    assert action is None
