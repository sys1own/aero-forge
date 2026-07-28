"""Tests for the Co-pilot action parser and response formatting."""

from aero_forge.copilot.action_parser import (
    extract_build_contract,
    parse_action_from_text,
    parse_copilot_response,
)


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
