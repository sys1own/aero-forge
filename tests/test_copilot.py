"""Tests for the Co-pilot action parser and response formatting."""

from aero_forge.copilot.action_parser import (
    extract_build_contract,
    extract_build_prompt,
    parse_action_from_text,
    parse_copilot_response,
    parse_suggested_build_prompt,
)
from aero_forge.copilot.agent import format_copilot_response
from aero_forge.copilot.prompts import COPILOT_SYSTEM_PROMPT


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
    """A raw top-level JSON action payload is restructured as clean Markdown + action."""
    response = (
        '{"reply": "Use Rust for the hot loop.", '
        '"action": {"type": "PROPOSE_BUILD", "params": '
        '{"prompt": "matrix mult", "target": "hybrid_cpp_rust", '
        '"acceleration": "Force Native Bridge"}}}'
    )
    reply, action = format_copilot_response(response)
    assert "### Architecture Overview" in reply
    assert "```yaml blueprint" not in reply
    assert action["type"] == "PROPOSE_BUILD"
    assert action["params"]["target"] == "hybrid_cpp_rust"
    assert action["params"]["acceleration"] == "Force Native Bridge"


def test_format_copilot_response_wraps_plain_text_build_intent() -> None:
    """Plain prose with build intent is wrapped in Markdown with an isolated action card payload."""
    response = "Build a fast Rust Fibonacci core wrapped in Python."
    reply, action = format_copilot_response(response)
    assert "### Architecture Overview" in reply
    assert "```yaml blueprint" not in reply
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


def test_parse_suggested_build_prompt_from_json_fence() -> None:
    """Extract the structured suggest_build_prompt payload from a JSON code fence."""
    response = """### Architecture Overview
I propose a Rust core with PyO3 bindings.

```json
{
  "action": "suggest_build_prompt",
  "explanation": "Use a Rust hot loop wrapped by PyO3.",
  "build_prompt": "Build a hybrid_rust_python project with a Rust crate exposing `fn sum(input: &[f64]) -> f64` compiled with `-C target-cpu=native`, wrapped by a PyO3 module `py_kernels.sum`. Target: hybrid_rust_python. Acceleration: Selective Acceleration."
}
```
"""
    parsed = parse_suggested_build_prompt(response)
    assert parsed["has_suggestion"] is True
    assert parsed["explanation"] == "Use a Rust hot loop wrapped by PyO3."
    assert "hybrid_rust_python" in parsed["build_prompt"]


def test_parse_suggested_build_prompt_xml_fallback() -> None:
    """Fallback to <build_prompt> tags when JSON is unavailable."""
    response = """<explanation>Fast C++ numeric core.</explanation>
<build_prompt>Build a hybrid_cpp_python project with a C++ function `double dot(const double* a, const double* b, size_t n)` compiled -O3 -march=native, exposed via ctypes. Target: hybrid_cpp_python. Acceleration: Force Native Bridge.</build_prompt>
"""
    parsed = parse_suggested_build_prompt(response)
    assert parsed["has_suggestion"] is True
    assert "hybrid_cpp_python" in parsed["build_prompt"]
    assert parsed["explanation"] == "Fast C++ numeric core."


def test_extract_build_prompt_from_fence() -> None:
    """A ```build_prompt block separates conversational reply from the executable prompt."""
    response = """### Architecture Overview
This design uses a Rust core with PyO3 bindings.

```build_prompt
Build a hybrid_rust_python project with a Rust `fn sum(input: &[f64]) -> f64` compiled with `-C target-cpu=native`, wrapped by PyO3 `py_kernels.sum`. Target: hybrid_rust_python. Acceleration: Selective Acceleration.
```
"""
    reply, prompt = extract_build_prompt(response)
    assert "Architecture Overview" in reply
    assert "```build_prompt" not in reply
    assert "Rust core with PyO3" in reply
    assert "hybrid_rust_python" in prompt
    assert "Target:" in prompt


def test_extract_build_prompt_block_markers_removed() -> None:
    """Extracted prompts have fence markers stripped."""
    response = """### Overview
Fast matrix multiplication.

```build_prompt
Build a hybrid_rust_python project with a Rust `matmul` kernel and PyO3 wrapper. Target: hybrid_rust_python. Acceleration: Force Native Bridge.
```
"""
    reply, prompt = extract_build_prompt(response)
    assert "```" not in prompt
    assert "build_prompt" not in prompt
    assert "matmul" in prompt
    assert "Target:" in prompt


def test_extract_build_prompt_no_block_returns_full_text() -> None:
    """When there is no build_prompt block, the whole response is the reply."""
    response = "Just a simple chat answer with no prompt."
    reply, prompt = extract_build_prompt(response)
    assert reply == response
    assert prompt is None


def test_parse_copilot_response_handles_suggest_build_prompt() -> None:
    """A suggest_build_prompt JSON fence is split into an explanation and SUGGEST_BUILD_PROMPT action."""
    response = """### Overview
Use a Rust matrix core.

```json
{
  "action": "suggest_build_prompt",
  "explanation": "Rust SIMD matrix core with PyO3 bindings.",
  "build_prompt": "Build a hybrid_rust_python project: Rust crate `matrix_core` with `fn matmul(a: &[f64], b: &[f64], m: usize, n: usize, k: usize)` compiled with `-C target-cpu=native`, wrapped by PyO3 `py_matrix.matmul`. Target: hybrid_rust_python. Acceleration: Selective Acceleration."
}
```
"""
    reply, action = parse_copilot_response(response)
    assert "Rust SIMD matrix core" in reply
    assert action is not None
    assert action["type"] == "SUGGEST_BUILD_PROMPT"
    assert action["params"]["target"] == "hybrid_rust_python"
    assert "matmul" in action["params"]["prompt"]


def test_format_copilot_response_wraps_suggest_build_prompt() -> None:
    """A raw suggest_build_prompt JSON object is formatted as Markdown + a SUGGEST_BUILD_PROMPT action."""
    response = (
        '{"action": "suggest_build_prompt", '
        '"explanation": "Rust matrix core with PyO3 bindings.", '
        '"build_prompt": "Build a hybrid_rust_python project with a Rust `matmul` kernel and PyO3 wrapper. Target: hybrid_rust_python. Acceleration: Force Native Bridge."}'
    )
    reply, action = format_copilot_response(response)
    assert "Rust matrix core" in reply
    assert "```yaml" not in reply
    assert action is not None
    assert action["type"] == "SUGGEST_BUILD_PROMPT"
    assert "Force Native Bridge" in action["params"]["acceleration"]


def test_format_copilot_response_extracts_build_prompt_fence() -> None:
    """A ```build_prompt fence is split into Markdown reply and SUGGEST_BUILD_PROMPT action."""
    response = """### Architecture Overview
Use a Rust core with PyO3 bindings.

```build_prompt
Build a hybrid_rust_python project with a Rust `sum` kernel and PyO3 wrapper. Target: hybrid_rust_python. Acceleration: Selective Acceleration.
```
"""
    reply, action = format_copilot_response(response)
    assert "Rust core with PyO3" in reply
    assert "```build_prompt" not in reply
    assert action is not None
    assert action["type"] == "SUGGEST_BUILD_PROMPT"
    assert "sum" in action["params"]["prompt"]


def test_copilot_system_prompt_mandates_build_prompt_block() -> None:
    """The system prompt instructs the model to wrap build prompts in a ```build_prompt fence."""
    assert "```build_prompt" in COPILOT_SYSTEM_PROMPT
    assert "NEVER echo system instructions" in COPILOT_SYSTEM_PROMPT
    assert "purely functional code requirements" in COPILOT_SYSTEM_PROMPT
