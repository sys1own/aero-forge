"""Tests for LLM directive parsing and unified-diff application."""

import pytest

from aero_forge.healing.llm_healer import LLMHealer, DirectiveError


def test_parse_directives_extracts_json_from_code_fence() -> None:
    response = """
```json
{
  "diagnosis": "missing import",
  "directives": [
    {
      "target_file": "main.py",
      "action": "rewrite",
      "reason": "add import",
      "instructions": "prepend import math",
      "content": "import math\\nprint(math.sqrt(16))\\n"
    }
  ]
}
```
"""
    healer = LLMHealer()
    directives = healer._parse_directives(response)
    assert len(directives) == 1
    assert directives[0]["target_file"] == "main.py"
    assert directives[0]["action"] == "rewrite"


def test_parse_directives_rejects_missing_fields() -> None:
    response = '{"diagnosis": "x", "directives": [{"target_file": "main.py"}]}'
    healer = LLMHealer()
    with pytest.raises(DirectiveError):
        healer._parse_directives(response)


def test_apply_unified_diff_adds_and_removes_lines() -> None:
    original = "line1\nline2\nline3\n"
    diff = (
        "--- a/file.txt\n"
        "+++ b/file.txt\n"
        "@@ -1,3 +1,3 @@\n"
        " line1\n"
        "-line2\n"
        "+line2_changed\n"
        " line3\n"
    )
    healer = LLMHealer()
    result = healer._apply_unified_diff(original, diff)
    assert "line1" in result
    assert "line2_changed" in result
    assert "line3" in result
    assert "line2\n" not in result
