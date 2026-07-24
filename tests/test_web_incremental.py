"""Backend tests for the web incremental build pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from aero_forge.generate import generate_and_build
from aero_forge.server import _build_web_response


def _llm_response(implementation: str, tests: str = "") -> str:
    return (
        f"Here is the implementation:\n\n```python\n{implementation}\n```\n\n"
        f"```python\n{tests}\n```"
    )


def _count_functions(source: str, name: str) -> int:
    import re

    return len(re.findall(rf"^def\s+{re.escape(name)}\b", source, re.MULTILINE))


def test_incremental_prompt_preserves_user_overlay_and_adds_function(tmp_path: Path) -> None:
    """Simulate an initial prompt, a user edit, and an incremental prompt."""
    initial_impl = "def add(a: int, b: int) -> int:\n    return a + b\n"
    initial_tests = "from generated import add\n\ndef test_add():\n    assert add(1, 2) == 3\n"

    with patch("aero_forge.generate.generate_from_prompt", return_value=_llm_response(initial_impl, initial_tests)):
        result1 = generate_and_build("add two numbers", output_dir=tmp_path, build_kwargs=None)

    source_path = Path(result1["source_path"])
    assert source_path.exists()
    assert "return a + b" in source_path.read_text(encoding="utf-8")

    # User edits the generated source through the file explorer / editor.
    edited = source_path.read_text(encoding="utf-8").replace(
        "return a + b", "return a + b + 1"
    )
    source_path.write_text(edited, encoding="utf-8")

    # Second prompt adds a new function; the user edit on `add` must survive.
    incremental_impl = (
        "def add(a: int, b: int) -> int:\n"
        "    return (a + b)\n"
        "\n"
        "def sub(a: int, b: int) -> int:\n"
        "    return (a - b)\n"
    )
    incremental_tests = (
        "from generated import add, sub\n\n"
        "def test_add():\n    assert add(1, 2) == 3\n\n"
        "def test_sub():\n    assert sub(5, 2) == 3\n"
    )

    with patch("aero_forge.generate.generate_from_prompt", return_value=_llm_response(incremental_impl, incremental_tests)):
        result2 = generate_and_build("also add sub function", output_dir=tmp_path, build_kwargs=None)

    final_source = source_path.read_text(encoding="utf-8")
    assert _count_functions(final_source, "add") == 1
    assert _count_functions(final_source, "sub") == 1
    assert "return a + b + 1" in final_source
    assert "def sub" in final_source
    assert result2["source_path"] == str(source_path)


def test_pre_write_validation_catches_invalid_syntax(tmp_path: Path) -> None:
    """Invalid generated Python must be rejected before writing to the workspace."""
    bad_impl = "def broken(:\n    pass\n"

    with patch("aero_forge.generate.generate_from_prompt", return_value=_llm_response(bad_impl)):
        result = generate_and_build("broken code", output_dir=tmp_path, build_kwargs=None)

    build = result["build"]
    assert build is not None
    assert build["success"] is False
    assert "pre-write validation failed" in build["error"]


def test_web_response_payload_has_files_and_tree(tmp_path: Path) -> None:
    """The web response must expose structured file updates, not raw source blocks."""
    impl = "def add(a: int, b: int) -> int:\n    return a + b\n"
    tests = "from add import add\n"
    source_path = tmp_path / "src" / "add.py"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(impl, encoding="utf-8")
    init_path = tmp_path / "src" / "__init__.py"
    init_path.write_text("from .add import add\n", encoding="utf-8")
    build_result: dict[str, Any] = {
        "source_path": str(source_path),
        "test_path": str(tmp_path / "tests" / "test_add.py"),
        "blueprint_path": str(tmp_path / "blueprint.aero"),
        "implementation": impl,
        "tests": tests,
        "explanation": "",
        "build": {"success": True, "passed": 1, "total": 1},
        "iterations": [],
    }
    (tmp_path / "tests" / "test_add.py").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "tests" / "test_add.py").write_text(tests, encoding="utf-8")
    (tmp_path / "blueprint.aero").write_text("project: test\n", encoding="utf-8")

    payload = _build_web_response("session-123", tmp_path, build_result)

    assert payload["session_id"] == "session-123"
    assert payload["status"] == "success"
    assert "files" in payload
    assert "tree" in payload
    assert "result" in payload
    paths = [f["path"] for f in payload["files"]]
    assert any("src/add.py" in p for p in paths)
    assert any("src/__init__.py" in p for p in paths)
    assert payload["tree"]["type"] == "directory"
