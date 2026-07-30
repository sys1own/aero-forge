"""Tests for builder feedback parsing, file tagging, and typing import guards."""

from pathlib import Path

import pytest

from aero_forge.builder.context import WorkspaceContext, get_workspace_context
from aero_forge.builder.feedback import FeedbackParser
from aero_forge.builder.orchestration import ensure_typing_imports, tag_files_for_feedback


def test_feedback_parser_normalizes_sandbox_paths(tmp_path: Path) -> None:
    raw = (
        "Traceback (most recent call last):\n"
        f'  File "{tmp_path}/main.py", line 2, in <module>\n'
        "    x = Any\n"
        "NameError: name 'Any' is not defined\n"
    )
    parser = FeedbackParser(tmp_path)
    result = parser.parse_traceback(raw)
    assert result["missing_symbol"] == "Any"
    assert result["references"][0]["file"] == "main.py"


def test_feedback_parser_strips_ephemeral_sandbox_path() -> None:
    raw = (
        "Traceback (most recent call last):\n"
        '  File "/tmp/aero-forge-sandboxes/session-abc/main.py", line 5, in <module>\n'
        "    x = List[int]()\n"
        "NameError: name 'List' is not defined\n"
    )
    parser = FeedbackParser(Path("/tmp/aero-forge-sandboxes/session-abc"))
    normalized = parser.normalize_paths(raw)
    assert "/tmp/aero-forge-sandboxes/session-abc/" not in normalized


def test_tag_files_for_feedback_detects_typing_error(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("x = Any\n", encoding="utf-8")
    log = "NameError: name 'Any' is not defined"
    tags = tag_files_for_feedback(tmp_path, log)
    assert any("[MODIFY: main.py]" in tag for tag in tags)


def test_tag_files_for_feedback_extracts_user_create_tags(tmp_path: Path) -> None:
    prompt = "Add helper [CREATE: src/utils.py] and fix main [MODIFY: main.py]"
    tags = tag_files_for_feedback(tmp_path, "no error", user_prompt=prompt)
    assert "[CREATE: src/utils.py]" in tags
    assert "[MODIFY: main.py]" in tags


def test_ensure_typing_imports_adds_any() -> None:
    source = "x: Any = 1\n"
    result = ensure_typing_imports(source)
    assert "from typing import" in result
    assert "Any" in result


def test_ensure_typing_imports_adds_list_and_dict() -> None:
    source = 'def f() -> List[Dict[str, Any]]:\n    return []\n'
    result = ensure_typing_imports(source)
    assert "from typing import Any, Dict, List" in result


def test_ensure_typing_imports_does_not_duplicate() -> None:
    source = "from typing import Any\nx: Any = 1\n"
    result = ensure_typing_imports(source)
    assert result.count("from typing import") == 1


def test_workspace_context_bundles_files(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("print('ok')\n", encoding="utf-8")
    ctx = get_workspace_context(tmp_path)
    bundle = ctx.bundle()
    assert bundle["workspace"] == str(tmp_path)
    assert "main.py" in bundle["files"]
