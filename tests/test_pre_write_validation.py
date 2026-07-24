"""Tests for pre-write validation and isolated workspace promotion."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from aero_forge.precision_shield.rust_shield import RustSemanticShield
from aero_forge.scaffold.pre_write_validator import (
    PreWriteValidator,
    ValidationError,
)
from aero_forge.scaffold.workspace import (
    OutOfTreeWorkspace,
    WorkspaceLocationError,
)


def test_python_syntax_validation_passes(tmp_path: Path) -> None:
    validator = PreWriteValidator()
    (tmp_path / "main.py").write_text("def foo():\n    return 42\n")
    result = validator.validate(tmp_path, language="python")
    assert result.succeeded
    assert result.return_code == 0


def test_python_syntax_validation_fails(tmp_path: Path) -> None:
    validator = PreWriteValidator()
    (tmp_path / "main.py").write_text("def foo(\n")
    with pytest.raises(ValidationError) as exc_info:
        validator.validate(tmp_path, language="python")
    assert "main.py" in exc_info.value.output or "invalid" in exc_info.value.output.lower()


def test_workspace_promotes_on_success(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    with OutOfTreeWorkspace(distribution_directory=dist) as ws:
        (ws.root / "main.py").write_text("x = 1\n")
    assert dist.is_dir()
    assert (dist / "main.py").read_text() == "x = 1\n"


def test_workspace_discards_on_failure(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    validator = PreWriteValidator()
    with pytest.raises(ValidationError):
        with OutOfTreeWorkspace(distribution_directory=dist) as ws:
            (ws.root / "main.py").write_text("def broken(\n")
            validator.validate_and_promote(ws, language="python")
    assert not dist.exists()


def test_out_of_tree_workspace_rejects_tool_path(tmp_path: Path) -> None:
    tool_root = Path(__file__).resolve().parents[1] / "aero_forge"
    with pytest.raises(WorkspaceLocationError):
        OutOfTreeWorkspace(distribution_directory=tool_root)


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo not installed")
def test_rust_pre_write_validation_catches_errors(tmp_path: Path) -> None:
    validator = PreWriteValidator()
    crate = tmp_path / "bad_crate"
    crate.mkdir()
    (crate / "Cargo.toml").write_text('[package]\nname = "bad"\nversion = "0.1.0"\nedition = "2021"\n')
    src = crate / "src"
    src.mkdir()
    (src / "lib.rs").write_text("fn broken { }\n")
    with pytest.raises(ValidationError):
        validator.validate(crate, language="rust")


def test_rust_shield_detects_rug_anchor() -> None:
    source = "use rug::Float;\nfn f() -> Float { Float::new(1) }\n"
    report = RustSemanticShield().apply(source)
    assert "rug" in report.anchors


def test_rust_shield_is_idempotent() -> None:
    source = "use rug::Float;\nfn f() -> Float { Float::new(1) }\n"
    report1 = RustSemanticShield().apply(source)
    report2 = RustSemanticShield().apply(report1.source)
    assert report2.applied == []
    assert report2.source == report1.source


def test_bare_dict_annotation_rejected(tmp_path: Path) -> None:
    validator = PreWriteValidator()
    (tmp_path / "main.py").write_text(
        "def evaluate_rule_tree(event: dict) -> int:\n    return 0\n"
    )
    with pytest.raises(ValidationError) as exc_info:
        validator.validate(tmp_path, language="python")
    assert "dict" in str(exc_info.value).lower()


def test_generic_dict_annotation_accepted(tmp_path: Path) -> None:
    validator = PreWriteValidator()
    (tmp_path / "main.py").write_text(
        "from typing import Any\n"
        "def evaluate_rule_tree(event: dict[str, Any]) -> int:\n    return 0\n"
    )
    result = validator.validate(tmp_path, language="python")
    assert result.succeeded


def test_raw_enum_rejected(tmp_path: Path) -> None:
    validator = PreWriteValidator()
    (tmp_path / "main.py").write_text(
        "from enum import Enum\n"
        "class State(Enum):\n    ON = 1\n"
    )
    with pytest.raises(ValidationError) as exc_info:
        validator.validate(tmp_path, language="python")
    assert "IntEnum" in str(exc_info.value)


def test_intenum_accepted(tmp_path: Path) -> None:
    validator = PreWriteValidator()
    (tmp_path / "main.py").write_text(
        "from enum import IntEnum\n"
        "class State(IntEnum):\n    ON = 1\n"
    )
    result = validator.validate(tmp_path, language="python")
    assert result.succeeded


def test_empty_matrix_return_rejected(tmp_path: Path) -> None:
    validator = PreWriteValidator()
    (tmp_path / "main.py").write_text(
        "def zero_matrix(rows: int, cols: int) -> list[list[int]]:\n"
        "    return []\n"
    )
    with pytest.raises(ValidationError) as exc_info:
        validator.validate(tmp_path, language="python")
    assert "zero-filled" in str(exc_info.value).lower()


def test_zero_matrix_filled_accepted(tmp_path: Path) -> None:
    validator = PreWriteValidator()
    (tmp_path / "main.py").write_text(
        "def zero_matrix(rows: int, cols: int) -> list[list[int]]:\n"
        "    if rows == 0 or cols == 0:\n"
        "        return [[0] * cols for _ in range(rows)]\n"
        "    return [[0] * cols for _ in range(rows)]\n"
    )
    result = validator.validate(tmp_path, language="python")
    assert result.succeeded


def test_dynamic_reflection_builtin_rejected(tmp_path: Path) -> None:
    validator = PreWriteValidator()
    (tmp_path / "main.py").write_text(
        "def blocked_multiply(a: int, b: int) -> int:\n"
        "    if hasattr(a, 'real'):\n"
        "        return a * b\n"
        "    return a * b\n"
    )
    with pytest.raises(ValidationError) as exc_info:
        validator.validate(tmp_path, language="python")
    assert "hasattr" in str(exc_info.value).lower()
    assert "isinstance" in str(exc_info.value).lower()


def test_eval_exec_rejected(tmp_path: Path) -> None:
    validator = PreWriteValidator()
    (tmp_path / "main.py").write_text("x = eval('1 + 1')\n")
    with pytest.raises(ValidationError) as exc_info:
        validator.validate(tmp_path, language="python")
    assert "eval" in str(exc_info.value).lower()


def test_intenum_allowed_and_plain_class_allowed(tmp_path: Path) -> None:
    validator = PreWriteValidator()
    (tmp_path / "main.py").write_text(
        "from enum import IntEnum\n"
        "from dataclasses import dataclass\n"
        "class State(IntEnum):\n    ON = 1\n"
        "@dataclass\n"
        "class Config:\n    value: int = 0\n"
    )
    result = validator.validate(tmp_path, language="python")
    assert result.succeeded


def test_multi_base_class_rejected(tmp_path: Path) -> None:
    validator = PreWriteValidator()
    (tmp_path / "main.py").write_text(
        "from enum import Enum\n"
        "class Serializable:\n    pass\n"
        "class State(Serializable, Enum):\n    ON = 1\n"
    )
    with pytest.raises(ValidationError) as exc_info:
        validator.validate(tmp_path, language="python")
    assert "multiple base classes" in str(exc_info.value).lower()


def test_validator_output_includes_message_and_path(tmp_path: Path) -> None:
    validator = PreWriteValidator()
    (tmp_path / "main.py").write_text("def evaluate_rule_tree(event: dict):\n    pass\n")
    with pytest.raises(ValidationError) as exc_info:
        validator.validate(tmp_path, language="python")
    assert "main.py" in exc_info.value.output
    assert "dict" in exc_info.value.output


def test_primitive_type_annotations_accepted(tmp_path: Path) -> None:
    """Standard built-ins (str, int, list[str], None) must pass primitive validation."""
    validator = PreWriteValidator()
    (tmp_path / "main.py").write_text(
        "def search(query: str, limit: int = 10) -> list[str]:\n"
        "    return [query] * limit\n"
    )
    result = validator.validate(tmp_path, language="python")
    assert result.succeeded


def test_bytes_and_none_return_accepted(tmp_path: Path) -> None:
    """``bytes`` and ``None`` return types are valid primitives."""
    validator = PreWriteValidator()
    (tmp_path / "main.py").write_text(
        "def read_chunk(path: str) -> bytes | None:\n"
        "    return None\n"
    )
    result = validator.validate(tmp_path, language="python")
    assert result.succeeded


def test_typing_optional_and_list_accepted(tmp_path: Path) -> None:
    """``typing.List`` and ``Optional`` generics of primitives are accepted."""
    validator = PreWriteValidator()
    (tmp_path / "main.py").write_text(
        "from typing import List, Optional\n"
        "def items(x: Optional[str]) -> List[int]:\n"
        "    return [1]\n"
    )
    result = validator.validate(tmp_path, language="python")
    assert result.succeeded


def test_primitive_subscript_rejected(tmp_path: Path) -> None:
    """Subscripting a primitive builtin like ``str[int]`` is not a valid type."""
    validator = PreWriteValidator()
    (tmp_path / "main.py").write_text("def bad(x: str[int]) -> None:\n    pass\n")
    with pytest.raises(ValidationError) as exc_info:
        validator.validate(tmp_path, language="python")
    assert "non-primitive" in str(exc_info.value).lower()


def test_mat_mul_empty_return_rewritten_and_validated(tmp_path: Path) -> None:
    """A generated ``mat_mul`` with a bare ``return []`` is auto-rewritten and passes validation."""
    from aero_forge.scaffold.pre_write_validator import rewrite_empty_matrix_returns

    source = (
        "def mat_mul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:\n"
        "    if not a or not b:\n"
        "        return []\n"
        "    rows = len(a)\n"
        "    cols = len(b[0]) if b and b[0] else 0\n"
        "    result = []\n"
        "    for i in range(rows):\n"
        "        row = []\n"
        "        for j in range(cols):\n"
        "            total = 0.0\n"
        "            for k in range(len(b)):\n"
        "                total += a[i][k] * b[k][j]\n"
        "            row.append(total)\n"
        "        result.append(row)\n"
        "    return result\n"
    )

    rewritten = rewrite_empty_matrix_returns(source)
    assert "return []" not in rewritten
    assert "[[0.0] * " in rewritten

    (tmp_path / "mat_mul.py").write_text(rewritten)
    validator = PreWriteValidator()
    result = validator.validate(tmp_path, language="python")
    assert result.succeeded


def test_int_matrix_zero_value(tmp_path: Path) -> None:
    """Rewrites for integer matrices use ``0`` instead of ``0.0``."""
    from aero_forge.scaffold.pre_write_validator import rewrite_empty_matrix_returns

    source = (
        "def zero_int(rows: int, cols: int) -> list[list[int]]:\n"
        "    if rows == 0 or cols == 0:\n"
        "        return []\n"
        "    return [[0] * cols for _ in range(rows)]\n"
    )
    rewritten = rewrite_empty_matrix_returns(source)
    assert "[[0] * " in rewritten
    (tmp_path / "zero_int.py").write_text(rewritten)
    result = PreWriteValidator().validate(tmp_path, language="python")
    assert result.succeeded
