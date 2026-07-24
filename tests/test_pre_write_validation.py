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
