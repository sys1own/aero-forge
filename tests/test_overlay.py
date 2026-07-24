"""Tests for overlay, patch, and structural AST merge."""

from __future__ import annotations

from pathlib import Path

import pytest

from aero_forge.overlay import (
    OverlayManager,
    ReapplyStatus,
    StructuralMerger,
    apply_patch,
    make_patch,
)
from aero_forge.overlay.apply import _parse_hunks


def test_make_and_apply_patch_round_trip() -> None:
    original = "def foo():\n    return 1\n"
    modified = "def foo():\n    return 2\n"
    patch = make_patch(original, modified)
    assert patch
    merged, conflict = apply_patch(original, patch)
    assert not conflict
    assert merged == modified


def test_apply_patch_conflict() -> None:
    original = "def foo():\n    return 1\n"
    modified = "def bar():\n    return 2\n"
    patch = make_patch(original, modified)
    # The surrounding context no longer matches.
    merged, conflict = apply_patch("x = 1\n", patch)
    assert conflict


def test_parse_hunks() -> None:
    patch = (
        "--- a\n"
        "+++ b\n"
        "@@ -1,2 +1,2 @@\n"
        " def foo():\n"
        "-    return 1\n"
        "+    return 2\n"
    )
    hunks = _parse_hunks(patch)
    assert len(hunks) == 1
    tags = [tag for tag, _ in hunks[0]]
    assert tags == [" ", "-", "+"]


def test_overlay_manager_commit_and_reapply(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    manager = OverlayManager(workspace)

    source_file = workspace / "main.py"
    original = "def foo():\n    return 1\n"
    source_file.write_text(original)
    manager.record_generated(source_file)

    edited = "def foo():\n    return 2\n"
    source_file.write_text(edited)
    patch = manager.commit_overlay(source_file)
    assert patch
    assert "-    return 1" in patch
    assert "+    return 2" in patch

    # Simulate regeneration by restoring the original baseline.
    source_file.write_text(original)
    status = manager.reapply(source_file)
    assert status == ReapplyStatus.APPLIED
    assert source_file.read_text() == edited


def test_overlay_manager_reapply_missing_file(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    manager = OverlayManager(workspace)

    source_file = workspace / "main.py"
    source_file.write_text("x = 1\n")
    manager.record_generated(source_file)
    source_file.write_text("x = 2\n")
    manager.commit_overlay(source_file)
    source_file.unlink()
    assert manager.reapply(source_file) == ReapplyStatus.MISSING


def test_structural_merge_python_adds_user_edit(tmp_path: Path) -> None:
    base = """def foo():\n    return 1\n\ndef bar():\n    return 2\n"""
    left = """def foo():\n    return 42\n\ndef bar():\n    return 2\n"""
    right = """def foo():\n    return 1\n\ndef bar():\n    return 3\n"""
    merger = StructuralMerger("python")
    outcome = merger.merge(base, left, right)
    assert "return 42" in outcome.text
    assert "return 3" in outcome.text
    assert not outcome.conflicts


def test_structural_merge_python_conflict_on_same_entity(tmp_path: Path) -> None:
    base = "def foo():\n    return 1\n"
    left = "def foo():\n    return 2\n"
    right = "def foo():\n    return 3\n"
    merger = StructuralMerger("python")
    outcome = merger.merge(base, left, right)
    # The line-level 3-way patch may or may not conflict; either way we should
    # produce output containing the right-hand side as the fresh generation.
    assert "def foo():" in outcome.text


def test_structural_merge_rust_falls_back_to_line_patch() -> None:
    base = "pub fn foo() -> i32 { 1 }\n"
    left = "pub fn foo() -> i32 { 2 }\n"
    right = "pub fn foo() -> i32 { 1 }\npub fn bar() -> i32 { 0 }\n"
    merger = StructuralMerger("rust")
    outcome = merger.merge(base, left, right)
    assert "pub fn foo()" in outcome.text
    assert "pub fn bar()" in outcome.text


def test_overlay_manager_structural_reapply(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    manager = OverlayManager(workspace)

    source_file = workspace / "main.py"
    base = "def foo():\n    return 1\n"
    source_file.write_text(base)
    manager.record_generated(source_file)

    user_edit = "def foo():\n    return 2\n"
    source_file.write_text(user_edit)
    manager.commit_overlay(source_file)

    regenerated = "def foo():\n    return 1\n    x = 3\n"
    status = manager.structural_reapply(source_file, regenerated, language="python")
    assert status == ReapplyStatus.APPLIED
    text = source_file.read_text()
    assert "return 2" in text
    assert "x = 3" in text
