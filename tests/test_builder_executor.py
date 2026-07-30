"""Tests for builder execution artifact filtering and workspace scope isolation."""

from __future__ import annotations

from pathlib import Path

import pytest

from aero_forge.builder.executor import (
    ExecutionReport,
    filter_artifact_paths,
    filter_to_scope,
    is_artifact_path,
    is_ignored_by_gitignore,
    parse_gitignore,
    should_report_path,
)


def test_is_artifact_path_rejects_build_and_runtime_artifacts() -> None:
    """Common generated artifacts are recognized and excluded."""
    assert is_artifact_path(Path(".venv/bin/python"))
    assert is_artifact_path(Path("venv/pyvenv.cfg"))
    assert is_artifact_path(Path("pyvenv.cfg"))
    assert is_artifact_path(Path("dist/libfoo.so"))
    assert is_artifact_path(Path("build/lib.so"))
    assert is_artifact_path(Path("target/release/libfoo.so"))
    assert is_artifact_path(Path("__pycache__/foo.cpython-310.pyc"))
    assert is_artifact_path(Path("foo.egg-info/PKG-INFO"))
    assert is_artifact_path(Path("src/lib.so"))
    assert is_artifact_path(Path("project.whl"))
    assert is_artifact_path(Path("workspace.aeroc"))


def test_is_artifact_path_accepts_source_and_config_files() -> None:
    """Source code and build config files are not treated as artifacts."""
    assert not is_artifact_path(Path("src/main.py"))
    assert not is_artifact_path(Path("src/lib.rs"))
    assert not is_artifact_path(Path("src/native.cpp"))
    assert not is_artifact_path(Path("pyproject.toml"))
    assert not is_artifact_path(Path("Cargo.toml"))
    assert not is_artifact_path(Path("tests/test_foo.py"))
    assert not is_artifact_path(Path("README.md"))


def test_filter_artifact_paths_strips_artifacts() -> None:
    """A list of workspace-relative paths is pruned of build artifacts."""
    paths = [
        "src/main.py",
        ".venv/bin/python",
        "dist/libfoo.so",
        "build/lib.so",
        "foo.egg-info/PKG-INFO",
        "pyvenv.cfg",
        "Cargo.toml",
        "target/release/libfoo.so",
        "tests/test_main.py",
        "project.whl",
    ]
    filtered = filter_artifact_paths(paths)
    assert filtered == [
        "Cargo.toml",
        "src/main.py",
        "tests/test_main.py",
    ]


def test_parse_gitignore_reads_patterns(tmp_path: Path) -> None:
    """parse_gitignore returns non-comment patterns from a .gitignore file."""
    gitignore = tmp_path / ".gitignore"
    gitignore.write_text(
        "*.so\n"
        ".venv/\n"
        "dist/\n"
        "# comment\n"
        "!keep.so\n",
        encoding="utf-8",
    )
    assert parse_gitignore(tmp_path) == ["*.so", ".venv/", "dist/"]


def test_is_ignored_by_gitignore_respects_patterns(tmp_path: Path) -> None:
    """Files matching .gitignore patterns are reported as ignored."""
    gitignore = tmp_path / ".gitignore"
    gitignore.write_text("*.so\n.venv/\ndist/\n", encoding="utf-8")
    patterns = parse_gitignore(tmp_path)
    assert is_ignored_by_gitignore(Path("src/lib.so"), patterns)
    assert is_ignored_by_gitignore(Path(".venv/bin/python"), patterns)
    assert is_ignored_by_gitignore(Path("dist/output"), patterns)
    assert not is_ignored_by_gitignore(Path("src/main.py"), patterns)


def test_should_report_path_considers_artifacts_and_gitignore(tmp_path: Path) -> None:
    """should_report_path excludes artifacts and anything ignored by .gitignore."""
    (tmp_path / ".gitignore").write_text("*.so\nbuild/\n", encoding="utf-8")
    assert should_report_path(Path("src/main.py"), tmp_path)
    assert should_report_path(Path("pyproject.toml"), tmp_path)
    assert not should_report_path(Path("src/lib.so"), tmp_path)
    assert not should_report_path(Path("build/lib.so"), tmp_path)
    assert not should_report_path(Path(".venv/bin/python"), tmp_path)


def test_execution_report_filters_paths(tmp_path: Path) -> None:
    """ExecutionReport.filter_paths applies both artifact and .gitignore filtering."""
    (tmp_path / ".gitignore").write_text("*.log\n", encoding="utf-8")
    report = ExecutionReport(tmp_path)
    paths = [
        "src/main.py",
        "src/output.log",
        ".venv/bin/python",
        "dist/lib.so",
        "pyproject.toml",
    ]
    assert report.filter_paths(paths) == ["pyproject.toml", "src/main.py"]


def test_filter_to_scope_keeps_only_task_files_and_config() -> None:
    """Out-of-scope file changes are dropped from the user-facing report."""
    files = [
        "src/accelerator/cli.py",
        "src/accelerator/new_module.rs",
        "src/accelerator/rolling_average.py",
        "src/accelerator/native.cpp",
        "pyproject.toml",
        "Cargo.toml",
        "README.md",
        "tests/test_unrelated.py",
        ".venv/bin/python",
    ]
    scope = [
        "src/accelerator/new_module.rs",
        "src/accelerator/rolling_average.py",
        "src/accelerator/native.cpp",
        "tests/test_rolling_average.py",
    ]
    filtered = filter_to_scope(files, scope)
    assert "src/accelerator/new_module.rs" in filtered
    assert "src/accelerator/rolling_average.py" in filtered
    assert "src/accelerator/native.cpp" in filtered
    assert "pyproject.toml" in filtered
    assert "Cargo.toml" in filtered
    assert "src/accelerator/cli.py" not in filtered
    assert "tests/test_unrelated.py" not in filtered
    assert ".venv/bin/python" not in filtered


def test_filter_to_scope_keeps_files_under_scope_directories() -> None:
    """Scope directories include all files underneath them."""
    files = ["src/foo/bar.py", "src/foo/baz.rs", "src/other/main.py", "pyproject.toml"]
    filtered = filter_to_scope(files, ["src/foo"])
    assert "src/foo/bar.py" in filtered
    assert "src/foo/baz.rs" in filtered
    assert "pyproject.toml" in filtered
    assert "src/other/main.py" not in filtered
