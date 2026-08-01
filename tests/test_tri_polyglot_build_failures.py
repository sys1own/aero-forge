"""Tests for tri-polyglot C++ include-path alignment and fail-fast behavior.

These tests verify that:
1. C++ headers emitted at the workspace root are discoverable by source files
   living in subdirectories such as ``cpp_core/`` or ``cpp_engine/src/``.
2. Native compilation stage failures raise ``BuildStageError`` immediately and
   do not let downstream ``pytest`` collection mask the root cause.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

import pytest

from aero_forge.blueprint import Blueprint, ContractEntry, ManifestEntry
from aero_forge.errors import BuildStageError
from aero_forge.orchestrator.stack_classifier import INTENT_TRI_POLYGLOT_RUST_CPP_PYTHON
from aero_forge.scaffold.cpp_materializer import (
    _collect_include_dirs,
    _find_cpp_compiler,
    _generate_native_cpp,
)
from aero_forge.scaffold.tri_polyglot_materializer import TriPolyglotMaterializer
from aero_forge.universal_builder import _run_polyglot_materializer


def test_collect_include_dirs_includes_workspace_root(tmp_path: Path) -> None:
    """A header at the workspace root must be on the C++ include path."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    source = workspace / "cpp_core" / "native.cpp"
    source.parent.mkdir(parents=True)
    header_paths: List[str] = ["cpp_kernel.hpp"]

    include_dirs = _collect_include_dirs(source, header_paths, workspace)

    assert str(workspace.resolve()) in include_dirs
    assert str(source.parent.resolve()) in include_dirs


def test_generate_native_cpp_uses_workspace_relative_header_paths() -> None:
    """``#include`` directives must be relative to the workspace root, not just
    basenames, so a single ``-I <workspace>`` resolves headers in any layout.
    """
    contract = ContractEntry(
        name="add",
        signature="def add(a: int, b: int) -> int",
    )
    source = _generate_native_cpp(
        "demo_cpp",
        [contract],
        header_includes=["cpp_engine/include/api.h"],
    )
    assert '#include "cpp_engine/include/api.h"' in source


def test_tri_polyglot_build_cpp_raises_build_stage_error(tmp_path: Path) -> None:
    """A failing ``g++`` invocation must raise ``BuildStageError`` with logs."""
    if not _find_cpp_compiler():
        pytest.skip("No C++ compiler available")

    workspace = tmp_path / "ws"
    workspace.mkdir()
    materializer = TriPolyglotMaterializer(workspace)

    cpp_pkg_name = "demo_cpp"
    cpp_source = workspace / "cpp_core" / "native.cpp"
    cpp_source.parent.mkdir(parents=True)
    cpp_source.write_text(
        '#include "missing_header_that_does_not_exist.hpp"\nint main(){}',
        encoding="utf-8",
    )

    with pytest.raises(BuildStageError) as exc_info:
        materializer._build_cpp(cpp_pkg_name, cpp_source, [])

    assert exc_info.value.stage == "cpp_compile"
    assert "missing_header_that_does_not_exist" in exc_info.value.logs


def test_tri_polyglot_materializer_places_root_header_on_include_path(
    tmp_path: Path,
) -> None:
    """A header emitted at the workspace root is found by a custom source dir."""
    if not _find_cpp_compiler():
        pytest.skip("No C++ compiler available")

    workspace = tmp_path / "ws"
    workspace.mkdir()
    blueprint = Blueprint(
        project="root_header_demo",
        architecture=INTENT_TRI_POLYGLOT_RUST_CPP_PYTHON,
        toolchains=["python", "rust", "cpp", "cargo"],
        manifest=[
            ManifestEntry(
                path="cpp_engine/src/kernels.cpp",
                lang="cpp",
                purpose="C-ABI source",
            ),
            ManifestEntry(
                path="api.h",
                lang="cpp",
                purpose="C-ABI header",
            ),
            ManifestEntry(
                path="python_interface/__init__.py",
                lang="python",
                purpose="package init",
            ),
            ManifestEntry(
                path="pyproject.toml",
                lang="toml",
                purpose="project manifest",
            ),
            ManifestEntry(
                path="tests/test_tri.py",
                lang="python",
                purpose="tests",
            ),
            ManifestEntry(path="README.md", lang="markdown", purpose="docs"),
        ],
        contracts=[
            ContractEntry(
                name="add",
                signature="def add(a: int, b: int) -> int",
            ),
        ],
    )

    materializer = TriPolyglotMaterializer(workspace)
    updated = materializer.materialize(blueprint, build=False)

    # The source must reference the root header with a workspace-relative path
    # and the include directories must cover the workspace root.
    source_text = (workspace / "cpp_engine" / "src" / "kernels.cpp").read_text(
        encoding="utf-8"
    )
    assert '#include "api.h"' in source_text

    cpp_source = workspace / "cpp_engine" / "src" / "kernels.cpp"
    include_dirs = _collect_include_dirs(cpp_source, ["api.h"], workspace)
    assert str(workspace.resolve()) in include_dirs

    # Materializer should have emitted the header at the workspace root.
    assert (workspace / "api.h").is_file()

    # The full C++ build should succeed despite source and header being in
    # different directories.
    materializer._build_cpp("root_header_demo_cpp", cpp_source, ["api.h"])
    so_candidates = list((workspace / "cpp_engine" / "src").glob("*.so"))
    assert so_candidates, "Expected compiled C++ .so in cpp_engine/src"


def test_run_polyglot_materializer_halts_on_cpp_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When C++ compilation raises ``BuildStageError``, pytest must not run."""
    workspace = tmp_path / "ws"
    workspace.mkdir()

    original_build_cpp = TriPolyglotMaterializer._build_cpp

    def _failing_build_cpp(
        self: TriPolyglotMaterializer,
        cpp_pkg_name: str,
        cpp_source: Path,
        header_paths: List[str],
    ) -> None:
        raise BuildStageError(
            "Synthetic C++ compile failure",
            stage="cpp_compile",
            logs="synthetic failure log",
        )

    monkeypatch.setattr(TriPolyglotMaterializer, "_build_cpp", _failing_build_cpp)

    blueprint = Blueprint(
        project="halt_demo",
        architecture=INTENT_TRI_POLYGLOT_RUST_CPP_PYTHON,
        toolchains=["python", "rust", "cpp", "cargo"],
        contracts=[
            ContractEntry(
                name="add",
                signature="def add(a: int, b: int) -> int",
            ),
        ],
    )

    result = _run_polyglot_materializer(
        "halt_demo",
        ["cpp", "rust", "python"],
        workspace,
        prompt="tri polyglot halt demo",
        blueprint=blueprint,
    )

    assert result["success"] is False
    assert result.get("stage") == "cpp_compile"
    assert "pytest_output" not in result
    assert "pytest_error" not in result
