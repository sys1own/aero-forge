"""Tests for dynamic C++ build target resolution in CppPolyglotMaterializer."""

from __future__ import annotations

import subprocess
import sys

import pytest
from pathlib import Path

from aero_forge.blueprint import Blueprint, ContractEntry
from aero_forge.blueprint.schema import ArtifactType, BlueprintV3, BuildArtifact, Metadata
from aero_forge.scaffold.cpp_materializer import (
    CppPolyglotMaterializer,
    _find_cpp_compiler,
)


def _write_v3_blueprint(workspace: Path, output_path: str = "cpp_native/libnative.so") -> None:
    blueprint = BlueprintV3(
        metadata=Metadata(
            schema_version="3.0.0",
            project_name="native_demo",
            status="finalized",
            generation_method="static_heuristic",
        ),
        build_pipeline=[
            BuildArtifact(
                id="native",
                type=ArtifactType.shared_library,
                source_files=["cpp_native/native.cpp"],
                output_path=output_path,
                compiler_flags=["-O3", "-march=native"],
            )
        ],
    )
    import yaml
    (workspace / "blueprint.aero").write_text(
        yaml.safe_dump(blueprint.model_dump(mode="json")), encoding="utf-8"
    )


@pytest.fixture
def skip_no_compiler() -> None:
    if not _find_cpp_compiler():
        pytest.skip("No C++ compiler available")


def test_resolve_cpp_build_config_from_v3_blueprint(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _write_v3_blueprint(workspace)

    blueprint = Blueprint(
        project="native_demo",
        architecture="hybrid_cpp_python",
        metadata={"schema_version": "2.0.0", "llm_initialized": "true"},
        contracts=[ContractEntry(name="add", signature="def add(a: int, b: int) -> int")],
    )
    materializer = CppPolyglotMaterializer(workspace)
    pkg_dir = workspace / "native_demo"
    pkg_dir.mkdir()
    dummy_cpp = pkg_dir / "native.cpp"
    dummy_cpp.write_text("// placeholder")

    config = materializer._resolve_cpp_build_config(blueprint, "native_demo", pkg_dir, dummy_cpp, [])

    assert config.source_files == [workspace / "cpp_native" / "native.cpp"]
    assert config.output_path.resolve() == (workspace / "cpp_native" / "libnative.so").resolve()
    assert "-O3" in config.compiler_flags
    assert "-march=native" in config.compiler_flags


@pytest.mark.integration
def test_materialize_uses_blueprint_cpp_artifact_paths_and_flags(
    tmp_path: Path, skip_no_compiler
) -> None:
    workspace = tmp_path / "demo"
    workspace.mkdir()
    _write_v3_blueprint(workspace)

    blueprint = Blueprint(
        project="native_demo",
        architecture="hybrid_cpp_python",
        metadata={"schema_version": "2.0.0", "llm_initialized": "true"},
        contracts=[ContractEntry(name="add", signature="def add(a: int, b: int) -> int")],
    )
    CppPolyglotMaterializer(workspace).materialize(blueprint, build=True)

    so_path = workspace / "cpp_native" / "libnative.so"
    assert so_path.is_file(), f"Expected compiled .so at {so_path}"
    assert (workspace / "cpp_native" / "native.cpp").is_file()
    assert (workspace / "native_demo" / "__init__.py").is_file()

    script = workspace / "smoke.py"
    script.write_text(
        "import sys\n"
        "sys.path.insert(0, '.')\n"
        "from native_demo import add\n"
        "assert add(2, 3) == 5\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(script)], cwd=workspace, capture_output=True, text=True
    )
    assert result.returncode == 0, f"C++ smoke test failed: {result.stderr}"
