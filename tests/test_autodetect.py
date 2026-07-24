"""Tests for prompt-assisted architecture autodetection."""

from pathlib import Path

from aero_forge.blueprint import FunctionSpec, generate_blueprint


def test_python_only_files_with_rust_prompt_become_polyglot(tmp_path: Path) -> None:
    """Even when only .py files exist on disk, a Rust prompt forces polyglot."""
    src = tmp_path / "src" / "core.py"
    src.parent.mkdir(parents=True)
    src.write_text(
        "def compute_batch(values: list[float]) -> list[float]:\n"
        "    return [v * 2.0 for v in values]\n"
    )

    blueprint = generate_blueprint(
        project="batch_processor",
        functions=[FunctionSpec(file=src, name="compute_batch")],
        output_dir=tmp_path / "dist",
        prompt="Python Rust orchestration build for batch processing with PyO3 and cargo",
    )

    assert blueprint.architecture == "hybrid_rust_python"
    assert "python" in blueprint.toolchains
    assert "rust" in blueprint.toolchains
    assert "cargo" in blueprint.toolchains
    assert blueprint.prompt is not None
    paths = {entry.path for entry in blueprint.manifest}
    assert "Cargo.toml" in paths
    assert "src/lib.rs" in paths


def test_python_prompt_stays_pure_python(tmp_path: Path) -> None:
    """A plain Python prompt does not add Rust artifacts."""
    src = tmp_path / "src" / "pure.py"
    src.parent.mkdir(parents=True)
    src.write_text("def add(a: int, b: int) -> int:\n    return a + b\n")

    blueprint = generate_blueprint(
        project="pure",
        functions=[FunctionSpec(file=src, name="add")],
        output_dir=tmp_path / "dist",
        prompt="Implement a Python sorting function",
    )

    assert blueprint.architecture == "pure_python"
    assert blueprint.toolchains == ["python"]
    assert blueprint.manifest == []
