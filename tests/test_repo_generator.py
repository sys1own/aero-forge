"""End-to-end repository generation tests for aero-forge."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from aero_forge.invisible_config import InvisibleConfigEngine, looks_like_lean_blueprint
from aero_forge.scaffold.universal_generator import UniversalRepoGenerator


@pytest.fixture
def generator(tmp_path: Path) -> UniversalRepoGenerator:
    return UniversalRepoGenerator(tmp_path)


def test_rust_repo_from_prompt(generator: UniversalRepoGenerator) -> None:
    prompt = "def add(a: int, b: int) -> int:\n    return a + b\n"
    result = generator.generate(prompt, target_language="rust", project_name="demo_rust")

    assert result.language == "rust"
    assert (result.root / "Cargo.toml").is_file()
    assert (result.root / "src" / "lib.rs").is_file()
    assert (result.root / "README.md").is_file()
    assert (result.root / "test_binding.py").is_file()
    assert (result.root / ".gitignore").is_file()

    lib_rs = (result.root / "src" / "lib.rs").read_text(encoding="utf-8")
    assert "pub fn add" in lib_rs
    cargo_toml = (result.root / "Cargo.toml").read_text(encoding="utf-8")
    assert 'name = "demo_rust"' in cargo_toml

    if shutil.which("cargo"):
        import subprocess

        fmt = subprocess.run(["cargo", "fmt"], cwd=result.root, capture_output=True, text=True)
        assert fmt.returncode == 0, fmt.stderr


def test_python_repo_from_prompt(generator: UniversalRepoGenerator) -> None:
    prompt = "def add(a: int, b: int) -> int:\n    return a + b\n"
    result = generator.generate(prompt, target_language="python", project_name="demo_python")

    assert result.language == "python"
    assert (result.root / "src" / "add.py").is_file()
    assert (result.root / "src" / "__init__.py").is_file()
    assert (result.root / "tests" / "test_add.py").is_file()
    assert (result.root / "pyproject.toml").is_file()
    assert (result.root / "requirements.txt").is_file()
    assert (result.root / "README.md").is_file()
    assert (result.root / ".gitignore").is_file()

    main_py = (result.root / "src" / "add.py").read_text(encoding="utf-8")
    assert "def add" in main_py
    init = (result.root / "src" / "__init__.py").read_text(encoding="utf-8")
    assert "from .add import add" in init
    pyproject = (result.root / "pyproject.toml").read_text(encoding="utf-8")
    assert 'name = "demo_python"' in pyproject

    import py_compile

    py_compile.compile(str(result.root / "src" / "add.py"), doraise=True)


def test_python_repo_multi_file_workspace(generator: UniversalRepoGenerator) -> None:
    """A generated Python workspace contains multiple supporting files."""
    prompt = (
        "def greet(name: str) -> str:\n"
        "    return f'hello {name}'\n"
    )
    result = generator.generate(prompt, target_language="python", project_name="greet_app")
    files = {f for f in result.files}
    assert {
        "src/greet.py", "src/__init__.py", "tests/test_greet.py",
        "pyproject.toml", "requirements.txt", "README.md", ".gitignore",
    } <= files


def test_incremental_feature_merge_preserves_user_edit(generator: UniversalRepoGenerator) -> None:
    initial = "def add(a: int, b: int) -> int:\n    return a + b\n"
    result = generator.generate(initial, target_language="python", project_name="demo")
    main = result.root / "src" / "add.py"

    # User makes a local edit to the generated entry file.
    original = main.read_text(encoding="utf-8")
    edited = original.replace("return (a + b)", "return (a + b + 1)")
    main.write_text(edited, encoding="utf-8")
    generator.commit_overlay(main)

    # A new prompt adds another function. The prior edit must be preserved.
    new_prompt = (
        "def add(a: int, b: int) -> int:\n"
        "    return a + b\n"
        "\n"
        "def sub(a: int, b: int) -> int:\n"
        "    return a - b\n"
    )
    generator.generate_with_overlay(new_prompt, target_language="python", project_name="demo")

    merged = main.read_text(encoding="utf-8")
    assert "return (a + b + 1)" in merged
    assert "def sub" in merged


def test_polyglot_blueprint_materializes_files(tmp_path: Path) -> None:
    from aero_forge.blueprint import Blueprint, ContractEntry, ManifestEntry
    from aero_forge.scaffold.polyglot_materializer import PolyglotMaterializer

    workspace = tmp_path / "poly"
    blueprint = Blueprint(
        project="polyglot_demo",
        architecture="hybrid_rust_python",
        toolchains=["python", "rust", "cargo"],
        manifest=[
            ManifestEntry(path="Cargo.toml", lang="toml", purpose="workspace manifest"),
            ManifestEntry(path="rust_core/Cargo.toml", lang="toml", purpose="crate manifest"),
            ManifestEntry(path="rust_core/src/lib.rs", lang="rust", purpose="Rust core"),
            ManifestEntry(path="aero_polyglot_runner/__init__.py", lang="python", purpose="package init"),
            ManifestEntry(path="aero_polyglot_runner/orchestrator.py", lang="python", purpose="Python orchestrator"),
            ManifestEntry(path="run_demo.py", lang="python", purpose="demo"),
            ManifestEntry(path="tests/test_polyglot.py", lang="python", purpose="tests"),
            ManifestEntry(path="pyproject.toml", lang="toml", purpose="Python packaging"),
            ManifestEntry(path="README.md", lang="markdown", purpose="docs"),
        ],
        contracts=[
            ContractEntry(name="fast_vector_transform", signature="def fast_vector_transform(v: list[float], scalar: float) -> list[float]"),
            ContractEntry(name="get_engine_status", signature="def get_engine_status() -> dict[str, str]"),
        ],
    )
    updated = PolyglotMaterializer(workspace).materialize(blueprint)

    assert (workspace / "Cargo.toml").is_file()
    assert (workspace / "rust_core" / "Cargo.toml").is_file()
    assert (workspace / "rust_core" / "src" / "lib.rs").is_file()
    assert (workspace / "aero_polyglot_runner" / "orchestrator.py").is_file()
    assert (workspace / "aero_polyglot_runner" / "__init__.py").is_file()
    assert (workspace / "run_demo.py").is_file()
    assert (workspace / "tests" / "test_polyglot.py").is_file()
    assert (workspace / "pyproject.toml").is_file()
    assert (workspace / "README.md").is_file()
    assert any(f.name == "fast_vector_transform" for f in updated.functions)
    assert any(f.name == "get_engine_status" for f in updated.functions)
    assert any(f.name == "PolyglotEngine" for f in updated.functions)


def test_invisible_config_generates_repo(tmp_path: Path) -> None:
    engine = InvisibleConfigEngine(tmp_path)
    content = (
        'project "invisible_demo"\n'
        'targets = ["rust"]\n'
        'source = "def add(a: int, b: int) -> int:\\n    return a + b\\n"\n'
    )
    assert looks_like_lean_blueprint(content)
    context = engine.build_context_from_source(content, output_dir=tmp_path / "out")
    assert context["project"] == "invisible_demo"
    repo = context["repo"]
    assert repo["language"] == "rust"
    assert "src/lib.rs" in repo["files"]
    assert (Path(repo["root"]) / "Cargo.toml").is_file()
