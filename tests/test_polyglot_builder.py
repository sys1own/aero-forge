"""Tests for the aero-forge polyglot builder / emitter core."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from aero_forge.builder import (
    ASTNode,
    ArtifactGenerator,
    BuildOutput,
    EngineSpec,
    binary_op,
    block,
    binding,
    build_engine,
    call,
    comment,
    dict_literal,
    field,
    function,
    get_emitter,
    list_literal,
    literal,
    module,
    param,
    reference,
    resolve_target_language,
    return_node,
    spec_from_python,
    struct,
)
from aero_forge.builder.emitters.base import EmitterError


@pytest.fixture
def kv_spec() -> EngineSpec:
    """A simple key-value store engine spec."""
    root = module(
        name="kv_engine",
        children=[
            comment("Key-value store engine"),
            struct(
                "KVStore",
                fields=[
                    field("data", "dict[str, int]"),
                ],
            ),
            function(
                "get",
                params=[param("store", "KVStore"), param("key", "str")],
                return_type="int",
                body=[
                    return_node(call("lookup", [reference("key")])),
                ],
            ),
            function(
                "sum_values",
                params=[param("values", "list[int]")],
                return_type="int",
                body=[
                    binding("total", literal(0), "int"),
                    return_node(reference("total")),
                ],
            ),
        ],
    )
    return EngineSpec(
        name="kv_engine",
        root=root,
        metadata={"language": "rust"},
    )


@pytest.fixture
def fib_spec() -> EngineSpec:
    """A fibonacci engine spec with an if statement."""
    then_body = block([return_node(literal(1))])
    else_body = block(
        [
            return_node(
                binary_op(
                    call("fib", [binary_op(reference("n"), "-", literal(1))]),
                    "+",
                    call("fib", [binary_op(reference("n"), "-", literal(2))]),
                )
            )
        ]
    )
    root = module(
        name="fib_engine",
        children=[
            function(
                "fib",
                params=[param("n", "int")],
                return_type="int",
                body=[
                    ASTNode(
                        kind="if",
                        children=[
                            binary_op(reference("n"), "<=", literal(1)),
                            then_body,
                            else_body,
                        ],
                    )
                ],
            )
        ],
    )
    return EngineSpec(name="fib_engine", root=root)


def test_language_router_from_extension() -> None:
    assert resolve_target_language(source_path="compute.py") == "python"
    assert resolve_target_language(source_path="compute.rs") == "rust"
    assert resolve_target_language(source_path="compute.cpp") == "cpp"


def test_language_router_context_override() -> None:
    assert (
        resolve_target_language(
            {"frameworks": {"language": "python"}},
            source_path="compute.rs",
        )
        == "python"
    )


def test_emitter_registry() -> None:
    assert get_emitter("rust").target_language == "rust"
    assert get_emitter("python").target_language == "python"
    assert get_emitter("cpp").target_language == "cpp"


def test_unknown_emitter_raises() -> None:
    with pytest.raises(EmitterError):
        get_emitter("fortran")


def test_rust_emitter(kv_spec: EngineSpec) -> None:
    output = build_engine(kv_spec, target_language="rust")
    assert isinstance(output, BuildOutput)
    assert output.language == "rust"
    assert "pub struct KVStore" in output.source
    assert "pub fn get" in output.source
    assert "pub fn sum_values" in output.source
    assert "HashMap" not in output.source or "data" in output.source


def test_python_emitter(kv_spec: EngineSpec) -> None:
    output = build_engine(kv_spec, target_language="python")
    assert output.language == "python"
    assert "class KVStore:" in output.source
    assert "def get(store: KVStore, key: str) -> int:" in output.source
    assert "def sum_values(values: list[int]) -> int:" in output.source


def test_cpp_emitter(kv_spec: EngineSpec) -> None:
    output = build_engine(kv_spec, target_language="cpp")
    assert output.language == "cpp"
    assert "struct KVStore" in output.source
    assert "int get(" in output.source or "auto get(" in output.source
    assert "int sum_values(" in output.source


def test_fib_rust_control_flow(fib_spec: EngineSpec) -> None:
    output = build_engine(fib_spec, target_language="rust")
    assert "pub fn fib" in output.source
    assert "if (n <= 1)" in output.source or "if n <= 1 {" in output.source
    assert "return 1;" in output.source


def test_fib_python_control_flow(fib_spec: EngineSpec) -> None:
    output = build_engine(fib_spec, target_language="python")
    assert "def fib(n: int) -> int:" in output.source
    assert "if (n <= 1):" in output.source or "if n <= 1:" in output.source
    assert "return 1" in output.source


def test_python_to_spec_round_trip() -> None:
    source = """
def add(a: int, b: int) -> int:
    return a + b
"""
    spec = spec_from_python(source, name="math_engine")
    output = build_engine(spec, target_language="python")
    assert "def add(a: int, b: int) -> int:" in output.source
    assert "return (a + b)" in output.source
    rust = build_engine(spec, target_language="rust")
    assert "pub fn add" in rust.source


def test_artifact_generator_kv_store() -> None:
    spec = EngineSpec(
        name="kv_demo",
        root=module(),
        metadata={"language": "rust"},
    )
    generator = ArtifactGenerator()
    artifact = generator.render(
        "kv_store.rs",
        spec,
        output_path="src/kv_store.rs",
        struct_name="MyStore",
        key_type="String",
        value_type="i64",
    )
    assert artifact.path == "src/kv_store.rs"
    assert "pub struct MyStore" in artifact.content
    assert "HashMap<String, i64>" in artifact.content


def test_artifact_generator_policy_evaluator() -> None:
    spec = EngineSpec(name="policy_demo", root=module())
    generator = ArtifactGenerator()
    artifact = generator.render(
        "policy_evaluator.rs",
        spec,
        output_path="src/policy.rs",
        struct_name="Policy",
    )
    assert "pub struct Policy" in artifact.content
    assert "pub fn evaluate" in artifact.content


def test_artifact_generator_bundle() -> None:
    spec = EngineSpec(name="bundled", root=module(), metadata={"language": "rust"})
    generator = ArtifactGenerator()
    bundle = generator.generate(
        spec,
        ["Cargo.toml", "README.md"],
        output_paths={"Cargo.toml": "Cargo.toml", "README.md": "README.md"},
    )
    paths = {a.path for a in bundle.artifacts}
    assert paths == {"Cargo.toml", "README.md"}
    cargo = next(a for a in bundle.artifacts if a.path == "Cargo.toml")
    assert 'name = "bundled"' in cargo.content


def test_emit_with_artifacts(kv_spec: EngineSpec) -> None:
    output = build_engine(
        kv_spec,
        target_language="rust",
        template_names=["Cargo.toml", "README.md"],
    )
    assert output.artifacts.artifacts
    assert any(a.path == "Cargo.toml" for a in output.artifacts.artifacts)


def test_dict_literal_emits() -> None:
    root = module(
        children=[
            function(
                "make_map",
                return_type="dict[str, int]",
                body=[
                    binding("m", dict_literal({"a": 1, "b": 2}), "dict[str, int]"),
                    return_node(reference("m")),
                ],
            )
        ]
    )
    spec = EngineSpec(name="dict_demo", root=root)
    rust = build_engine(spec, target_language="rust").source
    assert "HashMap" in rust
    python = build_engine(spec, target_language="python").source
    assert '"a": 1' in python


def test_list_literal_emits() -> None:
    root = module(
        children=[
            function(
                "make_list",
                return_type="list[int]",
                body=[
                    binding("xs", list_literal([1, 2, 3]), "list[int]"),
                    return_node(reference("xs")),
                ],
            )
        ]
    )
    spec = EngineSpec(name="list_demo", root=root)
    rust = build_engine(spec, target_language="rust").source
    assert "vec![" in rust
    cpp = build_engine(spec, target_language="cpp").source
    assert "std::vector" in cpp or "{" in cpp


@pytest.mark.integration
def test_polyglot_materializer_builds_shared_object(tmp_path: Path) -> None:
    """End-to-end: a hybrid blueprint is materialised, compiled, and tested."""
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
            ContractEntry(
                name="fast_vector_transform",
                signature="def fast_vector_transform(v: list[float], scalar: float) -> list[float]",
            ),
            ContractEntry(
                name="get_engine_status",
                signature="def get_engine_status() -> dict[str, str]",
            ),
        ],
    )
    updated = PolyglotMaterializer(workspace).materialize(blueprint, build=True)

    so_files = list((workspace / "dist").glob("*.so"))
    assert so_files, "Expected a compiled .so in dist/"

    script = workspace / "check_backend.py"
    script.write_text(
        'import sys\n'
        'sys.path.insert(0, ".")\n'
        'from aero_polyglot_runner.orchestrator import PolyglotEngine\n'
        'engine = PolyglotEngine()\n'
        'assert engine.backend == "rust"\n'
        'assert engine.fast_vector_transform([1.0, 2.0, 3.0], 2.0) == [2.0, 4.0, 6.0]\n'
        'assert engine.get_engine_status()["status"] == "ok"\n'
    )
    result = subprocess.run(
        ["python", str(script)], cwd=workspace, capture_output=True, text=True
    )
    assert result.returncode == 0, f"Native backend smoke test failed: {result.stderr}"

    pytest_result = subprocess.run(
        ["python", "-m", "pytest", "tests/test_polyglot.py", "-q"],
        cwd=workspace,
        capture_output=True,
        text=True,
    )
    assert pytest_result.returncode == 0, f"Generated tests failed:\n{pytest_result.stdout}\n{pytest_result.stderr}"

    assert any(f.name == "fast_vector_transform" for f in updated.functions)
    assert any(f.name == "get_engine_status" for f in updated.functions)
    assert any(f.name == "PolyglotEngine" for f in updated.functions)
