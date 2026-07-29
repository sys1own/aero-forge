"""Tests for the aero-forge polyglot builder / emitter core."""

from __future__ import annotations

import shutil
import subprocess
import sys
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
    with_gil_release,
)
from aero_forge.builder.emitters.base import EmitterError
from aero_forge.builder.language_router import is_cpp_friendly, should_accelerate_with_native
from aero_forge.scaffold.cargo_manifest import infer_dependencies


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


def test_language_router_cpp_heuristic() -> None:
    numeric_loop = "def sum_even(n: int) -> int:\n    total = 0\n    for i in range(n + 1):\n        if i % 2 == 0:\n            total += i\n    return total\n"
    assert should_accelerate_with_native(numeric_loop)
    assert is_cpp_friendly(numeric_loop)

    io_code = "def greet(name: str) -> str:\n    return 'hello ' + name\n"
    assert not is_cpp_friendly(io_code)
    assert not should_accelerate_with_native(io_code)

    numpy_code = "import numpy as np\ndef vector_dot(a, b):\n    return float(np.dot(a, b))\n"
    assert not is_cpp_friendly(numpy_code)


@pytest.mark.integration
def test_cpp_c_abi_shared_object(tmp_path: Path) -> None:
    """End-to-end: generate C-ABI source, compile it, and call the .so via ctypes."""
    import shutil
    import subprocess

    from aero_forge.builder.emitters.cpp_emitter import CppEmitter
    from aero_forge.native_bridge import _ctypes_loader_source

    source = """
def sum_even(n: int) -> int:
    total = 0
    for i in range(n + 1):
        if i % 2 == 0:
            total += i
    return total
"""
    spec = spec_from_python(source, name="sum_even")
    cpp = CppEmitter(c_abi=True).emit(spec)
    assert 'extern "C"' in cpp
    assert "AERO_EXPORT" in cpp

    cpp_path = tmp_path / "native.cpp"
    so_path = tmp_path / "libsum_even.so"
    cpp_path.write_text(cpp, encoding="utf-8")

    compiler = _find_cpp_compiler() or "g++"
    result = subprocess.run(
        [compiler, "-shared", "-fPIC", "-O2", "-std=c++17", "-o", str(so_path), str(cpp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert so_path.is_file()

    loader_source = _ctypes_loader_source(source, so_path, ["sum_even"])
    loader_path = tmp_path / "loader.py"
    loader_path.write_text(loader_source, encoding="utf-8")
    import importlib.util
    spec_loader = importlib.util.spec_from_file_location("cpp_loader", loader_path)
    assert spec_loader is not None
    mod = importlib.util.module_from_spec(spec_loader)
    spec_loader.loader.exec_module(mod)
    assert mod.sum_even(10) == 30
    assert mod.sum_even(100) == 2550


@pytest.mark.integration
def test_cpp_pybind11_polyglot_materializer_builds_and_runs(tmp_path: Path) -> None:
    """End-to-end: a C++/Python hybrid blueprint is materialised, compiled, and executed."""
    from aero_forge.blueprint import Blueprint, ContractEntry, ManifestEntry
    from aero_forge.scaffold.cpp_materializer import CppPolyglotMaterializer

    if not _find_cpp_compiler():
        pytest.skip("No C++ compiler available")

    workspace = tmp_path / "cpp_poly"
    blueprint = Blueprint(
        project="cpp_poly_demo",
        architecture="hybrid_cpp_python",
        toolchains=["python", "cpp", "setuptools"],
        manifest=[
            ManifestEntry(path="cpp_poly_demo/__init__.py", lang="python", purpose="package init"),
            ManifestEntry(path="cpp_poly_demo/native.cpp", lang="cpp", purpose="pybind11 extension source"),
            ManifestEntry(path="cpp_poly_demo/cli.py", lang="python", purpose="CLI module"),
            ManifestEntry(path="setup.py", lang="python", purpose="setuptools build script"),
            ManifestEntry(path="pyproject.toml", lang="toml", purpose="project manifest"),
            ManifestEntry(path="run_shell.py", lang="python", purpose="launcher"),
            ManifestEntry(path="tests/test_cli.py", lang="python", purpose="tests"),
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
    updated = CppPolyglotMaterializer(workspace).materialize(blueprint, build=True)
    so_files = list(workspace.glob("*.so")) + list((workspace / "cpp_poly_demo").glob("*.so"))
    assert so_files, "Expected a compiled .so"

    script = workspace / "check_backend.py"
    script.write_text(
        'import sys\n'
        'sys.path.insert(0, ".")\n'
        'from cpp_poly_demo import fast_vector_transform, get_engine_status\n'
        'assert fast_vector_transform([1.0, 2.0, 3.0], 2.0) == [2.0, 4.0, 6.0]\n'
        'assert get_engine_status()["status"] == "ok"\n'
    )
    result = subprocess.run(["python", str(script)], cwd=workspace, capture_output=True, text=True)
    assert result.returncode == 0, f"C++ hybrid smoke test failed: {result.stderr}"

    pytest_result = subprocess.run(
        ["python", "-m", "pytest", "tests/test_cli.py", "-q"],
        cwd=workspace,
        capture_output=True,
        text=True,
    )
    assert pytest_result.returncode == 0, f"Generated tests failed:\n{pytest_result.stdout}\n{pytest_result.stderr}"

    assert any(f.name == "fast_vector_transform" for f in updated.functions)


@pytest.mark.integration
def test_tri_polyglot_materializer_builds_and_runs(tmp_path: Path) -> None:
    """End-to-end: a Python + Rust + C++ tri-polyglot workspace is materialized, compiled, and executed."""
    from aero_forge.blueprint import Blueprint, ContractEntry, ManifestEntry
    from aero_forge.scaffold.tri_polyglot_materializer import TriPolyglotMaterializer

    if not _find_cpp_compiler():
        pytest.skip("No C++ compiler available")
    if not shutil.which("cargo"):
        pytest.skip("No Rust cargo available")

    workspace = tmp_path / "tri_poly"
    blueprint = Blueprint(
        project="tri_demo",
        architecture="tri_polyglot_rust_cpp_python",
        toolchains=["python", "rust", "cpp", "cargo"],
        manifest=[
            ManifestEntry(path="Cargo.toml", lang="toml", purpose="workspace manifest"),
            ManifestEntry(path="rust_core/Cargo.toml", lang="toml", purpose="PyO3 crate manifest"),
            ManifestEntry(path="rust_core/src/lib.rs", lang="rust", purpose="Rust core"),
            ManifestEntry(path="cpp_core/native.cpp", lang="cpp", purpose="C-ABI source"),
            ManifestEntry(path="tri_demo/__init__.py", lang="python", purpose="package init"),
            ManifestEntry(path="tri_demo/main.py", lang="python", purpose="CLI"),
            ManifestEntry(path="pyproject.toml", lang="toml", purpose="project manifest"),
            ManifestEntry(path="run_shell.py", lang="python", purpose="launcher"),
            ManifestEntry(path="tests/test_tri.py", lang="python", purpose="tests"),
            ManifestEntry(path="README.md", lang="markdown", purpose="docs"),
        ],
        contracts=[
            ContractEntry(
                name="fast_vector_transform",
                signature="def fast_vector_transform(v: list[float], scalar: float) -> list[float]",
            ),
            ContractEntry(
                name="validate_token",
                signature="def validate_token(token: str) -> bool",
            ),
            ContractEntry(
                name="get_engine_status",
                signature="def get_engine_status() -> dict[str, str]",
            ),
        ],
    )
    updated = TriPolyglotMaterializer(workspace).materialize(blueprint, build=True)

    assert (workspace / "cpp_core" / "native.cpp").is_file()
    assert (workspace / "rust_core" / "Cargo.toml").is_file()
    assert (workspace / "tri_demo" / "main.py").is_file()
    assert updated.architecture == "tri_polyglot_rust_cpp_python"

    cpp_so = next((p for p in (workspace / "cpp_core").glob("*.so")), None)
    assert cpp_so, "Expected compiled C++ .so"

    rust_candidates = list((workspace / "rust_core" / "target" / "release").glob("*.so")) + list((workspace / "target" / "release").glob("*.so"))
    rust_so = next((p for p in rust_candidates), None)
    assert rust_so, "Expected compiled Rust .so"

    # Smoke test the package entrypoint.
    smoke = workspace / "check_tri.py"
    smoke.write_text(
        "import sys\n"
        'sys.path.insert(0, ".")\n'
        "from tri_demo import fast_vector_transform, validate_token, get_engine_status\n"
        "assert fast_vector_transform([1.0, 2.0, 3.0], 2.0) == [2.0, 4.0, 6.0]\n"
        'assert validate_token("validtoken123") is True\n'
        'assert validate_token("short") is False\n'
        'assert get_engine_status().get("status") == "ok"\n'
        'print("tri-polyglot smoke ok")\n'
    )
    result = subprocess.run(["python", str(smoke)], cwd=workspace, capture_output=True, text=True)
    assert result.returncode == 0, f"Tri-polyglot smoke test failed: {result.stderr}"

    pytest_result = subprocess.run(
        ["python", "-m", "pytest", "tests/test_tri.py", "-q"],
        cwd=workspace,
        capture_output=True,
        text=True,
    )
    assert pytest_result.returncode == 0, f"Generated tri-polyglot tests failed:\n{pytest_result.stdout}\n{pytest_result.stderr}"

    accel_log = workspace / ".aero_forge_accel.log"
    if accel_log.is_file():
        log_text = accel_log.read_text(encoding="utf-8")
        assert "C++ dynamic shared" in log_text or "C-ABI" in log_text, "Expected C++ dispatch in accelerator log"
        assert "Rust PyO3" in log_text or "tri-polyglot" in log_text, "Expected Rust dispatch in accelerator log"


def _find_cpp_compiler() -> str | None:
    for name in ["g++", "clang++", "c++"]:
        if shutil.which(name):
            return name
    return None


@pytest.mark.integration
def test_hybrid_cpp_rust_materializer_builds_and_runs(tmp_path: Path) -> None:
    """End-to-end: a C++/Rust (no Python runtime) hybrid workspace is materialized, compiled, and run."""
    from aero_forge.blueprint import Blueprint, ContractEntry, ManifestEntry
    from aero_forge.scaffold.hybrid_cpp_rust_materializer import HybridCppRustMaterializer

    if not _find_cpp_compiler():
        pytest.skip("No C++ compiler available")
    if not shutil.which("cargo"):
        pytest.skip("No Rust cargo available")

    workspace = tmp_path / "hybrid_cpp_rust"
    blueprint = Blueprint(
        project="hybrid_demo",
        architecture="hybrid_cpp_rust",
        toolchains=["rust", "cpp", "cargo"],
        manifest=[
            ManifestEntry(path="Cargo.toml", lang="toml", purpose="Rust package manifest"),
            ManifestEntry(path="build.rs", lang="rust", purpose="C++ build and link script"),
            ManifestEntry(path="src/lib.rs", lang="rust", purpose="Rust library wrappers"),
            ManifestEntry(path="src/main.rs", lang="rust", purpose="Rust CLI binary"),
            ManifestEntry(path="src/cpp_core/native.cpp", lang="cpp", purpose="C-ABI math source"),
            ManifestEntry(path="tests/test_hybrid_cpp_rust.rs", lang="rust", purpose="Rust integration test"),
            ManifestEntry(path="README.md", lang="markdown", purpose="Project README"),
        ],
        contracts=[
            ContractEntry(
                name="fast_vector_transform",
                signature="def fast_vector_transform(v: list[float], scalar: float) -> list[float]",
            ),
        ],
    )
    updated = HybridCppRustMaterializer(workspace).materialize(blueprint, build=True)

    assert (workspace / "src" / "cpp_core" / "native.cpp").is_file()
    assert (workspace / "src" / "main.rs").is_file()
    assert (workspace / "src" / "lib.rs").is_file()
    assert (workspace / "build.rs").is_file()
    assert updated.architecture == "hybrid_cpp_rust"

    binary = workspace / "target" / "release" / updated.project
    if not binary.is_file():
        binary = workspace / "target" / "release" / updated.project.replace("-", "_")
    assert binary.is_file(), "Expected compiled Rust binary"

    run_result = subprocess.run(
        [str(binary), "--benchmark", "100"],
        cwd=workspace,
        capture_output=True,
        text=True,
    )
    assert run_result.returncode == 0, f"Binary run failed:\n{run_result.stdout}\n{run_result.stderr}"
    assert "Benchmark:" in run_result.stdout, "Expected benchmark output"

    test_result = subprocess.run(
        ["cargo", "test", "--release"],
        cwd=workspace,
        capture_output=True,
        text=True,
    )
    assert test_result.returncode == 0, f"Rust integration test failed:\n{test_result.stdout}\n{test_result.stderr}"
    assert (
        "test_hybrid_cpp_rust_fast_vector_transform ... ok" in test_result.stdout
        or "test_hybrid_cpp_rust_vector_transform ... ok" in test_result.stdout
    ), "Expected test to pass"

    accel_log = workspace / ".aero_forge_accel.log"
    if accel_log.is_file():
        log_text = accel_log.read_text(encoding="utf-8")
        assert "C++ selected for extern \"C\"" in log_text, "Expected C++ dispatch in accelerator log"
        assert "hybrid C++/Rust binary compiled successfully" in log_text, "Expected build success in accelerator log"


@pytest.mark.integration
def test_tri_polyglot_materializer_honors_module_graph_paths(tmp_path: Path) -> None:
    """Tri-polyglot materializer places Rust/C++/Python files at module_graph paths."""
    from aero_forge.blueprint import Blueprint, ContractEntry, ManifestEntry
    from aero_forge.scaffold.tri_polyglot_materializer import TriPolyglotMaterializer

    if not _find_cpp_compiler():
        pytest.skip("No C++ compiler available")
    if not shutil.which("cargo"):
        pytest.skip("No Rust cargo available")

    workspace = tmp_path / "tri_mc"
    blueprint = Blueprint(
        project="mc_engine",
        architecture="tri_polyglot_rust_cpp_python",
        toolchains=["python", "rust", "cpp", "cargo"],
        module_graph=[
            {"path": "crates/mc_engine/Cargo.toml", "lang": "toml", "purpose": "PyO3 crate manifest"},
            {"path": "crates/mc_engine/src/lib.rs", "lang": "rust", "purpose": "Rust core"},
            {"path": "src/cpp/native.cpp", "lang": "cpp", "purpose": "C-ABI source"},
            {"path": "src/python/__init__.py", "lang": "python", "purpose": "package init"},
            {"path": "src/python/main.py", "lang": "python", "purpose": "CLI"},
            {"path": "pyproject.toml", "lang": "toml", "purpose": "project manifest"},
            {"path": "run_shell.py", "lang": "python", "purpose": "launcher"},
            {"path": "tests/test_mc.py", "lang": "python", "purpose": "tests"},
            {"path": "README.md", "lang": "markdown", "purpose": "docs"},
        ],
        contracts=[
            ContractEntry(
                name="fast_vector_transform",
                signature="def fast_vector_transform(v: list[float], scalar: float) -> list[float]",
            ),
            ContractEntry(
                name="get_status",
                signature="def get_status() -> dict[str, str]",
            ),
        ],
        execution_strategy={
            "primary_entrypoint": {"path": "src/python/main.py", "runtime": "python3", "wrapper_generation": True},
            "cli_contract": {
                "parser_type": "argparse",
                "flags": [
                    {"name": "cmd", "short": "c", "type": "string", "required": True, "default": None, "choices": [], "help": "Command", "dest_var": "cmd"},
                    {"name": "simulations", "short": "n", "type": "int", "required": False, "default": 100000, "choices": [], "help": "Simulations", "dest_var": "simulations"},
                    {"name": "spot", "short": "s", "type": "float", "required": False, "default": 100.0, "choices": [], "help": "Spot", "dest_var": "spot"},
                    {"name": "strike", "short": "k", "type": "float", "required": False, "default": 100.0, "choices": [], "help": "Strike", "dest_var": "strike"},
                ],
            },
            "run_spec": {},
        },
    )
    updated = TriPolyglotMaterializer(workspace).materialize(blueprint, build=True)

    assert (workspace / "crates" / "mc_engine" / "Cargo.toml").is_file()
    assert (workspace / "crates" / "mc_engine" / "src" / "lib.rs").is_file()
    assert (workspace / "src" / "cpp" / "native.cpp").is_file()
    assert (workspace / "src" / "python" / "main.py").is_file()

    main_text = (workspace / "src" / "python" / "main.py").read_text(encoding="utf-8")
    assert "--simulations" in main_text
    assert "--spot" in main_text
    assert "--strike" in main_text

    cargo_result = subprocess.run(
        ["cargo", "test", "--manifest-path", "crates/mc_engine/Cargo.toml"],
        cwd=workspace,
        capture_output=True,
        text=True,
    )
    assert cargo_result.returncode == 0, f"Cargo test failed:\n{cargo_result.stdout}\n{cargo_result.stderr}"

    help_result = subprocess.run(
        [sys.executable, "-m", "src.python.main", "--help"],
        cwd=workspace,
        capture_output=True,
        text=True,
    )
    assert help_result.returncode == 0, help_result.stderr
    assert "--simulations" in help_result.stdout
    assert "--spot" in help_result.stdout
    assert "--strike" in help_result.stdout

    assert updated.architecture == "tri_polyglot_rust_cpp_python"


def test_rust_emitter_pyo3_array_and_submodules() -> None:
    """PyO3/NumPy signatures, submodules, GIL release, and dependency inference work end-to-end."""
    ops = module(
        name="ops",
        children=[
            function(
                "matrix_multiply",
                params=[param("a", "&PyArray2<f64>"), param("b", "&PyArray2<f64>")],
                return_type="PyResult<&PyArray2<f64>>",
                body=[
                    with_gil_release([
                        binding("out", literal(0.0), "f64"),
                    ]),
                    return_node(reference("out")),
                ],
            ),
        ],
    )
    spec = EngineSpec(
        name="array_bridge",
        root=module(
            name="array_bridge",
            children=[
                function(
                    "sum_arrays",
                    params=[param("py", "Python"), param("x", "&PyArray2<f64>")],
                    return_type="PyResult<f64>",
                    body=[
                        with_gil_release([
                            return_node(literal(0.0)),
                        ]),
                    ],
                ),
            ],
        ),
        metadata={
            "module_files": [
                {"path": "src/ops.rs", "root": ops},
            ],
            "pyo3": True,
            "numpy": True,
        },
    )

    output = build_engine(spec, target_language="rust")
    source = output.source

    assert "use pyo3::prelude::*;" in source
    assert "use numpy::PyArray2;" in source
    assert "pub mod ops;" in source
    assert "&PyArray2<f64>" in source
    assert "Python" in source
    assert "PyResult<f64>" in source
    assert "#[pyfunction" in source
    assert "#[pymodule]" in source
    assert "py.allow_threads" in source
    assert "wrap_pyfunction!(ops::matrix_multiply)" in source

    artifact_paths = {a.path for a in output.artifacts.artifacts}
    assert "src/ops.rs" in artifact_paths
    ops_artifact = next(a for a in output.artifacts.artifacts if a.path == "src/ops.rs")
    assert "pub fn matrix_multiply" in ops_artifact.content
    assert "&PyArray2<f64>" in ops_artifact.content

    deps = infer_dependencies(source + ops_artifact.content)
    assert "pyo3" in deps
    assert "numpy" in deps


def test_cargo_manifest_infers_rayon_for_par_iter() -> None:
    source = (
        "use rayon::prelude::*;\n"
        "fn compute(v: Vec<f64>) -> f64 {\n"
        "    v.par_iter().fold(|| 0.0, |a, b| a + b).reduce(|| 0.0, |a, b| a + b)\n"
        "}\n"
    )
    deps = infer_dependencies(source)
    assert "rayon" in deps
