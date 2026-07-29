"""Tests for the Layer 0 intent compiler and v2 blueprint integration."""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml

from aero_forge.builder.intent_compiler import IntentCompiler, IntentCompilerError
from aero_forge.orchestrator.orchestrator import DeterministicVerificationRunner
from aero_forge.scaffold.entrypoint_adapter import EntrypointAdapterEngine


def _valid_polyglot_json() -> Dict[str, Any]:
    return {
        "metadata": {
            "schema_version": "2.0.0",
            "project_name": "quantum_sim",
            "domain_target": "tri_polyglot_rust_cpp_python",
        },
        "execution_strategy": {
            "primary_entrypoint": {
                "path": "main.py",
                "runtime": "python3",
                "wrapper_generation": True,
            },
            "cli_contract": {
                "parser_type": "argparse",
                "flags": [
                    {
                        "name": "cmd",
                        "short": "c",
                        "type": "string",
                        "required": False,
                        "default": "run",
                        "choices": [],
                        "help": "Command to run",
                        "dest_var": "cmd",
                    },
                    {
                        "name": "precision",
                        "short": "p",
                        "type": "int",
                        "required": False,
                        "default": 256,
                        "choices": [],
                        "help": "Simulation precision",
                        "dest_var": "precision",
                    },
                ],
            },
            "run_spec": {
                "working_dir": ".",
                "env_vars": {},
                "timeout_seconds": 120,
            },
        },
        "abi_contracts": [
            {
                "contract_id": "unitary_step",
                "target_language": "cpp",
                "binding_framework": "c_abi",
                "export_symbol": "apply_unitary_step",
                "c_symbol_alias": "",
                "header_path": "cpp_core/quantum.h",
                "memory_model": "caller_allocates",
                "signature": {
                    "inputs": [
                        {"name": "amplitudes", "type": "double*"},
                        {"name": "count", "type": "i32"},
                    ],
                    "outputs": [{"name": "result", "type": "i32"}],
                },
            }
        ],
        "module_graph": [
            {"path": "main.py", "lang": "python", "purpose": "CLI entrypoint"},
            {"path": "src/python/engine.py", "lang": "python", "purpose": "Python orchestrator"},
            {"path": "rust_core/src/lib.rs", "lang": "rust", "purpose": "PyO3 state manager"},
            {"path": "cpp_core/quantum.cpp", "lang": "cpp", "purpose": "C-ABI math core"},
        ],
        "verification_nodes": [
            {
                "test_id": "cli_parses",
                "execution_cmd": [sys.executable, "main.py", "--cmd", "benchmark", "--precision", "512"],
                "expected_exit_code": 0,
                "stdout_match_patterns": [r"status=ok"],
                "stderr_prohibited_patterns": [r"Traceback"],
                "numerical_assertions": [
                    {
                        "target_metric": "unitary_determinant",
                        "expected_value": 1.0,
                        "absolute_tolerance": 1e-9,
                    }
                ],
            }
        ],
    }


def _make_mock_client(responses: List[str]):
    class MockClient:
        def __init__(self, responses: List[str]):
            self.responses = list(responses)
            self.calls: List[Any] = []

        def generate(self, messages, **kwargs):
            self.calls.append((messages, kwargs))
            return self.responses.pop(0) if self.responses else ""

    return MockClient(responses)


def test_compile_prompt_writes_valid_blueprint_aero(tmp_path: Path) -> None:
    client = _make_mock_client([json.dumps(_valid_polyglot_json())])
    compiler = IntentCompiler(llm_client=client)
    blueprint = compiler.compile_prompt(
        "Build a tri-language quantum simulator with Python CLI, Rust state, and C++ math",
        output_dir=str(tmp_path),
        project_name="quantum_sim",
    )

    assert blueprint.project == "quantum_sim"
    assert blueprint.architecture == "tri_polyglot_rust_cpp_python"
    assert blueprint.execution_strategy is not None
    assert len(blueprint.abi_contracts) == 1
    assert len(blueprint.verification_nodes) == 1

    blueprint_path = tmp_path / "blueprint.aero"
    assert blueprint_path.is_file()
    data = yaml.safe_load(blueprint_path.read_text(encoding="utf-8"))
    assert data["metadata"]["schema_version"] == "2.0.0"
    assert data["execution_strategy"]["primary_entrypoint"]["path"] == "main.py"


def _bad_json() -> str:
    # project_name is an int, which violates Dict[str, str].
    return json.dumps({"metadata": {"schema_version": "2.0.0", "project_name": 123}})


def test_compile_prompt_retries_on_schema_error(tmp_path: Path) -> None:
    good = json.dumps(_valid_polyglot_json())
    client = _make_mock_client([_bad_json(), good])
    compiler = IntentCompiler(llm_client=client, max_schema_retries=2)
    blueprint = compiler.compile_prompt("Build a quantum simulator", output_dir=str(tmp_path))
    assert blueprint.architecture == "tri_polyglot_rust_cpp_python"
    assert len(client.calls) == 2


def test_compile_prompt_raises_after_exhausted_retries(tmp_path: Path) -> None:
    client = _make_mock_client([_bad_json(), _bad_json(), _bad_json()])
    compiler = IntentCompiler(llm_client=client, max_schema_retries=2)
    with pytest.raises(IntentCompilerError):
        compiler.compile_prompt("Build a quantum simulator", output_dir=str(tmp_path))


def test_end_to_end_entrypoint_adapter_and_verification_runner(tmp_path: Path) -> None:
    client = _make_mock_client([json.dumps(_valid_polyglot_json())])
    compiler = IntentCompiler(llm_client=client)
    blueprint = compiler.compile_prompt(
        "Build a tri-language quantum simulator",
        output_dir=str(tmp_path),
    )

    # Synthesize main.py using the v2 execution strategy.
    engine_dir = tmp_path / "src" / "python"
    engine_dir.mkdir(parents=True, exist_ok=True)
    engine_dir.joinpath("__init__.py").write_text("")
    engine_dir.joinpath("engine.py").write_text(
        "def run_domain_task(args):\n"
        "    print(f'status=ok cmd={args.cmd} precision={args.precision}')\n"
        "    print('unitary_determinant=1.0000000002')\n"
        "    return 0\n"
    )

    strategy = blueprint.execution_strategy.model_dump()
    EntrypointAdapterEngine(strategy, str(tmp_path)).synthesize_root_entrypoint()

    main_path = tmp_path / "main.py"
    assert main_path.is_file()

    # Run the deterministic verification runner against the generated CLI.
    runner = DeterministicVerificationRunner(str(tmp_path), blueprint.verification_nodes)
    assert runner.run_all_verifications() is True


def test_compile_prompt_extracts_cargo_dependencies_and_module_graph(tmp_path: Path) -> None:
    """Prompt compilation extracts Cargo dependencies and Rust submodule paths."""
    prompt_json = {
        "metadata": {"schema_version": "2.0.0", "project_name": "array_bridge"},
        "execution_strategy": {
            "primary_entrypoint": {"path": "src/lib.rs", "runtime": "cargo", "wrapper_generation": True},
            "cli_contract": {"parser_type": "argparse", "flags": []},
            "run_spec": {"working_dir": ".", "timeout_seconds": 120},
        },
        "abi_contracts": [
            {
                "contract_id": "matrix_multiply",
                "target_language": "rust",
                "binding_framework": "pyo3",
                "export_symbol": "matrix_multiply",
                "c_symbol_alias": "",
                "header_path": "",
                "memory_model": "shared_pyo3",
                "signature": {
                    "inputs": [
                        {"name": "a", "type": "&PyArray2<f64>"},
                        {"name": "b", "type": "&PyArray2<f64>"},
                    ],
                    "outputs": [{"name": "result", "type": "&PyArray2<f64>"}],
                },
            }
        ],
        "module_graph": [
            {"path": "src/lib.rs", "lang": "rust", "purpose": "PyO3 module entrypoint"},
            {"path": "src/ops.rs", "lang": "rust", "purpose": "matrix ops submodule"},
            {"path": "tests/test_ops.rs", "lang": "rust", "purpose": "unit tests"},
        ],
        "verification_nodes": [],
        "cargo_dependencies": {"pyo3": "0.20.3", "numpy": "0.21", "rayon": "1.10"},
    }
    client = _make_mock_client([json.dumps(prompt_json)])
    compiler = IntentCompiler(llm_client=client)
    blueprint = compiler.compile_prompt(
        "Build a PyO3 Rust extension with numpy arrays, rayon parallelism, src/ops.rs submodule and tests/test_ops.rs",
        output_dir=str(tmp_path),
    )

    assert blueprint.project == "array_bridge"
    assert blueprint.architecture == "hybrid_rust_python"
    paths = {m.path for m in blueprint.manifest}
    assert "src/lib.rs" in paths
    assert "src/ops.rs" in paths
    assert "tests/test_ops.rs" in paths
    assert blueprint.cargo_dependencies == {"pyo3": "0.20.3", "numpy": "0.21", "rayon": "1.10"}
