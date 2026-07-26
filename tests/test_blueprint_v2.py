"""Tests for the Schema v2.0.0 blueprint contract validator."""

import pytest

from aero_forge.blueprint import BlueprintValidator


def _valid_v2_dict() -> dict:
    return {
        "metadata": {
            "schema_version": "2.0.0",
            "project_name": "demo",
            "domain_target": "hybrid_cpp_python",
        },
        "execution_strategy": {
            "primary_entrypoint": {
                "path": "main.py",
                "runtime": "python",
                "wrapper_generation": True,
            },
            "cli_contract": {
                "parser_type": "argparse",
                "flags": [
                    {
                        "name": "benchmark",
                        "short": "-b",
                        "type": "bool",
                        "required": False,
                        "default": False,
                        "help": "Run benchmark mode",
                        "dest_var": "benchmark_mode",
                    },
                    {
                        "name": "samples",
                        "short": "-n",
                        "type": "int",
                        "required": False,
                        "default": 1000,
                        "help": "Number of samples",
                        "dest_var": "samples",
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
                "contract_id": "ray_march",
                "target_language": "cpp",
                "binding_framework": "c_abi",
                "export_symbol": "march_rays_batch",
                "memory_model": "callee_allocates",
                "signature": {
                    "inputs": [
                        {"name": "origins", "type": "*mut f64"},
                        {"name": "dirs", "type": "*mut f64"},
                        {"name": "count", "type": "i32"},
                        {"name": "max_steps", "type": "i32"},
                        {"name": "hit_threshold", "type": "f64"},
                        {"name": "sphere_radius", "type": "f64"},
                    ],
                    "outputs": [
                        {"name": "hit_distances", "type": "*mut f64"},
                        {"name": "hit_count", "type": "i32"},
                    ],
                },
            },
        ],
        "module_graph": [
            {"name": "native_bridge", "language": "python", "depends_on": ["cpp_core"]},
        ],
        "verification_nodes": [
            {"kind": "pytest", "path": "tests/test_demo.py"},
        ],
    }


def test_valid_v2_blueprint() -> None:
    validator = BlueprintValidator(_valid_v2_dict())
    assert validator.blueprint.metadata["schema_version"] == "2.0.0"
    assert validator.validate_abi_integrity() is True
    assert validator.validate_cli_contract() is True


def test_invalid_abi_type_raises_type_error() -> None:
    data = _valid_v2_dict()
    data["abi_contracts"][0]["signature"]["inputs"].append(
        {"name": "label", "type": "std::string"}
    )
    validator = BlueprintValidator(data)
    with pytest.raises(TypeError):
        validator.validate_abi_integrity()


def test_malformed_cli_flag_raises_value_error() -> None:
    data = _valid_v2_dict()
    data["execution_strategy"]["cli_contract"]["flags"].append(
        {
            "name": "123-bad",
            "short": "-x",
            "type": "string",
            "required": False,
            "default": None,
            "help": "bad flag",
            "dest_var": "bad_var",
        }
    )
    validator = BlueprintValidator(data)
    with pytest.raises(ValueError):
        validator.validate_cli_contract()


def test_invalid_dest_var_raises_value_error() -> None:
    data = _valid_v2_dict()
    data["execution_strategy"]["cli_contract"]["flags"].append(
        {
            "name": "valid_flag",
            "short": "-v",
            "type": "string",
            "required": False,
            "default": None,
            "help": "bad dest",
            "dest_var": "not-valid",
        }
    )
    validator = BlueprintValidator(data)
    with pytest.raises(ValueError):
        validator.validate_cli_contract()


def test_v1_schema_upgrade() -> None:
    v1 = {
        "project": "legacy_project",
        "architecture": "pure_python",
        "toolchains": ["python"],
        "functions": [],
    }
    validator = BlueprintValidator(v1)
    assert validator.blueprint.metadata["schema_version"] == "2.0.0"
    assert validator.blueprint.metadata["project_name"] == "legacy_project"
    assert validator.blueprint.metadata["domain_target"] == "pure_python"
    assert validator.validate_abi_integrity() is True
    assert validator.validate_cli_contract() is True
