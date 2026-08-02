"""Tests for the EntrypointAdapterEngine CLI wrapper generator."""

import subprocess
import sys
from pathlib import Path

import pytest

from aero_forge.scaffold.entrypoint_adapter import EntrypointAdapterEngine


def _strategy(cmd_required: bool = False) -> dict:
    return {
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
                    "required": cmd_required,
                    "default": None,
                    "choices": [],
                    "help": "Command to execute",
                    "dest_var": "cmd",
                },
                {
                    "name": "precision",
                    "short": "p",
                    "type": "int",
                    "required": False,
                    "default": 256,
                    "choices": [],
                    "help": "Numeric precision",
                    "dest_var": "precision",
                },
                {
                    "name": "interactive",
                    "short": "i",
                    "type": "bool",
                    "required": False,
                    "default": False,
                    "choices": [],
                    "help": "Run in interactive mode",
                    "dest_var": "interactive",
                },
            ],
        },
        "run_spec": {
            "working_dir": ".",
            "env_vars": {},
            "timeout_seconds": 120,
        },
    }


def _write_engine(output_dir: Path) -> None:
    engine_dir = output_dir / "src" / "python"
    engine_dir.mkdir(parents=True, exist_ok=True)
    engine_dir.joinpath("__init__.py").write_text("")
    engine_dir.joinpath("engine.py").write_text(
        "def run_domain_task(args):\n"
        "    print(f'cmd={args.cmd} precision={args.precision} interactive={args.interactive}')\n"
        "    return 0\n"
    )


def test_synthesize_python_cli_wrapper(tmp_path: Path) -> None:
    _write_engine(tmp_path)
    engine = EntrypointAdapterEngine(_strategy(), str(tmp_path))
    main_path = engine.synthesize_root_entrypoint()

    assert Path(main_path).name == "main.py"
    assert Path(main_path).is_file()

    result = subprocess.run(
        [sys.executable, main_path, "--cmd", "benchmark", "--precision", "512"],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "cmd=benchmark" in result.stdout
    assert "precision=512" in result.stdout


def test_synthesize_with_short_flags(tmp_path: Path) -> None:
    _write_engine(tmp_path)
    engine = EntrypointAdapterEngine(_strategy(), str(tmp_path))
    main_path = engine.synthesize_root_entrypoint()

    result = subprocess.run(
        [sys.executable, main_path, "-c", "status", "-i"],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "cmd=status" in result.stdout
    assert "interactive=True" in result.stdout


def test_unsupported_runtime_returns_empty() -> None:
    strategy = _strategy()
    strategy["primary_entrypoint"]["runtime"] = "go"
    engine = EntrypointAdapterEngine(strategy, "/tmp")
    result = engine.synthesize_root_entrypoint()
    assert result == ""


def _strategy_with_custom_flags() -> dict:
    return {
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
                    "required": True,
                    "default": None,
                    "choices": [],
                    "help": "Sub-command to run",
                    "dest_var": "cmd",
                },
                {
                    "name": "simulations",
                    "short": "n",
                    "type": "int",
                    "required": False,
                    "default": 100000,
                    "choices": [],
                    "help": "Number of simulations",
                    "dest_var": "simulations",
                },
                {
                    "name": "spot",
                    "short": "s",
                    "type": "float",
                    "required": False,
                    "default": 100.0,
                    "choices": [],
                    "help": "Spot price",
                    "dest_var": "spot",
                },
                {
                    "name": "strike",
                    "short": "k",
                    "type": "float",
                    "required": False,
                    "default": 100.0,
                    "choices": [],
                    "help": "Strike price",
                    "dest_var": "strike",
                },
            ],
        },
        "run_spec": {
            "working_dir": ".",
            "env_vars": {},
            "timeout_seconds": 120,
        },
    }


def test_synthesize_cli_contract_with_custom_flags(tmp_path: Path) -> None:
    """Custom Monte-Carlo style CLI flags are registered in main.py."""
    engine_dir = tmp_path / "src" / "python"
    engine_dir.mkdir(parents=True, exist_ok=True)
    (engine_dir / "__init__.py").write_text("")
    (engine_dir / "engine.py").write_text(
        "def run_domain_task(args):\n"
        "    print(f'cmd={args.cmd} simulations={args.simulations} spot={args.spot} strike={args.strike}')\n"
        "    return 0\n"
    )

    engine = EntrypointAdapterEngine(_strategy_with_custom_flags(), str(tmp_path))
    main_path = engine.synthesize_root_entrypoint()

    result = subprocess.run(
        [sys.executable, main_path, "--cmd", "simulate", "--simulations", "500000", "--spot", "105.0", "--strike", "100.0"],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "cmd=simulate" in result.stdout
    assert "simulations=500000" in result.stdout
    assert "spot=105.0" in result.stdout
    assert "strike=100.0" in result.stdout
