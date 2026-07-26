"""Tests for DeterministicVerificationRunner."""

import subprocess
import sys
from pathlib import Path

import pytest

from aero_forge.orchestrator.orchestrator import DeterministicVerificationRunner


def _write_dummy_script(project_root: Path) -> None:
    script = project_root / "dummy_app.py"
    script.write_text(
        "import sys\n"
        "print('unitary_determinant=1.0000000005')\n"
        "print('status=ok')\n"
        "print('ready')\n"
        "sys.stderr.write('')\n"
        "sys.exit(0)\n"
    )


def test_stdout_patterns_and_exit_code(tmp_path: Path) -> None:
    _write_dummy_script(tmp_path)
    nodes = [
        {
            "test_id": "basic",
            "execution_cmd": [sys.executable, str(tmp_path / "dummy_app.py")],
            "expected_exit_code": 0,
            "stdout_match_patterns": [r"status=ok", r"ready"],
        }
    ]
    runner = DeterministicVerificationRunner(str(tmp_path), nodes)
    assert runner.run_all_verifications() is True


def test_numerical_tolerance_pass(tmp_path: Path) -> None:
    _write_dummy_script(tmp_path)
    nodes = [
        {
            "test_id": "numeric",
            "execution_cmd": [sys.executable, str(tmp_path / "dummy_app.py")],
            "expected_exit_code": 0,
            "stdout_match_patterns": [],
            "numerical_assertions": [
                {
                    "target_metric": "unitary_determinant",
                    "expected_value": 1.0,
                    "absolute_tolerance": 1e-9,
                }
            ],
        }
    ]
    runner = DeterministicVerificationRunner(str(tmp_path), nodes)
    assert runner.run_all_verifications() is True


def test_numerical_tolerance_fail(tmp_path: Path) -> None:
    _write_dummy_script(tmp_path)
    nodes = [
        {
            "test_id": "numeric_fail",
            "execution_cmd": [sys.executable, str(tmp_path / "dummy_app.py")],
            "expected_exit_code": 0,
            "numerical_assertions": [
                {
                    "target_metric": "unitary_determinant",
                    "expected_value": 2.0,
                    "absolute_tolerance": 1e-9,
                }
            ],
        }
    ]
    runner = DeterministicVerificationRunner(str(tmp_path), nodes)
    assert runner.run_all_verifications() is False


def test_prohibited_stderr_pattern(tmp_path: Path) -> None:
    script = tmp_path / "bad_app.py"
    script.write_text(
        "import sys\n"
        "print('running')\n"
        "sys.stderr.write('Traceback (most recent call last):\\n')\n"
        "sys.exit(0)\n"
    )
    nodes = [
        {
            "test_id": "stderr",
            "execution_cmd": [sys.executable, str(script)],
            "expected_exit_code": 0,
            "stdout_match_patterns": [r"running"],
            "stderr_prohibited_patterns": [r"Traceback"],
        }
    ]
    runner = DeterministicVerificationRunner(str(tmp_path), nodes)
    assert runner.run_all_verifications() is False


def test_missing_stdout_pattern_fails(tmp_path: Path) -> None:
    _write_dummy_script(tmp_path)
    nodes = [
        {
            "test_id": "missing",
            "execution_cmd": [sys.executable, str(tmp_path / "dummy_app.py")],
            "expected_exit_code": 0,
            "stdout_match_patterns": [r"not_in_output"],
        }
    ]
    runner = DeterministicVerificationRunner(str(tmp_path), nodes)
    assert runner.run_all_verifications() is False


def test_unexpected_exit_code_fails(tmp_path: Path) -> None:
    script = tmp_path / "exit1.py"
    script.write_text("import sys; sys.exit(1)\n")
    nodes = [
        {
            "test_id": "exit",
            "execution_cmd": [sys.executable, str(script)],
            "expected_exit_code": 0,
        }
    ]
    runner = DeterministicVerificationRunner(str(tmp_path), nodes)
    assert runner.run_all_verifications() is False
