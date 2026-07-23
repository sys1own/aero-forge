"""Unit tests for the simplified sandbox manager."""

import subprocess
from pathlib import Path

import pytest

from aero_forge.sandbox.manager import Sandbox


@pytest.fixture
def temp_source(tmp_path):
    source = tmp_path / "calc.py"
    source.write_text("def add(a, b):\n    return a + b\n")
    test = tmp_path / "test_calc.py"
    test.write_text("from calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n")
    return source


def test_sandbox_copies_files(temp_source):
    sandbox = Sandbox(temp_source, "add")
    assert (sandbox.root / "calc.py").is_file()
    assert (sandbox.root / "test_calc.py").is_file()
    sandbox.cleanup()


def test_sandbox_runs_pytest_successfully(temp_source):
    with Sandbox(temp_source, "add") as sandbox:
        result = sandbox.run_tests()
        assert result["passed"]
        assert result["returncode"] == 0


def test_sandbox_reports_failure(temp_source):
    source = temp_source.parent / "calc.py"
    source.write_text("def add(a, b):\n    return a - b\n")
    with Sandbox(source, "add") as sandbox:
        result = sandbox.run_tests()
        assert not result["passed"]
        assert result["returncode"] != 0
