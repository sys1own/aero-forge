"""Unit tests for the simplified sandbox manager."""

import subprocess
import sys
from pathlib import Path

import pytest

from aero_forge.errors import SemanticRegressionError
from aero_forge.sandbox.manager import Sandbox, TraceVerifier


@pytest.fixture
def temp_source(tmp_path):
    source = tmp_path / "calc.py"
    source.write_text("def add(a, b):\n    return a + b\n")
    test = tmp_path / "test_calc.py"
    test.write_text(
        "from calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    )
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


def _write_script(tmp_path: Path, name: str, body: str) -> Path:
    script = tmp_path / name
    script.write_text(body, encoding="utf-8")
    return script


def test_trace_verifier_passes_for_matching_output(tmp_path: Path) -> None:
    """Identical reference and target executions should verify cleanly."""
    ref = _write_script(tmp_path, "ref.py", "print('hello')\n")
    tgt = _write_script(tmp_path, "tgt.py", "print('hello')\n")

    verifier = TraceVerifier()
    result = verifier.verify([sys.executable, str(ref)], [sys.executable, str(tgt)])
    assert result["verification_passed"] is True
    assert result["semantic_delta"] == 0


def test_trace_verifier_fails_on_stdout_mismatch(tmp_path: Path) -> None:
    """A difference in stdout raises SemanticRegressionError with a diff report."""
    ref = _write_script(tmp_path, "ref.py", "print('hello')\n")
    tgt = _write_script(tmp_path, "tgt.py", "print('world')\n")

    verifier = TraceVerifier()
    with pytest.raises(SemanticRegressionError) as exc_info:
        verifier.verify([sys.executable, str(ref)], [sys.executable, str(tgt)])

    assert "stdout" in exc_info.value.report
    assert exc_info.value.delta == 1


def test_trace_verifier_fails_on_exit_code_mismatch(tmp_path: Path) -> None:
    """A non-zero target exit code raises SemanticRegressionError."""
    ref = _write_script(tmp_path, "ref.py", "print('ok')\n")
    tgt = _write_script(tmp_path, "tgt.py", "import sys; sys.exit(1)\n")

    verifier = TraceVerifier()
    with pytest.raises(SemanticRegressionError) as exc_info:
        verifier.verify([sys.executable, str(ref)], [sys.executable, str(tgt)])

    assert "returncode" in exc_info.value.report
    assert exc_info.value.delta >= 1


def test_trace_verifier_captures_stdin(tmp_path: Path) -> None:
    """The verifier should pass stdin to both reference and target."""
    ref = _write_script(tmp_path, "ref.py", "import sys; print(sys.stdin.read().strip())")
    tgt = _write_script(tmp_path, "tgt.py", "import sys; print(sys.stdin.read().strip())")

    verifier = TraceVerifier()
    result = verifier.verify(
        [sys.executable, str(ref)],
        [sys.executable, str(tgt)],
        input_text="hello",
    )
    assert result["verification_passed"] is True
    assert result["reference"].stdout == result["target"].stdout == "hello\n"
