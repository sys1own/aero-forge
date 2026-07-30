"""Tests for the healing pipeline: overlay persistence, pip cache busting, and signature-mismatch repair."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

from aero_forge.environment.env_manager import (
    install_workspace_editable,
    run_pip_in_workspace,
)
from aero_forge.healing.context_builder import ContextBuilder
from aero_forge.healing.evaluator import HealingStrategy, LogEvaluator
from aero_forge.healing.router import try_auto_fix
from aero_forge.orchestrator.error_classifier import (
    extract_signature_mismatch_symbol,
    get_signature_mismatch_expected_given,
    is_signature_mismatch,
)
from aero_forge.overlay.apply import apply_patch_to_disk, persist_text_to_disk


def test_persist_text_to_disk_is_immediately_visible_to_subprocess() -> None:
    """Writing through ``persist_text_to_disk`` must flush the file to disk."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "message.txt"
        persist_text_to_disk(path, "hello")
        # Use a fresh Python interpreter to read the file to be certain it is not
        # just cached in this process.
        script = f"print(open({str(path)!r}).read())"
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert result.stdout.strip() == "hello"


def test_apply_patch_to_disk_persists_and_runs() -> None:
    """A unified-diff patch applied to a workspace file must be visible to the runner."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        module = root / "demo.py"
        module.write_text("def compute():\n    return 1\n", encoding="utf-8")
        patch = """--- demo.py
+++ demo.py
@@ -1,2 +1,2 @@
 def compute():
-    return 1
+    return 2
"""
        apply_patch_to_disk(module, patch)
        result = subprocess.run(
            [sys.executable, "-c", f"import sys; sys.path.insert(0, {str(root)!r}); import demo; print(demo.compute())"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert result.stdout.strip() == "2"


def test_run_pip_in_workspace_disables_cache_and_sets_cwd(monkeypatch: Any) -> None:
    """``run_pip_in_workspace`` must set cwd and disable pip caches by default."""
    calls: list[Dict[str, Any]] = []
    original_run = subprocess.run

    def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        calls.append({"cmd": cmd, "kwargs": kwargs})
        class DummyProc:
            returncode = 0
            stdout = "ok"
            stderr = ""
        return DummyProc()

    monkeypatch.setattr(subprocess, "run", fake_run)

    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        run_pip_in_workspace(["install", "x"], workspace, sys.executable)

    assert len(calls) == 1
    call = calls[0]
    assert call["kwargs"]["cwd"] == str(workspace)
    assert call["kwargs"]["env"]["PIP_NO_CACHE_DIR"] == "1"


def test_install_workspace_editable_adds_cache_bust_args(monkeypatch: Any) -> None:
    """Editable install must include ``--no-cache-dir --force-reinstall --no-deps``."""
    calls: list[Dict[str, Any]] = []
    original_run = subprocess.run

    def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        calls.append({"cmd": cmd, "kwargs": kwargs})
        class DummyProc:
            returncode = 0
            stdout = "ok"
            stderr = ""
        return DummyProc()

    monkeypatch.setattr(subprocess, "run", fake_run)

    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        (workspace / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")
        install_workspace_editable(workspace, sys.executable)

    cmd = calls[0]["cmd"]
    assert "-e" in cmd
    assert "." in cmd
    assert "--no-cache-dir" in cmd
    assert "--force-reinstall" in cmd
    assert "--no-deps" in cmd


def test_signature_mismatch_regexes() -> None:
    """The error classifier should recognize Python positional-argument arity errors."""
    msg1 = "TypeError: compute() takes 1 positional argument but 2 were given"
    assert is_signature_mismatch(msg1)
    assert extract_signature_mismatch_symbol(msg1) == "compute"
    assert get_signature_mismatch_expected_given(msg1) == (1, 2)

    msg2 = "TypeError: MyClass.process() takes from 1 to 3 positional arguments but 5 were given"
    assert is_signature_mismatch(msg2)
    assert extract_signature_mismatch_symbol(msg2) == "process"
    assert get_signature_mismatch_expected_given(msg2) == (3, 5)

    msg3 = "TypeError: func() missing 1 required positional argument: 'x'"
    assert is_signature_mismatch(msg3)
    assert extract_signature_mismatch_symbol(msg3) == "func"

    assert not is_signature_mismatch("RuntimeError: boom")


def test_try_auto_fix_adds_variadic_arguments() -> None:
    """A signature mismatch should be repaired by inserting ``*args, **kwargs``."""
    code = "def add(a, b):\n    return a + b\n"
    error = "TypeError: add() takes 2 positional arguments but 3 were given"
    fixed = try_auto_fix(error, code)
    assert fixed is not None
    assert "def add(a, b, *args, **kwargs):" in fixed
    # Make sure the body is preserved and the result is still valid Python.
    assert "return a + b" in fixed
    compile(fixed, "<test>", "exec")


def test_try_auto_fix_on_no_args_function() -> None:
    """Variadic repair must also work for functions that currently take no arguments."""
    code = "def greet():\n    return 'hi'\n"
    error = "TypeError: greet() takes 0 positional arguments but 1 was given"
    fixed = try_auto_fix(error, code)
    assert fixed is not None
    assert "def greet(*args, **kwargs):" in fixed


def test_log_evaluator_returns_signature_mismatch_strategy() -> None:
    """The evaluator should surface signature mismatches as AST-healable."""
    evaluator = LogEvaluator()
    log = (
        "Traceback (most recent call last):\n"
        '  File "/tmp/ws/run.py", line 5, in <module>\n'
        "    result = compute(1, 2, 3)\n"
        "TypeError: compute() takes 2 positional arguments but 3 were given\n"
    )
    result = evaluator.evaluate_log("python run.py", 1, log)
    assert result["error_type"] == "signature_mismatch"
    assert result["ast_healable"] is True
    assert result["llm_healable"] is True
    assert result["healable"] is True


def test_context_builder_extracts_signature_context() -> None:
    """For a signature mismatch, the context builder should locate the definition."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "compute.py").write_text("def compute(a, b):\n    return a + b\n", encoding="utf-8")
        (root / "run.py").write_text(
            "from compute import compute\nprint(compute(1, 2, 3))\n",
            encoding="utf-8",
        )
        builder = ContextBuilder(root)
        log = (
            "Traceback (most recent call last):\n"
            f'  File "{root / "run.py"}", line 2, in <module>\n'
            "    print(compute(1, 2, 3))\n"
            "TypeError: compute() takes 2 positional arguments but 3 were given\n"
        )
        ctx = builder.build_failure_context(
            command="python run.py",
            exit_code=1,
            log_text=log,
            diagnosis={"error_type": "signature_mismatch"},
        )
        sig_ctx = ctx.get("signature_context")
        assert sig_ctx is not None
        assert sig_ctx["symbol"] == "compute"
        assert sig_ctx["definition"]["file"] == "compute.py"
        assert sig_ctx["definition"]["line"] == 1
        assert sig_ctx["caller"]["file"] == "run.py"


def test_context_builder_prompt_contains_signature_guidance() -> None:
    """The LLM prompt should include explicit instructions for arity repairs."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "compute.py").write_text("def compute(a, b):\n    return a + b\n", encoding="utf-8")
        builder = ContextBuilder(root)
        log = (
            f'  File "{root / "run.py"}", line 1, in <module>\n'
            "TypeError: compute() takes 2 positional arguments but 3 were given\n"
        )
        prompt = builder.build_prompt(
            command="python run.py",
            exit_code=1,
            log_text=log,
            diagnosis={"error_type": "signature_mismatch"},
        )
        assert "SIGNATURE MISMATCH GUIDANCE" in prompt
        assert "*args, **kwargs" in prompt
        assert "Mismatched symbol: compute" in prompt
