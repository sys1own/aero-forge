"""Tests for cross-compilation to additional Rust targets."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from aero_forge.blueprint import parse_blueprint
from aero_forge.build_runner import BuildRunner


def _target_installed(target: str) -> bool:
    try:
        result = subprocess.run(
            ["rustup", "target", "list", "--installed"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return target in result.stdout
    except (OSError, subprocess.TimeoutExpired):
        return False


def _host_target() -> str:
    try:
        result = subprocess.run(
            ["rustc", "-vV"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        for line in result.stdout.splitlines():
            if line.startswith("host:"):
                return line.split(":", 1)[1].strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return ""


@pytest.mark.skipif(
    not shutil.which("cargo") or not shutil.which("rustc"),
    reason="Rust toolchain not installed",
)
def test_build_runner_cross_compile_same_target(tmp_path):
    """Cross-compiling to the host target should still run tests and produce a .so."""
    source = tmp_path / "calc.py"
    test = tmp_path / "test_calc.py"
    source.write_text("def square(n):\n    return n * n\n")
    test.write_text(
        "from calc import square\ndef test_square():\n    assert square(4) == 16\n"
    )
    blueprint_path = tmp_path / "blueprint.aero"
    blueprint_path.write_text(
        "project: cross_test\n"
        "functions:\n"
        "  - file: calc.py\n"
        "    name: square\n"
        "    tests: [test_calc.py]\n"
        "llm:\n"
        "  provider: none\n"
        "output_dir: ./dist\n"
    )

    target = _host_target() or "x86_64-unknown-linux-gnu"
    if not _target_installed(target):
        pytest.skip(f"Target {target} not installed")

    bp = parse_blueprint(blueprint_path)
    runner = BuildRunner(bp, max_workers=1, target=target, cache_enabled=False)
    result = runner.build()

    assert result["success"] is True
    assert result["passed"] == 1
    assert any((tmp_path / "dist").glob("*.so"))


@pytest.mark.skipif(
    not shutil.which("cargo") or not shutil.which("rustc"),
    reason="Rust toolchain not installed",
)
def test_build_runner_cross_compile_windows(tmp_path):
    """Cross-compiling to Windows should produce a .dll without running host tests."""
    source = tmp_path / "calc.py"
    source.write_text("def square(n):\n    return n * n\n")
    blueprint_path = tmp_path / "blueprint.aero"
    blueprint_path.write_text(
        "project: cross_win_test\n"
        "functions:\n"
        "  - file: calc.py\n"
        "    name: square\n"
        "llm:\n"
        "  provider: none\n"
        "output_dir: ./dist\n"
    )

    target = "x86_64-pc-windows-gnu"
    if not _target_installed(target):
        pytest.skip(f"Target {target} not installed")
    if not shutil.which("x86_64-w64-mingw32-gcc"):
        pytest.skip("MinGW cross linker not installed")

    bp = parse_blueprint(blueprint_path)
    runner = BuildRunner(bp, max_workers=1, target=target, cache_enabled=False)
    result = runner.build()

    assert result["success"] is True
    assert result["passed"] == 1
    dll = next((tmp_path / "dist").glob("*.dll"), None)
    assert dll is not None
