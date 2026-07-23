"""Simplified sandbox manager for isolated test execution."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional


class Sandbox:
    """Isolate a source file and its test file in a temporary directory."""

    def __init__(
        self,
        source: Path,
        function_name: str,
        test_file: Optional[Path] = None,
        extra_files: Optional[list] = None,
        project_root: Optional[Path] = None,
    ):
        self.source = Path(source)
        self.function_name = function_name
        self.test_file = (
            Path(test_file)
            if test_file
            else self.source.parent / f"test_{self.source.stem}.py"
        )
        self.extra_files = [Path(f) for f in (extra_files or [])]
        self.project_root = project_root
        self._tmpdir = tempfile.TemporaryDirectory(prefix="aero-forge-sandbox-")
        self.root = Path(self._tmpdir.name)
        self._populate()

    @property
    def source_in_sandbox(self) -> Path:
        return self.root / self.source.name

    @property
    def test_in_sandbox(self) -> Path:
        return self.root / self.test_file.name

    def _populate(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        shutil.copy(self.source, self.source_in_sandbox)
        if self.test_file and self.test_file.is_file():
            shutil.copy(self.test_file, self.test_in_sandbox)
        for extra in self.extra_files:
            if extra.is_file():
                shutil.copy(extra, self.root / extra.name)

    def run_tests(self, timeout: int = 120) -> Dict[str, Any]:
        """Run pytest or unittest on the sandboxed test file."""
        if not self.test_in_sandbox.is_file():
            return {
                "passed": True,
                "returncode": 0,
                "logs": "No test file found; skipping.",
            }

        cmd = [sys.executable, "-m", "pytest", str(self.test_in_sandbox), "-v"]
        try:
            result = subprocess.run(
                cmd,
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "passed": False,
                "returncode": -1,
                "logs": f"Test execution timed out after {timeout}s.\n{exc}",
            }
        except FileNotFoundError:
            # pytest not installed; fall back to unittest discovery
            return self._run_unittest(timeout)

        return {
            "passed": result.returncode == 0,
            "returncode": result.returncode,
            "logs": result.stdout + result.stderr,
        }

    def _run_unittest(self, timeout: int) -> Dict[str, Any]:
        cmd = [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(self.root),
            "-p",
            f"test_{self.source.stem}.py",
            "-v",
        ]
        result = subprocess.run(
            cmd,
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "passed": result.returncode == 0,
            "returncode": result.returncode,
            "logs": result.stdout + result.stderr,
        }

    def cleanup(self) -> None:
        self._tmpdir.cleanup()

    def __enter__(self) -> "Sandbox":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.cleanup()
