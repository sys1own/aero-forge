"""Simplified sandbox manager for isolated test execution."""

from __future__ import annotations

import io
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional


class Sandbox:
    """Isolate a source file and its test files in a temporary directory."""

    def __init__(
        self,
        source: Path,
        function_name: str,
        test_file: Optional[Path] = None,
        test_paths: Optional[List[Path]] = None,
        extra_files: Optional[List[Path]] = None,
        project_root: Optional[Path] = None,
        root: Optional[Path] = None,
    ):
        self.source = Path(source)
        self.function_name = function_name

        if test_paths:
            self.test_paths = [Path(p) for p in test_paths if p]
        elif test_file:
            self.test_paths = [Path(test_file)]
        else:
            self.test_paths = [self.source.parent / f"test_{self.source.stem}.py"]

        self.test_file = self.test_paths[0]
        self.extra_files = [Path(f) for f in (extra_files or [])]
        self.project_root = project_root
        if root is None:
            self._tmpdir = tempfile.TemporaryDirectory(prefix="aero-forge-sandbox-")
            self.root = Path(self._tmpdir.name)
            self._own_root = True
        else:
            self._tmpdir = None
            self.root = Path(root)
            self.root.mkdir(parents=True, exist_ok=True)
            self._own_root = False
        self._populate()

    @property
    def source_in_sandbox(self) -> Path:
        return self.root / self.source.name

    @property
    def test_in_sandbox(self) -> Path:
        return self.root / self.test_file.name

    def _populate(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if self.source.is_file() and self.source.resolve() != self.source_in_sandbox.resolve():
            shutil.copy(self.source, self.source_in_sandbox)
        for test_path in self.test_paths:
            if test_path and test_path.is_file():
                shutil.copy(test_path, self.root / test_path.name)
        for extra in self.extra_files:
            if extra.is_file():
                shutil.copy(extra, self.root / extra.name)

    def run_tests(self, timeout: int = 120) -> Dict[str, Any]:
        """Run pytest or unittest on the sandboxed test files."""
        present_tests = [p for p in self.test_paths if p.is_file()]
        if not present_tests:
            return {
                "passed": True,
                "returncode": 0,
                "logs": "No test file found; skipping.",
            }

        cmd = [sys.executable, "-m", "pytest", str(self.root), "-v"]
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
            "test_*.py",
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
        if self._tmpdir is not None:
            self._tmpdir.cleanup()

    def __enter__(self) -> "Sandbox":
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._own_root:
            self.cleanup()


class SandboxManager:
    """Manage ephemeral, UUID-isolated sandbox directories for web requests."""

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self.base_dir = (
            Path(base_dir)
            if base_dir
            else Path(tempfile.gettempdir()) / "aero-forge-sandboxes"
        )
        self._sessions: Dict[str, Sandbox] = {}

    def _session_dir(self, session_id: str) -> Path:
        return (self.base_dir / session_id).resolve()

    def create_session_sandbox(self, session_id: str) -> Path:
        """Create and return a sandbox directory for ``session_id``."""
        session_dir = self._session_dir(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        source = session_dir / "source.py"
        if not source.is_file():
            source.write_text("# placeholder\n", encoding="utf-8")
        sandbox = Sandbox(
            source=source,
            function_name="main",
            root=session_dir,
        )
        self._sessions[session_id] = sandbox
        return session_dir

    def get_session_sandbox(self, session_id: str) -> Sandbox:
        """Return an existing ``Sandbox`` for ``session_id``."""
        if session_id not in self._sessions:
            self.create_session_sandbox(session_id)
        return self._sessions[session_id]

    def clean_session_sandbox(self, session_id: str) -> None:
        """Delete the sandbox directory for ``session_id``."""
        session_dir = self._session_dir(session_id)
        if session_dir.is_dir():
            shutil.rmtree(session_dir, ignore_errors=True)
        self._sessions.pop(session_id, None)

    def archive_session_sandbox(self, session_id: str) -> bytes:
        """Return a zip archive of the ``session_id`` sandbox as bytes."""
        session_dir = self._session_dir(session_id)
        if not session_dir.is_dir():
            raise ValueError(f"Sandbox for session '{session_id}' does not exist")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in session_dir.rglob("*"):
                if path.is_file():
                    zf.write(path, path.relative_to(session_dir))
        return buf.getvalue()
