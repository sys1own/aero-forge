"""E2E tests for the self-extracting ``workspace.aeroc.bin`` runner."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from aero_forge._native import compile_aeroc
from aero_forge.builder.aeroc_compiler import bundle_aeroc_executable


@pytest.mark.skipif(not shutil.which("cargo"), reason="cargo not available")
def test_aeroc_runner_executes_standalone(tmp_path: Path) -> None:
    """Compile, bundle, and execute a standalone workspace.aeroc.bin binary."""
    marker = "ran.txt"
    spec = {
        "nodes": ["verify"],
        "edges": {},
        "instructions": [
            {
                "op": "UNIT_VERIFY",
                "test_bin_ref": f"sh -c 'echo ok > {marker}'",
                "args": 0,
            },
            {"op": "HALT"},
        ],
        "sources": [],
        "flags": 0,
    }

    aeroc = tmp_path / "workspace.aeroc"
    compile_aeroc(json.dumps(spec), str(aeroc))

    binary = tmp_path / "workspace.aeroc.bin"
    bundle_aeroc_executable(str(aeroc), str(binary))
    assert binary.is_file()

    workspace = tmp_path / "run"
    workspace.mkdir()
    os.chmod(binary, 0o755)
    result = subprocess.run([str(binary), str(workspace)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (workspace / marker).read_text().strip() == "ok"
