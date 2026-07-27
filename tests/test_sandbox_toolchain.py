"""Integration tests for sandbox toolchain execution."""

import asyncio
import shutil
import subprocess
from pathlib import Path

import pytest

from aero_forge import runner
from aero_forge.bundle_repo import scaffold_native_crate


@pytest.mark.slow
@pytest.mark.skipif(not shutil.which("cargo"), reason="cargo not installed")
def test_maturin_develop_in_fresh_sandbox(tmp_path: Path) -> None:
    """Scaffold the native crate and run `maturin develop` in a clean .venv."""
    scaffold_native_crate(tmp_path, project_name="test-native")
    assert (tmp_path / "crates" / "native_core" / "Cargo.toml").is_file()

    resolved, env, _logs = asyncio.run(
        runner.resolve_command(
            "maturin develop --manifest-path crates/native_core/Cargo.toml",
            env={},
            sandbox_dir=tmp_path,
        )
    )

    proc = subprocess.run(
        resolved,
        shell=True,
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert proc.returncode == 0, f"maturin develop failed:\n{proc.stdout}\n{proc.stderr}"
    # maturin develop installs an importable native module.
    check = subprocess.run(
        [env.get("VIRTUAL_ENV", "").strip() and f"{env['VIRTUAL_ENV']}/bin/python" or shutil.which('python3'), "-c", "import aero_forge_native"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert check.returncode == 0, f"Module import failed: {check.stderr}"
