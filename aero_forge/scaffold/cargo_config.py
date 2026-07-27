"""Hardware-optimized Cargo configuration writer for exported/scaffolded projects."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


CARGO_CONFIG_TOML = """\
[build]
rustflags = ["-C", "target-cpu=native"]

[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
panic = "abort"
"""


def write_cargo_config(
    workspace_dir: Path,
    config_text: Optional[str] = None,
) -> Path:
    """Write ``.cargo/config.toml`` with release-profile optimizations.

    If ``config_text`` is supplied it is written verbatim; otherwise the
    default hardware-optimized configuration is used.
    """
    workspace_dir = Path(workspace_dir).resolve()
    cargo_dir = workspace_dir / ".cargo"
    cargo_dir.mkdir(parents=True, exist_ok=True)
    config_path = cargo_dir / "config.toml"
    config_path.write_text(config_text or CARGO_CONFIG_TOML, encoding="utf-8")
    return config_path
