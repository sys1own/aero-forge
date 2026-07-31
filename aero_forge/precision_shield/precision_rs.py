"""Write `src/precision.rs` compatibility shims for `rug::Float` / `rug::Complex`."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from .rust_shield import EXTENSION_TRAITS, _TRAIT_SENTINEL


def ensure_precision_traits(project_root: Path) -> Tuple[Path, bool]:
    """Ensure a `src/precision.rs` file exists containing the AeroNegMutExt shims.

    Returns the path to the file and a flag indicating whether traits were newly
    written (True) or already present (False).
    """
    precision_rs = Path(project_root) / "src" / "precision.rs"
    existing = precision_rs.read_text(encoding="utf-8") if precision_rs.is_file() else ""
    if _TRAIT_SENTINEL in existing:
        return precision_rs, False
    precision_rs.parent.mkdir(parents=True, exist_ok=True)
    precision_rs.write_text(existing + ("\n\n" if existing else "") + EXTENSION_TRAITS, encoding="utf-8")
    return precision_rs, True
