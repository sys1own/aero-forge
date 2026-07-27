"""Dual-engine accelerator: attempts to load the Rust/PyO3 extension,
falls back to pure-Python reference implementations on import failure or when
``AERO_DISABLE_NATIVE=1`` is set.
"""

import os
from typing import Optional

from aero_forge.accelerator.contracts import GraphEngineABC, HasherABC


def is_native() -> bool:
    """Return True when the Rust/PyO3 native extension is active."""
    return _NATIVE is not None


def _try_load_native() -> Optional[object]:
    if os.environ.get("AERO_DISABLE_NATIVE") == "1":
        return None
    try:
        from aero_forge._native import (  # type: ignore[import-not-found]
            GraphEngine,
            Hasher,
            hash_bytes,
            hash_file,
        )

        return {
            "Hasher": Hasher,
            "GraphEngine": GraphEngine,
            "hash_bytes": hash_bytes,
            "hash_file": hash_file,
        }
    except Exception:
        return None


_NATIVE = _try_load_native()

if _NATIVE is not None:
    Hasher: type = _NATIVE["Hasher"]
    GraphEngine: type = _NATIVE["GraphEngine"]
    hash_bytes = _NATIVE["hash_bytes"]
    hash_file = _NATIVE["hash_file"]
else:
    from aero_forge._fallback.hasher import Hasher, hash_bytes, hash_file
    from aero_forge._fallback.graph_engine import GraphEngine


__all__ = ["Hasher", "GraphEngine", "hash_bytes", "hash_file", "is_native"]
