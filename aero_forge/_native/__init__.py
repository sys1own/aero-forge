"""Native PyO3 accelerator extension package.

The compiled extension is loaded from ``aero_forge_native``.
"""

try:
    from .aero_forge_native import (  # type: ignore[import-not-found]
        GraphEngine,
        Hasher,
        hash_bytes,
        hash_file,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The aero_forge_native extension is not compiled. "
        "Run 'cargo build --release' in aero_forge/_native."
    ) from exc

__all__ = ["Hasher", "GraphEngine", "hash_bytes", "hash_file"]
