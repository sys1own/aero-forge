"""Native PyO3 accelerator extension package.

The compiled extension is loaded from ``aero_forge_native``.
"""

try:
    from .aero_forge_native import (  # type: ignore[import-not-found]
        GraphEngine,
        Hasher,
        compile_aeroc,
        hash_bytes,
        hash_file,
        run_aeroc,
        unpack_aeroc,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The aero_forge_native extension is not compiled. "
        "Run 'cargo build --release' in aero_forge/_native."
    ) from exc

__all__ = ["Hasher", "GraphEngine", "compile_aeroc", "run_aeroc", "unpack_aeroc", "hash_bytes", "hash_file"]
