"""Native PyO3 accelerator extension package.

The compiled extension is loaded from ``aero_forge_native``.
"""

try:
    from .aero_forge_native import (  # type: ignore[import-not-found]
        GraphEngine,
        Hasher,
        HinEngine,
        compile_aeroc,
        enforce_repair_isolation_py,
        evaluate_hin_energy,
        hash_bytes,
        hash_file,
        reduce_hin_uast,
        repair_uast_expression,
        run_aeroc,
        unpack_aeroc,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The aero_forge_native extension is not compiled. "
        "Run 'cargo build --release' in aero_forge/_native."
    ) from exc

__all__ = [
    "Hasher",
    "GraphEngine",
    "HinEngine",
    "compile_aeroc",
    "run_aeroc",
    "unpack_aeroc",
    "hash_bytes",
    "hash_file",
    "reduce_hin_uast",
    "evaluate_hin_energy",
    "repair_uast_expression",
    "enforce_repair_isolation_py",
]
