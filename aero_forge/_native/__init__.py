"""Native PyO3 accelerator extension package.

The compiled extension is loaded from ``aero_forge_native``.
"""

try:
    from .aero_forge_native import (  # type: ignore[import-not-found]
        ASTRewritePatch,
        GoIProofNet,
        GoIWavefrontSolverNative,
        GraphEngine,
        Hasher,
        HinEngine,
        PreWriteHealer,
        compile_aeroc,
        enforce_repair_isolation_py,
        evaluate_hin_energy,
        hash_bytes,
        hash_file,
        reduce_hin_uast,
        repair_uast_expression,
        run_aeroc,
        unpack_aeroc,
        verify_goi_proof_net,
        verify_hin_boundary_layouts,
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
    "GoIProofNet",
    "GoIWavefrontSolverNative",
    "verify_goi_proof_net",
    "ASTRewritePatch",
    "PreWriteHealer",
    "compile_aeroc",
    "run_aeroc",
    "unpack_aeroc",
    "hash_bytes",
    "hash_file",
    "reduce_hin_uast",
    "evaluate_hin_energy",
    "repair_uast_expression",
    "enforce_repair_isolation_py",
    "verify_hin_boundary_layouts",
]
