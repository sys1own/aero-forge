"""Holographic Invariant Storage (HIS) interface for the IntentCompiler.

This module wraps the `aero_forge_his` Rust extension and exposes a high-level
`HolographicContext` class that can build an invariant vector from a target goal
and a safety constraint, then measure cosine-similarity drift between the
invariant and a noisy runtime context.
"""

from __future__ import annotations

import hashlib
from typing import List, Optional

try:
    import aero_forge_his as _his
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "aero_forge_his is not installed. Build it with: "
        "cd aero_forge_his && maturin develop --release"
    ) from exc


DIMENSION = _his.dimension()


def _to_bipolar(values: List[int]) -> List[int]:
    """Coerce a vector to {-1, 1} bipolar values."""
    return [1 if v >= 0 else -1 for v in values]


def bind(a: List[int], b: List[int]) -> List[int]:
    """Bind two bipolar vectors with element-wise multiplication."""
    return _his.bind(a, b)


def bundle(a: List[int], b: List[int]) -> List[int]:
    """Bundle two bipolar vectors with element-wise addition (returns i32)."""
    return _his.bundle(a, b)


def cleanup(v: List[int]) -> List[int]:
    """Threshold a real-valued vector back to bipolar {-1, 1}."""
    return _his.cleanup(v)


def invariant(goal: List[int], safety: List[int]) -> List[int]:
    """Create an invariant vector: H_inv = goal ⊗ safety."""
    return _his.invariant(goal, safety)


def restore(hinv: List[int], context: List[int], noise: float = 1.0) -> List[int]:
    """Restore a clean vector from an invariant and a noisy context."""
    return _his.restore(hinv, context, noise)


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two real-valued vectors in [-1, 1]."""
    return _his.cosine_similarity(a, b)


def random_bipolar(seed: int = 0) -> List[int]:
    """Return a reproducible random 10,000-dimensional bipolar vector."""
    return _his.random_bipolar(seed)


def ones() -> List[int]:
    """Return a 10,000-dimensional vector filled with +1."""
    return _his.ones()


def dimension() -> int:
    """Return the canonical HIS dimensionality (10,000)."""
    return _his.dimension()


def _symbol_seed(symbol: str, base_seed: int) -> int:
    """Deterministic 64-bit seed from a symbol name."""
    digest = hashlib.sha256(symbol.encode("utf-8")).hexdigest()[:16]
    return base_seed ^ int(digest, 16)


def intent_vector(symbols: List[str], *, seed: int = 0) -> List[int]:
    """Encode a list of functional-intent symbols as a single bipolar vector.

    Each symbol is mapped to a deterministic random 10,000-dimensional bipolar
    vector and the vectors are superposed (bundled) and cleaned. The result is
    reproducible for the same symbol set and seed.
    """
    if not symbols:
        return ones()
    vectors = [random_bipolar(_symbol_seed(symbol, seed)) for symbol in symbols]
    summed = [sum(vals) for vals in zip(*vectors)]
    return cleanup(summed)


class HolographicContext:
    """High-level wrapper for tracking intent drift with HIS.

    Example:
        ctx = HolographicContext()
        hinv = ctx.build_invariant(goal, safety)
        drift = ctx.measure_drift(noisy_context)
    """

    def __init__(self, seed: int = 0) -> None:
        self._goal: Optional[List[int]] = None
        self._safety: Optional[List[int]] = None
        self._hinv: Optional[List[int]] = None
        self._seed = seed

    def build_invariant(
        self, goal: List[int], safety: List[int]
    ) -> List[int]:
        """Bind a target goal to a safety constraint and cache the invariant."""
        if len(goal) != DIMENSION:
            raise ValueError(
                f"goal dimension must be {DIMENSION}, got {len(goal)}"
            )
        if len(safety) != DIMENSION:
            raise ValueError(
                f"safety dimension must be {DIMENSION}, got {len(safety)}"
            )
        self._goal = _to_bipolar(goal)
        self._safety = _to_bipolar(safety)
        self._hinv = invariant(self._goal, self._safety)
        return self._hinv

    def measure_drift(
        self, context: List[float], *, noise: float = 1.0
    ) -> float:
        """Return cosine similarity between the invariant and a context.

        A high value (close to 1.0) means the context is aligned with the
        invariant; a low or negative value indicates drift.
        """
        if self._hinv is None:
            raise RuntimeError("build_invariant() must be called first")
        if len(context) != DIMENSION:
            raise ValueError(
                f"context dimension must be {DIMENSION}, got {len(context)}"
            )
        # Convert the bipolar invariant to floats for cosine similarity.
        hinv_floats = [float(x) for x in self._hinv]
        return cosine_similarity(hinv_floats, context)

    def restore_context(
        self, context: List[int], *, noise: float = 1.0
    ) -> List[int]:
        """Clean up a noisy context against the stored invariant."""
        if self._hinv is None:
            raise RuntimeError("build_invariant() must be called first")
        return restore(self._hinv, context, noise)

    def build_invariant_from_symbols(
        self,
        symbols: List[str],
        *,
        safety: Optional[List[int]] = None,
    ) -> List[int]:
        """Build an invariant from a list of functional-intent symbols.

        The ``goal`` vector is derived from ``symbols`` and ``safety`` defaults
        to the all-ones vector. This is the entry point expected by the
        ``IntentCompiler`` when comparing a prompt's functional intent against a
        generated blueprint.
        """
        goal = intent_vector(symbols, seed=self._seed)
        safety_vec = safety if safety is not None else ones()
        return self.build_invariant(goal, safety_vec)

    def measure_symbol_drift(self, symbols: List[str]) -> float:
        """Measure cosine-similarity drift for a set of functional-intent symbols.

        Returns a value in ``[-1, 1]``. A value near ``1.0`` means the symbol set
        is strongly aligned with the invariant; a value near ``0.0`` or negative
        indicates drift.
        """
        context = [float(x) for x in intent_vector(symbols, seed=self._seed)]
        return self.measure_drift(context)

    @property
    def dimension(self) -> int:
        return DIMENSION

    @property
    def hinv(self) -> Optional[List[int]]:
        return self._hinv
