"""Unit tests for the Holographic Invariant Storage (HIS) interface."""

import pytest

from aero_forge.builder.holographic import (
    HolographicContext,
    bind,
    bundle,
    cleanup,
    cosine_similarity,
    dimension,
    intent_vector,
    invariant,
    ones,
    random_bipolar,
    restore,
)


def test_dimension_is_ten_thousand():
    assert dimension() == 10_000


def test_ones_returns_positive_vector():
    v = ones()
    assert len(v) == dimension()
    assert all(x == 1 for x in v)


def test_random_bipolar_is_bipolar():
    v = random_bipolar(seed=42)
    assert len(v) == dimension()
    assert all(x in (-1, 1) for x in v)
    # A second seed must produce a different vector.
    v2 = random_bipolar(seed=7)
    assert any(a != b for a, b in zip(v, v2))


def test_bind_is_elementwise_xor_in_bipolar_space():
    a = [1, -1, 1, -1]
    b = [1, 1, -1, -1]
    result = bind(a, b)
    assert result == [1, -1, -1, 1]


def test_bundle_adds_values():
    a = [1, -1, 1]
    b = [1, 1, -1]
    result = bundle(a, b)
    assert result == [2, 0, 0]


def test_cleanup_thresholds_to_bipolar():
    assert cleanup([2, -3, 0, 5, -1]) == [1, -1, 1, 1, -1]


def test_invariant_is_bipolar():
    goal = random_bipolar(seed=1)
    safety = random_bipolar(seed=2)
    hinv = invariant(goal, safety)
    assert len(hinv) == dimension()
    assert all(x in (-1, 1) for x in hinv)
    # Binding with the same vectors returns the same result (deterministic).
    assert invariant(goal, safety) == hinv


def test_restore_is_bipolar():
    goal = random_bipolar(seed=3)
    safety = random_bipolar(seed=4)
    hinv = invariant(goal, safety)
    noisy = random_bipolar(seed=5)
    restored = restore(hinv, noisy)
    assert len(restored) == dimension()
    assert all(x in (-1, 1) for x in restored)


def test_cosine_perfect_match():
    a = [1.0] * 100
    b = [1.0] * 100
    assert cosine_similarity(a, b) == pytest.approx(1.0)


def test_cosine_orthogonal():
    a = [1.0] * 100
    b = [-1.0] * 100
    assert cosine_similarity(a, b) == pytest.approx(-1.0)


def test_holographic_context_drift():
    ctx = HolographicContext(seed=0)
    goal = ones()
    safety = random_bipolar(seed=9)
    ctx.build_invariant(goal, safety)
    # The invariant aligned with itself should yield a high cosine similarity.
    drift = ctx.measure_drift([float(x) for x in ctx.hinv])
    assert drift == pytest.approx(1.0)
    # An orthogonal random context should have near-zero drift.
    random_ctx = [float(x) for x in random_bipolar(seed=10)]
    drift_random = ctx.measure_drift(random_ctx)
    assert -0.3 < drift_random < 0.3


def test_dimension_mismatch_raises():
    with pytest.raises(ValueError):
        HolographicContext().build_invariant([1, -1], [1, 1])


def test_measure_drift_before_invariant_raises():
    with pytest.raises(RuntimeError):
        HolographicContext().measure_drift([1.0] * dimension())


def test_intent_vector_reproducible():
    symbols = ["aligner", "main", "tests"]
    assert intent_vector(symbols) == intent_vector(symbols)
    assert intent_vector(symbols) != intent_vector(["other"])


def test_symbol_drift_matches_exact_intent():
    ctx = HolographicContext(seed=7)
    symbols = ["smith_waterman", "aligner", "main"]
    ctx.build_invariant_from_symbols(symbols)
    assert ctx.measure_symbol_drift(symbols) == pytest.approx(1.0, abs=1e-9)


def test_symbol_drift_drops_when_intent_missing():
    ctx = HolographicContext(seed=7)
    ctx.build_invariant_from_symbols(["smith_waterman", "aligner", "main"])
    # A blueprint missing the core symbol should drift away from the invariant.
    drift = ctx.measure_symbol_drift(["main"])
    assert 0.0 < drift < 1.0
