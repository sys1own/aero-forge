"""Tests for the batch analytics Rust core and Python wrappers."""

from __future__ import annotations

import pytest

from batch_analytics import (
    AggregateResult,
    Record,
    aggregate_batch,
    detect_outliers,
    validate_window,
)


def test_empty_records():
    assert aggregate_batch([], 2) == []


def test_single_window():
    r1 = Record("a", 1.0, 0.0)
    r2 = Record("b", 3.0, 1.0)
    results = aggregate_batch([r1, r2], 2)
    assert len(results) == 1
    assert results[0].window_start == 0
    assert results[0].mean == pytest.approx(2.0)
    assert results[0].std == pytest.approx(1.0)


def test_multiple_windows():
    records = [Record(str(i), float(i), float(i)) for i in range(5)]
    results = aggregate_batch(records, 2)
    assert len(results) == 3
    assert results[0].mean == pytest.approx(0.5)
    assert results[1].mean == pytest.approx(2.5)
    assert results[2].mean == pytest.approx(4.0)


def test_window_size_one():
    records = [Record("a", 2.0, 0.0), Record("b", 4.0, 1.0)]
    results = aggregate_batch(records, 1)
    assert len(results) == 2
    assert results[0].mean == pytest.approx(2.0)
    assert results[1].mean == pytest.approx(4.0)


def test_invalid_window():
    with pytest.raises(ValueError):
        validate_window(0)


def test_detect_outliers():
    results = [
        AggregateResult(0, 1.0, 0.0, []),
        AggregateResult(2, 1.0, 0.0, []),
        AggregateResult(4, 100.0, 0.0, []),
    ]
    assert detect_outliers(results, 1.0) == [2]


def test_detect_outliers_empty():
    assert detect_outliers([], 1.5) == []
    assert detect_outliers([AggregateResult(0, 5.0, 0.0, [])], 1.5) == []


def test_record_repr():
    r = Record("x", 1.5, 2.0)
    assert "x" in repr(r) and "1.5" in repr(r)
