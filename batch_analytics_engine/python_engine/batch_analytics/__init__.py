"""Public Python package for the batch-analytics-engine.

This module re-exports the Rust native extension symbols when available.
If the compiled extension cannot be loaded (missing dynamic library, ABI
mismatch, etc.) it falls back to a pure-Python reference implementation
so callers can still develop and test the API surface.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterator, List, Sequence, Union

_NATIVE = False


def _chunks(seq: Sequence[object], n: int) -> Iterator[Sequence[object]]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


try:
    from batch_analytics._native import (  # type: ignore[import-not-found]
        AggregateResult,
        Record,
        aggregate_batch,
        detect_outliers,
        validate_window,
    )

    _NATIVE = True
except ImportError as exc:  # pragma: no cover - fallback path
    warnings.warn(
        f"Rust extension not available, using pure-Python fallback: {exc}",
        RuntimeWarning,
        stacklevel=1,
    )

    @dataclass
    class Record:  # type: ignore[no-redef]
        id: str
        value: float
        timestamp: float

    @dataclass
    class AggregateResult:  # type: ignore[no-redef]
        window_start: int
        mean: float
        std: float
        outliers: List[int]

    def validate_window(window: int) -> None:  # type: ignore[misc]
        """Pure-Python validation mirror of the Rust validator."""
        if window <= 0:
            raise ValueError(f"window size must be positive, got {window}")

    def aggregate_batch(records: Sequence[Record], window: int) -> List[AggregateResult]:  # type: ignore[misc]
        """Pure-Python fallback for window aggregation."""
        validate_window(window)
        if not records:
            return []

        results: List[AggregateResult] = []
        for idx, chunk in enumerate(_chunks(records, window)):
            values = [r.value for r in chunk]
            n = len(values)
            mean = sum(values) / n
            variance = sum((v - mean) ** 2 for v in values) / n
            std = variance**0.5
            results.append(
                AggregateResult(window_start=idx * window, mean=mean, std=std, outliers=[])
            )
        return results

    def detect_outliers(results: Sequence[AggregateResult], threshold: float) -> List[int]:  # type: ignore[misc]
        """Pure-Python fallback for Z-score outlier detection."""
        if not results or threshold <= 0.0:
            return []

        means = [r.mean for r in results]
        n = len(means)
        mean = sum(means) / n
        variance = sum((m - mean) ** 2 for m in means) / n
        std = variance**0.5

        if std == 0.0:
            return []

        return [
            i
            for i, result in enumerate(results)
            if abs(result.mean - mean) / std > threshold
        ]


__all__ = [
    "Record",
    "AggregateResult",
    "aggregate_batch",
    "detect_outliers",
    "validate_window",
    "_NATIVE",
]
