"""Benchmark harness for the batch-analytics-engine.

Compares the Rust/PyO3 native extension against equivalent pure-Python
implementations for `aggregate_batch`, `detect_outliers`, and `validate_window`.

Run from the `batch_analytics_engine` workspace root:

    python python_engine/benchmarks/bench_batch.py

Optionally save results:

    python python_engine/benchmarks/bench_batch.py --json --csv -o results
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

# Import the public package.  When the Rust extension is available the names
# below are native; otherwise they are the pure-Python fallback from
# `batch_analytics/__init__.py`.
import batch_analytics as ba
from batch_analytics import AggregateResult, Record, aggregate_batch, detect_outliers, validate_window


# ---------------------------------------------------------------------------
# Pure-Python reference implementations
# ---------------------------------------------------------------------------


def _chunks(seq: Sequence[Any], n: int):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


@dataclass(frozen=True)
class _PureAggregateResult:
    window_start: int
    mean: float
    std: float
    outliers: List[int]


def pure_aggregate_batch(records: Sequence[Record], window: int) -> List[_PureAggregateResult]:
    """Reference pure-Python implementation of aggregate_batch."""
    if window <= 0:
        raise ValueError(f"window size must be positive, got {window}")
    if not records:
        return []

    results: List[_PureAggregateResult] = []
    for idx, chunk in enumerate(_chunks(records, window)):
        values = [r.value for r in chunk]
        n = len(values)
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / n
        std = math.sqrt(variance)
        results.append(_PureAggregateResult(idx * window, mean, std, []))
    return results


def pure_detect_outliers(
    results: Sequence[_PureAggregateResult], threshold: float
) -> List[int]:
    """Reference pure-Python implementation of detect_outliers."""
    if not results or threshold <= 0.0:
        return []

    means = [r.mean for r in results]
    n = len(means)
    mean = sum(means) / n
    variance = sum((m - mean) ** 2 for m in means) / n
    std = math.sqrt(variance)

    if std == 0.0:
        return []

    return [i for i, result in enumerate(results) if abs(result.mean - mean) / std > threshold]


def pure_validate_window(window: int) -> None:
    """Reference pure-Python implementation of validate_window."""
    if window <= 0:
        raise ValueError(f"window size must be positive, got {window}")


# ---------------------------------------------------------------------------
# Input data generators
# ---------------------------------------------------------------------------


def make_records(n: int, seed: int = 42) -> List[Record]:
    """Create a batch of `n` records with a few injected outliers."""
    rng = random.Random(seed)
    records: List[Record] = []
    for i in range(n):
        value = rng.random()
        # Inject a small number of extreme values so outlier detection has work.
        if i % max(n // 10, 1000) == 0 and i != 0:
            value = 10.0 + rng.random()
        records.append(Record(str(i), value, float(i)))
    return records


def choose_window(size: int) -> int:
    """Choose a window that keeps the number of aggregate results reasonable."""
    return max(10, size // 1000)


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------


def _warmup(fn: Callable[..., Any], *args: Any) -> None:
    try:
        fn(*args)
    except Exception:
        pass


def _time_once(fn: Callable[..., Any], args: Tuple[Any, ...]) -> float:
    start = time.perf_counter()
    fn(*args)
    return time.perf_counter() - start


def benchmark_native_pure(
    native_fn: Callable[..., Any],
    native_args: Tuple[Any, ...],
    pure_fn: Callable[..., Any],
    pure_args: Tuple[Any, ...],
    n_reps: int,
    calls_per_rep: int = 1,
) -> Tuple[float, float, float, float]:
    """Return (native_mean, native_stdev, pure_mean, pure_stdev) in microseconds."""
    _warmup(native_fn, *native_args)
    _warmup(pure_fn, *pure_args)

    native_times: List[float] = []
    pure_times: List[float] = []

    for _ in range(n_reps):
        native_times.append(_time_once(native_fn, native_args))
        pure_times.append(_time_once(pure_fn, pure_args))

    # Convert to per-call microseconds.
    factor = 1_000_000.0 / calls_per_rep
    native_per_call = [t * factor for t in native_times]
    pure_per_call = [t * factor for t in pure_times]

    return (
        statistics.mean(native_per_call),
        statistics.stdev(native_per_call) if len(native_per_call) > 1 else 0.0,
        statistics.mean(pure_per_call),
        statistics.stdev(pure_per_call) if len(pure_per_call) > 1 else 0.0,
    )


def _fmt_us(value: float, stdev: float) -> str:
    if value >= 1000:
        return f"{value / 1000:.2f} ms (+/- {stdev / 1000:.2f} ms)"
    return f"{value:.2f} us (+/- {stdev:.2f} us)"


@dataclass
class BenchmarkRow:
    function: str
    size: int
    window: int
    native_mean_us: float
    native_stdev_us: float
    pure_mean_us: float
    pure_stdev_us: float
    speedup: float
    native_available: bool


def print_results(rows: List[BenchmarkRow]) -> None:
    print(f"\nAero-Forge batch-analytics-engine benchmark (native available: {ba._NATIVE})")
    print("=" * 100)
    print(
        f"{'Function':<22} {'Size':>10} {'Window':>10} {'Native':>22} {'Pure':>22} {'Speedup':>10}"
    )
    print("-" * 100)
    for r in rows:
        print(
            f"{r.function:<22} {r.size:>10,} {r.window:>10,} "
            f"{_fmt_us(r.native_mean_us, r.native_stdev_us):>22} "
            f"{_fmt_us(r.pure_mean_us, r.pure_stdev_us):>22} "
            f"{r.speedup:>9.2f}x"
        )
    print("=" * 100)


def save_results(rows: List[BenchmarkRow], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "bench_batch_results.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump([asdict(r) for r in rows], fh, indent=2)
    print(f"Saved JSON results to {json_path}")

    csv_path = output_dir / "bench_batch_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        if rows:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].__dict__.keys()))
            writer.writeheader()
            for r in rows:
                writer.writerow(r.__dict__)
    print(f"Saved CSV results to {csv_path}")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_sensible_outputs(records: List[Record], window: int) -> None:
    """Crash early with a clear message if native and pure diverge."""
    native_agg = aggregate_batch(records[: min(len(records), window * 3)], window)
    pure_agg = pure_aggregate_batch(records[: min(len(records), window * 3)], window)

    assert len(native_agg) == len(pure_agg), "aggregate_batch result length mismatch"
    for n, p in zip(native_agg, pure_agg):
        assert math.isclose(n.mean, p.mean, rel_tol=1e-9, abs_tol=1e-9), (
            f"aggregate_batch mean mismatch: native={n.mean}, pure={p.mean}"
        )
        assert math.isclose(n.std, p.std, rel_tol=1e-6, abs_tol=1e-9), (
            f"aggregate_batch std mismatch: native={n.std}, pure={p.std}"
        )

    if native_agg:
        threshold = 2.0
        native_out = detect_outliers(native_agg, threshold)
        pure_out = pure_detect_outliers(pure_agg, threshold)
        assert sorted(native_out) == sorted(pure_out), (
            f"detect_outliers mismatch: native={native_out}, pure={pure_out}"
        )

    validate_window(window)
    pure_validate_window(window)
    with _expecting_value_error():
        validate_window(0)
    with _expecting_value_error():
        pure_validate_window(0)


class _expecting_value_error:
    def __enter__(self) -> "_expecting_value_error":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            raise AssertionError("expected ValueError was not raised")
        if issubclass(exc_type, ValueError):
            return True
        return False


# ---------------------------------------------------------------------------
# Main benchmark driver
# ---------------------------------------------------------------------------


def _bench_aggregate(size: int) -> BenchmarkRow:
    window = choose_window(size)
    records = make_records(size)
    verify_sensible_outputs(records, window)

    # More repetitions for smaller/faster workloads; cap total work.
    n_reps = max(3, min(1000, 1_000_000 // size))

    native_mean, native_stdev, pure_mean, pure_stdev = benchmark_native_pure(
        aggregate_batch,
        (records, window),
        pure_aggregate_batch,
        (records, window),
        n_reps=n_reps,
    )

    speedup = pure_mean / native_mean if native_mean > 0 else 0.0
    return BenchmarkRow(
        function="aggregate_batch",
        size=size,
        window=window,
        native_mean_us=native_mean,
        native_stdev_us=native_stdev,
        pure_mean_us=pure_mean,
        pure_stdev_us=pure_stdev,
        speedup=speedup,
        native_available=ba._NATIVE,
    )


def _bench_detect_outliers(size: int) -> BenchmarkRow:
    window = choose_window(size)
    records = make_records(size)
    native_results = aggregate_batch(records, window)
    pure_results = pure_aggregate_batch(records, window)

    threshold = 2.0
    n_reps = max(10, min(1000, 100_000 // max(len(native_results), 1)))

    native_mean, native_stdev, pure_mean, pure_stdev = benchmark_native_pure(
        detect_outliers,
        (native_results, threshold),
        pure_detect_outliers,
        (pure_results, threshold),
        n_reps=n_reps,
    )

    speedup = pure_mean / native_mean if native_mean > 0 else 0.0
    return BenchmarkRow(
        function="detect_outliers",
        size=size,
        window=window,
        native_mean_us=native_mean,
        native_stdev_us=native_stdev,
        pure_mean_us=pure_mean,
        pure_stdev_us=pure_stdev,
        speedup=speedup,
        native_available=ba._NATIVE,
    )


def _bench_validate_window(size: int) -> BenchmarkRow:
    window = size
    # validate_window is very fast, so call it `size` times per timed block.
    calls_per_rep = size
    n_reps = max(3, min(10, 5_000_000 // size))

    def native_rep() -> None:
        for _ in range(calls_per_rep):
            validate_window(window)

    def pure_rep() -> None:
        for _ in range(calls_per_rep):
            pure_validate_window(window)

    _warmup(native_rep)
    _warmup(pure_rep)

    native_times: List[float] = []
    pure_times: List[float] = []
    for _ in range(n_reps):
        native_times.append(_time_once(native_rep, ()))
        pure_times.append(_time_once(pure_rep, ()))

    factor = 1_000_000.0 / calls_per_rep
    native_per_call = [t * factor for t in native_times]
    pure_per_call = [t * factor for t in pure_times]

    native_mean = statistics.mean(native_per_call)
    native_stdev = statistics.stdev(native_per_call) if len(native_per_call) > 1 else 0.0
    pure_mean = statistics.mean(pure_per_call)
    pure_stdev = statistics.stdev(pure_per_call) if len(pure_per_call) > 1 else 0.0
    speedup = pure_mean / native_mean if native_mean > 0 else 0.0

    return BenchmarkRow(
        function="validate_window",
        size=size,
        window=window,
        native_mean_us=native_mean,
        native_stdev_us=native_stdev,
        pure_mean_us=pure_mean,
        pure_stdev_us=pure_stdev,
        speedup=speedup,
        native_available=ba._NATIVE,
    )


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark batch-analytics-engine native vs pure Python."
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        help="Override dataset sizes to benchmark.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Write results to a JSON file.",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Write results to a CSV file.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Directory for output files (default: same directory as this script).",
    )
    args = parser.parse_args(argv)

    sizes = args.sizes or [1_000, 10_000, 100_000, 1_000_000]
    rows: List[BenchmarkRow] = []

    for size in sizes:
        rows.append(_bench_aggregate(size))
        rows.append(_bench_detect_outliers(size))
        rows.append(_bench_validate_window(size))

    print_results(rows)

    if args.json or args.csv:
        save_results(rows, args.output_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
