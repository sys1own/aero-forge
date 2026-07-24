"""Benchmark the Rust extension against a pure-Python reference."""

from __future__ import annotations

import random
import sys
import time
from typing import List

from batch_analytics import Record, aggregate_batch


def make_records(n: int) -> List[Record]:
    return [Record(str(i), random.random(), float(i)) for i in range(n)]


def pure_aggregate(records: List[Record], window: int) -> List[dict]:
    """Reference pure-Python implementation."""
    results: List[dict] = []
    for i in range(0, len(records), window):
        chunk = records[i : i + window]
        values = [r.value for r in chunk]
        n = len(values)
        mean = sum(values) / n
        std = (sum((v - mean) ** 2 for v in values) / n) ** 0.5
        results.append({"window_start": i, "mean": mean, "std": std})
    return results


def bench(fn, records: List[Record], window: int, reps: int) -> float:
    start = time.perf_counter()
    for _ in range(reps):
        fn(records, window)
    return time.perf_counter() - start


def main() -> int:
    records = make_records(1000)
    window = 10
    reps = 100
    native_time = bench(aggregate_batch, records, window, reps)
    pure_time = bench(pure_aggregate, records, window, reps)
    print(f"native: {native_time:.4f}s for {reps} calls")
    print(f"pure:   {pure_time:.4f}s for {reps} calls")
    if native_time > 0:
        print(f"speedup: {pure_time / native_time:.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
