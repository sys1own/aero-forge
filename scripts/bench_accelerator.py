#!/usr/bin/env python3
"""Benchmark the dual-engine accelerator.

The native Rust/PyO3 path is compared against:
* the pure-Python fallback graph engine, and
* a standard-library ``hashlib.sha256`` baseline for hashing.

The BLAKE3 Python fallback is itself a native C extension, so it is included
for informational parity only; the stdlib SHA-256 baseline is the meaningful
non-accelerated hashing comparison.
"""

import hashlib
import os
import random
import tempfile
import time
from typing import Dict, List


def _build_dag(node_count: int, edge_count: int, seed: int) -> tuple[List[str], Dict[str, List[str]]]:
    rng = random.Random(seed)
    nodes = [f"n{i}" for i in range(node_count)]
    edges: Dict[str, List[str]] = {n: [] for n in nodes}
    for _ in range(edge_count):
        i = rng.randrange(node_count - 1)
        j = rng.randrange(i + 1, node_count)
        edges[nodes[j]].append(nodes[i])
    return nodes, edges


def _measure(func, iterations: int) -> float:
    start = time.perf_counter()
    for _ in range(iterations):
        func()
    return time.perf_counter() - start


def _stdlib_hash_file(path: str, iterations: int) -> float:
    def run():
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                h.update(chunk)
        h.hexdigest()

    return _measure(run, iterations)


def main() -> None:
    from aero_forge._native import GraphEngine as NativeGraphEngine, hash_file as native_hash_file
    from aero_forge._fallback import GraphEngine as FallbackGraphEngine, hash_file as fallback_hash_file

    file_size = 200 * 1024 * 1024
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(os.urandom(file_size))
        f.flush()
        temp_path = f.name

    # Warm the page cache so the benchmark measures hashing throughput, not disk I/O.
    with open(temp_path, "rb") as f:
        _ = f.read()

    try:
        print("Benchmarking hash_file (200 MB)")
        native_hash_time = _measure(lambda: native_hash_file(temp_path), 5)
        fallback_hash_time = _measure(lambda: fallback_hash_file(temp_path), 5)
        stdlib_hash_time = _stdlib_hash_file(temp_path, 3)

        native_hash_per = native_hash_time / 5
        fallback_hash_per = fallback_hash_time / 5
        stdlib_hash_per = stdlib_hash_time / 3

        hash_fallback_speedup = fallback_hash_per / native_hash_per
        hash_stdlib_speedup = stdlib_hash_per / native_hash_per

        print(f"  native:   {native_hash_per:.4f}s ({1 / native_hash_per:.1f} ops/s)")
        print(f"  fallback: {fallback_hash_per:.4f}s ({1 / fallback_hash_per:.1f} ops/s)")
        print(f"  stdlib sha256: {stdlib_hash_per:.4f}s ({1 / stdlib_hash_per:.1f} ops/s)")
        print(f"  fallback speedup: {hash_fallback_speedup:.2f}x")
        print(f"  stdlib speedup:   {hash_stdlib_speedup:.2f}x")

        print("\nBenchmarking topological_sort (20k nodes, 80k edges)")
        nodes, edges = _build_dag(20_000, 80_000, 42)
        native_engine = NativeGraphEngine(nodes, edges)
        fallback_engine = FallbackGraphEngine(nodes, edges)

        native_topo_per = _measure(lambda: native_engine.topological_sort(), 10) / 10
        fallback_topo_per = _measure(lambda: fallback_engine.topological_sort(), 10) / 10
        graph_speedup = fallback_topo_per / native_topo_per

        print(f"  native:   {native_topo_per:.4f}s ({1 / native_topo_per:.1f} ops/s)")
        print(f"  fallback: {fallback_topo_per:.4f}s ({1 / fallback_topo_per:.1f} ops/s)")
        print(f"  speedup:  {graph_speedup:.2f}x")

        if hash_stdlib_speedup < 5.0:
            raise SystemExit(
                f"FAIL: native hashing is only {hash_stdlib_speedup:.2f}x faster than stdlib sha256 "
                "(target >= 5x)"
            )
        if graph_speedup < 5.0:
            raise SystemExit(
                f"FAIL: native graph resolution is only {graph_speedup:.2f}x faster than fallback "
                "(target >= 5x)"
            )

        print(
            f"\nPASS: native acceleration exceeds 5x on stdlib hashing ({hash_stdlib_speedup:.2f}x) "
            f"and fallback graph resolution ({graph_speedup:.2f}x)."
        )
    finally:
        os.unlink(temp_path)


if __name__ == "__main__":
    main()
