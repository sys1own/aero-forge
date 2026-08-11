#!/usr/bin/env python3
"""Verification: domain-driven blueprinting for complex polyglot builds.

Exercises three high-complexity prompts against a deterministic MockLLM to
confirm that GraphPolyglotMaterializer:
- Does not collapse distinct functional domains into a single node.
- Emits a correct manifest, contracts, and toolchain files per node.
- Builds native artifacts and a runnable entrypoint.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from aero_forge.builder.materializers.graph_materializer import (
    GraphPolyglotMaterializer,
)


class MockLLM:
    """Deterministic LLM that returns the source payload for the requested node."""

    def __init__(self, responses: dict):
        self.calls = 0
        self._responses = responses

    def generate(self, messages, *, temperature=None, max_tokens=None):
        self.calls += 1
        user = messages[1]["content"]
        node_id = ""
        # The materializer wraps the node spec in a JSON block.
        m = re.search(r'"node_id"\s*:\s*"([^"]+)"', user)
        if m:
            node_id = m.group(1)
        if node_id in self._responses:
            return self._responses[node_id]
        # Fallback for emitter-plugin synthesis attempts.
        if "EmitterPlugin" in user:
            return (
                "__AERO_LOGIC_START__\n"
                "```zig:src/fallback.zig\n"
                "export fn fast_math_kernel(x: i64) i64 { return x; }\n"
                "```\n"
                "__AERO_LOGIC_END__"
            )
        return ""


def _run_build_sh(workspace: Path) -> tuple:
    build_sh = workspace / "build.sh"
    if not build_sh.is_file():
        return 1, "", "build.sh missing"
    result = subprocess.run(
        ["bash", str(build_sh)],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode, result.stdout, result.stderr


def _check_log(log_path: Path, *phrases: str) -> None:
    text = log_path.read_text(encoding="utf-8") if log_path.is_file() else ""
    for phrase in phrases:
        assert phrase in text, f"Expected log phrase not found: {phrase!r}"


def build_iot_gateway():
    """Build 1: Edge Computing IoT Gateway (C++ + Rust + Python)."""
    cpp_src = """__AERO_LOGIC_START__
```cpp:signal_processing.cpp
#include <cstdint>
extern "C" int64_t process_sensor_signal(int64_t x) { return x * 2; }
```
__AERO_LOGIC_END__"""

    rust_src = """__AERO_LOGIC_START__
```rust:broker/src/lib.rs
#[no_mangle]
pub extern "C" fn authenticate_device(x: i64) -> i64 {
    x * 3
}
```
__AERO_LOGIC_END__"""

    test_src = """__AERO_LOGIC_START__
```python:tests/test_all.py
def test_signal_processing_positive(): assert True
def test_signal_processing_zero(): assert True
def test_signal_processing_negative(): assert True
def test_signal_processing_large(): assert True
def test_signal_processing_small(): assert True
def test_broker_positive(): assert True
def test_broker_zero(): assert True
def test_broker_negative(): assert True
def test_broker_large(): assert True
def test_broker_small(): assert True
def test_dashboard_imports(): assert True
def test_dashboard_run(): assert True
def test_dashboard_pathlib(): from pathlib import Path; assert Path(".").exists()
def test_dashboard_ctypes(): import ctypes; assert ctypes.CDLL is not None
def test_project_structure(): assert True
```
__AERO_LOGIC_END__"""

    spec = {
        "project": "iot_gateway",
        "metadata": {
            "architecture": "tri_polyglot_rust_cpp_python",
            "llm_initialized": True,
            "prompt": (
                "Build a tri_polyglot_rust_cpp_python IoT Gateway. Use a C++ shared library "
                "for real-time sensor signal processing. Use Rust for an asynchronous message "
                "broker. The Python CLI should provide a dashboard. The signal processing "
                "logic lives in a hardware/ package and the broker in core/auth/."
            ),
        },
        "nodes": [
            {
                "node_id": "signal_processing",
                "lang": "cpp",
                "toolchain": "cmake",
                "source_files": ["signal_processing.cpp"],
                "exports": ["process_sensor_signal"],
            },
            {
                "node_id": "broker",
                "lang": "rust",
                "toolchain": "cargo",
                "source_files": ["src/lib.rs"],
                "exports": ["authenticate_device"],
            },
            {
                "node_id": "dashboard",
                "lang": "python",
                "toolchain": "python",
                "source_files": ["dashboard/main.py"],
                "exports": [],
            },
            {
                "node_id": "tests",
                "lang": "python",
                "toolchain": "python",
                "source_files": ["tests/test_all.py"],
                "exports": [],
            },
        ],
        "edges": [
            {
                "source": "signal_processing",
                "target": "dashboard",
                "symbol": "process_sensor_signal",
                "boundary_type": "c_abi",
                "args": ["int64"],
                "return_type": "int64",
            },
            {
                "source": "broker",
                "target": "dashboard",
                "symbol": "authenticate_device",
                "boundary_type": "c_abi",
                "args": ["int64"],
                "return_type": "int64",
            },
        ],
        "primary_entrypoint": "dashboard/main.py",
    }

    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp) / "iot"
        log = Path(tmp) / "accel.log"
        os.environ["AERO_FORGE_ACCEL_LOG"] = str(log)
        materializer = GraphPolyglotMaterializer(
            workspace_root=workspace,
            llm_client=MockLLM(
                {
                    "signal_processing": cpp_src,
                    "broker": rust_src,
                    "tests": test_src,
                }
            ),
        )
        result = materializer.materialize(spec, build=True)
        artifacts = [a["path"] for a in result.get("artifacts", [])]
        assert any("signal_processing/CMakeLists.txt" in a for a in artifacts)
        assert any("broker/Cargo.toml" in a for a in artifacts)
        assert any("broker/src/lib.rs" in a for a in artifacts)
        assert any("dashboard/main.py" in a for a in artifacts)
        assert any("tests/test_all.py" in a for a in artifacts)

        _check_log(
            log,
            "Node Materialization Verified: signal_processing",
            "Node Materialization Verified: broker",
            "Test density verified",
        )

        rc, stdout, stderr = _run_build_sh(workspace)
        assert rc == 0, f"build.sh failed: {stderr}\n{stdout}"
        assert "process_sensor_signal(42) = 84" in stdout

    print("PASS: IoT Gateway build")


def build_hft_simulator():
    """Build 2: High-Frequency Trading Simulator (Rust + Python)."""
    rust_src = """__AERO_LOGIC_START__
```rust:matching_engine/src/lib.rs
#[no_mangle]
pub extern "C" fn process_order(x: i64) -> i64 { x + 1 }

#[no_mangle]
pub extern "C" fn get_market_depth(x: i64) -> i64 { x * 2 }
```
__AERO_LOGIC_END__"""

    py_src = """__AERO_LOGIC_START__
```python:strategies/mean_reversion.py
def mean_reversion(x: int) -> int:
    return x // 2
```
__AERO_LOGIC_END__"""

    test_src = """__AERO_LOGIC_START__
```python:tests/test_all.py
def test_process_order_positive(): assert True
def test_process_order_zero(): assert True
def test_process_order_negative(): assert True
def test_process_order_large(): assert True
def test_process_order_small(): assert True
def test_get_market_depth_positive(): assert True
def test_get_market_depth_zero(): assert True
def test_get_market_depth_negative(): assert True
def test_get_market_depth_large(): assert True
def test_get_market_depth_small(): assert True
def test_mean_reversion_positive(): assert True
def test_mean_reversion_zero(): assert True
def test_mean_reversion_negative(): assert True
def test_mean_reversion_large(): assert True
def test_mean_reversion_small(): assert True
```
__AERO_LOGIC_END__"""

    spec = {
        "project": "hft_simulator",
        "metadata": {
            "architecture": "hybrid_rust_python",
            "llm_initialized": True,
            "prompt": (
                "Build a modular hybrid_rust_python HFT simulator. Implement a Rust crate "
                "for a high-concurrency order matching engine. The Python side must include "
                "a strategies/ module for user-defined trading logic and a main.py for backtesting. "
                "The matching engine must expose an FFI contract for process_order and get_market_depth. "
                "Every module must have comprehensive tests verifying state consistency."
            ),
        },
        "nodes": [
            {
                "node_id": "matching_engine",
                "lang": "rust",
                "toolchain": "cargo",
                "source_files": ["src/lib.rs"],
                "exports": ["process_order", "get_market_depth"],
            },
            {
                "node_id": "strategies",
                "lang": "python",
                "toolchain": "python",
                "source_files": ["strategies/mean_reversion.py"],
                "exports": ["mean_reversion"],
            },
            {
                "node_id": "main",
                "lang": "python",
                "toolchain": "python",
                "source_files": ["main/entry.py"],
                "exports": [],
            },
            {
                "node_id": "tests",
                "lang": "python",
                "toolchain": "python",
                "source_files": ["tests/test_all.py"],
                "exports": [],
            },
        ],
        "edges": [
            {
                "source": "matching_engine",
                "target": "main",
                "symbol": "process_order",
                "boundary_type": "c_abi",
                "args": ["int64"],
                "return_type": "int64",
            },
            {
                "source": "matching_engine",
                "target": "main",
                "symbol": "get_market_depth",
                "boundary_type": "c_abi",
                "args": ["int64"],
                "return_type": "int64",
            },
        ],
        "primary_entrypoint": "main/entry.py",
    }

    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp) / "hft"
        log = Path(tmp) / "accel.log"
        os.environ["AERO_FORGE_ACCEL_LOG"] = str(log)
        materializer = GraphPolyglotMaterializer(
            workspace_root=workspace,
            llm_client=MockLLM(
                {
                    "matching_engine": rust_src,
                    "strategies": py_src,
                    "tests": test_src,
                }
            ),
        )
        result = materializer.materialize(spec, build=True)
        artifacts = [a["path"] for a in result.get("artifacts", [])]
        assert any("matching_engine/Cargo.toml" in a for a in artifacts)
        assert any("matching_engine/src/lib.rs" in a for a in artifacts)
        assert any("strategies/mean_reversion.py" in a for a in artifacts)
        assert any("main/entry.py" in a for a in artifacts)
        assert any("tests/test_all.py" in a for a in artifacts)

        _check_log(
            log,
            "Node Materialization Verified: matching_engine",
            "Node Materialization Verified: strategies",
            "Test density verified",
        )

        rc, stdout, stderr = _run_build_sh(workspace)
        assert rc == 0, f"build.sh failed: {stderr}\n{stdout}"
        assert "process_order(42) = 43" in stdout

    print("PASS: HFT Simulator build")


def build_genomic_sharder():
    """Build 3: Distributed Genomic Data Sharder (Zig + Rust + Python)."""
    rust_src = """__AERO_LOGIC_START__
```rust:rust_kmer/src/lib.rs
#[no_mangle]
pub extern "C" fn kmer_hash(x: i64) -> i64 {
    x * 5
}
```
__AERO_LOGIC_END__"""

    spec = {
        "project": "genomic_sharder",
        "metadata": {
            "architecture": "graph_polyglot",
            "llm_initialized": True,
            "prompt": (
                "Build a graph_polyglot system for sharding large DNA files. "
                "JIT-synthesize a Zig module for high-speed file I/O. Use Rust for "
                "calculating k-mer frequency hashes. The Python controller orchestrates "
                "the Zig and Rust nodes to shard a 10GB FASTA file into multiple compressed "
                "archives. Every node must be verified by the GoI Proof Net to ensure "
                "zero-deadlock scheduling."
            ),
        },
        "nodes": [
            {
                "node_id": "zig_io",
                "lang": "zig",
                "toolchain": "zig",
                "source_files": ["src/read_fasta_chunk.zig"],
                "exports": ["read_fasta_chunk"],
            },
            {
                "node_id": "rust_kmer",
                "lang": "rust",
                "toolchain": "cargo",
                "source_files": ["src/lib.rs"],
                "exports": ["kmer_hash"],
            },
            {
                "node_id": "python_controller",
                "lang": "python",
                "toolchain": "python",
                "source_files": ["python_controller/main.py"],
                "exports": [],
            },
        ],
        "edges": [
            {
                "source": "zig_io",
                "target": "python_controller",
                "symbol": "read_fasta_chunk",
                "boundary_type": "c_abi",
                "args": ["int64"],
                "return_type": "int64",
            },
            {
                "source": "rust_kmer",
                "target": "python_controller",
                "symbol": "kmer_hash",
                "boundary_type": "c_abi",
                "args": ["int64"],
                "return_type": "int64",
            },
        ],
        "primary_entrypoint": "python_controller/main.py",
    }

    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp) / "sharder"
        log = Path(tmp) / "accel.log"
        os.environ["AERO_FORGE_ACCEL_LOG"] = str(log)
        materializer = GraphPolyglotMaterializer(
            workspace_root=workspace,
            llm_client=MockLLM({"rust_kmer": rust_src}),
        )
        result = materializer.materialize(spec, build=True)
        artifacts = [a["path"] for a in result.get("artifacts", [])]
        assert any("zig_io/build.zig" in a for a in artifacts)
        assert any("zig_io/src/read_fasta_chunk.zig" in a for a in artifacts)
        assert any("rust_kmer/Cargo.toml" in a for a in artifacts)
        assert any("rust_kmer/src/lib.rs" in a for a in artifacts)
        assert any("python_controller/main.py" in a for a in artifacts)

        _check_log(
            log,
            "Node Materialization Verified: zig_io",
            "Node Materialization Verified: rust_kmer",
            "zig build succeeded",
            "cargo build succeeded",
        )

        rc, stdout, stderr = _run_build_sh(workspace)
        assert rc == 0, f"build.sh failed: {stderr}\n{stdout}"
        assert "read_fasta_chunk(42) = 84" in stdout

    print("PASS: Genomic Sharder build")


def main():
    build_iot_gateway()
    build_hft_simulator()
    build_genomic_sharder()
    print("\nAll domain-driven verification builds passed.")


if __name__ == "__main__":
    main()
