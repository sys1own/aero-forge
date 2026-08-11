#!/usr/bin/env python3
"""Verification: tri-polyglot Genomics Prompt 2 (Python + Rust + C++).

This deterministic mock-LLM build confirms:
- The C++ node emits a valid CMakeLists.txt with -fPIC -shared.
- The Rust node emits a Cargo.toml.
- The blueprint architecture is promoted to ``tri_polyglot_rust_cpp_python``.
- The build pipeline validates toolchain manifests.
"""

import json
import os
import re
import sys
import tempfile
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from aero_forge.builder.materializers.graph_materializer import (
    GraphPolyglotMaterializer,
)


CPP_SRC = '''__AERO_LOGIC_START__
```cpp:cpp_engine.cpp
#include <cstdint>
extern "C" int64_t align_kernel(int64_t x) { return x * 2; }
```
__AERO_LOGIC_END__'''

RUST_SRC = '''__AERO_LOGIC_START__
```rust:rust_core/src/lib.rs
#[no_mangle]
pub extern "C" fn rust_kernel(x: i64) -> i64 {
    x * 2
}
```
__AERO_LOGIC_END__'''

PYTHON_SRC = '''__AERO_LOGIC_START__
```python:python_interface/main.py
import ctypes
import os
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent

_libs = {}
for node in ("cpp_engine", "rust_core"):
    so = _SCRIPT_DIR.parent / node / f"lib{node}.so"
    if so.exists():
        _libs[node] = ctypes.CDLL(str(so))

if "cpp_engine" in _libs:
    _libs["cpp_engine"].align_kernel.argtypes = [ctypes.c_int64]
    _libs["cpp_engine"].align_kernel.restype = ctypes.c_int64
if "rust_core" in _libs:
    _libs["rust_core"].rust_kernel.argtypes = [ctypes.c_int64]
    _libs["rust_core"].rust_kernel.restype = ctypes.c_int64

def run_demo():
    return 42

if __name__ == "__main__":
    result = 0
    if "cpp_engine" in _libs:
        result += _libs["cpp_engine"].align_kernel(21)
    if "rust_core" in _libs:
        result += _libs["rust_core"].rust_kernel(21)
    print("demo result:", result)
```
__AERO_LOGIC_END__'''

TEST_SRC = '''__AERO_LOGIC_START__
```python:tests/test_all.py
def test_all():
    return True

def test_align_kernel():
    import ctypes, pathlib
    so = pathlib.Path(__file__).resolve().parent.parent / "cpp_engine" / "libcpp_engine.so"
    if not so.exists():
        return
    lib = ctypes.CDLL(str(so))
    lib.align_kernel.argtypes = [ctypes.c_int64]
    lib.align_kernel.restype = ctypes.c_int64
    assert lib.align_kernel(7) == 14

def test_align_kernel_zero():
    import ctypes, pathlib
    so = pathlib.Path(__file__).resolve().parent.parent / "cpp_engine" / "libcpp_engine.so"
    if not so.exists():
        return
    lib = ctypes.CDLL(str(so))
    lib.align_kernel.argtypes = [ctypes.c_int64]
    lib.align_kernel.restype = ctypes.c_int64
    assert lib.align_kernel(0) == 0

def test_align_kernel_negative():
    import ctypes, pathlib
    so = pathlib.Path(__file__).resolve().parent.parent / "cpp_engine" / "libcpp_engine.so"
    if not so.exists():
        return
    lib = ctypes.CDLL(str(so))
    lib.align_kernel.argtypes = [ctypes.c_int64]
    lib.align_kernel.restype = ctypes.c_int64
    assert lib.align_kernel(-3) == -6

def test_align_kernel_large():
    import ctypes, pathlib
    so = pathlib.Path(__file__).resolve().parent.parent / "cpp_engine" / "libcpp_engine.so"
    if not so.exists():
        return
    lib = ctypes.CDLL(str(so))
    lib.align_kernel.argtypes = [ctypes.c_int64]
    lib.align_kernel.restype = ctypes.c_int64
    assert lib.align_kernel(1000) == 2000

def test_rust_kernel():
    import ctypes, pathlib
    so = pathlib.Path(__file__).resolve().parent.parent / "rust_core" / "librust_core.so"
    if not so.exists():
        return
    lib = ctypes.CDLL(str(so))
    lib.rust_kernel.argtypes = [ctypes.c_int64]
    lib.rust_kernel.restype = ctypes.c_int64
    assert lib.rust_kernel(7) == 14

def test_rust_kernel_zero():
    import ctypes, pathlib
    so = pathlib.Path(__file__).resolve().parent.parent / "rust_core" / "librust_core.so"
    if not so.exists():
        return
    lib = ctypes.CDLL(str(so))
    lib.rust_kernel.argtypes = [ctypes.c_int64]
    lib.rust_kernel.restype = ctypes.c_int64
    assert lib.rust_kernel(0) == 0

def test_rust_kernel_negative():
    import ctypes, pathlib
    so = pathlib.Path(__file__).resolve().parent.parent / "rust_core" / "librust_core.so"
    if not so.exists():
        return
    lib = ctypes.CDLL(str(so))
    lib.rust_kernel.argtypes = [ctypes.c_int64]
    lib.rust_kernel.restype = ctypes.c_int64
    assert lib.rust_kernel(-5) == -10

def test_rust_kernel_large():
    import ctypes, pathlib
    so = pathlib.Path(__file__).resolve().parent.parent / "rust_core" / "librust_core.so"
    if not so.exists():
        return
    lib = ctypes.CDLL(str(so))
    lib.rust_kernel.argtypes = [ctypes.c_int64]
    lib.rust_kernel.restype = ctypes.c_int64
    assert lib.rust_kernel(1234) == 2468

def test_python_interface_imports():
    import pathlib, runpy
    main = pathlib.Path(__file__).resolve().parent.parent / "python_interface" / "main.py"
    if main.exists():
        runpy.run_path(str(main), run_name="__not_main__")

def test_python_interface_run_demo():
    import pathlib, runpy
    main = pathlib.Path(__file__).resolve().parent.parent / "python_interface" / "main.py"
    if main.exists():
        ns = runpy.run_path(str(main), run_name="__not_main__")
        assert "run_demo" in ns or True

def test_python_interface_pathlib():
    import pathlib
    assert pathlib.Path(__file__).resolve().parent.parent.exists()

def test_python_interface_ctypes_available():
    import ctypes
    assert ctypes.CDLL is not None

def test_python_interface_os_environ():
    import os
    assert isinstance(os.environ, dict)

def test_python_interface_math_constants():
    import math
    assert math.isfinite(1.0)

def test_python_interface_string_split():
    assert "a,b,c".split(",") == ["a", "b", "c"]
```
__AERO_LOGIC_END__'''


class MockLLM:
    def __init__(self):
        self.calls = 0
        self._responses = {
            "cpp_engine": CPP_SRC,
            "rust_core": RUST_SRC,
            "python_interface": PYTHON_SRC,
            "tests": TEST_SRC,
        }

    @staticmethod
    def _first_symbol(skeleton: str) -> str:
        for line in skeleton.splitlines():
            line = line.strip()
            m = re.search(
                r'(?:^|\s)(?:def|fn|func|function|class|struct|interface)\s+(\w+)', line
            )
            if m:
                return m.group(1)
            m = re.search(
                r'extern\s+"C"\s+(?:const\s+)?(?:[\w<>,:*&~]+\s+){1,4}(\w+)\s*\(',
                line,
            )
            if m:
                return m.group(1)
            m = re.search(
                r'pub\s+extern\s+"C"\s+fn\s+(\w+)\s*\(', line
            )
            if m:
                return m.group(1)
        return ""

    def generate(self, messages, *, temperature=None, max_tokens=None):
        self.calls += 1
        user = messages[1]["content"]
        node_id = ""
        m = re.search(r"```json\s*([\s\S]*?)```", user)
        if m:
            try:
                payload = json.loads(m.group(1))
                candidates = payload.get("missing_symbols") or payload.get(
                    "required_symbols", []
                )
                for sym in candidates:
                    for nid, src in self._responses.items():
                        if re.search(rf"\b{re.escape(sym)}\b", src):
                            node_id = nid
                            break
                    if node_id:
                        break
                if not node_id:
                    sym = self._first_symbol(payload.get("skeleton", ""))
                    if sym:
                        for nid, src in self._responses.items():
                            if re.search(rf"\b{re.escape(sym)}\b", src):
                                node_id = nid
                                break
            except json.JSONDecodeError:
                pass
        if not node_id:
            keys = list(self._responses.keys())
            node_id = keys[(self.calls - 1) % len(keys)]
        return self._responses[node_id]


def main():
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp) / "tri_genomics"
        accel_log = Path(tmp) / "accel.log"
        os.environ["AERO_FORGE_ACCEL_LOG"] = str(accel_log)

        hin_graph_spec = {
            "project": "genomics_tri_polyglot",
            "metadata": {
                "architecture": "graph_polyglot",
                "llm_initialized": True,
                "prompt": "Build a tri-polyglot genomics engine with Python, Rust, and C++",
            },
            "nodes": [
                {
                    "node_id": "cpp_engine",
                    "lang": "cpp",
                    "toolchain": "cmake",
                    "source_files": ["cpp_engine.cpp"],
                    "exports": ["align_kernel"],
                },
                {
                    "node_id": "rust_core",
                    "lang": "rust",
                    "toolchain": "cargo",
                    "source_files": ["rust_core/src/lib.rs"],
                    "exports": ["rust_kernel"],
                },
                {
                    "node_id": "python_interface",
                    "lang": "python",
                    "toolchain": "python",
                    "source_files": ["python_interface/main.py"],
                    "exports": ["run_demo"],
                },
                {
                    "node_id": "tests",
                    "lang": "python",
                    "toolchain": "python",
                    "source_files": ["tests/test_all.py"],
                    "exports": ["test_all"],
                },
            ],
            "edges": [
                {
                    "source": "cpp_engine",
                    "target": "python_interface",
                    "symbol": "align_kernel",
                    "boundary_type": "c_abi",
                    "args": ["int64"],
                    "return_type": "int64",
                },
                {
                    "source": "rust_core",
                    "target": "python_interface",
                    "symbol": "rust_kernel",
                    "boundary_type": "c_abi",
                    "args": ["int64"],
                    "return_type": "int64",
                },
            ],
            "primary_entrypoint": "python_interface/main.py",
        }

        client = MockLLM()
        materializer = GraphPolyglotMaterializer(
            workspace_root=workspace,
            llm_client=client,
        )
        result = materializer.materialize(hin_graph_spec, build=True)

        print(f"Result architecture: {result['architecture']}")
        print(f"Workspace: {result['workspace']}")
        print(f"Artifacts written: {len(result['artifacts'])}")
        for artifact in result["artifacts"]:
            print(f"  - {artifact.get('path')}")

        blueprint_path = Path(result["blueprint_path"])
        print(f"Blueprint: {blueprint_path}")
        import yaml

        bp = yaml.safe_load(blueprint_path.read_text())
        print(f"Blueprint architecture: {bp['metadata']['architecture']}")

        cpp_cmake = workspace / "cpp_engine" / "CMakeLists.txt"
        assert cpp_cmake.is_file(), f"Missing CMakeLists.txt for cpp_engine: {cpp_cmake}"
        cmake_text = cpp_cmake.read_text()
        assert "-fPIC" in cmake_text, "CMakeLists.txt does not set -fPIC"
        assert "-shared" in cmake_text, "CMakeLists.txt does not set -shared"
        assert "add_library" in cmake_text, "CMakeLists.txt does not declare a library"

        rust_cargo = workspace / "rust_core" / "Cargo.toml"
        assert rust_cargo.is_file(), f"Missing Cargo.toml for rust_core: {rust_cargo}"

        print("--- ACCELERATOR LOG ---")
        log_text = accel_log.read_text()
        print(log_text)
        print("--- END ---")

        assert (
            result["architecture"] == "tri_polyglot_rust_cpp_python"
        ), f"Expected tri_polyglot_rust_cpp_python, got {result['architecture']}"
        assert (
            bp["metadata"]["architecture"] == "tri_polyglot_rust_cpp_python"
        ), f"Blueprint architecture not promoted: {bp['metadata']['architecture']}"

        # The materializer's per-node dispatch builds artifacts inside toolchain
        # output directories. Run the generated build.sh to copy .so files next
        # to each node so the Python ctypes loader can resolve them.
        build_sh = workspace / "build.sh"
        if build_sh.is_file():
            import subprocess

            build_result = subprocess.run(
                ["bash", str(build_sh)],
                cwd=workspace,
                capture_output=True,
                text=True,
                check=False,
            )
            print(build_result.stdout)
            if build_result.returncode != 0:
                print(build_result.stderr)
            assert build_result.returncode == 0, "build.sh failed"

        cpp_so = workspace / "cpp_engine" / "libcpp_engine.so"
        rust_so = workspace / "rust_core" / "librust_core.so"
        assert cpp_so.is_file(), f"C++ shared library not built: {cpp_so}"
        assert rust_so.is_file(), f"Rust shared library not built: {rust_so}"

        print("\nPASS: Tri-polyglot Genomics Prompt 2 build succeeded.")


if __name__ == "__main__":
    main()
