#!/usr/bin/env python3
"""Stress test: tri-language (Rust + Go + Python) graph_polyglot build with high test density."""

import json
import os
import re
import sys
import tempfile
from pathlib import Path

# Ensure repo source is importable.
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from aero_forge.builder.materializers.graph_materializer import (
    GraphPolyglotMaterializer,
)


class MockLLM:
    """Deterministic LLM client that returns fenced source for each node."""

    def __init__(self):
        self.calls = 0
        self._responses = self._build_responses()

    @staticmethod
    def _build_responses():
        rust_src = '''
__AERO_LOGIC_START__
```rust:rust_core/src/lib.rs
pub fn process(x: i64) -> i64 {
    x * 2
}

#[no_mangle]
pub extern "C" fn process_c(x: i64) -> i64 {
    process(x)
}
```
__AERO_LOGIC_END__
'''
        go_src = '''
__AERO_LOGIC_START__
```go:go_core/main.go
package main

//export transform
func transform(x int) int {
    return x * 2
}

func main() {}
```
__AERO_LOGIC_END__
'''
        py_src = '''
__AERO_LOGIC_START__
```python:python_interface/main.py
def run_demo(x: int) -> int:
    return x * 2

if __name__ == "__main__":
    print(run_demo(3))
```
__AERO_LOGIC_END__
'''
        tests = '''
__AERO_LOGIC_START__
```python:tests/test_all.py
def tests():
    return True

# process: 5 tests per symbol
def test_process_success():
    assert process(2) == 4

def test_process_empty():
    assert process(0) == 0

def test_process_large():
    assert process(1000000) == 2000000

def test_process_negative():
    assert process(-3) == -6

def test_process_error():
    try:
        process("not an int")
    except TypeError:
        pass

# transform: 5 tests per symbol
def test_transform_success():
    assert transform(2) == 4

def test_transform_empty():
    assert transform(0) == 0

def test_transform_large():
    assert transform(1000000) == 2000000

def test_transform_negative():
    assert transform(-3) == -6

def test_transform_error():
    try:
        transform("not an int")
    except TypeError:
        pass

# run_demo: 5 tests per symbol
def test_run_demo_success():
    assert run_demo(2) == 4

def test_run_demo_empty():
    assert run_demo(0) == 0

def test_run_demo_large():
    assert run_demo(1000000) == 2000000

def test_run_demo_negative():
    assert run_demo(-3) == -6

def test_run_demo_error():
    try:
        run_demo("not an int")
    except TypeError:
        pass
```
__AERO_LOGIC_END__
'''
        return {
            "rust_core": rust_src,
            "go_core": go_src,
            "python_interface": py_src,
            "tests": tests,
        }

    def generate(self, messages, *, temperature=None, max_tokens=None):
        self.calls += 1
        user = messages[1]["content"]
        # The payload is a JSON code block; find the skeleton to infer node.
        m = re.search(r"```json\s*([\s\S]*?)```", user)
        node_id = ""
        if m:
            try:
                payload = json.loads(m.group(1))
                skeleton = payload.get("skeleton", "")
                # First line of the skeleton typically contains the node id comment.
                first = skeleton.splitlines()[0] if skeleton else ""
                for nid in self._responses:
                    if nid in first:
                        node_id = nid
                        break
                if not node_id:
                    # Fallback: required_symbols can identify the node.
                    for sym in payload.get("required_symbols", []):
                        for nid, src in self._responses.items():
                            if sym in src:
                                node_id = nid
                                break
                        if node_id:
                            break
            except json.JSONDecodeError:
                pass
        if not node_id:
            # Fallback to round-robin based on call count.
            keys = list(self._responses.keys())
            node_id = keys[(self.calls - 1) % len(keys)]
        return self._responses[node_id]


def main():
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp) / "tri_workspace"
        accel_log = Path(tmp) / "accel.log"
        os.environ["AERO_FORGE_ACCEL_LOG"] = str(accel_log)

        hin_graph_spec = {
            "project": "tri_stress",
            "metadata": {
                "architecture": "graph_polyglot",
                "llm_initialized": True,
                "prompt": "Build a Rust/Go/Python tri-language data pipeline",
            },
            "nodes": [
                {
                    "node_id": "rust_core",
                    "lang": "rust",
                    "source_files": ["rust_core/src/lib.rs"],
                    "exports": ["process"],
                },
                {
                    "node_id": "go_core",
                    "lang": "go",
                    "source_files": ["go_core/main.go"],
                    "exports": ["transform"],
                },
                {
                    "node_id": "python_interface",
                    "lang": "python",
                    "source_files": ["python_interface/main.py"],
                    "exports": ["run_demo"],
                },
                {
                    "node_id": "tests",
                    "lang": "python",
                    "source_files": ["tests/test_all.py"],
                    "exports": ["tests"],
                },
            ],
            "edges": [
                {
                    "source": "go_core",
                    "target": "python_interface",
                    "symbol": "transform",
                    "boundary_type": "cgo",
                    "args": ["int64"],
                    "return_type": "int64",
                }
            ],
            "primary_entrypoint": "python_interface/main.py",
        }

        client = MockLLM()
        materializer = GraphPolyglotMaterializer(
            workspace_root=workspace,
            llm_client=client,
        )
        result = materializer.materialize(hin_graph_spec, build=False)

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

        # Validate stage logs.
        log_text = accel_log.read_text()
        print("\n--- ACCELERATOR LOG ---")
        print(log_text)
        print("--- END ACCELERATOR LOG ---\n")

        assert "Stage 1:" in log_text, "Wavefront stage name missing from ACCELERATOR LOG"
        assert "Stage 2:" in log_text, "Second wavefront stage missing from ACCELERATOR LOG"
        assert "Rust Core Initialization" in log_text, "Rust stage label missing"
        assert "Go Boundary Synthesis" in log_text, "Go stage label missing"
        assert "Python Driver Synthesis" in log_text, "Python stage label missing"

        assert result["architecture"] == "graph_polyglot", f"Expected graph_polyglot, got {result['architecture']}"
        assert bp["metadata"]["architecture"] == "graph_polyglot", "Blueprint architecture not elevated"

        # Verify test density.
        tests_dir = workspace / "tests"
        test_files = list(tests_dir.glob("test_*.py"))
        test_functions = []
        for tf in test_files:
            tree = __import__("ast").parse(tf.read_text())
            for node in __import__("ast").walk(tree):
                if isinstance(node, __import__("ast").FunctionDef) and node.name.startswith("test_"):
                    test_functions.append(f"{tf.name}::{node.name}")
        print(f"Test functions found: {len(test_functions)}")
        assert len(test_functions) >= 15, f"Expected >=15 tests, found {len(test_functions)}"

        print("\nPASS: Tri-language graph_polyglot build passed test-density and wavefront logging gates.")


if __name__ == "__main__":
    main()
