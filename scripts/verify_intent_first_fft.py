"""Deterministic verification of the Intent-First UAST materialization path.

This script simulates an LLM that returns a UAST JSON sketch for a pure_python
FFT implementation. The engine must:

1. Parse the UAST sketch.
2. Run the SMT attribute resolver (``conj`` -> ``conjugate``).
3. Lower the sketch to Python source with HIN AST normalization.
4. Pass syntax, GoI, and HIN active-pair verification.
5. Emit ``main.py`` containing ``.conjugate()`` even though the sketch
   contained ``cmath.conj(twiddle)``.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aero_forge.builder.materializers.graph_materializer import (
    GraphPolyglotMaterializer,
)


def _make_llm_client():
    """Return a mock client that emits a UAST sketch with ``cmath.conj``."""
    uast = {
        "type": "Module",
        "body": [
            {
                "type": "FunctionDef",
                "name": "fft",
                "args": {
                    "type": "arguments",
                    "args": [{"type": "arg", "arg": "signal"}],
                },
                "body": [
                    {
                        "type": "Assign",
                        "targets": [{"type": "Name", "id": "n"}],
                        "value": {
                            "type": "Call",
                            "func": {"type": "Name", "id": "len"},
                            "args": [{"type": "Name", "id": "signal"}],
                        },
                    },
                    {
                        "type": "If",
                        "test": {
                            "type": "Compare",
                            "left": {"type": "Name", "id": "n"},
                            "ops": [{"type": "LtE"}],
                            "comparators": [{"type": "Constant", "value": 1}],
                        },
                        "body": [
                            {"type": "Return", "value": {"type": "Name", "id": "signal"}}
                        ],
                        "orelse": [],
                    },
                    {
                        "type": "Assign",
                        "targets": [{"type": "Name", "id": "twiddle"}],
                        "value": {
                            "type": "Call",
                            "func": {"type": "Name", "id": "complex"},
                            "args": [
                                {"type": "Constant", "value": 0.0},
                                {"type": "Constant", "value": 1.0},
                            ],
                        },
                    },
                    {
                        "type": "Return",
                        "value": {
                            "type": "Call",
                            "func": {
                                "type": "Attribute",
                                "value": {"type": "Name", "id": "cmath"},
                                "attr": "conj",
                            },
                            "args": [{"type": "Name", "id": "twiddle"}],
                        },
                    },
                ],
                "decorator_list": [],
                "returns": None,
            }
        ],
    }
    response = (
        "__AERO_LOGIC_START__\n"
        "```uast:main.py\n"
        f"{json.dumps(uast)}\n"
        "```\n"
        "__AERO_LOGIC_END__"
    )
    client = MagicMock()
    client.generate.return_value = response
    return client


def main() -> int:
    log_path = Path(tempfile.gettempdir()) / "aero_intent_first_fft.log"
    os.environ["AERO_FORGE_ACCEL_LOG"] = str(log_path)

    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp) / "fft_workspace"
        workspace.mkdir()

        spec = {
            "project": "intent_first_fft",
            "metadata": {
                "architecture": "pure_python",
                "prompt": "Build a pure_python FFT library.",
            },
            "architecture": "pure_python",
            "primary_entrypoint": "main.py",
            "build_script": "build.sh",
            "nodes": [
                {
                    "node_id": "main",
                    "lang": "python",
                    "toolchain": "python",
                    "source_files": ["main.py"],
                }
            ],
            "edges": [],
        }

        materializer = GraphPolyglotMaterializer(
            workspace, llm_client=_make_llm_client()
        )
        materializer.materialize(spec, build=False)

        main_py = workspace / "main.py"
        blueprint = workspace / "blueprint.aero"
        if not main_py.exists():
            print("FAIL: main.py was not materialized")
            return 1

        source = main_py.read_text()
        print("--- emitted main.py ---")
        print(source)
        print("--- end main.py ---")

        if "conj" in source and ".conjugate()" not in source:
            print("FAIL: LLM attribute 'conj' was not resolved to 'conjugate'")
            return 1

        if ".conjugate()" not in source:
            print("FAIL: expected .conjugate() in emitted source")
            return 1

        import py_compile
        py_compile.compile(str(main_py), doraise=True)

        if not blueprint.exists():
            print("FAIL: blueprint.aero was not generated")
            return 1

        blueprint_text = blueprint.read_text()
        if "llm_initialized: true" not in blueprint_text:
            print("FAIL: blueprint.aero does not report llm_initialized: true")
            return 1

        log = log_path.read_text() if log_path.exists() else ""
        if "HIN verification passed" not in log:
            print("FAIL: HIN verification was not recorded in the accelerator log")
            return 1

        print("PASS: Intent-first FFT materialization emits .conjugate() and passes verification.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
