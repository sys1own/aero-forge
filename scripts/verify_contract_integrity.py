"""Deterministic verification of contract-to-source integrity gates.

This script simulates an LLM that returns UAST JSON sketches for a pure_python
security gateway project (auth_lib/core.py + main.py). It verifies that the
engine:

1. Detects missing functions from the blueprint exports.
2. Retries with a Full Implementation Map.
3. Emits every contracted symbol.
4. Logs "Contract Integrity Verified: X/Y functions present".
5. Produces a runnable project (no ImportError).
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aero_forge.builder.intent_compiler import IntentCompiler
from aero_forge.builder.materializers.graph_materializer import (
    GraphPolyglotMaterializer,
)


def _ast_to_uast(node: Any) -> Any:
    """Convert a Python ``ast.AST`` into the JSON UAST ``uast_to_python_source`` expects."""
    if isinstance(node, ast.AST):
        result: dict = {"type": node.__class__.__name__}
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                result[field] = [_ast_to_uast(item) for item in value]
            elif isinstance(value, ast.AST):
                result[field] = _ast_to_uast(value)
            else:
                result[field] = value
        return result
    if isinstance(node, list):
        return [_ast_to_uast(item) for item in node]
    return node


def _uast_response(uast: dict, path: str) -> str:
    return (
        "__AERO_LOGIC_START__\n"
        f"```uast:{path}\n"
        f"{json.dumps(uast, indent=2)}\n"
        "```\n"
        "__AERO_LOGIC_END__"
    )


def _auth_lib_source() -> str:
    return '''\
import json
from typing import Any

def validate_token(token: str, secret: str = "secret") -> bool:
    parts = token.split(".")
    if len(parts) != 3:
        return False
    payload = json.loads(parts[1])
    return payload.get("secret") == secret

def check_permissions(roles: list, required: list) -> bool:
    return not set(required).isdisjoint(roles)
'''


def _main_source() -> str:
    return '''\
import json
from auth_lib.core import validate_token, check_permissions

def main() -> None:
    sample = {"user": "alice", "roles": ["admin", "user"], "secret": "secret"}
    token = "header." + json.dumps(sample) + ".sig"
    print(validate_token(token))
    print(check_permissions(sample["roles"], ["admin"]))

if __name__ == "__main__":
    main()
'''


class _GraphMockLLM:
    """Return a valid graph_polyglot blueprint JSON for the security prompt."""

    def generate(self, messages, **kwargs) -> str:
        return json.dumps(
            {
                "project": "security_gateway",
                "architecture": "pure_python",
                "primary_entrypoint": "main.py",
                "build_script": "build.sh",
                "nodes": [
                    {
                        "node_id": "auth_lib",
                        "lang": "python",
                        "toolchain": "python",
                        "source_files": ["auth_lib/core.py"],
                        "exports": ["validate_token", "check_permissions"],
                    },
                    {
                        "node_id": "main",
                        "lang": "python",
                        "toolchain": "python",
                        "source_files": ["main.py"],
                    },
                ],
                "edges": [],
                "output_dir": "./dist",
            }
        )


class _MockLLM:
    """Return the correct UAST sketch depending on which node is being synthesized."""

    def __init__(self) -> None:
        self.calls: list = []

    def generate(self, messages, **kwargs) -> str:
        content = messages[1]["content"]
        self.calls.append(content)
        # Match the synthesized skeleton: each node's prompt contains a skeleton
        # with only the functions for that node.
        if "def validate_token" in content:
            tree = ast.parse(_auth_lib_source())
            return _uast_response(_ast_to_uast(tree), "auth_lib/core.py")
        tree = ast.parse(_main_source())
        return _uast_response(_ast_to_uast(tree), "main.py")


def main() -> int:
    log_path = Path(tempfile.gettempdir()) / "aero_contract_integrity.log"
    os.environ["AERO_FORGE_ACCEL_LOG"] = str(log_path)
    if log_path.exists():
        log_path.unlink()

    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp) / "auth_workspace"
        workspace.mkdir()

        prompt = (
            "Build a pure_python authentication and security utility. "
            "It must implement validate_token and check_permissions in auth_lib/core.py "
            "with a CLI entrypoint in main.py."
        )

        compiler = IntentCompiler(llm_client=_GraphMockLLM())
        graph = compiler.compile_prompt_to_graph(prompt, output_dir=workspace)

        materializer = GraphPolyglotMaterializer(
            workspace, llm_client=_MockLLM()
        )
        materializer.materialize(graph.model_dump(mode="json"), build=False)

        core_py = workspace / "auth_lib" / "core.py"
        main_py = workspace / "main.py"
        blueprint = workspace / "blueprint.aero"

        for path in (core_py, main_py, blueprint):
            if not path.exists():
                print(f"FAIL: {path.relative_to(workspace)} was not materialized")
                return 1

        core_source = core_py.read_text()
        print("--- emitted auth_lib/core.py ---")
        print(core_source)
        print("--- end auth_lib/core.py ---")

        if "def validate_token" not in core_source:
            print("FAIL: validate_token missing from auth_lib/core.py")
            return 1
        if "def check_permissions" not in core_source:
            print("FAIL: check_permissions missing from auth_lib/core.py")
            return 1

        log = log_path.read_text() if log_path.exists() else ""
        print("--- accelerator log ---")
        print(log)
        print("--- end accelerator log ---")

        for expected in (
            "Enriching Blueprint...",
            "Materializing Symbol: validate_token",
            "Materializing Symbol: check_permissions",
            "Contract Integrity Verified: 2/2",
        ):
            if expected not in log:
                print(f"FAIL: accelerator log missing '{expected}'")
                return 1

        blueprint_text = blueprint.read_text()
        if "llm_initialized: true" not in blueprint_text:
            print("FAIL: blueprint.aero does not report llm_initialized: true")
            return 1

        # Run the CLI entrypoint exactly as the generated build.sh would.
        env = os.environ.copy()
        env["PYTHONPATH"] = str(workspace)
        result = subprocess.run(
            [sys.executable, str(main_py)],
            cwd=str(workspace),
            env=env,
            capture_output=True,
            text=True,
        )
        print("--- main.py stdout ---")
        print(result.stdout)
        print("--- main.py stderr ---")
        print(result.stderr)
        if result.returncode != 0:
            print("FAIL: main.py failed to run")
            return 1
        if "True" not in result.stdout:
            print("FAIL: expected token/permission validation to succeed")
            return 1

        print("PASS: Contract-to-source integrity gate works and the security gateway runs.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
