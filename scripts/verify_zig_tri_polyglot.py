#!/usr/bin/env python3
"""High-complexity tri-polyglot build verification with a Zig kernel.

Prompt: "Build a high-performance math kernel in Zig for calculating prime
numbers. Link this Zig kernel to a Python CLI via a C-ABI zero-copy bridge.
Ensure the engine JIT-synthesizes the Zig emitter and validates the boundary
contract."
"""

import json
import tempfile
from pathlib import Path

from aero_forge.builder.builder import ProactivePolyglotBuilder

PROMPT = (
    "Build a high-performance math kernel in Zig for calculating prime numbers. "
    "Link this Zig kernel to a Python CLI via a C-ABI zero-copy bridge. "
    "Ensure the engine JIT-synthesizes the Zig emitter and validates the boundary contract."
)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="aero_forge_zig_") as tmp:
        output_dir = Path(tmp)
        builder = ProactivePolyglotBuilder()
        result = builder.synthesize_and_build(
            prompt=PROMPT,
            output_dir=output_dir,
            llm_provider="deepseek",
            llm_model="deepseek-chat",
            max_retries=2,
        )
        print(json.dumps(result, indent=2, default=str))

        success = result.get("build", {}).get("success", False)
        if not success:
            print("\nBuild failed. See output above.")
            return 1

        # Verify the Zig kernel file was materialized.
        zig_files = list(output_dir.rglob("*.zig"))
        python_files = list(output_dir.rglob("*.py"))
        print(f"\nMaterialized Zig files: {zig_files}")
        print(f"Materialized Python files: {python_files}")
        if not zig_files:
            print("ERROR: No Zig source files were materialized.")
            return 1
        if not python_files:
            print("ERROR: No Python source files were materialized.")
            return 1

        print("\nZig tri-polyglot build verification passed.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
