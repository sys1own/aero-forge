"""Deterministic verification of static data symbol materialization.

This script simulates an LLM for the Genomics Prompt 1 (pure_python
bioinformatics library) and verifies that the Proactive Formal Synthesis Engine:

1. Flags ``blosum62`` as a mandatory data payload in the Compacted Functional Matrix.
2. Emits the data constant as a non-trivial top-level dictionary.
3. Passes Atomic Symbol Assembly for both ``blosum62`` and ``smith_waterman``.
4. Produces a runnable ``main.py`` CLI entrypoint.
"""

from __future__ import annotations

import json
import os
import re
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


def _source_response(source: str, path: str) -> str:
    return (
        "__AERO_LOGIC_START__\n"
        f"```python:{path}\n"
        f"{source}\n"
        "```\n"
        "__AERO_LOGIC_END__"
    )


_ALIGNER_SOURCE = '''\
blosum62 = {
    "A": {"A": 2, "C": -1, "G": -1, "T": -1},
    "C": {"A": -1, "C": 2, "G": -1, "T": -1},
    "G": {"A": -1, "C": -1, "G": 2, "T": -1},
    "T": {"A": -1, "C": -1, "G": -1, "T": 2},
}


def smith_waterman(seq1: str, seq2: str) -> int:
    gap = -2
    m = len(seq1)
    n = len(seq2)
    score = []
    for i in range(m + 1):
        row = []
        for j in range(n + 1):
            row.append(0)
        score.append(row)
    max_score = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match = score[i - 1][j - 1] + blosum62[seq1[i - 1]][seq2[j - 1]]
            delete = score[i - 1][j] + gap
            insert = score[i][j - 1] + gap
            new_score = max(0, match, delete, insert)
            score[i][j] = new_score
            if new_score > max_score:
                max_score = new_score
    return max_score
'''


_MAIN_SOURCE = '''\
import sys
from genomics.aligner import smith_waterman


def main() -> None:
    if len(sys.argv) < 3:
        seq1 = "ACGT"
        seq2 = "ACGT"
    else:
        seq1 = sys.argv[1]
        seq2 = sys.argv[2]
    print(smith_waterman(seq1, seq2))


if __name__ == "__main__":
    main()
'''


class _GraphMockLLM:
    """Return a valid graph_polyglot blueprint JSON for the genomics prompt."""

    def generate(self, messages: Any, **kwargs: Any) -> str:
        return json.dumps(
            {
                "project": "genomics_alignment",
                "architecture": "pure_python",
                "primary_entrypoint": "main.py",
                "build_script": "build.sh",
                "nodes": [
                    {
                        "node_id": "genomics",
                        "lang": "python",
                        "toolchain": "python",
                        "source_files": ["genomics/aligner.py"],
                        "exports": ["smith_waterman", "blosum62"],
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
    """Return the correct source depending on which node is being synthesized."""

    def generate(self, messages: Any, **kwargs: Any) -> str:
        content = messages[1]["content"]
        # The compacted context contains every symbol; the skeleton isolates
        # the symbols that belong to the current node.
        skeleton = content
        match = re.search(r"```json\s*([\s\S]*?)\s*```", content)
        if match:
            try:
                payload = json.loads(match.group(1))
                skeleton = payload.get("skeleton", content)
            except json.JSONDecodeError:
                pass
        if "def smith_waterman" in skeleton or "blosum62" in skeleton:
            return _source_response(_ALIGNER_SOURCE, "genomics/aligner.py")
        return _source_response(_MAIN_SOURCE, "main.py")


def main() -> int:
    log_path = Path(tempfile.gettempdir()) / "aero_genomics.log"
    os.environ["AERO_FORGE_ACCEL_LOG"] = str(log_path)
    if log_path.exists():
        log_path.unlink()

    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp) / "genomics_workspace"
        workspace.mkdir()

        prompt = (
            "Build a pure_python bioinformatics library for DNA sequence alignment. "
            "Implement the Smith-Waterman local alignment algorithm using standard nested "
            "list[list[int]] for the scoring matrix. The project must have a clean API in "
            "genomics/aligner.py and a CLI entrypoint in main.py that accepts two DNA strings "
            "and reports the optimal alignment score."
        )

        compiler = IntentCompiler(llm_client=_GraphMockLLM())
        graph = compiler.compile_prompt_to_graph(prompt, output_dir=workspace)

        materializer = GraphPolyglotMaterializer(workspace, llm_client=_MockLLM())
        materializer.materialize(graph.model_dump(mode="json"), build=False)

        aligner_py = workspace / "genomics" / "aligner.py"
        main_py = workspace / "main.py"
        blueprint = workspace / "blueprint.aero"

        for path in (aligner_py, main_py, blueprint):
            if not path.exists():
                print(f"FAIL: {path.relative_to(workspace)} was not materialized")
                return 1

        blueprint_size = blueprint.stat().st_size
        print(f"blueprint.aero size: {blueprint_size} bytes")
        if blueprint_size <= 3000:
            print("FAIL: blueprint.aero did not grow beyond 3000 bytes")
            return 1

        aligner_source = aligner_py.read_text()
        print("--- emitted genomics/aligner.py ---")
        print(aligner_source)
        print("--- end genomics/aligner.py ---")

        if "blosum62" not in aligner_source.lower():
            print("FAIL: blosum62 data constant missing from genomics/aligner.py")
            return 1
        if "def smith_waterman" not in aligner_source:
            print("FAIL: smith_waterman alignment logic missing from genomics/aligner.py")
            return 1

        log = log_path.read_text() if log_path.exists() else ""
        print("--- accelerator log ---")
        print(log)
        print("--- end accelerator log ---")

        for expected in (
            "Enriching Blueprint...",
            "Materializing Symbol: blosum62",
            "Materializing Symbol: smith_waterman",
            "HIN Node Saturation Verified: blosum62",
            "Atomic Symbol Assembly verified for genomics",
            "Contract Integrity Verified",
        ):
            if expected not in log:
                print(f"FAIL: accelerator log missing '{expected}'")
                return 1

        blueprint_text = blueprint.read_text()
        if "llm_initialized: true" not in blueprint_text:
            print("FAIL: blueprint.aero does not report llm_initialized: true")
            return 1

        env = os.environ.copy()
        env["PYTHONPATH"] = str(workspace)
        result = subprocess.run(
            [sys.executable, str(main_py), "ACGT", "AGCT"],
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
        try:
            score = int(result.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            print("FAIL: main.py did not print an integer alignment score")
            return 1
        if score <= 0:
            print(f"FAIL: expected a positive alignment score, got {score}")
            return 1

        print(
            f"PASS: Genomics Prompt 1 materialized with blosum62 and smith_waterman "
            f"(blueprint={blueprint_size} bytes, score={score})."
        )
        return 0


if __name__ == "__main__":
    sys.exit(main())
