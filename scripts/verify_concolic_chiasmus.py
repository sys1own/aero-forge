#!/usr/bin/env python3
"""Verification script for the Logitext/Cottontail feedback loop.

Combines ``concolic.py`` (Z3 manifest verification) and ``chiasmus.py``
(Tree-sitter -> Prolog facts) to produce formal feedback for an LLM when a
generated manifest is inconsistent or the repository has unsafe transitions.
"""

import tempfile
from pathlib import Path

from aero_forge.builder.chiasmus import LogicEngine, PrologFactEmitter, RefinementFeedback
from aero_forge.builder.concolic import ConcolicManifestVerifier


def _create_bad_repo(tmp: Path) -> Path:
    """Create a repo with a Python -> Rust import missing a proper boundary."""
    (tmp / "main.py").write_text('import rust_core\nrust_core.calc()\n')
    (tmp / "rust_core.rs").write_text('pub fn calc() {}\n')
    return tmp


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        repo = _create_bad_repo(Path(tmp))

        # Chiasmus: emit facts and detect unsafe FFI.
        facts = PrologFactEmitter().emit_facts(repo)
        engine = LogicEngine()
        engine.load_facts(facts)
        unsafe = engine.unsafe_ffi_transitions()

        # Concolic: verify a manifest that claims a C-ABI boundary for Rust/Python.
        manifest = {
            "architecture": "hybrid_rust_python",
            "nodes": [
                {"node_id": "main", "lang": "python", "toolchain": "python"},
                {"node_id": "rust_core", "lang": "rust", "toolchain": "cargo"},
            ],
            "edges": [
                {"source": "main", "target": "rust_core", "boundary_type": "C_ABI"}
            ],
            "functional_intent": [],
        }
        verifier = ConcolicManifestVerifier(manifest)
        result = verifier.verify()

        feedback = RefinementFeedback(
            unsat_core=result.conflicting_rules,
            unsafe_ffi=unsafe,
            summary="Formal verification detected a boundary mismatch.",
        )
        print(feedback.to_llm_message())

        assert not result.satisfiable, "C-ABI boundary for Rust/Python must be UNSAT"
        assert unsafe, "Python->Rust import without boundary must be flagged"
        print("\nConcolic/Chiasmus verification passed.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
