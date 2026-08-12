#!/usr/bin/env python3
"""Verification script for the Six-Phase IntentCompiler pipeline.

Exercises the deterministic stages without calling an LLM:
1. Context binding (HIS)
2. Topology mapping (FoGE)
3. Category-theoretic bootstrap (Adjoint)
4. Bounded intent completion prompt materialization
5. Concolic feedback (Z3)
6. SHACL firewall + Prolog/Chiasmus verification
"""

import tempfile
from pathlib import Path

from aero_forge.builder.intent_compiler import IntentCompiler


def _stage_1_2_3(compiler: IntentCompiler, output_dir: Path) -> None:
    classification = {
        "architecture": "hybrid_rust_python",
        "functional_intent": [
            {"symbol_name": "calculate_var", "type": "algorithm", "requirement_level": "required"},
            {"symbol_name": "analyze_portfolio", "type": "function", "requirement_level": "required"},
        ],
    }
    hctx = compiler._six_phase_bind_context(classification)
    topology = compiler._six_phase_topology_prefix(output_dir)
    skeleton = compiler._six_phase_bootstrap_skeleton(classification, topology)

    assert hctx.hinv is not None, "HIS invariant must be bound"
    assert topology.get("encoded") or not (output_dir / "main.py").is_file(), "topology result must be present"
    assert skeleton.get("nodes"), "Adjoint skeleton must contain nodes"
    print("[Phase 1-3] HIS bound, topology encoded, skeleton bootstrapped.")


def _stage_4(compiler: IntentCompiler, classification: dict, hctx, topology, skeleton) -> str:
    prompt = compiler._six_phase_user_content(
        "Build a financial risk analyzer.", classification, hctx, topology, skeleton
    )
    assert "Manifest skeleton" in prompt, "bounded prompt must include skeleton"
    assert "Topological prefix" in prompt, "bounded prompt must include topology"
    print("[Phase 4] Bounded completion prompt assembled.")
    return prompt


def _stage_5_6(compiler: IntentCompiler, bad_manifest: dict, output_dir: Path) -> None:
    feedback = compiler._six_phase_formal_feedback(bad_manifest, output_dir)
    assert feedback, "formal feedback must detect the bad manifest"
    assert "boundary" in feedback.lower() or "SHACL" in feedback, "feedback must reference a boundary/SHACL violation"
    print("[Phase 5-6] Concolic/SHACL/Chiasmus feedback generated.")


def main() -> int:
    compiler = IntentCompiler(llm_client=None)

    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp)
        (output_dir / "main.py").write_text("import rust_core\n")

        _stage_1_2_3(compiler, output_dir)

        classification = {
            "architecture": "hybrid_rust_python",
            "functional_intent": [
                {"symbol_name": "calculate_var", "type": "algorithm"},
                {"symbol_name": "analyze_portfolio", "type": "function"},
            ],
        }
        hctx = compiler._six_phase_bind_context(classification)
        topology = compiler._six_phase_topology_prefix(output_dir)
        skeleton = compiler._six_phase_bootstrap_skeleton(classification, topology)
        _stage_4(compiler, classification, hctx, topology, skeleton)

        # A manifest with an invalid C-ABI boundary for Rust/Python.
        bad_manifest = {
            "architecture": "hybrid_rust_python",
            "nodes": [
                {"node_id": "main", "lang": "python", "toolchain": "python", "exports": ["main"]},
                {"node_id": "rust_core", "lang": "rust", "toolchain": "cargo", "exports": ["calculate_var"]},
            ],
            "edges": [{"source": "main", "target": "rust_core", "boundary_type": "C_ABI"}],
            "functional_intent": [{"symbol_name": "main"}],
        }
        _stage_5_6(compiler, bad_manifest, output_dir)

    print("Six-phase pipeline verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
