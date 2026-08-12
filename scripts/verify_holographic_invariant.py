#!/usr/bin/env python3
"""Verification script for Holographic Invariant Storage (HIS).

This demonstrates the HIS backend with a genomics-style functional intent.
It builds an invariant from a prompt's required symbols and then measures drift
against a "hollow" blueprint that is missing the core algorithm. The intent is
to show that the `IntentCompiler` can now quantify how far an LLM-generated
blueprint has drifted from the requested functional intent.
"""

from aero_forge.builder.holographic import HolographicContext


def main() -> int:
    prompt_symbols = [
        "smith_waterman",
        "sequence_alignment",
        "scoring_matrix",
        "aligner",
        "main",
    ]
    full_blueprint_symbols = [
        "smith_waterman",
        "sequence_alignment",
        "scoring_matrix",
        "aligner",
        "main",
        "tests",
    ]
    hollow_blueprint_symbols = ["main"]

    ctx = HolographicContext(seed=0xA3A0)
    ctx.build_invariant_from_symbols(prompt_symbols)

    full_drift = ctx.measure_symbol_drift(full_blueprint_symbols)
    hollow_drift = ctx.measure_symbol_drift(hollow_blueprint_symbols)

    print(f"Prompt symbols:        {prompt_symbols}")
    print(f"Full blueprint drift:  {full_drift:.4f}")
    print(f"Hollow blueprint drift: {hollow_drift:.4f}")

    assert full_drift > hollow_drift, "full blueprint should drift less than hollow"
    assert full_drift > 0.6, "full blueprint should align strongly with the invariant"
    assert hollow_drift < 0.5, "hollow blueprint should show significant drift"
    print("HIS verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
