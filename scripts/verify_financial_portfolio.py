#!/usr/bin/env python3
"""Verification: Financial Portfolio Risk Analyzer structural enrichment.

Confirms that the tiered intent enrichment and structural integrity guard reject a
``pure_python`` misclassification for a hybrid Rust/Python prompt and produce a
valid, non-collapsed ``PolyglotGraphBlueprint`` with the required Rust symbols.
"""

import sys
import tempfile
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from aero_forge.builder.intent_compiler import IntentCompiler
from aero_forge.blueprint.schema import PolyglotGraphBlueprint

PROMPT = (
    "Build a financial portfolio risk analyzer using a hybrid_rust_python architecture. "
    "The project must implement a Monte Carlo simulation kernel in Rust to calculate "
    "Value at Risk (VaR) for large portfolios. Provide a clean Python API and a CLI "
    "entrypoint in main.py that accepts a CSV of positions and returns risk metrics. "
    "Include unit tests for every exported symbol."
)

REQUIRED_SYMBOLS = {
    "monte_carlo_simulate",
    "calculate_var",
    "portfolio_risk",
    "var",
    "risk_metrics",
}


def main() -> int:
    compiler = IntentCompiler(
        provider="deepseek",
        model="deepseek-chat",
        max_schema_retries=3,
    )
    output_dir = Path(tempfile.mkdtemp(prefix="aero_forge_financial_portfolio_"))
    graph = compiler.compile_prompt_to_graph(PROMPT, output_dir=output_dir)

    assert isinstance(graph, PolyglotGraphBlueprint), "Expected PolyglotGraphBlueprint"
    assert (
        graph.architecture == "hybrid_rust_python"
    ), f"Expected hybrid_rust_python, got {graph.architecture!r}"
    assert graph.nodes, "Graph must contain at least one node"
    node_langs = {n.lang for n in graph.nodes}
    assert "rust" in node_langs, f"Expected a Rust node, got {node_langs!r}"
    assert "python" in node_langs, f"Expected a Python node, got {node_langs!r}"

    functional_names = {f.symbol_name for f in graph.functional_intent}
    node_exports = set()
    for node in graph.nodes:
        node_exports.update(node.exports or [])
        node_exports.add(node.node_id)
    edge_symbols = {e.symbol for e in graph.edges}
    covered = functional_names & (node_exports | edge_symbols)
    assert covered, "At least one functional_intent symbol must appear in nodes/edges"
    assert (REQUIRED_SYMBOLS & covered) or (
        REQUIRED_SYMBOLS & node_exports
    ), f"Expected at least one of {REQUIRED_SYMBOLS} in nodes or covered intents"

    print(
        f"OK: architecture={graph.architecture}, nodes={len(graph.nodes)}, "
        f"edges={len(graph.edges)}, functional_intent={len(graph.functional_intent)}"
    )
    print(f"Workspace: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
