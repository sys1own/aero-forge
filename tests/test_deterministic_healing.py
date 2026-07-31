"""Tests for the deterministic proof-theoretic self-healing engine."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from aero_forge._native import (
    enforce_repair_isolation_py,
    evaluate_hin_energy,
    repair_uast_expression,
)
from aero_forge.blueprint import ContractEntry
from aero_forge.healing.healer import ContractSynthesizer, DeterministicHealer


def _arena(*nodes: dict) -> str:
    """Serialize node dicts to the arena JSON format expected by evaluate_hin_energy."""
    return json.dumps(list(nodes))


def test_hin_energy_zero_on_valid_graph() -> None:
    """A reduced (empty) HIN graph has zero stalled, wires, dangling and E(G)=0."""
    result = json.loads(evaluate_hin_energy("[]"))
    assert result["stalled"] == 0
    assert result["wires"] == 0
    assert result["dangling"] == 0
    assert result["total"] == 0.0


def test_hin_energy_defect_detection() -> None:
    """Stalled active principal-principal pairs and dangling ports raise E(G)."""
    # Two Value nodes wired principal-to-principal: same kind => stalled active pair.
    stalled = json.loads(
        evaluate_hin_energy(
            _arena(
                {
                    "id": "a",
                    "kind": "Value",
                    "ports": [
                        {
                            "name": "p",
                            "is_principal": True,
                            "target_node": "b",
                            "target_port": "p",
                        }
                    ],
                },
                {
                    "id": "b",
                    "kind": "Value",
                    "ports": [
                        {
                            "name": "p",
                            "is_principal": True,
                            "target_node": "a",
                            "target_port": "p",
                        }
                    ],
                },
            )
        )
    )
    assert stalled["stalled"] == 1
    assert stalled["wires"] == 1
    assert stalled["dangling"] == 0
    assert stalled["total"] == 15.0

    # A single Value node with an unconnected principal port => dangling.
    dangling = json.loads(
        evaluate_hin_energy(
            _arena(
                {
                    "id": "a",
                    "kind": "Value",
                    "ports": [{"name": "p", "is_principal": True}],
                }
            )
        )
    )
    assert dangling["stalled"] == 0
    assert dangling["wires"] == 0
    assert dangling["dangling"] == 1
    assert dangling["total"] == 2.0


def test_egraph_ast_rewrite_performance() -> None:
    """repair_uast_expression optimizes x + 0 -> x in well under 1.0 ms."""
    expr = json.dumps(
        {
            "type": "call",
            "function": "+",
            "arguments": [
                {"type": "reference", "name": "x"},
                {"type": "literal", "value": 0},
            ],
        }
    )

    # Warmup.
    _ = repair_uast_expression(expr)

    iterations = 1000
    start = time.perf_counter()
    for _ in range(iterations):
        out = repair_uast_expression(expr)
    elapsed = (time.perf_counter() - start) / iterations

    assert elapsed < 1.0e-3, f"average rewrite took {elapsed:.6f}s"
    assert json.loads(out) == {"type": "reference", "name": "x"}


def test_enforce_repair_isolation_py_zero_perturbation() -> None:
    """A zero perturbation on a stable identity/zero matrix is isolated."""
    base = json.dumps(
        {
            "U": [[1.0, 0.0], [0.0, 1.0]],
            "M": [[0.0, 0.0], [0.0, 0.0]],
        }
    )
    delta = json.dumps([[0.0, 0.0], [0.0, 0.0]])
    result = json.loads(enforce_repair_isolation_py(base, delta))
    assert result["isolated"] is True
    assert result["radius"] < 1.0
    assert result["bound"] == 1.0
    assert result["support_rows"] == []
    assert result["support_cols"] == []


def test_contract_synthesizer_emits_canonical_wrappers() -> None:
    """ContractSynthesizer produces the three canonical FFI morphism templates."""
    contracts = [
        ContractEntry(
            name="add",
            signature="add(a: float, b: float) -> float",
            language="python",
            python_name="add",
        )
    ]
    synth = ContractSynthesizer(contracts)

    pyo3 = synth.synthesize_missing_morphism("add", "python/rust")
    assert pyo3["language_pair"] == "python/rust"
    assert "#[pyfunction]" in pyo3["rust_stub"]
    assert "aero_forge_native import add" in pyo3["python_stub"]

    c_abi = synth.synthesize_missing_morphism("add", "rust/cpp")
    assert c_abi["language_pair"] == "rust/cpp"
    assert "#[no_mangle]" in c_abi["rust_stub"]
    assert 'extern "C"' in c_abi["cpp_header"]
    assert "ctypes" in c_abi["python_stub"]

    zero_copy = synth.synthesize_missing_morphism("add", "rust/rust")
    assert zero_copy["language_pair"] == "rust/rust"
    assert "AeroZeroCopy" in zero_copy["rust_stub"]


def test_integrated_healing_pipeline(tmp_path: Path) -> None:
    """DeterministicHealer detects a missing import, rewrites AST, and patches the file."""
    source = tmp_path / "main.py"
    source.write_text("print(math.sqrt(16))\n", encoding="utf-8")

    log_text = (
        "Traceback (most recent call last):\n"
        '  File "main.py", line 1, in <module>\n'
        "    print(math.sqrt(16))\n"
        "NameError: name 'math' is not defined\n"
    )

    healer = DeterministicHealer(tmp_path)
    result = healer.heal(
        error_logs=log_text,
        command="python main.py",
        exit_code=1,
    )

    assert result["status"] == "success"
    assert result["strategy_used"] == "ast"
    assert result["target_file"] == "main.py"
    assert result["patched_files"] == ["main.py"]
    assert "import math" in result["diff"]
    assert source.read_text(encoding="utf-8").startswith("import math")


def test_integrated_healing_pipeline_with_goi_validation(tmp_path: Path) -> None:
    """A stable GoI boundary is validated before AST repair proceeds."""
    source = tmp_path / "main.py"
    source.write_text("print(math.sqrt(16))\n", encoding="utf-8")

    log_text = "NameError: name 'math' is not defined\n"
    base_matrix = {"U": [[1.0, 0.0], [0.0, 1.0]], "M": [[0.0, 0.0], [0.0, 0.0]]}
    delta_matrix = [[0.0, 0.0], [0.0, 0.0]]

    healer = DeterministicHealer(tmp_path)
    result = healer.execute_healing_pass(
        error_log=log_text,
        source_text=source.read_text(encoding="utf-8"),
        source_path=Path("main.py"),
        base_matrix=base_matrix,
        delta_matrix=delta_matrix,
        apply=True,
    )

    assert result["status"] == "success"
    assert result["goi_result"]["isolated"] is True
    assert result["strategy_used"] == "ast"
    assert "import math" in result["diff"]
