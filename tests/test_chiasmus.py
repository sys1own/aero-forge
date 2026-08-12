"""Unit tests for Chiasmus Tree-sitter -> Prolog facts engine."""

import tempfile
from pathlib import Path

from aero_forge.builder.chiasmus import (
    LogicEngine,
    PrologFactEmitter,
    RefinementFeedback,
    analyze_repository,
)


def test_emit_python_facts(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text(
        "from aero_forge.core import run\n\ndef main():\n    run()\n"
    )
    emitter = PrologFactEmitter()
    facts = emitter.emit_facts(tmp_path)
    assert any('imports("main", "aero_forge.core.run")' in f for f in facts)
    assert any('exports("main", "main")' in f for f in facts)
    assert any('calls("main.main", "run")' in f for f in facts)


def test_cycle_detection():
    facts = [
        'imports("a", "b").',
        'imports("b", "c").',
        'imports("c", "a").',
    ]
    engine = LogicEngine()
    engine.load_facts(facts)
    cycles = engine.detect_cycles()
    assert len(cycles) == 1
    assert set(cycles[0]) == {"a", "b", "c"}


def test_unsafe_ffi_detected():
    facts = [
        'lang("main", "python").',
        'lang("rust_core", "rust").',
        'imports("main", "rust_core").',
    ]
    engine = LogicEngine()
    engine.load_facts(facts)
    unsafe = engine.unsafe_ffi_transitions()
    assert len(unsafe) == 1
    assert unsafe[0]["source_lang"] == "python"
    assert unsafe[0]["target_lang"] == "rust"


def test_boundary_fact_prevents_unsafe_flag():
    facts = [
        'lang("main", "python").',
        'lang("rust_core", "rust").',
        'imports("main", "rust_core").',
        'boundary("main", "rust_core", "PYO3_MATURIN").',
    ]
    engine = LogicEngine()
    engine.load_facts(facts)
    assert engine.unsafe_ffi_transitions() == []


def test_trace_returns_path():
    facts = [
        'imports("a", "b").',
        'imports("b", "c").',
    ]
    engine = LogicEngine()
    engine.load_facts(facts)
    trace = engine.derivation_trace("a", "c")
    assert trace is not None
    assert len(trace) == 2
    assert trace[0]["source"] == "a"
    assert trace[-1]["target"] == "c"


def test_analyze_repository_detects_python_cycle(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("import b\n")
    (tmp_path / "b.py").write_text("import a\n")
    feedback = analyze_repository(tmp_path)
    assert feedback.cycles
    assert "1 cycle(s)" in feedback.summary


def test_refinement_feedback_formatting():
    feedback = RefinementFeedback(
        unsat_core=["node_x toolchain invalid"],
        cycles=[["a", "b", "a"]],
        unsafe_ffi=[
            {
                "source": "py",
                "target": "rs",
                "source_lang": "python",
                "target_lang": "rust",
                "relation": "imports",
            }
        ],
    )
    text = feedback.to_llm_message()
    assert "Unsatisfiable core" in text
    assert "Dependency cycles" in text
    assert "Unsafe cross-language" in text
