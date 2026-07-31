"""Tests for the Rust HIN (MELL interaction-net) engine bridge."""

import json
import os
import sys
from pathlib import Path

import pytest

from aero_forge.translator.aero_frontend import python_source_to_uast

# The native extension is optional until the workspace tool-chain builds it.
HIN = pytest.importorskip("aero_forge.hin_engine")


def _load_aero_future_modules():
    """Load aero-future's Python HIN VM for parity checks if present."""
    base = Path(__file__).resolve().parents[2] / "aero-future"
    if not base.is_dir():
        base = Path(os.environ.get("AERO_FUTURE_PATH", "/home/ubuntu/repos/aero-future"))
    if base.is_dir() and str(base) not in sys.path:
        sys.path.insert(0, str(base))
    try:
        from core.hin_vm import HINNetwork
        from core.translator import UASTToHINTranslator

        return UASTToHINTranslator, HINNetwork
    except Exception:
        return None, None


def test_native_bridge_available():
    assert HIN.native_available()


def test_identity_function_reduction():
    source = "def f(x):\n    return x\ny = f(42)\ny"
    uast = python_source_to_uast(source)
    result = HIN.reduce_uast(uast)
    assert result["native"]
    assert result["steps"] > 0
    assert isinstance(result["graph"], list)


def test_nested_function_reduction():
    source = "def f(x):\n    return x\ndef g(y):\n    return f(y)\ny = g(7)\ny"
    uast = python_source_to_uast(source)
    result = HIN.reduce_uast(uast)
    assert result["native"]
    assert result["steps"] > 0


def test_conditional_reduction_parity():
    source = "def f(x):\n    return x\nif f(True):\n    1\nelse:\n    0\n"
    uast = python_source_to_uast(source)
    result = HIN.reduce_uast(uast)
    assert result["native"]
    assert result["steps"] > 0

    PythonTranslator, PythonNetwork = _load_aero_future_modules()
    if PythonTranslator is None:
        pytest.skip("aero-future reference VM not available")

    t = PythonTranslator()
    net = t.translate_uast(uast)
    net.run_to_completion()
    assert len(net.nodes) == len(result["graph"])


def test_hin_engine_class_api():
    HinEngine = pytest.importorskip("aero_forge_native").HinEngine
    source = "def f(x):\n    return x\nf(42)"
    uast = python_source_to_uast(source)
    engine = HinEngine()
    engine.build_from_json(json.dumps(uast))
    steps = engine.reduce_to_completion(1_000_000)
    assert steps > 0
    graph = json.loads(engine.to_json())
    assert isinstance(graph, list)
