"""Tests for FallbackManager proactive collection AST healing."""

from aero_forge.builder.fallback_manager import FallbackManager, HeuristicWarning


def test_remediates_dict_get_to_subscript():
    source = "def f(d):\n    return d.get('k')\n"
    manager = FallbackManager()
    changed, new_source, diagnostics = manager.remediate_collection_ast(source)
    assert changed
    assert "d['k']" in new_source or 'd["k"]' in new_source
    assert not diagnostics


def test_remediates_dict_kwargs_to_literal():
    source = "def f():\n    return dict(a=1, b=2)\n"
    manager = FallbackManager()
    changed, new_source, diagnostics = manager.remediate_collection_ast(source)
    assert changed
    assert "{" in new_source
    assert "'a':" in new_source or '"a":' in new_source


def test_remediates_empty_dict_call():
    source = "def f():\n    return dict()\n"
    manager = FallbackManager()
    changed, new_source, diagnostics = manager.remediate_collection_ast(source)
    assert changed
    assert "{}" in new_source


def test_heuristic_warning_is_importable():
    assert issubclass(HeuristicWarning, Exception)
