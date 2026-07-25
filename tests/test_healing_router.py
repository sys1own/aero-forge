"""Tests for ``aero_forge.healing.router``."""

import pytest

from aero_forge.healing.router import try_auto_fix


def test_adds_missing_stdlib_import() -> None:
    code = "x = math.sqrt(2)\n"
    fixed = try_auto_fix("NameError: name 'math' is not defined", code)
    assert fixed is not None
    assert "import math" in fixed


def test_fixes_local_variable_typo() -> None:
    code = (
        "def test_example():\n"
        "    rstats = RollingStats(10)\n"
        "    assert r_stats.get_values() == []\n"
    )
    fixed = try_auto_fix("NameError: name 'r_stats' is not defined", code)
    assert fixed is not None
    assert "rstats.get_values()" in fixed
    assert "r_stats" not in fixed


def test_no_fix_when_name_is_unrelated() -> None:
    code = "def test_example():\n    assert missing_thing == 1\n"
    fixed = try_auto_fix("NameError: name 'missing_thing' is not defined", code)
    assert fixed is None
