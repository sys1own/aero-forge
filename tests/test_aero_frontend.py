"""Tests for the UAST frontend AST lowering."""

import pytest

from aero_forge.errors import UnsupportedError
from aero_forge.translator.aero_frontend import python_source_to_uast


def test_safe_stdlib_imports_lowering():
    source = (
        "import io\n"
        "import sys\n"
        "import time\n"
        "import math\n"
        "\n"
        "def f(x: int) -> int:\n"
        "    print(\"start\")\n"
        "    io.StringIO(\"data\")\n"
        "    sys.version\n"
        "    time.time()\n"
        "    math.pi\n"
        "    return x\n"
    )
    uast = python_source_to_uast(source)
    assert uast["type"] == "module"
    assert len(uast["children"]) == 1
    assert uast["children"][0]["type"] == "function_declaration"


def test_generic_attribute_fallback():
    source = (
        "def f(x: int) -> int:\n"
        "    obj = object()\n"
        "    obj.some_attr\n"
        "    obj.some_method()\n"
        "    return x\n"
    )
    uast = python_source_to_uast(source)
    assert uast["type"] == "module"
    assert len(uast["children"]) == 1


def test_unsafe_io_still_raises():
    source = (
        "def f(x: int) -> int:\n"
        "    os.system(\"bad\")\n"
        "    return x\n"
    )
    with pytest.raises(UnsupportedError):
        python_source_to_uast(source)
