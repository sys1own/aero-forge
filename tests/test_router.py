"""Unit tests for the self-healing and AST routing routers."""

from aero_forge.healing.router import try_auto_fix
from aero_forge.orchestrator.router import (
    GENERAL_PURPOSE,
    HIN_COMPUTE,
    CodeRouteClassifier,
    classify,
)


def test_adds_missing_math_import():
    code = "def sqrt_x(x):\n    return math.sqrt(x)\n"
    error = "NameError: name 'math' is not defined"
    fixed = try_auto_fix(error, code)
    assert fixed is not None
    assert "import math" in fixed


def test_no_fix_for_unknown_error():
    code = "def f(x):\n    return x\n"
    assert try_auto_fix("some random error", code) is None


# ---------------------------------------------------------------------------
# CodeRouteClassifier tests
# ---------------------------------------------------------------------------


def test_pure_numeric_dot_product_routes_to_hin():
    source = (
        "def dot(a: list[float], b: list[float]) -> float:\n"
        "    s = 0.0\n"
        "    for i in range(len(a)):\n"
        "        s += a[i] * b[i]\n"
        "    return s\n"
    )
    result = classify(source, ["dot"])
    assert result["route"] == HIN_COMPUTE
    assert "dot" in result["target_functions"]


def test_mandelbrot_loop_routes_to_hin():
    source = (
        "def mandelbrot(c: complex, max_iter: int) -> int:\n"
        "    z = 0 + 0j\n"
        "    for n in range(max_iter):\n"
        "        if abs(z) > 2.0:\n"
        "            return n\n"
        "        z = z * z + c\n"
        "    return max_iter\n"
    )
    result = classify(source, ["mandelbrot"])
    assert result["route"] == HIN_COMPUTE
    assert "mandelbrot" in result["target_functions"]


def test_vector_operation_routes_to_hin():
    source = (
        "def scale(v: list[float], k: float) -> list[float]:\n"
        "    out: list[float] = []\n"
        "    for x in v:\n"
        "        out.append(x * k)\n"
        "    return out\n"
    )
    result = classify(source, ["scale"])
    assert result["route"] == HIN_COMPUTE
    assert "scale" in result["target_functions"]


def test_file_reading_routes_to_general():
    source = (
        "def read_data(path: str) -> str:\n"
        "    with open(path) as f:\n"
        "        return f.read()\n"
    )
    result = classify(source, ["read_data"])
    assert result["route"] == GENERAL_PURPOSE
    assert result["target_functions"] == []


def test_dictionary_manipulation_routes_to_general():
    source = (
        "def freq(items: list[str]) -> dict[str, int]:\n"
        "    out = {}\n"
        "    for x in items:\n"
        "        out[x] = out.get(x, 0) + 1\n"
        "    return out\n"
    )
    result = classify(source, ["freq"])
    assert result["route"] == GENERAL_PURPOSE
    assert result["target_functions"] == []


def test_f_string_routes_to_general():
    source = (
        "def greet(name: str) -> str:\n"
        "    return f'Hello {name}'\n"
    )
    result = classify(source, ["greet"])
    assert result["route"] == GENERAL_PURPOSE


def test_unannotated_dynamic_function_routes_to_general():
    source = (
        "def add_and_report(a, b):\n"
        "    print(a + b)\n"
        "    return a + b\n"
    )
    result = classify(source, ["add_and_report"])
    assert result["route"] == GENERAL_PURPOSE


def test_web_route_routes_to_general():
    source = (
        "def handle(data: dict) -> str:\n"
        "    return data.get('name', 'guest')\n"
    )
    result = classify(source, ["handle"])
    assert result["route"] == GENERAL_PURPOSE
    assert result["target_functions"] == []


def test_classifier_class_returns_same_payload():
    source = (
        "def square(x: int) -> int:\n"
        "    return x * x\n"
    )
    payload = CodeRouteClassifier(source, ["square"]).classify()
    direct = classify(source, ["square"])
    assert payload == direct
    assert payload["route"] == HIN_COMPUTE
    assert payload["target_functions"] == ["square"]


def test_hin_function_calling_general_callee_routes_general():
    source = (
        "def helper(x: str) -> str:\n"
        "    return f'value: {x}'\n"
        "\n"
        "def compute(a: int) -> int:\n"
        "    return a + len(helper(str(a)))\n"
    )
    result = classify(source, ["compute"])
    assert result["route"] == GENERAL_PURPOSE
    assert "compute" not in result["target_functions"]
