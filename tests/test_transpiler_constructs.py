"""Tests for newly supported transpiler constructs."""

import shutil
from pathlib import Path

import pytest

from aero_forge.orchestrator.orchestrator import Orchestrator


@pytest.mark.skipif(
    not shutil.which("cargo") or not shutil.which("rustc"),
    reason="Rust toolchain not installed",
)
def _run_case(tmp_path: Path, name: str, source: str, test_code: str) -> dict:
    src = tmp_path / f"{name}.py"
    src.write_text(source)
    test = tmp_path / f"test_{name}.py"
    test.write_text(f"from {name} import {name}\n\n{test_code}")
    orchestrator = Orchestrator(
        src,
        function_name=name,
        test_path=test,
        max_iterations=2,
        use_llm=False,
    )
    result = orchestrator.run()
    assert result["success"], result.get("error", "")
    return result


@pytest.mark.skipif(
    not shutil.which("cargo") or not shutil.which("rustc"),
    reason="Rust toolchain not installed",
)
def test_list_comprehension_over_range(tmp_path: Path) -> None:
    source = (
        "def list_comp(n: int) -> list[int]:\n" "    return [x * x for x in range(n)]\n"
    )
    _run_case(
        tmp_path,
        "list_comp",
        source,
        "def test_list_comp():\n    assert list_comp(5) == [0, 1, 4, 9, 16]\n",
    )


@pytest.mark.skipif(
    not shutil.which("cargo") or not shutil.which("rustc"),
    reason="Rust toolchain not installed",
)
def test_list_comprehension_over_list(tmp_path: Path) -> None:
    source = (
        "def double_list(arr: list[int]) -> list[int]:\n"
        "    return [x * 2 for x in arr]\n"
    )
    _run_case(
        tmp_path,
        "double_list",
        source,
        "def test_double_list():\n    assert double_list([1, 2, 3]) == [2, 4, 6]\n",
    )


@pytest.mark.skipif(
    not shutil.which("cargo") or not shutil.which("rustc"),
    reason="Rust toolchain not installed",
)
def test_nested_list_comprehension_matrix(tmp_path: Path) -> None:
    source = (
        "def zero_matrix(rows: int, cols: int) -> list[list[int]]:\n"
        "    return [[0 for _ in range(cols)] for _ in range(rows)]\n"
    )
    _run_case(
        tmp_path,
        "zero_matrix",
        source,
        "def test_zero_matrix():\n    assert zero_matrix(2, 3) == [[0, 0, 0], [0, 0, 0]]\n",
    )


@pytest.mark.skipif(
    not shutil.which("cargo") or not shutil.which("rustc"),
    reason="Rust toolchain not installed",
)
def test_tuple_unpacking_in_while(tmp_path: Path) -> None:
    source = (
        "def fib(n: int) -> int:\n"
        "    a, b = 0, 1\n"
        "    while n > 0:\n"
        "        a, b = b, a + b\n"
        "        n -= 1\n"
        "    return a\n"
    )
    _run_case(
        tmp_path,
        "fib",
        source,
        "def test_fib():\n    assert fib(10) == 55\n",
    )


@pytest.mark.skipif(
    not shutil.which("cargo") or not shutil.which("rustc"),
    reason="Rust toolchain not installed",
)
def test_enumerate_loop(tmp_path: Path) -> None:
    source = (
        "def sum_enum(arr: list[int]) -> int:\n"
        "    total = 0\n"
        "    for i, x in enumerate(arr):\n"
        "        total += i + x\n"
        "    return total\n"
    )
    _run_case(
        tmp_path,
        "sum_enum",
        source,
        "def test_sum_enum():\n    assert sum_enum([10, 20, 30]) == 63\n",
    )


@pytest.mark.skipif(
    not shutil.which("cargo") or not shutil.which("rustc"),
    reason="Rust toolchain not installed",
)
def test_zip_loop(tmp_path: Path) -> None:
    source = (
        "def sum_pairs(a: list[int], b: list[int]) -> int:\n"
        "    total = 0\n"
        "    for x, y in zip(a, b):\n"
        "        total += x * y\n"
        "    return total\n"
    )
    _run_case(
        tmp_path,
        "sum_pairs",
        source,
        "def test_sum_pairs():\n    assert sum_pairs([1, 2, 3], [4, 5, 6]) == 32\n",
    )


@pytest.mark.skipif(
    not shutil.which("cargo") or not shutil.which("rustc"),
    reason="Rust toolchain not installed",
)
def test_list_concatenation(tmp_path: Path) -> None:
    source = (
        "def concat(a: list[int], b: list[int]) -> list[int]:\n" "    return a + b\n"
    )
    _run_case(
        tmp_path,
        "concat",
        source,
        "def test_concat():\n    assert concat([1, 2], [3, 4]) == [1, 2, 3, 4]\n",
    )


@pytest.mark.skipif(
    not shutil.which("cargo") or not shutil.which("rustc"),
    reason="Rust toolchain not installed",
)
def test_complex_literal_rejected(tmp_path: Path) -> None:
    source = "def bad(x: int) -> int:\n    return x + 1j\n"
    src = tmp_path / "bad.py"
    src.write_text(source)
    test = tmp_path / "test_bad.py"
    test.write_text("from bad import bad\ndef test_bad():\n    assert bad(0) == 1\n")
    orchestrator = Orchestrator(
        src,
        function_name="bad",
        test_path=test,
        max_iterations=1,
        use_llm=False,
    )
    result = orchestrator.run()
    assert not result["success"]
    assert "Complex numbers are not supported" in result.get("error", "")


@pytest.mark.skipif(
    not shutil.which("cargo") or not shutil.which("rustc"),
    reason="Rust toolchain not installed",
)
def test_return_arity_with_empty_list(tmp_path: Path) -> None:
    source = (
        "def primes_up_to(n: int) -> list[int]:\n"
        "    if n < 2:\n"
        "        return []\n"
        "    return [2, 3]\n"
    )
    _run_case(
        tmp_path,
        "primes_up_to",
        source,
        "def test_primes_up_to():\n    assert primes_up_to(1) == []\n    assert primes_up_to(5) == [2, 3]\n",
    )


@pytest.mark.skipif(
    not shutil.which("cargo") or not shutil.which("rustc"),
    reason="Rust toolchain not installed",
)
def test_subscript_tuple_swap(tmp_path: Path) -> None:
    source = (
        "def reverse_in_place(arr: list[int]) -> list[int]:\n"
        "    i = 0\n"
        "    j = len(arr) - 1\n"
        "    while i < j:\n"
        "        arr[i], arr[j] = arr[j], arr[i]\n"
        "        i += 1\n"
        "        j -= 1\n"
        "    return arr\n"
    )
    _run_case(
        tmp_path,
        "reverse_in_place",
        source,
        "def test_reverse():\n    assert reverse_in_place([1, 2, 3, 4]) == [4, 3, 2, 1]\n",
    )


@pytest.mark.skipif(
    not shutil.which("cargo") or not shutil.which("rustc"),
    reason="Rust toolchain not installed",
)
def test_sorted_builtin(tmp_path: Path) -> None:
    source = (
        "def median(values: list[float]) -> float:\n"
        "    s = sorted(values)\n"
        "    n = len(s)\n"
        "    if n % 2 == 1:\n"
        "        return s[n // 2]\n"
        "    return (s[n // 2 - 1] + s[n // 2]) / 2.0\n"
    )
    _run_case(
        tmp_path,
        "median",
        source,
        "def test_median():\n    assert median([3.0, 1.0, 2.0]) == 2.0\n",
    )


@pytest.mark.skipif(
    not shutil.which("cargo") or not shutil.which("rustc"),
    reason="Rust toolchain not installed",
)
def test_int_cast_with_float_math(tmp_path: Path) -> None:
    source = "def integer_sqrt(n: int) -> int:\n" "    return int(n ** 0.5)\n"
    _run_case(
        tmp_path,
        "integer_sqrt",
        source,
        "def test_integer_sqrt():\n    assert integer_sqrt(17) == 4\n",
    )


@pytest.mark.skipif(
    not shutil.which("cargo") or not shutil.which("rustc"),
    reason="Rust toolchain not installed",
)
def test_list_comprehension_filter_and_step(tmp_path: Path) -> None:
    source = (
        "def primes_sieve(n: int) -> list[int]:\n"
        "    sieve = [True] * (n + 1)\n"
        "    sieve[0] = sieve[1] = False\n"
        "    for i in range(2, int(n ** 0.5) + 1):\n"
        "        if sieve[i]:\n"
        "            for j in range(i * i, n + 1, i):\n"
        "                sieve[j] = False\n"
        "    return [i for i in range(2, n + 1) if sieve[i]]\n"
    )
    _run_case(
        tmp_path,
        "primes_sieve",
        source,
        "def test_primes_sieve():\n    assert primes_sieve(10) == [2, 3, 5, 7]\n",
    )


@pytest.mark.skipif(
    not shutil.which("cargo") or not shutil.which("rustc"),
    reason="Rust toolchain not installed",
)
def test_nested_function_rejected(tmp_path: Path) -> None:
    source = (
        "def outer(x: int) -> int:\n"
        "    def inner(y: int) -> int:\n"
        "        return y + 1\n"
        "    return inner(x)\n"
    )
    src = tmp_path / "nested.py"
    src.write_text(source)
    test = tmp_path / "test_nested.py"
    test.write_text(
        "from nested import outer\ndef test_outer():\n    assert outer(1) == 2\n"
    )
    orchestrator = Orchestrator(
        src,
        function_name="outer",
        test_path=test,
        max_iterations=1,
        use_llm=False,
    )
    result = orchestrator.run()
    assert not result["success"]
    assert "Nested functions" in result.get("error", "")


@pytest.mark.skipif(
    not shutil.which("cargo") or not shutil.which("rustc"),
    reason="Rust toolchain not installed",
)
def test_mixed_float_int_math(tmp_path: Path) -> None:
    source = (
        "import math\n\n"
        "def dft_real(vals: list[float]) -> list[float]:\n"
        "    n = len(vals)\n"
        "    out = []\n"
        "    for k in range(n):\n"
        "        s = 0.0\n"
        "        for t in range(n):\n"
        "            angle = 2 * math.pi * t * k / n\n"
        "            s += vals[t] * math.cos(angle)\n"
        "        out.append(s)\n"
        "    return out\n"
    )
    _run_case(
        tmp_path,
        "dft_real",
        source,
        "def test_dft():\n    r = dft_real([1.0, 0.0])\n    assert abs(r[0] - 1.0) < 0.01\n",
    )


@pytest.mark.skipif(
    not shutil.which("cargo") or not shutil.which("rustc"),
    reason="Rust toolchain not installed",
)
def test_sorted_empty_list_guard(tmp_path: Path) -> None:
    source = (
        "def most_frequent(values: list[int]) -> int:\n"
        "    s = sorted(values)\n"
        "    return s[len(values) // 2]\n"
    )
    _run_case(
        tmp_path,
        "most_frequent",
        source,
        "def test_most_frequent():\n"
        "    assert most_frequent([1, 2, 2, 3]) == 2\n"
        "    assert most_frequent([]) == -1\n",
    )


@pytest.mark.skipif(
    not shutil.which("cargo") or not shutil.which("rustc"),
    reason="Rust toolchain not installed",
)
def test_tuple_subscript(tmp_path: Path) -> None:
    source = (
        "def tuple_first(t: tuple[float, float]) -> float:\n"
        "    return t[0]\n"
    )
    _run_case(
        tmp_path,
        "tuple_first",
        source,
        "def test_tuple_first():\n"
        "    assert tuple_first((1.0, 2.0)) == 1.0\n"
        "    assert tuple_first((3.0, 4.0)) == 3.0\n",
    )


@pytest.mark.skipif(
    not shutil.which("cargo") or not shutil.which("rustc"),
    reason="Rust toolchain not installed",
)
def test_local_function_call(tmp_path: Path) -> None:
    source = (
        "def helper(x: int) -> int:\n"
        "    return x + 1\n"
        "\n"
        "def main(n: int) -> int:\n"
        "    return helper(n)\n"
    )
    _run_case(
        tmp_path,
        "main",
        source,
        "def test_main():\n"
        "    assert main(5) == 6\n"
        "    assert main(0) == 1\n",
    )


@pytest.mark.skipif(
    not shutil.which("cargo") or not shutil.which("rustc"),
    reason="Rust toolchain not installed",
)
def test_matrix_chained_subscript(tmp_path: Path) -> None:
    source = (
        "def matrix_sum(m: list[list[int]]) -> int:\n"
        "    s = 0\n"
        "    for i in range(len(m)):\n"
        "        for j in range(len(m[i])):\n"
        "            s += m[i][j]\n"
        "    return s\n"
    )
    _run_case(
        tmp_path,
        "matrix_sum",
        source,
        "def test_matrix_sum():\n    assert matrix_sum([[1, 2], [3, 4]]) == 10\n",
    )


@pytest.mark.skipif(
    not shutil.which("cargo") or not shutil.which("rustc"),
    reason="Rust toolchain not installed",
)
def test_sliced_list_subscript(tmp_path: Path) -> None:
    source = (
        "def sum_even(signal: list[float]) -> float:\n"
        "    s = signal[::2]\n"
        "    total = 0.0\n"
        "    for i in range(len(s)):\n"
        "        total += s[i]\n"
        "    return total\n"
    )
    _run_case(
        tmp_path,
        "sum_even",
        source,
        "def test_sum_even():\n    assert sum_even([1.0, 2.0, 3.0, 4.0]) == 4.0\n",
    )


@pytest.mark.skipif(
    not shutil.which("cargo") or not shutil.which("rustc"),
    reason="Rust toolchain not installed",
)
def test_function_return_subscript(tmp_path: Path) -> None:
    source = (
        "def cooley_tukey_fft(sig: list[float]) -> list[float]:\n"
        "    return sig\n"
        "\n"
        "def first_mag(sig: list[float]) -> float:\n"
        "    return cooley_tukey_fft(sig)[0]\n"
    )
    _run_case(
        tmp_path,
        "first_mag",
        source,
        "def test_first_mag():\n    assert first_mag([5.0, 6.0]) == 5.0\n",
    )


@pytest.mark.skipif(
    not shutil.which("cargo") or not shutil.which("rustc"),
    reason="Rust toolchain not installed",
)
def test_safe_stdlib_imports_and_calls(tmp_path: Path) -> None:
    source = (
        "import io\n"
        "import sys\n"
        "import time\n"
        "import math\n"
        "\n"
        "def safe_stdlib_calls(x: int) -> int:\n"
        "    print(\"start\")\n"
        "    io.StringIO(\"data\")\n"
        "    sys.version\n"
        "    time.time()\n"
        "    math.pi\n"
        "    return x * 2\n"
    )
    _run_case(
        tmp_path,
        "safe_stdlib_calls",
        source,
        "def test_safe_stdlib_calls():\n    assert safe_stdlib_calls(5) == 10\n",
    )


@pytest.mark.skipif(
    not shutil.which("cargo") or not shutil.which("rustc"),
    reason="Rust toolchain not installed",
)
def test_infinity_constants(tmp_path: Path) -> None:
    source = (
        "import math\n"
        "def infinity_constants(x: float) -> float:\n"
        "    if x > 0:\n"
        "        return float('inf')\n"
        "    if x < 0:\n"
        "        return -math.inf\n"
        "    return 'infinity'\n"
    )
    _run_case(
        tmp_path,
        "infinity_constants",
        source,
        "def test_infinity_constants():\n"
        "    assert infinity_constants(1) == float('inf')\n"
        "    assert infinity_constants(-1) == float('-inf')\n"
        "    assert infinity_constants(0) == float('inf')\n",
    )
