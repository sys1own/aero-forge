"""Tests for the synthetic test generator."""

from aero_forge.scaffold.test_generator import generate_smoke_tests


def test_simple_types() -> None:
    impl = (
        "def orchestrate_hybrid_rust_python(project_name: str, functions: list) -> str:\n"
        "    return project_name\n"
    )
    tests = generate_smoke_tests(impl, module_name="my_module")
    assert "from my_module import orchestrate_hybrid_rust_python" in tests
    assert 'orchestrate_hybrid_rust_python("project_name", [])' in tests
    assert "orchestrate_hybrid_rust_python(1" not in tests


def test_generic_list_and_dict() -> None:
    impl = (
        "def evaluate(scores: list[list[float]], weights: dict[str, float]) -> float:\n"
        "    return 1.0\n"
    )
    tests = generate_smoke_tests(impl)
    assert "evaluate([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]," in tests
    assert "\"project_name\":" in tests
    assert "isinstance(result, (int, float))" in tests


def test_no_integer_fallback_for_str_or_list() -> None:
    impl = (
        "def build_payload(name: str, items: list[str], mapping: dict[str, int]) -> bool:\n"
        "    return True\n"
    )
    tests = generate_smoke_tests(impl)
    assert 'build_payload("project_name", ["project_name", "project_name", "project_name"],' in tests
    assert "build_payload(1" not in tests
    assert "[1, 2, 3]" not in tests
    assert "{\"project_name\": 1}" in tests or '{"project_name": 1}' in tests


def test_tuple_and_optional() -> None:
    impl = (
        "def pair(a: int, b: float | None) -> tuple[int, float]:\n"
        "    return (a, b or 0.0)\n"
    )
    tests = generate_smoke_tests(impl)
    assert "pair(1, 1.0)" in tests
    assert "isinstance(result, tuple)" in tests
