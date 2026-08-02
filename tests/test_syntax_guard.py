"""Tests for the Rust/C/C++ syntax truncation guard."""

from __future__ import annotations

from pathlib import Path

import ast

from aero_forge.scaffold.syntax_guard import (
    ClassInitNormalizer,
    ensure_typing_imports,
    normalize_python_module,
    repair_file,
    repair_source,
    repair_workspace,
)


def test_repair_missing_closing_brace() -> None:
    source = "fn main() {\n    let x = 1;\n"
    repaired = repair_source(source)
    assert "}\n" in repaired
    assert repaired.count("{") == repaired.count("}")


def test_repair_unbalanced_parens_and_brackets() -> None:
    source = "fn foo(a: [i32; 3] {\n    bar(1, 2\n"
    repaired = repair_source(source)
    assert repaired.count("(") == repaired.count(")")
    assert repaired.count("[") == repaired.count("]")


def test_strip_trailing_doc_comments() -> None:
    source = "fn main() {}\n///\n/// dangling doc\n"
    repaired = repair_source(source)
    assert "///" not in repaired
    assert "fn main() {}" in repaired


def test_strip_unclosed_block_comment() -> None:
    source = "fn main() {}\n/* dangling block comment\n"
    repaired = repair_source(source)
    assert "/*" not in repaired
    assert "fn main() {}" in repaired


def test_no_change_for_valid_file(tmp_path: Path) -> None:
    path = tmp_path / "valid.rs"
    path.write_text("fn main() {\n    let x = 1;\n}\n", encoding="utf-8")
    assert repair_file(path) is False
    assert path.read_text(encoding="utf-8") == "fn main() {\n    let x = 1;\n}\n"


def test_repair_workspace(tmp_path: Path) -> None:
    (tmp_path / "a.rs").write_text("fn a() {\n", encoding="utf-8")
    (tmp_path / "b.cpp").write_text("int main() {\n", encoding="utf-8")
    (tmp_path / "readme.txt").write_text("hello", encoding="utf-8")
    changed = repair_workspace(tmp_path)
    assert len(changed) == 2
    assert all(p.suffix in {".rs", ".cpp"} for p in changed)


def test_ensure_typing_imports_injects_any_from_annotation() -> None:
    source = "x: Any = 1\n"
    result = ensure_typing_imports(source)
    assert "from typing import Any" in result


def test_ensure_typing_imports_injects_multiple_used_names() -> None:
    source = (
        "def process(items: List[Dict[str, Any]], callback: Callable[[int], Optional[str]]) -> Tuple[int, ...]:\n"
        "    return (0,)\n"
    )
    result = ensure_typing_imports(source)
    assert "from typing import" in result
    for name in ("Any", "Callable", "Dict", "List", "Optional", "Tuple"):
        assert name in result


def test_ensure_typing_imports_no_duplicate_when_already_imported() -> None:
    source = "from typing import Any, List\nx: Any = 1\ny: List[int] = []\n"
    result = ensure_typing_imports(source)
    assert result.count("from typing import") == 1
    assert "Any" in result
    assert "List" in result


def test_ensure_typing_imports_does_not_add_unused_names() -> None:
    source = "x = 1\n"
    result = ensure_typing_imports(source)
    assert "from typing import" not in result


def test_class_init_normalizer_injects_default_init() -> None:
    source = "class Foo:\n    x: int = 1\n"
    tree = ast.parse(source)
    normalized = ClassInitNormalizer().visit(tree)
    ast.fix_missing_locations(normalized)
    assert "def __init__" in ast.unparse(normalized)


def test_class_init_normalizer_preserves_existing_init() -> None:
    source = "class Foo:\n    def __init__(self, a: int):\n        self.a = a\n"
    tree = ast.parse(source)
    normalized = ClassInitNormalizer().visit(tree)
    assert ast.dump(normalized) == ast.dump(ast.parse(source))


def test_normalize_python_module_generates_init_from_fields() -> None:
    source = "class Counter:\n    value: int\n"
    result = normalize_python_module(source)
    assert "def __init__(self, value: int):" in result
    assert "self.value = value" in result


def test_normalize_python_module_generates_permissive_init_for_empty_class() -> None:
    source = "class Empty:\n    pass\n"
    result = normalize_python_module(source)
    assert "def __init__(self, *args, **kwargs):" in result


def test_normalize_python_module_preserves_source_when_no_changes() -> None:
    source = "class Foo:\n    def __init__(self):\n        pass\n"
    result = normalize_python_module(source)
    assert result == source
