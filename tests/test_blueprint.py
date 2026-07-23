"""Tests for blueprint parsing, validation, and generation."""

from pathlib import Path

import pytest

from aero_forge.blueprint import (
    Blueprint,
    FunctionSpec,
    discover_functions,
    generate_blueprint,
    parse_blueprint,
    write_blueprint,
)


def test_parse_aero_blueprint(tmp_path):
    src = tmp_path / "compute.py"
    test = tmp_path / "test_compute.py"
    src.write_text("def f(x): return x")
    test.write_text("")
    blueprint_path = tmp_path / "blueprint.aero"
    blueprint_path.write_text(
        'project: "my_project"\n'
        "functions:\n"
        "  - file: compute.py\n"
        "    name: f\n"
        "    tests: [test_compute.py]\n"
        "output_dir: ./dist\n"
    )

    bp = parse_blueprint(blueprint_path)
    assert bp.project == "my_project"
    assert len(bp.functions) == 1
    assert bp.functions[0].name == "f"
    assert bp.output_dir == tmp_path / "dist"


def test_parse_yaml_blueprint(tmp_path):
    src = tmp_path / "compute.py"
    test = tmp_path / "test_compute.py"
    src.write_text("def g(x): return x")
    test.write_text("")
    blueprint_path = tmp_path / "blueprint.yaml"
    blueprint_path.write_text(
        "project: yaml_project\n"
        "functions:\n"
        "  - file: compute.py\n"
        "    name: g\n"
        "    tests:\n"
        "      - test_compute.py\n"
        "output_dir: ./dist\n"
    )

    bp = parse_blueprint(blueprint_path)
    assert bp.project == "yaml_project"
    assert bp.functions[0].file == src.resolve()


def test_parse_blueprint_missing_file_raises(tmp_path):
    blueprint_path = tmp_path / "blueprint.aero"
    blueprint_path.write_text(
        "project: bad\n"
        "functions:\n"
        "  - file: missing.py\n"
        "    name: missing\n"
    )
    with pytest.raises(ValueError, match="missing file"):
        parse_blueprint(blueprint_path)


def test_discover_functions(tmp_path):
    source = tmp_path / "example.py"
    source.write_text(
        "def public_one():\n"
        "    pass\n"
        "def public_two():\n"
        "    pass\n"
        "def _private():\n"
        "    pass\n"
    )
    functions = discover_functions(source)
    assert {f.name for f in functions} == {"public_one", "public_two"}


def test_generate_and_write_blueprint(tmp_path):
    src = tmp_path / "example.py"
    test = tmp_path / "test_example.py"
    src.write_text("def f(): pass")
    test.write_text("")
    bp = generate_blueprint(
        project="gen",
        functions=[
            FunctionSpec(file=src, name="f", tests=[test]),
        ],
        output_dir=tmp_path / "dist",
    )
    out = tmp_path / "out.aero"
    write_blueprint(bp, out)
    written = out.read_text()
    assert "project: gen" in written
    assert "f" in written


def test_parse_compile_all_wildcard(tmp_path):
    source = tmp_path / "utils.py"
    source.write_text(
        "def add(a, b):\n"
        "    return a + b\n"
        "def mul(a, b):\n"
        "    return a * b\n"
    )
    blueprint_path = tmp_path / "blueprint.aero"
    blueprint_path.write_text(
        "project: wild\n"
        "functions:\n"
        "  - file: utils.py\n"
        '    name: "*"\n'
    )

    bp = parse_blueprint(blueprint_path)
    assert bp.functions[0].compile_all is True
    assert bp.functions[0].name == "*"


def test_parse_compile_all_flag(tmp_path):
    source = tmp_path / "utils.py"
    source.write_text("def f(): pass\n")
    blueprint_path = tmp_path / "blueprint.yaml"
    blueprint_path.write_text(
        "project: all\n"
        "functions:\n"
        "  - file: utils.py\n"
        "    compile_all: true\n"
    )

    bp = parse_blueprint(blueprint_path)
    assert bp.functions[0].compile_all is True
