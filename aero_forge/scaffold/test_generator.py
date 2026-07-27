"""Synthetic pytest generation from implementation signatures."""

from __future__ import annotations

import ast
import math
from typing import Any, Dict, List, Optional, Tuple


# Names that suggest a function operates on statistical / aggregated data.
_STATISTICAL_HINTS = {
    "anomaly",
    "anomalies",
    "detect",
    "z_score",
    "zscore",
    "stat",
    "stats",
    "statistics",
    "metric",
    "metrics",
    "aggregate",
    "aggregation",
    "mean",
    "std",
    "stddev",
    "variance",
    "deviation",
    "outlier",
    "outliers",
}

# Dict keys that hold integer counts (do not compare with exact magic numbers).
_COUNT_KEYS = {
    "anomalies",
    "count",
    "total",
    "num",
    "flags",
}

# Dict keys that hold floating-point statistical results.
_FLOAT_KEYS = {
    "mean",
    "std",
    "stddev",
    "variance",
    "z_score",
    "zscore",
    "peak",
    "max",
    "min",
    "sum",
    "value",
}


def _base_name(node: ast.AST) -> str:
    """Return the final attribute/name part of a type expression."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _type_args(node: ast.AST) -> List[ast.AST]:
    """Return the argument nodes inside a generic subscript."""
    if isinstance(node, ast.Index):  # pragma: no cover  # Python 3.8 compatibility
        return _type_args(node.value)
    if isinstance(node, ast.Tuple):
        return list(node.elts)
    return [node]


def _is_none(node: Optional[ast.AST]) -> bool:
    if node is None:
        return True
    if isinstance(node, ast.Constant) and node.value is None:
        return True
    if isinstance(node, ast.Name) and node.id == "None":
        return True
    return False


def _scalar_literal(name: str) -> str:
    """Return a valid Python literal string for a scalar type name."""
    mapping = {
        "str": '"project_name"',
        "int": "1",
        "float": "1.0",
        "bool": "True",
        "bytes": 'b"test"',
        "None": "None",
    }
    return mapping.get(name, "None")


def _value_for_annotation(node: Optional[ast.AST], *, generic_fallback: bool = True) -> str:
    """Produce a single Python literal string matching an annotation."""
    if node is None or _is_none(node):
        return "None"

    if isinstance(node, ast.Constant):
        return repr(node.value)

    if isinstance(node, ast.Name):
        name = node.id
        if name in {"list", "List", "Sequence"}:
            return "[]" if generic_fallback else '["fn_a", "fn_b"]'
        if name in {"dict", "Dict", "Mapping"}:
            return "{}"
        if name in {"tuple", "Tuple"}:
            return "(1, 2)"
        if name in {"set", "Set"}:
            return "{1, 2}"
        return _scalar_literal(name)

    if isinstance(node, ast.Attribute):
        return _value_for_annotation(ast.Name(id=_base_name(node)), generic_fallback=generic_fallback)

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        left = _value_for_annotation(node.left, generic_fallback=generic_fallback)
        right = _value_for_annotation(node.right, generic_fallback=generic_fallback)
        if left == "None":
            return right
        if right == "None":
            return left
        return left

    if isinstance(node, ast.Subscript):
        base = _base_name(node.value).lower()
        args = _type_args(node.slice)
        if base in {"list", "sequence"}:
            if not args:
                return "[]"
            inner = _value_for_annotation(args[0], generic_fallback=False)
            if inner.startswith("[") or inner.startswith("{") or inner.startswith("("):
                return f"[{inner}, {inner}]"
            return f"[{inner}, {inner}, {inner}]"
        if base in {"dict", "mapping"}:
            if len(args) >= 2:
                key = _value_for_annotation(args[0], generic_fallback=False)
                val = _value_for_annotation(args[1], generic_fallback=False)
                return f"{{{key}: {val}}}"
            return "{}"
        if base in {"tuple", "tuple_"}:
            parts = [_value_for_annotation(a, generic_fallback=False) for a in args]
            if len(parts) == 1:
                return f"({parts[0]},)"
            return f"({', '.join(parts)})"
        if base in {"set", "frozen_set"}:
            if args:
                inner = _value_for_annotation(args[0], generic_fallback=False)
                if inner in {"None", "[]", "{}", "()"}:
                    return "set()"
                return f"{{{inner}, {inner}}}"
            return "{1, 2}"
        if base == "optional":
            if args and not _is_none(args[0]):
                return _value_for_annotation(args[0], generic_fallback=False)
            return "None"
        if base == "union":
            for arg in args:
                if not _is_none(arg):
                    return _value_for_annotation(arg, generic_fallback=False)
            return "None"
        if base == "literal":
            for arg in args:
                if isinstance(arg, ast.Constant):
                    return repr(arg.value)
            return _value_for_annotation(args[0], generic_fallback=False)
        # Unknown generic: fall back to the base name as a scalar/container.
        return _value_for_annotation(ast.Name(id=_base_name(node.value)), generic_fallback=generic_fallback)

    return "None"


def _type_name(node: Optional[ast.AST]) -> str:
    """Return a normalized base type name for an annotation."""
    if node is None or _is_none(node):
        return "none"
    if isinstance(node, ast.Name):
        return node.id.lower()
    if isinstance(node, ast.Attribute):
        return node.attr.lower()
    if isinstance(node, ast.Subscript):
        return _type_name(node.value)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        left = _type_name(node.left)
        right = _type_name(node.right)
        if left == "none":
            return right
        if right == "none":
            return left
        return left
    return ""


def _inner_type_name(node: Optional[ast.AST]) -> str:
    """Return the element type name for list/tuple subscripts."""
    if node is None:
        return ""
    if isinstance(node, ast.Subscript):
        args = _type_args(node.slice)
        if args:
            return _type_name(args[0])
    return ""


def _is_statistical(name: str, returns: Optional[ast.AST]) -> bool:
    """Heuristic: does this function look like a statistical/anomaly aggregator?"""
    lowered = name.lower()
    if any(hint in lowered for hint in _STATISTICAL_HINTS):
        return True
    return_type = _type_name(returns)
    if return_type in {"dict", "mapping"}:
        return True
    if return_type in {"tuple"}:
        inner = _inner_type_name(returns)
        if inner in {"float", "f64", "f32"}:
            return True
    return False


def _return_dict_keys(func: ast.FunctionDef) -> List[str]:
    """Extract string keys from literal ``return {...}`` statements in *func*."""
    keys: List[str] = []
    for node in ast.walk(func):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
            for key in node.value.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    keys.append(key.value)
                elif isinstance(key, ast.Str):  # pragma: no cover  # Python < 3.8
                    keys.append(key.s)
    return keys


def _statistical_fixture(arg_name: str, arg_type: str, inner: str) -> Optional[str]:
    """Return an outlier-rich fixture literal for a statistical argument, or None."""
    lowered = arg_name.lower()
    type_lower = arg_type.lower()
    if "threshold" in lowered and type_lower in {"float", "f64", "f32"}:
        return "3.0"
    if "window" in lowered or "capacity" in lowered or "size" in lowered:
        if type_lower in {"int", "i64", "i32"}:
            return "100"
    if type_lower in {"list", "sequence"}:
        if inner in {"float", "f64", "f32"}:
            # 100 normal values plus one >10-sigma outlier so Z > 3.0 is guaranteed.
            if "data" in lowered or "values" in lowered or "samples" in lowered or "metrics" in lowered:
                return "[1.0] * 100 + [10000.0]"
            return "[1.0, 2.0, 3.0, 10000.0]"
        if inner in {"int", "i64", "i32"}:
            if "data" in lowered or "values" in lowered or "samples" in lowered or "metrics" in lowered:
                return "[1] * 100 + [10000]"
            return "[1, 2, 3, 10000]"
    return None


def _dict_key_assertions(keys: List[str], has_outlier_fixture: bool) -> List[str]:
    """Generate tolerant assertions for each known key in a returned dict."""
    if not keys:
        return ["    assert isinstance(result, dict)"]

    lines = ["    assert isinstance(result, dict)"]
    for key in keys:
        lowered = key.lower()
        if lowered in _COUNT_KEYS:
            if lowered == "anomalies" and has_outlier_fixture:
                lines.append(
                    f'    assert isinstance(result["{key}"], int) and result["{key}"] >= 1'
                )
            else:
                lines.append(
                    f'    assert isinstance(result["{key}"], int) and result["{key}"] >= 0'
                )
        elif lowered in _FLOAT_KEYS:
            lines.append(
                f'    assert isinstance(result["{key}"], float) and math.isfinite(result["{key}"])'
            )
        else:
            lines.append(f'    assert "{key}" in result')
    return lines


def _parity_check(value_expr: str, value_type: str) -> str:
    """Return an assertion that compares two deterministic calls of the same function."""
    if value_type in {"float", "f64", "f32"}:
        return f"    assert math.isclose({value_expr}, result2, rel_tol=1e-9)"
    if value_type in {"int", "i64", "i32"}:
        return f"    assert {value_expr} == result2"
    if value_type in {"list", "sequence"}:
        return f"    assert len({value_expr}) == len(result2) and all(math.isclose(a, b, rel_tol=1e-9) for a, b in zip({value_expr}, result2))"
    if value_type in {"dict", "mapping"}:
        return (
            f"    assert {value_expr}.keys() == result2.keys()\n"
            f"    for _k in {value_expr}:\n"
            f"        _v1, _v2 = {value_expr}[_k], result2[_k]\n"
            f"        if isinstance(_v1, float):\n"
            f"            assert math.isclose(_v1, _v2, rel_tol=1e-9)\n"
            f"        else:\n"
            f"            assert _v1 == _v2"
        )
    if value_type in {"tuple"}:
        return f"    assert all(math.isclose(a, b, rel_tol=1e-9) for a, b in zip({value_expr}, result2))"
    return f"    assert {value_expr} == result2"


def _statistical_assertion(call: str, returns: Optional[ast.AST], func: ast.FunctionDef, has_outlier_fixture: bool) -> str:
    """Generate a robust, parity-aware assertion block for a statistical function."""
    return_type = _type_name(returns)
    lines: List[str] = [f"    result = {call}"]

    if return_type == "dict":
        keys = _return_dict_keys(func)
        lines.extend(_dict_key_assertions(keys, has_outlier_fixture))
        # Add deterministic parity check with a second invocation.
        lines.append(f"    result2 = {call}")
        lines.append(_parity_check("result", return_type))
    elif return_type == "tuple":
        lines.append("    assert isinstance(result, tuple)")
        lines.append(f"    result2 = {call}")
        lines.append(_parity_check("result", return_type))
    elif return_type in {"float", "int"}:
        if return_type == "float":
            lines.append("    assert isinstance(result, float) and math.isfinite(result)")
        else:
            lines.append("    assert isinstance(result, int)")
        lines.append(f"    result2 = {call}")
        lines.append(_parity_check("result", return_type))
    elif return_type in {"list", "sequence"}:
        lines.append("    assert isinstance(result, list)")
        lines.append(f"    result2 = {call}")
        lines.append(_parity_check("result", return_type))
    else:
        lines.append("    assert result is not None")

    return "\n".join(lines)


def _assertion_for_call(call: str, returns: Optional[ast.AST]) -> str:
    """Return a pytest assertion body for a function call."""
    name = _type_name(returns)
    if name in {"none"}:
        return f"    result = {call}\n    assert result is None"
    if name == "bool":
        return f"    assert {call} in (True, False)"
    if name in {"list", "sequence"}:
        return f"    result = {call}\n    assert isinstance(result, list)"
    if name in {"dict", "mapping"}:
        return f"    result = {call}\n    assert isinstance(result, dict)"
    if name in {"tuple"}:
        return f"    result = {call}\n    assert isinstance(result, tuple)"
    if name in {"set", "frozenset"}:
        return f"    result = {call}\n    assert isinstance(result, set)"
    if name in {"int", "float"}:
        return f"    result = {call}\n    assert isinstance(result, (int, float))"
    if name == "str":
        return f"    result = {call}\n    assert isinstance(result, str)"
    if name == "bytes":
        return f"    result = {call}\n    assert isinstance(result, bytes)"
    return f"    result = {call}\n    assert result is not None"


def _find_repl_commands(tree: ast.AST) -> List[Tuple[str, str]]:
    """Return [(class_name, do_command_name)] for cmd.Cmd subclasses in *tree*."""
    repls: List[Tuple[str, str]] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        is_cmd = False
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id == "Cmd":
                is_cmd = True
            elif isinstance(base, ast.Attribute) and base.attr == "Cmd":
                is_cmd = True
            elif isinstance(base, ast.Name) and base.id == "cmd":
                is_cmd = True
        if not is_cmd:
            continue
        for member in node.body:
            if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)) and member.name.startswith("do_"):
                repls.append((node.name, member.name[3:]))
    return repls


def generate_smoke_tests(implementation: str, module_name: str = "generated") -> str:
    """Generate pytest smoke tests from the implementation when none were provided.

    Uses AST inspection of parameter annotations so synthesized test arguments
    match declared types (``str`` gets a string, ``list`` gets a list, etc.).
    For statistical / anomaly-detection functions, the generated tests use
    outlier-rich fixtures, soft ``math.isclose`` assertions, and deterministic
    parity checks instead of fragile exact scalar comparisons.

    If the source contains a ``cmd.Cmd`` REPL class, also generate tests that
    instantiate the shell and call ``onecmd(...)`` programmatically.
    """
    try:
        tree = ast.parse(implementation)
    except SyntaxError:
        return ""

    test_lines: List[str] = []

    repl_commands = _find_repl_commands(tree)
    if repl_commands:
        classes = sorted({cls for cls, _ in repl_commands})
        class_name = classes[0]
        commands = [cmd for cls, cmd in repl_commands if cls == class_name]
        test_lines.append("import io")
        test_lines.append("import sys")
        test_lines.append("")
        test_lines.append(f"from {module_name} import {class_name}")
        test_lines.append("")
        for cmd in commands:
            test_lines.append(f"def test_repl_{cmd}():\n    shell = {class_name}()\n    shell.onecmd('{cmd}')\n")
        test_lines.append(f"def test_repl_quit():\n    shell = {class_name}()\n    assert shell.onecmd('quit') is True\n")
        test_lines.append("")

    for item in tree.body:
        if not isinstance(item, ast.FunctionDef):
            continue
        name = item.name
        if name.startswith("_"):
            continue

        statistical = _is_statistical(name, item.returns)
        arg_info: List[Tuple[str, str, str]] = []
        arg_values: List[str] = []
        has_outlier_fixture = False

        for arg in item.args.args + item.args.posonlyargs + item.args.kwonlyargs:
            if arg.arg in {"self", "cls"}:
                continue
            arg_type = _type_name(arg.annotation)
            inner = _inner_type_name(arg.annotation)
            if statistical:
                fixture = _statistical_fixture(arg.arg, arg_type, inner)
                if fixture is not None:
                    if "10000.0" in fixture or "10000]" in fixture:
                        has_outlier_fixture = True
                    arg_values.append(fixture)
                    continue
            arg_values.append(_value_for_annotation(arg.annotation))

        call = f"{name}({', '.join(arg_values)})"
        if statistical:
            assertion = _statistical_assertion(call, item.returns, item, has_outlier_fixture)
        else:
            assertion = _assertion_for_call(call, item.returns)
        test_lines.append(f"def test_{name}():\n{assertion}\n")

    if not test_lines:
        return ""

    all_names = sorted(
        item.name
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and not item.name.startswith("_")
    )
    imports = ["import math", ""]
    imports.extend(f"from {module_name} import {n}" for n in all_names)
    return "\n".join(imports) + "\n\n" + "\n".join(test_lines)


def _abi_type_to_py(abi_type: str) -> str:
    """Map a supported ABI scalar/pointer type to a Python type hint."""
    t = (abi_type or "").strip().lower()
    if t in {"f64", "double", "float"}:
        return "float"
    if t in {"i32", "i64", "int", "int32_t", "int64_t", "usize", "u32"}:
        return "int"
    if t in {"bool"}:
        return "bool"
    if t in {"const char*", "char*", "c_str"}:
        return "str"
    if t in {"void"}:
        return "None"
    if "*" in t:
        # Pointer types are treated as a flat list of the underlying scalar.
        inner = t.replace("*", "").replace("mut", "").replace("const", "").strip()
        return f"list[{_abi_type_to_py(inner)}]"
    return "Any"


def _abi_contract_to_signature(contract: Any) -> str:
    """Convert an ``ABIContract`` signature dict into a Python signature string."""
    sig = contract.signature or {}
    inputs = sig.get("inputs", [])
    outputs = sig.get("outputs", [])
    name = contract.export_symbol or contract.contract_id
    args = ", ".join(
        f"{entry.get('name', 'arg')}: {_abi_type_to_py(entry.get('type', ''))}"
        for entry in inputs
    )
    ret = "None"
    if outputs:
        ret = _abi_type_to_py(outputs[0].get("type", ""))
        if len(outputs) > 1:
            ret = f"tuple[{', '.join(_abi_type_to_py(o.get('type', '')) for o in outputs)}]"
    return f"def {name}({args}) -> {ret}:"


def _contract_to_signature(contract: Any) -> str:
    """Return a Python signature string from a ``ContractEntry`` or ``ABIContract``."""
    if hasattr(contract, "signature") and isinstance(contract.signature, str):
        sig = contract.signature.strip()
        if sig:
            if not sig.endswith(":"):
                sig += ":"
            return sig
    if hasattr(contract, "signature") and isinstance(contract.signature, dict):
        return _abi_contract_to_signature(contract)
    return ""


def _contract_source(contracts: List[Any]) -> str:
    """Build a synthetic Python source containing one stub per contract."""
    lines: List[str] = []
    for contract in contracts:
        sig = _contract_to_signature(contract)
        if sig:
            lines.append(sig)
            lines.append("    pass")
            lines.append("")
    return "\n".join(lines)


def _sample_arg_for_py_type(type_hint: str) -> str:
    """Return a concrete Python literal for a Python type hint."""
    t = (type_hint or "").strip().lower().replace(" ", "")
    if t == "str":
        return '"test"'
    if t == "int":
        return "1"
    if t == "float":
        return "1.0"
    if t == "bool":
        return "True"
    if t == "none":
        return "None"
    if t.startswith("list["):
        inner = t[5:-1].strip() if t.endswith("]") else "float"
        inner_lit = _sample_arg_for_py_type(inner)
        if inner_lit.startswith("[") or inner_lit.startswith("{") or inner_lit.startswith("("):
            return f"[{inner_lit}, {inner_lit}]"
        return f"[{inner_lit}, {inner_lit}, {inner_lit}]"
    if t.startswith("dict["):
        return "{}"
    if t.startswith("tuple["):
        return "(1, 2)"
    return "None"


def _verification_tests(verification_nodes: List[Dict[str, Any]]) -> str:
    """Generate pytest subprocess tests from blueprint verification nodes."""
    if not verification_nodes:
        return ""
    lines: List[str] = ["import re", "import subprocess", "import sys", ""]
    for idx, node in enumerate(verification_nodes):
        test_id = node.get("test_id") or f"verification_{idx}"
        safe_id = "".join(c if c.isalnum() or c == "_" else "_" for c in str(test_id))
        cmd = node.get("execution_cmd", "")
        if isinstance(cmd, list):
            cmd = " ".join(str(x) for x in cmd)
        expected = node.get("expected_exit_code", 0)
        patterns = node.get("stdout_match_patterns", [])
        prohibited = node.get("stderr_prohibited_patterns", [])
        numerical = node.get("numerical_assertions", [])

        lines.append(f"def test_{safe_id}():")
        lines.append(f'    result = subprocess.run({cmd!r}, shell=True, capture_output=True, text=True, timeout=120)')
        lines.append(f'    assert result.returncode == {expected}, result.stderr')
        for pattern in patterns:
            lines.append(f'    assert re.search({pattern!r}, result.stdout), {pattern!r}')
        for pattern in prohibited:
            lines.append(f'    assert not re.search({pattern!r}, result.stderr), {pattern!r}')
        for assertion in numerical:
            metric = assertion.get("target_metric", "")
            expected_val = assertion.get("expected_value", 0)
            atol = assertion.get("absolute_tolerance", 1e-9)
            lines.append(f'    match = re.search(r"{metric}=([-+]?\\d*\\.\\d+|\\d+)", result.stdout)')
            lines.append(f'    assert match, "metric {metric} not found"')
            lines.append(f'    assert abs(float(match.group(1)) - {expected_val}) <= {atol}')
        lines.append("")
    return "\n".join(lines)


def generate_blueprint_tests(
    blueprint: "Blueprint",
    module_name: str,
    *,
    source_contracts: Optional[List[Any]] = None,
) -> str:
    """Generate a pytest file from the contracts and verification nodes in *blueprint*.

    The generated tests import functions from *module_name* and exercise them with
    literal arguments derived from their type annotations.  Verification nodes are
    turned into subprocess tests that assert exit codes, stdout patterns, and
    numerical tolerances.
    """
    from aero_forge.blueprint import ABIContract, ContractEntry

    contracts = source_contracts or []
    if not contracts:
        contracts = list(getattr(blueprint, "contracts", []) or [])
    if not contracts:
        contracts = list(getattr(blueprint, "abi_contracts", []) or [])

    impl_source = _contract_source(contracts)
    smoke = generate_smoke_tests(impl_source, module_name=module_name) if impl_source else ""
    verification = _verification_tests(getattr(blueprint, "verification_nodes", []) or [])

    parts = [p for p in [smoke, verification] if p]
    return "\n".join(parts)
