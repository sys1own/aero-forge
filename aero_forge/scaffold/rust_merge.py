"""Post-process LLM-generated Rust to fix common LLM mistakes.

- Duplicate ``impl`` block methods: LLMs often emit a ``#[pymethods]`` ``impl``
  whose methods mirror an ordinary ``impl`` for the same struct. Rust rejects the
  duplicates, so we rename the Python-exposed wrappers to ``_py_<name>`` and add
  ``#[pyo3(name = "...")]`` aliases.
- PyO3 0.20 ``#[pymodule]`` signature: newer LLM training data uses
  ``&Bound<'_, PyModule>`` from PyO3 0.21+, but aero-forge pins ``pyo3 = 0.20.3``.
  We rewrite the module init function to the 0.20 form
  ``fn name(_py: Python, m: &PyModule) -> PyResult<()>``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Matches a Rust ``impl`` header and captures any attribute lines above it.
_IMPL_HEADER_RE = re.compile(
    r"(?P<attrs>(?:^[ \t]*#\[[^\]]+\][ \t]*\n)*)"
    r"^[ \t]*impl[ \t]+(?P<struct>\w+)(?:[ \t]*for[ \t]*\w+)?[ \t]*\{",
    re.MULTILINE,
)


@dataclass
class Method:
    name: str
    attrs: List[str] = field(default_factory=list)
    text: str = ""
    body: str = ""
    start: int = 0
    end: int = 0


@dataclass
class ImplBlock:
    start: int
    end: int
    is_pymethods: bool
    struct_name: str
    text: str
    methods: List[Method] = field(default_factory=list)


def _is_inside_string_or_comment(line: str, pos: int) -> bool:
    """Return True if *pos* in *line* is inside a string literal or line comment."""
    in_string = False
    string_char = ""  # type: str
    escape = False
    for i, ch in enumerate(line):
        if i == pos:
            return in_string or (not in_string and "//" in line[i:])
        if i > pos:
            break
        if not in_string:
            if ch == "/" and i + 1 < len(line) and line[i + 1] == "/":
                # rest of line is a comment
                if pos > i:
                    return True
                return False
            if ch in ('"', "'"):
                in_string = True
                string_char = ch
                continue
        else:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == string_char:
                in_string = False
                string_char = ""
    return False


def _matching_brace(text: str, open_pos: int) -> int:
    """Return the index of the brace matching the ``{`` at *open_pos*.

    Skips braces inside ``//`` comments and double-quoted strings.  Single
    quotes are not treated as string delimiters because they are used for
    Rust lifetimes (``'_``) and would otherwise swallow braces.
    """
    depth = 0
    in_line_comment = False
    in_string = False
    escape = False
    for i in range(open_pos, len(text)):
        ch = text[i]
        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
            continue
        if in_string:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = False
            continue
        if ch == "/" and i + 1 < len(text) and text[i + 1] == "/":
            in_line_comment = True
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
    raise ValueError("Could not find matching brace")


def _method_boundaries(text: str, body_start: int, body_end: int) -> List[Tuple[int, int]]:
    """Return (start, end) character positions of each top-level method in an impl body."""
    boundaries: List[Tuple[int, int]] = []
    pos = body_start
    while pos < body_end:
        # Find next top-level opening brace (method body or nested block).
        next_open = text.find("{", pos)
        if next_open == -1 or next_open >= body_end:
            break
        # Walk back to the start of the item (attribute/function signature lines).
        item_start = next_open
        while item_start > body_start:
            prev = text.rfind("\n", body_start, item_start)
            if prev == -1:
                break
            line = text[prev + 1 : item_start]
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                item_start = prev
                continue
            if re.match(r"(?:async\s+)?(?:unsafe\s+)?fn\s+\w+", stripped):
                item_start = prev + 1
                continue
            break
        match_end = _matching_brace(text, next_open)
        boundaries.append((item_start, match_end + 1))
        pos = match_end + 1
    return boundaries


def _parse_method(text: str, start: int, end: int) -> Optional[Method]:
    """Parse a method from its attribute lines through closing brace."""
    snippet = text[start:end]
    lines = snippet.splitlines(keepends=True)
    attrs: List[str] = []
    sig_lines: List[str] = []
    body_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#["):
            attrs.append(stripped)
            continue
        if re.match(r"(?:async\s+)?(?:unsafe\s+)?fn\s+\w+", stripped):
            sig_lines.append(line)
            # Body may open on this line or the next.
            brace_idx = line.find("{")
            if brace_idx != -1:
                body_start = sum(len(lines[j]) for j in range(i)) + brace_idx
            else:
                body_start = sum(len(lines[j]) for j in range(i + 1))
                if i + 1 < len(lines):
                    body_start += lines[i + 1].find("{")
            break
    if not sig_lines:
        return None
    m = re.search(r"fn\s+(\w+)", "".join(sig_lines))
    name = m.group(1) if m else ""
    body = text[start + body_start : end]
    return Method(name=name, attrs=attrs, text=snippet, body=body, start=start, end=end)


def _is_wrapper(method: Method) -> bool:
    """Return True if the method body just delegates to a method of the same name."""
    body = method.body.strip()
    if body.startswith("{") and body.endswith("}"):
        body = body[1:-1].strip()
    # Strip trailing semicolon optional.
    body = body.rstrip(";")
    # self.foo(...) or StructName::foo(...) or Self::foo(...)
    patterns = [
        rf"^self\.{re.escape(method.name)}\s*\(.*\)$",
        rf"^(?:Self|(?:[A-Z][A-Za-z0-9_]*))::{re.escape(method.name)}\s*\(.*\)$",
    ]
    return any(re.search(p, body) for p in patterns)


def _extract_struct_methods(block: ImplBlock) -> Dict[str, Method]:
    """Index methods in an impl block by name."""
    result: Dict[str, Method] = {}
    for method in block.methods:
        if method.name:
            result[method.name] = method
    return result


def merge_rust_impl_blocks(source: str) -> str:
    """Rewrite duplicate inherent/PyO3 methods into a compilable layout.

    For each struct that has both an ordinary ``impl`` and a ``#[pymethods] impl``,
    any method name that appears in both is renamed inside the ``#[pymethods]``
    block to ``_py_<name>`` and given a ``#[pyo3(name = "<name>")]`` alias.  The
    body is left unchanged so ``self.<name>(...)`` resolves to the inherent
    implementation at runtime, while Python still sees the original method name.
    """
    try:
        return _merge_rust_impl_blocks(source)
    except Exception:
        # The file may be malformed in ways the lightweight parser cannot handle;
        # leave it for the compiler diagnostic rather than crashing the build.
        return source


def _merge_rust_impl_blocks(source: str) -> str:
    blocks: List[ImplBlock] = []
    for m in _IMPL_HEADER_RE.finditer(source):
        open_brace = source.find("{", m.start("attrs"))
        if open_brace == -1:
            continue
        close_brace = _matching_brace(source, open_brace)
        attrs_text = m.group("attrs") or ""
        is_pymethods = "#[pymethods]" in attrs_text
        block = ImplBlock(
            start=m.start(),
            end=close_brace + 1,
            is_pymethods=is_pymethods,
            struct_name=m.group("struct"),
            text=source[m.start() : close_brace + 1],
        )
        body_start = open_brace + 1
        body_end = close_brace
        for meth_start, meth_end in _method_boundaries(source, body_start, body_end):
            method = _parse_method(source, meth_start, meth_end)
            if method and method.name:
                block.methods.append(method)
        blocks.append(block)

    if not blocks:
        return source

    # Group blocks by struct.
    by_struct: Dict[str, List[ImplBlock]] = {}
    for block in blocks:
        by_struct.setdefault(block.struct_name, []).append(block)

    replacements: List[Tuple[int, int, str]] = []
    for struct_name, struct_blocks in by_struct.items():
        non_pymethods = [b for b in struct_blocks if not b.is_pymethods]
        pymethods = [b for b in struct_blocks if b.is_pymethods]
        if not pymethods or not non_pymethods:
            continue
        inherent_names = set()
        for b in non_pymethods:
            inherent_names.update(_extract_struct_methods(b).keys())

        for pb in pymethods:
            new_methods: List[str] = []
            for method in pb.methods:
                if method.name not in inherent_names:
                    new_methods.append(method.text)
                    continue
                # Rename the Python-exposed wrapper and keep the Python name via pyo3.
                py_name = method.name
                new_name = f"_py_{method.name}"
                # Do not add a duplicate #[pyo3(name)] if already present.
                attrs = method.attrs.copy()
                has_name_attr = any(
                    re.match(r'#\[pyo3\s*\(\s*name\s*=', attr) for attr in attrs
                )
                if not has_name_attr:
                    attrs.append(f'#[pyo3(name = "{py_name}")]')
                # Rewrite the function signature name.
                text = method.text
                text = re.sub(
                    rf"\bfn\s+{re.escape(method.name)}\b",
                    f"fn {new_name}",
                    text,
                    count=1,
                )
                # If the wrapper is literally ``self.name(...)``, leaving it as-is
                # now correctly dispatches to the inherent method because the
                # ``#[pymethods]`` method has a different Rust name.
                if "#[new]" in attrs and method.name == "new":
                    # Keep #[new] marker; the inherent constructor becomes the Python constructor.
                    pass
                attr_text = "\n".join(attrs)
                if attr_text:
                    attr_text += "\n"
                # Insert attributes right before the signature.
                text = re.sub(r"^(\s*)fn\s+", rf"\1{attr_text}\1fn ", text, count=1, flags=re.MULTILINE)
                new_methods.append(text)

            new_block = f"#[pymethods]\nimpl {struct_name} {{\n" + "\n".join(new_methods) + "\n}"
            replacements.append((pb.start, pb.end, new_block))

    if not replacements:
        return source

    # Apply replacements from end to start so indices remain valid.
    parts: List[str] = []
    last = len(source)
    for start, end, text in sorted(replacements, reverse=True):
        parts.append(source[end:last])
        parts.append(text)
        last = start
    parts.append(source[0:last])
    return "".join(reversed(parts))


def fix_pymodule_signature(source: str) -> str:
    """Normalize ``#[pymodule]`` function signatures to PyO3 0.20.x form.

    LLMs trained on newer PyO3 code emit ``fn name(m: &Bound<'_, PyModule>)``,
    which is not valid for the pinned ``pyo3 = 0.20.3`` dependency.  We rewrite
    it to ``fn name(_py: Python, m: &PyModule) -> PyResult<()>`.
    """
    return re.sub(
        r"(?P<before>(?:^[ \t]*//[^\n]*\n)*)"
        r"#\[pymodule\][ \t]*\n"
        r"fn\s+(?P<name>\w+)\s*\(\s*m\s*:\s*&\s*Bound\s*<\s*[^>]*PyModule[^>]*>\s*\)\s*(?:->\s*PyResult<\(\)>\s*)?\{",
        r"\g<before>#[pymodule]\nfn \g<name>(_py: Python, m: &PyModule) -> PyResult<()> {",
        source,
        flags=re.MULTILINE,
    )


def fix_rust_core_impls(workspace_dir: Path) -> bool:
    """Apply Rust post-processing fixes to ``rust_core/src/lib.rs`` if present.

    Returns ``True`` when a file was rewritten.
    """
    lib_rs = workspace_dir / "rust_core" / "src" / "lib.rs"
    if not lib_rs.is_file():
        return False
    original = lib_rs.read_text(encoding="utf-8")
    fixed = merge_rust_impl_blocks(original)
    fixed = fix_pymodule_signature(fixed)
    if fixed != original:
        lib_rs.write_text(fixed, encoding="utf-8")
        return True
    return False
