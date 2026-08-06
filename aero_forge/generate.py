"""Prompt-driven code generation for Aero-Forge."""

from __future__ import annotations

import ast
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from aero_forge.blueprint import (
    Blueprint,
    FunctionSpec,
    discover_functions,
    generate_blueprint,
)
from aero_forge.build_runner import BuildRunner
from aero_forge.config import ConfigOverride, Tier
from aero_forge.errors import UserError
from aero_forge.builder.builder import ProactivePolyglotBuilder
from aero_forge.builder.intent_compiler import IntentCompiler
from aero_forge.healing.healer import DeterministicHealer
from aero_forge.llm.clients import get_llm_client
from aero_forge.overlay import OverlayManager
from aero_forge.overlay.store import OverlayStore
from aero_forge.prompts import get_default_template, get_template
from aero_forge.scaffold.pre_write_validator import PreWriteValidator, ValidationError
from aero_forge.scaffold.test_generator import generate_smoke_tests
from aero_forge.algorithms import (
    Algorithm,
    algorithm_prompt_context,
    find_algorithm,
    get_algorithm,
    select_algorithm,
)

logger = logging.getLogger("aero_forge.generate")


CODE_FENCE_RE = re.compile(
    r"```(?:\w*)\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


class GenerationError(Exception):
    """Raised when prompt-driven code generation fails."""


def _is_universal_project_prompt(prompt: str) -> bool:
    """Return True when the prompt asks for a non-Python language bridge."""
    lower = prompt.lower()
    non_python_languages = [
        "zig",
        "mojo",
        "golang",
        "c#",
        "csharp",
        "java",
        "kotlin",
        "swift",
        "d language",
        "nim",
        "fortran",
    ]
    has_non_python = any(lang in lower for lang in non_python_languages)
    # "go" is ambiguous (e.g. "go ahead"), so only treat it as a language when
    # paired with project/FFI markers.
    if not has_non_python and " go " in lower:
        has_non_python = True
    project_markers = [
        "c-abi",
        "c abi",
        "bridge",
        "link",
        "kernel",
        "emitter",
        "toolchain",
        "shared library",
        "ctypes",
        "zero-copy",
        "jit-synthes",
    ]
    return has_non_python and any(m in lower for m in project_markers)


DEFAULT_SYSTEM_PROMPT = get_default_template().system_prompt


# Words ignored when deriving a module name from the user's prompt.
_STOP_WORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "from",
    "as",
    "is",
    "are",
    "be",
    "being",
    "been",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "can",
    "shall",
    "this",
    "that",
    "these",
    "those",
    "i",
    "you",
    "he",
    "she",
    "it",
    "we",
    "they",
    "me",
    "him",
    "her",
    "us",
    "them",
    "my",
    "your",
    "his",
    "its",
    "our",
    "their",
    "what",
    "which",
    "who",
    "when",
    "where",
    "why",
    "how",
    "all",
    "each",
    "every",
    "some",
    "any",
    "no",
    "write",
    "implement",
    "create",
    "build",
    "generate",
    "make",
    "function",
    "program",
    "code",
    "algorithm",
    "routine",
    "method",
    "fast",
    "optimized",
    "quick",
    "simple",
}

# Names too generic to use as a module name.
_GENERIC_NAMES = {"main", "run", "solve", "helper", "generated", "app", "test"}


_PYTHON_KEYWORDS = {
    "False",
    "None",
    "True",
    "and",
    "as",
    "assert",
    "async",
    "await",
    "break",
    "class",
    "continue",
    "def",
    "del",
    "elif",
    "else",
    "except",
    "finally",
    "for",
    "from",
    "global",
    "if",
    "import",
    "in",
    "is",
    "lambda",
    "nonlocal",
    "not",
    "or",
    "pass",
    "raise",
    "return",
    "try",
    "while",
    "with",
    "yield",
}


def _sanitize_module_name(name: str) -> str:
    """Convert *name* into a valid Python module identifier."""
    name = re.sub(r"[^A-Za-z0-9]+", "_", name)
    # Convert CamelCase to snake_case.
    name = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
    name = re.sub(r"_+", "_", name).strip("_")
    if not name or name[0].isdigit() or name in _PYTHON_KEYWORDS or name in _GENERIC_NAMES:
        name = "engine"
    return name[:40]


def _detect_public_names(source: str) -> List[str]:
    """Return public top-level function and class names from *source*."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    names: List[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not node.name.startswith("_"):
            names.append(node.name)
    return names


def _derive_module_name(prompt: str, implementation: str, existing: Optional[str] = None) -> str:
    """Pick a domain-specific Python module name from context and code."""
    if existing:
        return existing
    for name in _detect_public_names(implementation):
        if name not in _GENERIC_NAMES:
            return _sanitize_module_name(name)
    words = re.findall(r"[A-Za-z]+", prompt or "")
    filtered = [w for w in words if w.lower() not in _STOP_WORDS and len(w) > 1]
    if filtered:
        # Use the first 1-3 meaningful words to keep names concise but descriptive.
        return _sanitize_module_name("_".join(filtered[:3]))
    return "generated"


def _derive_export_names(source: str) -> List[str]:
    """Return public top-level functions/classes that ``src/__init__.py`` should re-export."""
    return _detect_public_names(source)


def _find_generated_python_paths(output_dir: Path) -> Tuple[Path, Path]:
    """Return the primary implementation and test paths in ``output_dir``.

    Falls back to ``src/generated.py`` / ``tests/test_generated.py`` when no
    generated source has been written yet.
    """
    src_dir = output_dir / "src"
    tests_dir = output_dir / "tests"
    candidates = [p for p in src_dir.glob("*.py") if p.name != "__init__.py"]
    if candidates:
        source_path = candidates[0]
        test_path = tests_dir / f"test_{source_path.stem}.py"
        if not test_path.is_file():
            alt = tests_dir / f"test_{source_path.stem}.py"
            test_path = alt
        return source_path, test_path
    return src_dir / "generated.py", tests_dir / "test_generated.py"


def _rewrite_generated_imports(tests: str, module_name: str) -> str:
    """Point tests at the real module name instead of the ``generated`` placeholder."""
    if not tests:
        return tests
    tests = re.sub(r"\bfrom\s+generated\s+import\b", f"from {module_name} import", tests)
    tests = re.sub(r"\bimport\s+generated\b", f"import {module_name}", tests)
    return tests


def _build_user_prompt(
    prompt: str,
    constraints: Optional[str] = None,
    algorithm_context: Optional[str] = None,
) -> str:
    parts = [f"Request: {prompt}"]
    if constraints:
        parts.append(f"Constraints: {constraints}")
    if algorithm_context:
        parts.append(algorithm_context)
    parts.append(
        "\nReturn the Python implementation in a ```python block and the pytest "
        "tests in a second ```python block. The test file must import from "
        "`generated` (e.g. `from generated import function_name`).\n"
    )
    return "\n".join(parts)


def extract_code_blocks(text: str) -> List[Tuple[Optional[str], str]]:
    """Extract all ```...``` code fences from ``text``.

    Returns a list of ``(language_hint, code)`` tuples. The hint is the token
    after the opening backticks, if any.
    """
    blocks: List[Tuple[Optional[str], str]] = []
    pattern = re.compile(r"```\s*(\w*)\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
    for match in pattern.finditer(text):
        lang = match.group(1).lower() or None
        code = match.group(2).strip("\n")
        blocks.append((lang, code))
    return blocks


def parse_generated_response(text: str) -> Tuple[str, str]:
    """Parse LLM response into (implementation, tests).

    Falls back to treating the first Python block as implementation and all
    remaining blocks as tests, or to extracting plain ``def`` functions from
    markdown-free responses.
    """
    blocks = extract_code_blocks(text)
    python_blocks: List[str] = []
    if blocks:
        python_blocks = [code for lang, code in blocks if lang in (None, "python", "py")]
        if not python_blocks:
            python_blocks = [code for _, code in blocks]

    if not python_blocks:
        # No markdown fences; extract plain ``def`` functions from raw text.
        impl, tests = _extract_functions_from_text(text)
        if not impl:
            raise GenerationError("No code blocks or function definitions found in LLM response")
        return impl, tests

    if len(python_blocks) >= 2:
        return python_blocks[0], python_blocks[1]

    # Single block: split at a test function boundary if present.
    source = python_blocks[0]
    match = re.search(r"\n(?=def test_)", source)
    if match:
        impl = source[: match.start()]
        tests = source[match.start() + 1 :]
        return impl, tests

    # Could not find separate tests; return the whole block as implementation
    # and let the caller generate smoke tests if needed.
    return source, ""


def extract_explanation(text: str) -> str:
    """Extract a free-form explanation section from an LLM response.

    Looks for an '## Explanation' or '### Explanation' markdown section and
    returns the text up to the next heading or code fence.  Returns an empty
    string if no explanation section is found.
    """
    match = re.search(
        r"(?:^|\n)\s*#+\s*Explanation\s*\n(.*?)(?=\n\s*#+ |\n```|\Z)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    # Fallback: look for an explicit EXPLANATION: marker.
    match = re.search(
        r"(?:^|\n)\s*EXPLANATION:\s*(.*?)(?=\n\s*[A-Z][A-Z_\s]{2,}:\s|\Z)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    return ""


def _extract_functions_from_text(text: str) -> Tuple[str, str]:
    """Extract the first implementation and any test functions from raw text."""
    lines = text.splitlines()
    boundaries: List[int] = []
    for i, line in enumerate(lines):
        if line.startswith("def "):
            boundaries.append(i)
    if not boundaries:
        return "", ""
    blocks: List[str] = []
    for idx, start in enumerate(boundaries):
        end = boundaries[idx + 1] if idx + 1 < len(boundaries) else len(lines)
        blocks.append("\n".join(lines[start:end]).strip())
    impl_blocks = [b for b in blocks if not b.startswith("def test_")]
    test_blocks = [b for b in blocks if b.startswith("def test_")]
    impl = impl_blocks[0] if impl_blocks else blocks[0]
    # Auto-generate a minimal test if none were provided.
    tests = "\n\n".join(test_blocks) if test_blocks else ""
    return impl, tests


def generate_from_prompt(
    prompt: str,
    *,
    constraints: Optional[str] = None,
    llm_provider: Optional[str] = None,
    model: Optional[str] = None,
    max_retries: int = 3,
    max_tokens: Optional[int] = None,
    system_prompt: Optional[str] = None,
    prompt_template: Optional[str] = None,
    algorithm_library: bool = False,
    selected_algorithm: Optional[str] = None,
    discover: bool = False,
    explain: bool = False,
    config_override: Optional[ConfigOverride] = None,
) -> str:
    """Call the configured LLM and return the raw generated text."""
    client = get_llm_client(
        llm_provider,
        model=model,
        max_retries=max_retries,
        config_override=config_override,
        tier=Tier.REASONING,
    )
    if client is None:
        raise GenerationError(
            f"LLM provider '{llm_provider}' is not configured or no API key is set"
        )

    if prompt_template:
        template = get_template(prompt_template)
    elif system_prompt:
        from aero_forge.prompts import PromptTemplate

        template = PromptTemplate("custom", system_prompt)
    else:
        template = get_default_template()

    selected: Optional[Algorithm] = None
    if selected_algorithm:
        selected = get_algorithm(selected_algorithm)
    elif algorithm_library:
        selected = select_algorithm(
            prompt, llm_provider=llm_provider, model=model, config_override=config_override
        )
        if selected is None and not discover:
            raise GenerationError(
                "No library algorithm matched the prompt. Use --discover to "
                "design a new algorithm."
            )
    else:
        selected = find_algorithm(prompt)

    algorithm_context = algorithm_prompt_context(
        prompt, selected=selected, algorithm_library=algorithm_library
    )
    user_prompt = _build_user_prompt(prompt, constraints, algorithm_context)
    if algorithm_library and selected:
        user_prompt += (
            "\nAdapt the selected reference algorithm to the request. "
            "Only use the algorithm above; do not invent a different approach."
        )
    if algorithm_library and selected is None and discover:
        user_prompt += (
            "\nNo existing algorithm in the library matched this request. "
            "Design a novel algorithm, explain your approach, and implement it."
        )
    if explain:
        user_prompt += (
            "\nAfter the code blocks, add an '## Explanation' section describing "
            "the algorithm choice, complexity, and tradeoffs."
        )

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": template.system_prompt},
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    if max_tokens is None:
        max_tokens = int(os.getenv("AERO_FORGE_MAX_TOKENS", "4096"))
    response = client.generate(messages, temperature=0.2, max_tokens=max_tokens)
    if not response:
        raise GenerationError("LLM returned an empty response")
    return response


def write_generated_project(
    output_dir: Path,
    implementation: str,
    tests: str,
    project_name: str = "generated_project",
    prompt: str = "",
    constraints: Optional[str] = None,
    module_name: Optional[str] = None,
    validate: bool = True,
) -> Tuple[Path, Path, Blueprint]:
    """Write implementation, tests, and a blueprint to ``output_dir``.

    The module filename is derived from the primary function/class in
    *implementation* or from the *prompt* domain context so workspaces use
    descriptive names instead of generic ``generated.py``.

    Runs pre-write validation and performs an active structural merge when a
    previous generated baseline exists, preserving user edits from the workspace.
    Returns ``(source_path, test_path, blueprint)``.
    """
    # Normalise generated text to end with a single newline so structural merges
    # and downstream line-oriented tools behave consistently.
    if not implementation.endswith("\n"):
        implementation += "\n"
    if tests and not tests.endswith("\n"):
        tests += "\n"

    # Auto-correct common generator mistakes such as ``return []`` in
    # matrix/array functions before validation sees them.
    from aero_forge.scaffold.pre_write_validator import rewrite_empty_matrix_returns

    implementation = rewrite_empty_matrix_returns(implementation)

    src_dir = output_dir / "src"
    tests_dir = output_dir / "tests"
    src_dir.mkdir(parents=True, exist_ok=True)
    tests_dir.mkdir(parents=True, exist_ok=True)

    # Preserve the existing module name across incremental rewrites so overlays
    # remain anchored to the same file.
    if module_name is None:
        existing_modules = [p.stem for p in src_dir.glob("*.py") if p.name != "__init__.py"]
        existing_stem = existing_modules[0] if existing_modules else None
        module_name = _derive_module_name(prompt, implementation, existing=existing_stem)

    # Point smoke/generated tests at the real module name.
    tests = _rewrite_generated_imports(tests, module_name)

    source_path = src_dir / f"{module_name}.py"
    test_path = tests_dir / f"test_{module_name}.py"

    if validate:
        validator = PreWriteValidator(context={}, language="python")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / f"{module_name}.py").write_text(implementation, encoding="utf-8")
            validator.validate(tmp_path, language="python")

    # Active structural merge: preserve committed user overlays on re-generation.
    overlay_store = OverlayStore(
        output_dir,
        build_cache_dir=".aero/build_cache",
        overlays_dir=".aero/overlays",
    )
    overlay_manager = OverlayManager(output_dir, store=overlay_store)
    if source_path.is_file():
        reapply_status = overlay_manager.structural_reapply(
            source_path, implementation, language="python"
        )
        if reapply_status.name == "APPLIED":
            implementation = source_path.read_text(encoding="utf-8")
        else:
            source_path.write_text(implementation, encoding="utf-8")
    else:
        source_path.write_text(implementation, encoding="utf-8")
    overlay_manager.record_generated(source_path)

    test_path.write_text(tests, encoding="utf-8")
    overlay_manager.record_generated(test_path)

    # Expose generated public functions/classes through the package root.
    export_names = _derive_export_names(implementation)
    init_path = src_dir / "__init__.py"
    if export_names:
        init_lines = [f"from .{module_name} import {', '.join(export_names)}", ""]
        init_lines.append("__all__ = [" + ", ".join(f'"{n}"' for n in export_names) + "]")
        init_path.write_text("\n".join(init_lines) + "\n", encoding="utf-8")
    else:
        init_path.write_text("# Generated Aero-Forge module\n", encoding="utf-8")

    functions = [
        FunctionSpec(
            file=source_path,
            name=name,
            tests=[test_path],
        )
        for name in _detect_function_names(implementation)
    ]
    blueprint = generate_blueprint(
        project=project_name,
        functions=functions,
        output_dir=output_dir / "dist",
        prompt=prompt,
        constraints=constraints,
    )
    blueprint_path = output_dir / "blueprint.aero"
    from aero_forge.blueprint import write_blueprint

    write_blueprint(blueprint, blueprint_path)

    return source_path, test_path, blueprint


def _detect_function_names(source: str) -> List[str]:
    """Return the public top-level function names in ``source``.

    Falls back to token-based discovery on syntax errors.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return _token_function_names(source)
    seen: set[str] = set()
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            if node.name not in seen:
                seen.add(node.name)
                names.append(node.name)
    return names


def _token_function_names(source: str) -> List[str]:
    """Simple fallback regex extraction of function names."""
    names = re.findall(r"^\s*def\s+([A-Za-z_]\w*)", source, re.MULTILINE)
    return [n for n in names if not n.startswith("_")]


GeneratedProject = Tuple[Path, Path, Blueprint, str, str, str]


def _review_code(
    implementation: str,
    prompt: str,
    constraints: Optional[str],
    llm_provider: Optional[str],
    model: Optional[str],
    max_retries: int,
    prompt_template: Optional[str] = None,
    config_override: Optional[ConfigOverride] = None,
) -> str:
    """Ask the LLM to review and improve generated code.

    Returns the corrected implementation. If the LLM is unavailable or the
    response cannot be parsed, the original implementation is returned.
    """
    client = get_llm_client(
        llm_provider,
        model=model,
        max_retries=max_retries,
        config_override=config_override,
        tier=Tier.REASONING,
    )
    if client is None:
        return implementation

    system = (
        "You are a senior engineer doing a strict code review. Check the code "
        "for correctness, performance, security, and style. If you find issues, "
        "output a corrected version in a ```python block. If no issues are "
        "found, return the original code unchanged."
    )
    user = (
        f"Original request: {prompt}\n"
        f"Constraints: {constraints or 'None'}\n\n"
        f"Implementation to review:\n```python\n{implementation}\n```\n\n"
        "Provide a brief review note and the corrected code in a single "
        "```python block."
    )
    response = client.generate(
        [
            {
                "role": "system",
                "content": (
                    get_template(prompt_template).system_prompt
                    if prompt_template
                    else get_default_template().system_prompt
                ),
            },
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    if not response:
        return implementation
    try:
        blocks = extract_code_blocks(response)
        for _, code in blocks:
            if code.strip():
                return code
    except Exception:
        pass
    return implementation


def _flatten_nested_functions(source: str) -> str:
    """Lift nested function definitions to module level so HIN/Rust can emit them.

    Only safe when the nested function does not close over outer local variables.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    builtins = {*dir(__builtins__)} if isinstance(__builtins__, dict) else {*dir(__builtins__)}
    new_top_level: List[ast.FunctionDef] = []

    def _assigned_names(body: List[ast.stmt]) -> Set[str]:
        names: Set[str] = set()
        for stmt in body:
            for child in ast.walk(stmt):
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                    names.add(child.id)
        return names

    for top_node in list(tree.body):
        if not isinstance(top_node, ast.FunctionDef):
            continue
        outer_locals = {arg.arg for arg in top_node.args.args}
        outer_locals.update(a.arg for a in getattr(top_node.args, "posonlyargs", []))
        outer_locals.update(a.arg for a in getattr(top_node.args, "kwonlyargs", []))
        outer_locals.add(top_node.name)
        outer_locals.update(_assigned_names(top_node.body))

        body = list(top_node.body)
        for idx in range(len(body) - 1, -1, -1):
            stmt = body[idx]
            if not isinstance(stmt, ast.FunctionDef):
                continue
            nested = stmt
            nested_locals = {arg.arg for arg in nested.args.args}
            nested_locals.update(_assigned_names(nested.body))
            free_vars: Set[str] = set()
            for child in ast.walk(nested):
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                    if child.id in builtins or child.id in nested_locals:
                        continue
                    if child.id in outer_locals:
                        free_vars.add(child.id)
            if free_vars:
                continue

            old_name = nested.name
            new_name = f"{top_node.name}_{old_name}"
            for child in ast.walk(nested):
                if isinstance(child, ast.Name) and child.id == old_name:
                    child.id = new_name
            nested.name = new_name
            ast.fix_missing_locations(nested)
            new_top_level.append(nested)

            body[idx] = ast.Pass()
            for child_stmt in body:
                for child in ast.walk(child_stmt):
                    if isinstance(child, ast.Name) and child.id == old_name:
                        child.id = new_name
            top_node.body = body

    if new_top_level:
        tree.body = new_top_level + tree.body
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)
    return source


def _normalize_numeric_literals(source: str) -> str:
    """Fix malformed numeric literals produced by some LLMs.

    Some model outputs include underscores adjacent to the decimal point
    (``1_.0`` or ``1._0``) which are invalid Python.  Remove the underscore
    while leaving valid digit-grouping underscores like ``1_000.0`` intact.
    """
    source = re.sub(r"(\d+)_\.(\d+)", r"\1.\2", source)
    source = re.sub(r"(\d+)\._(\d+)", r"\1.\2", source)
    return source


def sanitize_generated_code(source: str) -> str:
    """Remove unsupported constructs that commonly appear in LLM output.

    This is a router-level cleanup: it strips ``raise`` and ``assert``
    statements because the Aero-Forge transpiler does not support them,
    while preserving as much of the generated numeric function as possible.
    """
    source = _normalize_numeric_literals(source)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    class Sanitizer(ast.NodeTransformer):
        def visit_Raise(self, node: ast.Raise) -> ast.AST:  # type: ignore[misc]
            return ast.Pass()

        def visit_Assert(self, node: ast.Assert) -> ast.AST:  # type: ignore[misc]
            return ast.Pass()

    sanitized = ast.unparse(Sanitizer().visit(tree))
    return _flatten_nested_functions(sanitized)


def _has_float_literal(node: ast.AST) -> bool:
    """Return True when *node* contains any float Constant."""
    return any(isinstance(n, ast.Constant) and isinstance(n.value, float) for n in ast.walk(node))


def _is_literal_expression(node: ast.AST) -> bool:
    """Return True for simple literal containers used as expected values."""
    return isinstance(node, (ast.Constant, ast.List, ast.Tuple, ast.Set, ast.Dict))


def _is_nested_list_literal(node: ast.expr) -> bool:
    """Return True when ``node`` is a list literal containing other lists."""
    return isinstance(node, ast.List) and any(
        isinstance(elt, ast.List) for elt in node.elts
    )


def _is_pytest_approx(node: ast.AST) -> bool:
    """Return True if *node* is already ``pytest.approx(...)``."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "pytest"
        and node.func.attr == "approx"
    )


def _is_pytest_approx_call(node: ast.expr) -> Optional[ast.expr]:
    """Return the inner argument if ``node`` is ``pytest.approx(...)`` or ``pytest.approx(value, rel=...)``."""
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "pytest"
        and func.attr == "approx"
    ):
        if node.args:
            return node.args[0]
    return None


def _normalize_float_assertions(tests: str) -> str:
    """Rewrite equality assertions on floats to use ``pytest.approx`` safely.

    Floating-point arithmetic in compiled Rust can produce tiny differences
    (e.g. ``1.7000000000000002`` instead of ``1.7``). For scalar/flat values we
    wrap the expected literal in ``pytest.approx``. For nested matrices we
    emit a row-level helper so ``pytest.approx`` is not applied to nested
    containers (which raises a TypeError).
    """
    try:
        tree = ast.parse(tests)
    except SyntaxError:
        return tests

    def _collect_nested_matrix_vars(tree: ast.AST) -> Set[str]:
        """Return names assigned to a nested list literal anywhere in *tree*."""
        names: Set[str] = set()
        for stmt in ast.walk(tree):
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name) and _is_nested_list_literal(stmt.value):
                        names.add(target.id)
        return names

    needs_nested_helper = False

    class ApproxTransformer(ast.NodeTransformer):
        def __init__(self, nested_vars: Set[str]) -> None:
            self.nested_vars = nested_vars

        def _is_nested_expected(self, expr: ast.expr) -> bool:
            return _is_nested_list_literal(expr) or (
                isinstance(expr, ast.Name) and expr.id in self.nested_vars
            )

        def visit_Assert(self, node: ast.Assert) -> ast.AST:
            nonlocal needs_nested_helper
            test = node.test
            if (
                not isinstance(test, ast.Compare)
                or len(test.ops) != 1
                or not isinstance(test.ops[0], ast.Eq)
            ):
                return node
            left = test.left
            right = test.comparators[0]

            # Case 1: the LLM already wrapped the expected value in
            # ``pytest.approx``.  Replace with a row-wise approx helper when
            # the expectation is a nested matrix.
            approx_inner = _is_pytest_approx_call(left)
            if approx_inner is not None:
                expected, actual = approx_inner, right
                if self._is_nested_expected(expected):
                    needs_nested_helper = True
                    new_test = ast.Call(
                        func=ast.Name(id="_aero_approx_nested", ctx=ast.Load()),
                        args=[actual, expected],
                        keywords=[],
                    )
                    return ast.Assert(test=new_test, msg=node.msg)
                return node
            approx_inner = _is_pytest_approx_call(right)
            if approx_inner is not None:
                expected, actual = approx_inner, left
                if self._is_nested_expected(expected):
                    needs_nested_helper = True
                    new_test = ast.Call(
                        func=ast.Name(id="_aero_approx_nested", ctx=ast.Load()),
                        args=[actual, expected],
                        keywords=[],
                    )
                    return ast.Assert(test=new_test, msg=node.msg)
                return node

            left_has_float = _has_float_literal(left)
            right_has_float = _has_float_literal(right)
            if not left_has_float and not right_has_float:
                return node
            if _is_literal_expression(right) and right_has_float:
                expected, actual = right, left
            elif _is_literal_expression(left) and left_has_float:
                expected, actual = left, right
            else:
                return node
            if self._is_nested_expected(expected):
                needs_nested_helper = True
                new_test = ast.Call(
                    func=ast.Name(id="_aero_approx_nested", ctx=ast.Load()),
                    args=[actual, expected],
                    keywords=[],
                )
                return ast.Assert(test=new_test, msg=node.msg)
            approx_call = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="pytest", ctx=ast.Load()),
                    attr="approx",
                    ctx=ast.Load(),
                ),
                args=[expected],
                keywords=[],
            )
            new_test = ast.Compare(
                left=actual,
                ops=[ast.Eq()],
                comparators=[approx_call],
            )
            new_test.lineno = getattr(test, "lineno", None)
            new_test.col_offset = getattr(test, "col_offset", None)
            return ast.Assert(test=new_test, msg=node.msg)

    nested_vars = _collect_nested_matrix_vars(tree)
    tree = ApproxTransformer(nested_vars).visit(tree)
    ast.fix_missing_locations(tree)
    result = ast.unparse(tree)
    helper = (
        "def _aero_approx_nested(actual, expected):\n"
        "    assert len(actual) == len(expected)\n"
        "    for a_row, e_row in zip(actual, expected):\n"
        "        assert len(a_row) == len(e_row)\n"
        "        for a, e in zip(a_row, e_row):\n"
        "            assert a == pytest.approx(e)\n"
        "    return True\n"
    )
    if needs_nested_helper:
        prefix = ""
        if "import pytest" not in result:
            prefix += "import pytest\n"
        if "def _aero_approx_nested" not in result:
            prefix += helper + "\n"
        result = prefix + result
    elif "import pytest" not in result:
        result = "import pytest\n" + result
    return result


def generate_project(
    prompt: str,
    *,
    constraints: Optional[str] = None,
    output_dir: Path = Path("."),
    project_name: str = "generated_project",
    llm_provider: Optional[str] = None,
    model: Optional[str] = None,
    max_retries: int = 3,
    max_tokens: Optional[int] = None,
    prompt_template: Optional[str] = None,
    algorithm_library: bool = False,
    selected_algorithm: Optional[str] = None,
    discover: bool = False,
    explain: bool = False,
    review: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None,
    config_override: Optional[ConfigOverride] = None,
) -> GeneratedProject:
    """Generate code from a prompt and write the project files.

    Returns ``(source_path, test_path, blueprint, implementation, tests, explanation)``.
    """
    if progress_callback:
        progress_callback("Generating code from your prompt...")
    response = generate_from_prompt(
        prompt,
        constraints=constraints,
        llm_provider=llm_provider,
        model=model,
        max_retries=max_retries,
        max_tokens=max_tokens,
        prompt_template=prompt_template,
        algorithm_library=algorithm_library,
        selected_algorithm=selected_algorithm,
        discover=discover,
        explain=explain,
        config_override=config_override,
    )
    implementation, tests = parse_generated_response(response)
    implementation = sanitize_generated_code(implementation)
    if review:
        implementation = _review_code(
            implementation,
            prompt,
            constraints,
            llm_provider,
            model,
            max_retries,
            prompt_template=prompt_template,
            config_override=config_override,
        )
        implementation = sanitize_generated_code(implementation)
    explanation = extract_explanation(response) if explain else ""

    # Derive a domain-specific module name once so smoke tests and the saved
    # file are consistent.
    existing_modules = (
        [p.stem for p in (output_dir / "src").glob("*.py") if p.name != "__init__.py"]
        if (output_dir / "src").is_dir()
        else []
    )
    module_name = _derive_module_name(
        prompt, implementation, existing=existing_modules[0] if existing_modules else None
    )
    if not tests.strip():
        tests = generate_smoke_tests(implementation, module_name=module_name)
    else:
        tests = _rewrite_generated_imports(tests, module_name)
    tests = _normalize_numeric_literals(tests)
    tests = _normalize_float_assertions(tests)
    source_path, test_path, blueprint = write_generated_project(
        output_dir,
        implementation,
        tests,
        project_name=project_name,
        prompt=prompt,
        constraints=constraints,
        module_name=module_name,
    )
    if progress_callback:
        progress_callback("Code written; ready to compile.")
    return source_path, test_path, blueprint, implementation, tests, explanation


def generate_and_build(
    prompt: str,
    *,
    constraints: Optional[str] = None,
    output_dir: Path = Path("."),
    project_name: str = "generated_project",
    llm_provider: Optional[str] = None,
    model: Optional[str] = None,
    max_retries: int = 3,
    max_tokens: Optional[int] = None,
    max_iterations: int = 5,
    build_kwargs: Optional[Dict[str, Any]] = None,
    optimize: bool = False,
    prompt_template: Optional[str] = None,
    algorithm_library: bool = False,
    selected_algorithm: Optional[str] = None,
    variants: int = 1,
    discover: bool = False,
    explain: bool = False,
    review: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None,
    config_override: Optional[ConfigOverride] = None,
    target_language: Optional[str] = None,
    acceleration_policy: Optional[str] = None,
    engine_backend: Optional[str] = None,
    wavefront_parallelism: Optional[int] = None,
    precision_shield_mode: Optional[str] = None,
    hin_jit_opt_level: Optional[int] = None,
) -> Dict[str, Any]:
    """Generate code from a prompt and optionally build/optimize it.

    Returns a dictionary describing the generated files and build result.
    """
    if wavefront_parallelism is not None:
        try:
            wavefront_parallelism = int(wavefront_parallelism)
        except (TypeError, ValueError):
            wavefront_parallelism = None
    if hin_jit_opt_level is not None:
        try:
            hin_jit_opt_level = int(hin_jit_opt_level)
        except (TypeError, ValueError):
            hin_jit_opt_level = None

    if (build_kwargs is not None or optimize) and _is_universal_project_prompt(prompt):
        # Project-level prompts that name a non-Python language and an FFI
        # bridge should be routed through the graph materializer, which can
        # JIT-synthesize the required emitter plugin and validate boundaries.
        output_dir.mkdir(parents=True, exist_ok=True)
        return ProactivePolyglotBuilder().synthesize_and_build(
            prompt,
            output_dir,
            llm_provider=llm_provider,
            llm_model=model,
            prompt_template=prompt_template,
            max_retries=max_retries,
        )

    if variants > 1:
        from aero_forge.variants import generate_variants, select_best_variant

        variant_results = generate_variants(
            prompt,
            variants=variants,
            output_dir=output_dir,
            project_name=project_name,
            constraints=constraints,
            llm_provider=llm_provider,
            model=model,
            max_retries=max_retries,
            max_tokens=max_tokens,
            prompt_template=prompt_template,
            algorithm_library=algorithm_library,
            selected_algorithm=selected_algorithm,
            discover=discover,
            explain=explain,
            review=review,
            config_override=config_override,
            engine_backend=engine_backend,
            wavefront_parallelism=wavefront_parallelism,
            precision_shield_mode=precision_shield_mode,
            hin_jit_opt_level=hin_jit_opt_level,
        )
        best = select_best_variant(variant_results, output_dir=output_dir)
        best["variants"] = variant_results
        return best

    try:
        (
            source_path,
            test_path,
            blueprint,
            implementation,
            tests,
            explanation,
        ) = generate_project(
            prompt,
            constraints=constraints,
            output_dir=output_dir,
            project_name=project_name,
            llm_provider=llm_provider,
            model=model,
            max_retries=max_retries,
            max_tokens=max_tokens,
            prompt_template=prompt_template,
            algorithm_library=algorithm_library,
            selected_algorithm=selected_algorithm,
            discover=discover,
            explain=explain,
            review=review,
            config_override=config_override,
        )
    except ValidationError as exc:
        return {
            "source_path": "",
            "test_path": "",
            "blueprint_path": str(output_dir / "blueprint.aero"),
            "implementation": "",
            "tests": "",
            "explanation": "",
            "build": {
                "success": False,
                "error": "pre-write validation failed",
                "logs": exc.output,
            },
            "iterations": [],
        }
    except GenerationError as exc:
        # When the single-function generator returns an empty/malformed response,
        # the prompt is likely a project-level polyglot request.  Route it through
        # the proactive graph builder, which JIT-synthesizes missing emitter plugins.
        if "empty" in str(exc).lower() or "malformed" in str(exc).lower():
            return ProactivePolyglotBuilder().synthesize_and_build(
                prompt,
                output_dir,
                llm_provider=llm_provider,
                llm_model=model,
                prompt_template=prompt_template,
                max_retries=max_retries,
            )
        raise

    result: Dict[str, Any] = {
        "source_path": str(source_path),
        "test_path": str(test_path),
        "blueprint_path": str(output_dir / "blueprint.aero"),
        "implementation": implementation,
        "tests": tests,
        "explanation": explanation,
        "build": None,
        "iterations": [],
        "target_language": target_language or "auto",
        "acceleration_policy": acceleration_policy or "selective",
        "engine_backend": engine_backend or "default",
        "wavefront_parallelism": wavefront_parallelism,
        "precision_shield_mode": precision_shield_mode,
        "hin_jit_opt_level": hin_jit_opt_level,
    }

    if progress_callback:
        progress_callback("Compiling to Rust...")

    if optimize:
        result["iterations"] = optimize_generated_code(
            output_dir=output_dir,
            prompt=prompt,
            constraints=constraints,
            llm_provider=llm_provider,
            model=model,
            max_retries=max_retries,
            max_iterations=max_iterations,
            prompt_template=prompt_template,
            progress_callback=progress_callback,
            config_override=config_override,
        )
        result["build"] = result["iterations"][-1].get("build") if result["iterations"] else None
    elif build_kwargs is not None:
        # Derive BuildRunner settings from the UI/CLI acceleration selections.
        derived_kwargs: Dict[str, Any] = {
            "engine_backend": engine_backend,
            "precision_shield_mode": precision_shield_mode,
            "hin_jit_opt_level": hin_jit_opt_level,
        }
        if wavefront_parallelism is not None and wavefront_parallelism > 0:
            derived_kwargs["max_workers"] = wavefront_parallelism
        if target_language:
            tl = str(target_language).lower()
            if tl == "wasm":
                derived_kwargs["target"] = "wasm32-unknown-unknown"
        if engine_backend:
            eb = str(engine_backend).lower().replace("-", "_")
            if eb in ("hin_gpu", "gpu", "hin_cuda", "hin_vulkan"):
                derived_kwargs["gpu"] = True
            if eb in ("hin_wasm", "wasm"):
                derived_kwargs["target"] = "wasm32-unknown-unknown"
        # Merge with caller-supplied build_kwargs; explicit acceleration args win.
        build_kwargs = {**build_kwargs, **derived_kwargs}
        bp = Blueprint.model_validate(
            {
                "project": project_name,
                "functions": [
                    {
                        "file": str(source_path),
                        "name": name,
                        "tests": [str(test_path)],
                    }
                    for name in _detect_function_names(implementation)
                ],
                "output_dir": str(output_dir / "dist"),
            }
        )
        if project_name != "core" and llm_provider and llm_provider != "none":
            try:
                from aero_forge.blueprint import write_blueprint

                compiler = IntentCompiler(
                    provider=llm_provider,
                    model=model,
                    max_retries=max_retries,
                    config_override=config_override,
                )
                intent_bp = compiler.compile_prompt(
                    prompt,
                    output_dir=None,
                    project_name=project_name,
                )
                bp.metadata = intent_bp.metadata
                bp.execution_strategy = intent_bp.execution_strategy
                bp.abi_contracts = intent_bp.abi_contracts
                bp.module_graph = intent_bp.module_graph
                # Verification nodes are left empty for the single-function path
                # because the generated source name is not known by the compiler.
                bp.verification_nodes = []
                write_blueprint(bp, output_dir / "blueprint.aero")
            except Exception as exc:
                logger.warning("IntentCompiler enrichment failed in generate_and_build: %s", exc)
        if progress_callback:
            progress_callback("Running tests...")
        runner = BuildRunner(bp, **build_kwargs, config_override=config_override)
        result["build"] = runner.build()
        if progress_callback:
            build = result["build"] or {}
            status = "passed" if build.get("success") else "failed"
            progress_callback(f"Build {status}.")

    return result


def optimize_generated_code(
    output_dir: Path,
    prompt: str,
    *,
    constraints: Optional[str] = None,
    llm_provider: Optional[str] = None,
    model: Optional[str] = None,
    max_retries: int = 3,
    max_iterations: int = 5,
    prompt_template: Optional[str] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    config_override: Optional[ConfigOverride] = None,
) -> List[Dict[str, Any]]:
    """Iteratively compile, benchmark, and optimize generated code.

    Runs at least three iterations when ``max_iterations >= 3`` so there is a
    baseline, an optimized candidate, and a validation run. After that the loop
    stops when the benchmark no longer improves.
    """
    import time

    iterations: List[Dict[str, Any]] = []
    source_path, test_path = _find_generated_python_paths(output_dir)
    previous_time: Optional[float] = None

    for iteration in range(1, max_iterations + 1):
        if progress_callback:
            progress_callback(f"Optimization iteration {iteration}/{max_iterations}...")
        implementation = source_path.read_text(encoding="utf-8")
        bp = Blueprint.model_validate(
            {
                "project": f"generated_project_iter_{iteration}",
                "functions": [
                    {
                        "file": str(source_path),
                        "name": name,
                        "tests": [str(test_path)],
                    }
                    for name in _detect_function_names(implementation)
                ],
                "output_dir": str(output_dir / "dist"),
            }
        )

        start = time.perf_counter()
        if progress_callback:
            progress_callback("Compiling optimized version...")
        runner = BuildRunner(
            bp,
            max_workers=1,
            cache_enabled=False,
            config_override=config_override,
        )
        build_result = runner.build()
        elapsed = time.perf_counter() - start

        iteration_result: Dict[str, Any] = {
            "iteration": iteration,
            "build": build_result,
            "benchmark_seconds": elapsed,
        }
        iterations.append(iteration_result)

        if not build_result.get("success"):
            error_log = "\n".join(r.get("logs", "") for r in build_result.get("results", []))
            fixed = _ask_for_fix(
                implementation,
                error_log,
                prompt,
                constraints,
                llm_provider,
                model,
                max_retries,
                prompt_template=prompt_template,
                config_override=config_override,
            )
            if fixed:
                source_path.write_text(fixed, encoding="utf-8")
            continue

        # Ask the LLM to optimize the working implementation.
        if iteration < 3 or (previous_time is not None and elapsed < previous_time * 0.99):
            previous_time = elapsed
            optimized = _ask_for_optimize(
                implementation,
                elapsed,
                prompt,
                constraints,
                llm_provider,
                model,
                max_retries,
                prompt_template=prompt_template,
                config_override=config_override,
            )
            if optimized:
                source_path.write_text(optimized, encoding="utf-8")
        else:
            break

    return iterations


def _ask_for_optimize(
    implementation: str,
    elapsed: float,
    prompt: str,
    constraints: Optional[str],
    llm_provider: Optional[str],
    model: Optional[str],
    max_retries: int,
    prompt_template: Optional[str] = None,
    config_override: Optional[ConfigOverride] = None,
) -> Optional[str]:
    """Ask the LLM to optimize a working implementation.

    Returns the optimized Python source, or None if the request failed.
    """
    client = get_llm_client(
        llm_provider,
        model=model,
        max_retries=max_retries,
        config_override=config_override,
        tier=Tier.REASONING,
    )
    if client is None:
        return None

    system = (
        "You are an expert Python and Rust engineer. The implementation below "
        "already compiles and passes tests. Make it faster or more efficient "
        "while preserving the public function name(s) and behavior. "
        "Return only the improved Python implementation in a single fenced code block."
    )
    user = (
        f"Original request: {prompt}\n"
        f"Constraints: {constraints or 'None'}\n\n"
        f"Current implementation:\n```python\n{implementation}\n```\n\n"
        f"Last build/benchmark took {elapsed:.6f} seconds. "
        "Optimize the implementation and return the code only."
    )
    response = client.generate(
        [
            {
                "role": "system",
                "content": (
                    get_template(prompt_template).system_prompt
                    if prompt_template
                    else get_default_template().system_prompt
                ),
            },
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    if not response:
        return None
    try:
        blocks = extract_code_blocks(response)
        for lang, code in blocks:
            if lang in (None, "python", "py"):
                return code
        return blocks[0][1]
    except Exception:
        return None


def _ask_for_fix(
    implementation: str,
    error_log: str,
    prompt: str,
    constraints: Optional[str],
    llm_provider: Optional[str],
    model: Optional[str],
    max_retries: int,
    prompt_template: Optional[str] = None,
    config_override: Optional[ConfigOverride] = None,
) -> Optional[str]:
    """Apply deterministic proof-theoretic repair to a failing implementation.

    The build/repair loop never calls an LLM; healing is performed by
    ``DeterministicHealer`` using HIN energy, e-graph rewriting, and FFI
    morphism synthesis.
    """
    healer = DeterministicHealer(Path("."))
    result = healer.execute_healing_pass(
        error_log=error_log,
        source_text=implementation,
    )
    return result.get("patch")
