"""Prompt-driven code generation for Aero-Forge."""

from __future__ import annotations

import ast
import logging
import re
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from aero_forge.blueprint import Blueprint, FunctionSpec, discover_functions
from aero_forge.build_runner import BuildRunner
from aero_forge.config import ConfigOverride
from aero_forge.errors import UserError
from aero_forge.llm.clients import get_llm_client
from aero_forge.overlay import OverlayManager
from aero_forge.overlay.store import OverlayStore
from aero_forge.prompts import get_default_template, get_template
from aero_forge.scaffold.pre_write_validator import PreWriteValidator, ValidationError
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


DEFAULT_SYSTEM_PROMPT = get_default_template().system_prompt


# Words ignored when deriving a module name from the user's prompt.
_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "as", "is", "are", "be", "being", "been", "have",
    "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "can", "shall", "this", "that", "these", "those", "i",
    "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their", "what", "which", "who", "when",
    "where", "why", "how", "all", "each", "every", "some", "any", "no",
    "write", "implement", "create", "build", "generate", "make", "function",
    "program", "code", "algorithm", "routine", "method", "fast", "optimized",
    "quick", "simple",
}

# Names too generic to use as a module name.
_GENERIC_NAMES = {"main", "run", "solve", "helper", "generated", "app", "test"}


_PYTHON_KEYWORDS = {
    "False", "None", "True", "and", "as", "assert", "async", "await", "break",
    "class", "continue", "def", "del", "elif", "else", "except", "finally",
    "for", "from", "global", "if", "import", "in", "is", "lambda", "nonlocal",
    "not", "or", "pass", "raise", "return", "try", "while", "with", "yield",
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
        python_blocks = [
            code for lang, code in blocks if lang in (None, "python", "py")
        ]
        if not python_blocks:
            python_blocks = [code for _, code in blocks]

    if not python_blocks:
        # No markdown fences; extract plain ``def`` functions from raw text.
        impl, tests = _extract_functions_from_text(text)
        if not impl:
            raise GenerationError(
                "No code blocks or function definitions found in LLM response"
            )
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

    response = client.generate(messages, temperature=0.2)
    if not response:
        raise GenerationError("LLM returned an empty response")
    return response


def write_generated_project(
    output_dir: Path,
    implementation: str,
    tests: str,
    project_name: str = "generated_project",
    prompt: str = "",
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

    src_dir = output_dir / "src"
    tests_dir = output_dir / "tests"
    src_dir.mkdir(parents=True, exist_ok=True)
    tests_dir.mkdir(parents=True, exist_ok=True)

    # Preserve the existing module name across incremental rewrites so overlays
    # remain anchored to the same file.
    if module_name is None:
        existing_modules = [
            p.stem for p in src_dir.glob("*.py") if p.name != "__init__.py"
        ]
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

    blueprint = Blueprint(
        project=project_name,
        functions=[
            FunctionSpec(
                file=source_path,
                name=name,
                tests=[test_path],
            )
            for name in _detect_function_names(implementation)
        ],
        output_dir=output_dir / "dist",
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


def generate_smoke_tests(implementation: str, module_name: str = "generated") -> str:
    """Generate pytest smoke tests from the implementation when none were provided."""
    try:
        tree = ast.parse(implementation)
    except SyntaxError:
        return ""

    examples: Dict[str, Any] = {
        "int": "5",
        "float": "1.5",
        "bool": "True",
        "List[int]": "[1, 2, 3]",
        "list": "[1, 2, 3]",
        "List[float]": "[1.0, 2.0, 3.0]",
        "List[List[int]]": "[[1, 2], [3, 4]]",
        "List[List[float]]": "[[1.0, 2.0], [3.0, 4.0]]",
    }

    def example_for(node: Optional[ast.AST]) -> str:
        if node is None:
            return "1"
        if isinstance(node, ast.Name):
            return examples.get(node.id, "1")
        if isinstance(node, ast.Subscript):
            base = getattr(node.value, "id", "")
            if base == "List":
                slice_node = node.slice
                if isinstance(slice_node, ast.Name):
                    inner = examples.get(slice_node.id, "1")
                    if inner.startswith("["):
                        return inner
                    return f"[{inner}, {inner}, {inner}]"
        return "1"

    lines = [f"from {module_name} import {{name}}\n\n"]
    test_lines: List[str] = []
    for item in tree.body:
        if not isinstance(item, ast.FunctionDef):
            continue
        name = item.name
        if name.startswith("_"):
            continue
        args = [example_for(arg.annotation) for arg in item.args.args]
        call = f"{name}({', '.join(args)})"
        return_annotation = item.returns
        if isinstance(return_annotation, ast.Name) and return_annotation.id == "bool":
            assertion = f"    assert {call} in (True, False)"
        else:
            assertion = f"    result = {call}\n    assert result is not None"
        test_lines.append(f"def test_{name}():\n{assertion}\n")

    if not test_lines:
        return ""
    impl_names = sorted(
        {
            item.name
            for item in tree.body
            if isinstance(item, ast.FunctionDef) and not item.name.startswith("_")
        }
    )
    imports = "\n".join(f"from {module_name} import {n}" for n in impl_names)
    # Rebuild with a single import line to avoid repeated imports.
    return imports + "\n\n" + "\n".join(test_lines)


def sanitize_generated_code(source: str) -> str:
    """Remove unsupported constructs that commonly appear in LLM output.

    This is a router-level cleanup: it strips ``raise`` and ``assert``
    statements because the Aero-Forge transpiler does not support them,
    while preserving as much of the generated numeric function as possible.
    """
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
    return sanitized


def generate_project(
    prompt: str,
    *,
    constraints: Optional[str] = None,
    output_dir: Path = Path("."),
    project_name: str = "generated_project",
    llm_provider: Optional[str] = None,
    model: Optional[str] = None,
    max_retries: int = 3,
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
    existing_modules = [
        p.stem for p in (output_dir / "src").glob("*.py") if p.name != "__init__.py"
    ] if (output_dir / "src").is_dir() else []
    module_name = _derive_module_name(
        prompt, implementation, existing=existing_modules[0] if existing_modules else None
    )
    if not tests.strip():
        tests = generate_smoke_tests(implementation, module_name=module_name)
    else:
        tests = _rewrite_generated_imports(tests, module_name)
    source_path, test_path, blueprint = write_generated_project(
        output_dir,
        implementation,
        tests,
        project_name=project_name,
        prompt=prompt,
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
) -> Dict[str, Any]:
    """Generate code from a prompt and optionally build/optimize it.

    Returns a dictionary describing the generated files and build result.
    """
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
            prompt_template=prompt_template,
            algorithm_library=algorithm_library,
            selected_algorithm=selected_algorithm,
            discover=discover,
            explain=explain,
            review=review,
            config_override=config_override,
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

    result: Dict[str, Any] = {
        "source_path": str(source_path),
        "test_path": str(test_path),
        "blueprint_path": str(output_dir / "blueprint.aero"),
        "implementation": implementation,
        "tests": tests,
        "explanation": explanation,
        "build": None,
        "iterations": [],
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
        result["build"] = (
            result["iterations"][-1].get("build") if result["iterations"] else None
        )
    elif build_kwargs is not None:
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
            error_log = "\n".join(
                r.get("logs", "") for r in build_result.get("results", [])
            )
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
        if iteration < 3 or (
            previous_time is not None and elapsed < previous_time * 0.99
        ):
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
    """Ask the LLM to fix compilation errors in the generated implementation."""
    client = get_llm_client(
        llm_provider,
        model=model,
        max_retries=max_retries,
        config_override=config_override,
    )
    if client is None:
        return None

    system = (
        "You are an expert Python and Rust engineer. The implementation below "
        "was generated from a user request but failed to compile or pass tests. "
        "Fix only the implementation; keep the same function signature and public function "
        "names. If the error is an IndexError, out-of-bounds access, or any out-of-order "
        "execution issue, make sure all lists, tuples, dictionaries, and data structures "
        "are fully initialized and populated before any calculation, indexing, or method call. "
        "Return the corrected Python code in a single fenced code block."
    )
    user = (
        f"Original request: {prompt}\n"
        f"Constraints: {constraints or 'None'}\n\n"
        f"Implementation:\n```python\n{implementation}\n```\n\n"
        f"Compiler/test errors:\n```\n{error_log[:2000]}\n```\n\n"
        "Return the corrected implementation only."
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
