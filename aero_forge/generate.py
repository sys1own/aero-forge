"""Prompt-driven code generation for Aero-Forge."""

from __future__ import annotations

import ast
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from aero_forge.blueprint import Blueprint, FunctionSpec, discover_functions
from aero_forge.build_runner import BuildRunner
from aero_forge.llm.clients import get_llm_client
from aero_forge.prompts import get_default_template, get_template
from aero_forge.algorithms import algorithm_prompt_context

logger = logging.getLogger("aero_forge.generate")


CODE_FENCE_RE = re.compile(
    r"```(?:\w*)\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


class GenerationError(Exception):
    """Raised when prompt-driven code generation fails."""


DEFAULT_SYSTEM_PROMPT = get_default_template().system_prompt


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
) -> str:
    """Call the configured LLM and return the raw generated text."""
    client = get_llm_client(llm_provider, model=model, max_retries=max_retries)
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

    algorithm_context = algorithm_prompt_context(prompt)
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": template.system_prompt},
        {
            "role": "user",
            "content": _build_user_prompt(prompt, constraints, algorithm_context),
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
) -> Tuple[Path, Path, Blueprint]:
    """Write implementation, tests, and a blueprint to ``output_dir``.

    Returns ``(source_path, test_path, blueprint)``.
    """
    src_dir = output_dir / "src"
    tests_dir = output_dir / "tests"
    src_dir.mkdir(parents=True, exist_ok=True)
    tests_dir.mkdir(parents=True, exist_ok=True)

    source_path = src_dir / "generated.py"
    test_path = tests_dir / "test_generated.py"
    source_path.write_text(implementation, encoding="utf-8")
    test_path.write_text(tests, encoding="utf-8")

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
    return [
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
    ]


def _token_function_names(source: str) -> List[str]:
    """Simple fallback regex extraction of function names."""
    names = re.findall(r"^\s*def\s+([A-Za-z_]\w*)", source, re.MULTILINE)
    return [n for n in names if not n.startswith("_")]


GeneratedProject = Tuple[Path, Path, Blueprint, str, str]


def generate_smoke_tests(implementation: str) -> str:
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

    lines = ["from generated import {name}\n\n"]
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
    imports = "\n".join(f"from generated import {n}" for n in impl_names)
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
) -> GeneratedProject:
    """Generate code from a prompt and write the project files.

    Returns ``(source_path, test_path, blueprint, implementation, tests)``.
    """
    response = generate_from_prompt(
        prompt,
        constraints=constraints,
        llm_provider=llm_provider,
        model=model,
        max_retries=max_retries,
        prompt_template=prompt_template,
    )
    implementation, tests = parse_generated_response(response)
    implementation = sanitize_generated_code(implementation)
    if not tests.strip():
        tests = generate_smoke_tests(implementation)
    source_path, test_path, blueprint = write_generated_project(
        output_dir, implementation, tests, project_name=project_name
    )
    return source_path, test_path, blueprint, implementation, tests


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
) -> Dict[str, Any]:
    """Generate code from a prompt and optionally build/optimize it.

    Returns a dictionary describing the generated files and build result.
    """
    (
        source_path,
        test_path,
        blueprint,
        implementation,
        tests,
    ) = generate_project(
        prompt,
        constraints=constraints,
        output_dir=output_dir,
        project_name=project_name,
        llm_provider=llm_provider,
        model=model,
        max_retries=max_retries,
        prompt_template=prompt_template,
    )

    result: Dict[str, Any] = {
        "source_path": str(source_path),
        "test_path": str(test_path),
        "blueprint_path": str(output_dir / "blueprint.aero"),
        "implementation": implementation,
        "tests": tests,
        "build": None,
        "iterations": [],
    }

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
        runner = BuildRunner(bp, **build_kwargs)
        result["build"] = runner.build()

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
) -> List[Dict[str, Any]]:
    """Iteratively compile, benchmark, and optimize generated code.

    Runs at least three iterations when ``max_iterations >= 3`` so there is a
    baseline, an optimized candidate, and a validation run. After that the loop
    stops when the benchmark no longer improves.
    """
    import time

    iterations: List[Dict[str, Any]] = []
    source_path = output_dir / "src" / "generated.py"
    test_path = output_dir / "tests" / "test_generated.py"
    previous_time: Optional[float] = None

    for iteration in range(1, max_iterations + 1):
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
        runner = BuildRunner(bp, max_workers=1, cache_enabled=False)
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
) -> Optional[str]:
    """Ask the LLM to optimize a working implementation.

    Returns the optimized Python source, or None if the request failed.
    """
    client = get_llm_client(llm_provider, model=model, max_retries=max_retries)
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
) -> Optional[str]:
    """Ask the LLM to fix compilation errors in the generated implementation."""
    client = get_llm_client(llm_provider, model=model, max_retries=max_retries)
    if client is None:
        return None

    system = (
        "You are an expert Python and Rust engineer. The implementation below "
        "was generated from a user request but failed to compile. Fix only the "
        "implementation; keep the same function signature and public function "
        "names. Return the corrected Python code in a single fenced code block."
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
