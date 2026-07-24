"""Generate a production-grade Python-Rust monorepo from a prompt.

The monorepo pipeline deliberately separates *what* to compute (described in
the prompt) from *how* to package it.  It uses the existing transpile/build
machinery to generate a tested Python core, then packages that core as a Rust
shared library with a Python async service, benchmarks, and a Cargo workspace.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from aero_forge.config import ConfigOverride
from aero_forge.generate import generate_and_build
from aero_forge.translator.translator import TargetMode

logger = logging.getLogger("aero_forge.monorepo")


def _sanitize_name(name: str) -> str:
    """Return a valid Python/Cargo package identifier."""
    name = re.sub(r"[^A-Za-z0-9_\-]+", "-", name.strip()).strip("-").lower()
    name = name.replace("-", "_")
    if not name or name[0].isdigit():
        name = "engine"
    return name


def _function_names_from_source(source: str) -> List[str]:
    """Return public top-level function names from Python source."""
    import ast

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    return [
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
    ]


def _default_pyproject(
    package_name: str,
    project_name: str,
    dependencies: List[str],
) -> str:
    deps = (
        "\n".join(f'    "{dep}",' for dep in dependencies).rstrip(",")
        if dependencies
        else ""
    )
    deps_block = f"\ndependencies = [\n{deps}\n]\n" if deps else ""
    return (
        "[build-system]\n"
        'requires = ["setuptools>=61"]\n'
        'build-backend = "setuptools.build_meta"\n\n'
        "[project]\n"
        f'name = "{project_name}"\n'
        'version = "0.1.0"\n'
        f'description = "Aero-Forge generated Python-Rust monorepo: {project_name}"\n'
        'readme = "README.md"\n'
        'requires-python = ">=3.9"\n'
        f"{deps_block}\n"
        "[tool.setuptools]\n"
        'package-dir = {"" = "src"}\n'
        f'packages = ["{package_name}"]\n\n'
        "[tool.setuptools.package-data]\n"
        f'{package_name} = ["*.so"]\n\n'
        "[tool.pytest.ini_options]\n"
        'pythonpath = ["src"]\n'
        'testpaths = ["tests"]\n'
    )


def _top_level_cargo_toml() -> str:
    return (
        "[workspace]\n"
        'members = ["rust_core"]\n'
        'resolver = "2"\n'
    )


def _service_source(
    package_name: str,
    function_names: List[str],
    primary: str,
) -> str:
    return f'''"""Async HTTP service exposing the generated Rust core.

Run with:

    python service.py

Then POST to http://localhost:8080/evaluate with a JSON body such as:

    {{
      "scores": [[1.0, 2.0], [3.0, 4.0]],
      "weights": [0.5, 0.5],
      "criteria_types": ["benefit", "benefit"]
    }}
"""
import pathlib
import sys
from typing import Any

try:
    from {package_name} import {primary}
except ImportError:
    _src = pathlib.Path(__file__).parent / "src"
    sys.path.insert(0, str(_src))
    from {package_name} import {primary}

try:
    from aiohttp import web
except ImportError:  # pragma: no cover - aiohttp optional for pure import tests
    web = None  # type: ignore


async def evaluate(request: Any) -> Any:
    """Evaluate the core function on a JSON payload."""
    if web is None:
        raise RuntimeError("aiohttp is not installed")
    data = await request.json()
    result = {primary}(
        data["scores"],
        data["weights"],
        data["criteria_types"],
    )
    return web.json_response({{"result": result}})


async def health(request: Any) -> Any:
    """Health check endpoint."""
    if web is None:
        raise RuntimeError("aiohttp is not installed")
    return web.json_response({{"status": "ok"}})


if __name__ == "__main__":
    if web is None:
        raise SystemExit("aiohttp is required to run the service. Install it with: pip install aiohttp")
    app = web.Application()
    app.router.add_post("/evaluate", evaluate)
    app.router.add_get("/health", health)
    web.run_app(app, host="0.0.0.0", port=8080)
'''


def _bench_source(
    package_name: str,
    function_names: List[str],
    primary: str,
) -> str:
    return f'''"""Benchmark the native extension against the pure Python fallback."""
import random
import sys
import pathlib
import time

try:
    from {package_name} import {primary} as native
    from {package_name}._pure import {primary} as pure
except ImportError:
    _src = pathlib.Path(__file__).parent / "src"
    sys.path.insert(0, str(_src))
    from {package_name} import {primary} as native
    from {package_name}._pure import {primary} as pure


def make_input(rows: int = 100, cols: int = 10):
    scores = [[random.uniform(0, 100) for _ in range(cols)] for _ in range(rows)]
    weights = [1.0 / cols] * cols
    criteria_types = ["benefit"] * cols
    return scores, weights, criteria_types


def bench(fn, scores, weights, criteria_types, reps: int = 1000):
    start = time.perf_counter()
    for _ in range(reps):
        fn(scores, weights, criteria_types)
    return time.perf_counter() - start


def main() -> int:
    scores, weights, criteria_types = make_input()
    reps = 1000
    native_elapsed = bench(native, scores, weights, criteria_types, reps=reps)
    pure_elapsed = bench(pure, scores, weights, criteria_types, reps=reps)
    print(f"native: {{native_elapsed:.4f}}s for {{reps}} calls")
    print(f"pure:   {{pure_elapsed:.4f}}s for {{reps}} calls")
    if native_elapsed > 0:
        print(f"speedup: {{pure_elapsed / native_elapsed:.2f}}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


def module_name_from_tests(tests: str) -> str:
    """Best-guess the module name used by generated tests."""
    match = re.search(r"^from\s+([A-Za-z_][A-Za-z0-9_]*)\s+import", tests, re.MULTILINE)
    if match:
        return match.group(1)
    return "generated"


def _rewrite_test_imports(tests: str, package_name: str) -> str:
    """Point generated tests at the packaged module."""
    expected = {"generated", module_name_from_tests(tests)}
    lines = tests.splitlines(keepends=True)
    result: List[str] = []
    for line in lines:
        if line.startswith("from ") and " import " in line:
            parts = line.split(" import ", 1)
            mod = parts[0].split()[1]
            if mod in expected:
                line = f"from {package_name} import {parts[1].lstrip()}"
        result.append(line)
    return "".join(result)


def _run_cargo_build(crate_dir: Path) -> subprocess.CompletedProcess:
    logger.info("Building Rust core in %s", crate_dir)
    return subprocess.run(
        ["cargo", "build", "--release"],
        cwd=crate_dir,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )


def _run_pytest(python_engine_dir: Path) -> subprocess.CompletedProcess:
    logger.info("Running Python tests in %s", python_engine_dir)
    return subprocess.run(
        ["python", "-m", "pytest", "tests", "-q"],
        cwd=python_engine_dir,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


def generate_monorepo(
    prompt: str,
    output_dir: Path,
    *,
    constraints: Optional[str] = None,
    project_name: str = "monorepo",
    llm_provider: Optional[str] = None,
    model: Optional[str] = None,
    max_retries: int = 3,
    max_tokens: Optional[int] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    config_override: Optional[ConfigOverride] = None,
) -> Dict[str, Any]:
    """Build a Python-Rust monorepo from a computational prompt.

    The function first generates and compiles a small Python core using the
    existing transpile/build machinery, then packages it into a workspace with:

    - ``rust_core/`` — a Cargo crate exposing the core via PyO3.
    - ``python_engine/`` — a Python package that loads the compiled extension,
      with a pure-Python fallback, an async HTTP service, benchmarks, and pytest
      tests.
    - top-level ``Cargo.toml`` workspace and ``README.md``.

    Returns a dictionary with ``success``, ``files``, ``cargo_output``,
    ``pytest_output``, and ``build`` details.
    """
    if progress_callback:
        progress_callback("Generating computational core...")

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    core_dir = output_dir / ".aero_core"
    if core_dir.exists():
        shutil.rmtree(core_dir)
    core_dir.mkdir(parents=True, exist_ok=True)

    core_result = generate_and_build(
        prompt,
        constraints=constraints,
        output_dir=core_dir,
        project_name="core",
        llm_provider=llm_provider,
        model=model,
        max_retries=max_retries,
        max_tokens=max_tokens,
        build_kwargs={
            "max_workers": 1,
            "cache_enabled": False,
            "target_mode": TargetMode.PYO3,
        },
        config_override=config_override,
    )

    build = core_result.get("build") or {}
    if not build.get("success"):
        return {
            "success": False,
            "error": "Core generation/build failed",
            "core_error": build.get("error", ""),
            "core_logs": build.get("logs", ""),
        }

    source_path = Path(core_result["source_path"])
    test_path = Path(core_result["test_path"])
    implementation = core_result.get("implementation", "") or source_path.read_text(
        encoding="utf-8"
    )
    tests = core_result.get("tests", "") or test_path.read_text(encoding="utf-8")
    function_names = _function_names_from_source(implementation)
    if not function_names:
        return {
            "success": False,
            "error": "No public functions found in generated core",
            "core_logs": build.get("logs", ""),
        }

    primary = function_names[0]
    artifact = Path(build["results"][0]["artifact"])
    crate_candidates = list((core_dir / "dist" / "native_rust").iterdir())
    crate_dir = crate_candidates[0] if len(crate_candidates) == 1 else None

    if not crate_dir or not crate_dir.is_dir():
        return {
            "success": False,
            "error": "Generated Rust crate not found",
            "core_logs": build.get("logs", ""),
        }

    if progress_callback:
        progress_callback("Packaging monorepo...")

    package_name = "python_engine"
    safe_project = _sanitize_name(project_name)

    rust_dir = output_dir / "rust_core"
    python_dir = output_dir / "python_engine"
    pkg_dir = python_dir / "src" / package_name
    tests_dir = python_dir / "tests"

    if rust_dir.exists():
        shutil.rmtree(rust_dir)
    if python_dir.exists():
        shutil.rmtree(python_dir)

    shutil.copytree(crate_dir, rust_dir)
    pkg_dir.mkdir(parents=True, exist_ok=True)
    tests_dir.mkdir(parents=True, exist_ok=True)

    # Copy loader and shared library into the package.
    loader_src = core_dir / "dist" / f"{source_path.stem}.py"
    if loader_src.is_file():
        shutil.copy(loader_src, pkg_dir / f"{primary}.py")
    shutil.copy(artifact, pkg_dir / artifact.name)

    # Pure Python fallback.
    (pkg_dir / "_pure.py").write_text(implementation, encoding="utf-8")

    # Package __init__.py tries native loader, then pure fallback.
    all_names = ", ".join(repr(n) for n in function_names)
    init_lines = [
        f'"""{safe_project} generated by aero-forge."""',
        "",
        "try:",
        f"    from .{primary} import {', '.join(function_names)}",
        "except Exception as _exc:",
        f"    from ._pure import {', '.join(function_names)}",
        "",
        f"__all__ = [{all_names}]",
    ]
    (pkg_dir / "__init__.py").write_text("\n".join(init_lines) + "\n", encoding="utf-8")

    # Tests rewritten to import from the package.
    (tests_dir / f"test_{primary}.py").write_text(
        _rewrite_test_imports(tests, package_name),
        encoding="utf-8",
    )

    # Python packaging and extra files.
    (python_dir / "pyproject.toml").write_text(
        _default_pyproject(
            package_name,
            safe_project,
            dependencies=["aiohttp>=3.9"],
        ),
        encoding="utf-8",
    )
    (python_dir / "service.py").write_text(
        _service_source(package_name, function_names, primary),
        encoding="utf-8",
    )
    (python_dir / "bench.py").write_text(
        _bench_source(package_name, function_names, primary),
        encoding="utf-8",
    )

    # Top-level workspace files.
    (output_dir / "Cargo.toml").write_text(_top_level_cargo_toml(), encoding="utf-8")
    (output_dir / "README.md").write_text(
        f"# {safe_project}\n\n"
        "Aero-Forge generated Python-Rust monorepo.\n\n"
        "## Build\n\n"
        "```bash\n"
        "cd rust_core && cargo build --release\n"
        "cd ../python_engine && python -m pytest tests -q\n"
        "```\n",
        encoding="utf-8",
    )

    if progress_callback:
        progress_callback("Building Rust core in monorepo...")

    cargo = _run_cargo_build(rust_dir)
    if cargo.returncode != 0:
        return {
            "success": False,
            "error": "Rust core failed to build in monorepo",
            "cargo_output": cargo.stdout,
            "cargo_error": cargo.stderr,
        }

    # Prefer the freshly built shared library, but keep the original if absent.
    built_so = rust_dir / "target" / "release" / artifact.name
    if built_so.is_file():
        shutil.copy(built_so, pkg_dir / built_so.name)

    if progress_callback:
        progress_callback("Running Python tests...")

    pytest = _run_pytest(python_dir)
    success = pytest.returncode == 0

    files = sorted(
        str(p.relative_to(output_dir))
        for p in output_dir.rglob("*")
        if p.is_file() and ".aero_core" not in p.parts
    )

    return {
        "success": success,
        "project_name": safe_project,
        "package_name": package_name,
        "primary_function": primary,
        "files": files,
        "core_build": build,
        "cargo_output": cargo.stdout,
        "cargo_error": cargo.stderr,
        "pytest_output": pytest.stdout,
        "pytest_error": pytest.stderr,
    }
