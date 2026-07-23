"""Orchestrate the LLM → transpile → compile → test → heal loop."""

from __future__ import annotations

import ast
import importlib.machinery
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

from aero_forge.errors import (
    UnsupportedError,
    UserError,
    check_toolchain,
    classify_cargo_error,
)
from aero_forge.healing.router import try_auto_fix
from aero_forge.precision_shield.shield import Shield
from aero_forge.sandbox.manager import Sandbox
from aero_forge.scaffold.engine import Engine, _find_function, _rust_identifier
from aero_forge.translator import UASTToHINTranslator, python_source_to_uast


class ForgeError(Exception):
    """Raised when the forge loop cannot produce a passing function."""


class Orchestrator:
    """Drive the generate/transpile/build/test loop."""

    def __init__(
        self,
        source_path: str | Path,
        function_name: str,
        test_path: Optional[str | Path] = None,
        max_iterations: int = 5,
        use_llm: bool = True,
        model: str = "gpt-4",
    ):
        self.source_path = Path(source_path)
        self.function_name = function_name
        self.test_path = Path(test_path) if test_path else None
        self.max_iterations = max_iterations
        self.use_llm = use_llm
        self.model = model
        self._llm_client: Any = None
        self._cargo_target = Path.home() / ".cache" / "aero-forge" / "target"

    def _llm(self) -> Any:
        if self._llm_client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise UserError(
                    "openai package is not installed; install it or use --no-llm"
                ) from exc
            key = os.getenv("OPENAI_API_KEY")
            if not key:
                raise UserError(
                    "OPENAI_API_KEY is not set. Set it or run with --no-llm."
                )
            self._llm_client = OpenAI(api_key=key)
        return self._llm_client

    def run(self) -> Dict[str, Any]:
        """Run the build/heal loop and return the final result."""
        check_toolchain()
        if not self.source_path.is_file():
            raise UserError(f"Source file not found: {self.source_path}")

        original_source = self.source_path.read_text(encoding="utf-8")
        source = original_source

        for iteration in range(1, self.max_iterations + 1):
            with Sandbox(
                self.source_path, self.function_name, self.test_path
            ) as sandbox:
                sandbox.source_in_sandbox.write_text(source, encoding="utf-8")

                try:
                    artifact = self._compile_to_native(source, sandbox.root)
                except _BuildFailure as exc:
                    log = exc.log
                    fixed = self._attempt_fix(source, log)
                    if fixed is None:
                        raise ForgeError(
                            f"Build failed and could not be fixed after {iteration} iteration(s).\n\n{log}"
                        ) from exc
                    source = fixed
                    continue

                self._install_native_module(sandbox, artifact)

                result = sandbox.run_tests()
                if result["passed"]:
                    self._merge_back(sandbox, artifact)
                    return {
                        "success": True,
                        "iterations": iteration,
                        "artifact": str(artifact),
                        "logs": result["logs"],
                    }

                fixed = self._attempt_fix(source, result["logs"])
                if fixed is None:
                    raise ForgeError(
                        f"Tests failed and could not be fixed after {iteration} iteration(s).\n\n{result['logs']}"
                    )
                source = fixed

        raise ForgeError("Maximum iterations exceeded without a passing result.")

    def _compile_to_native(self, source: str, sandbox_root: Path) -> Path:
        """Transpile ``source`` to Rust, build it, and return the compiled .so path."""
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            raise _BuildFailure(
                f"Syntax error in source: {exc} (line {exc.lineno})"
            ) from exc

        if _find_function(tree, self.function_name) is None:
            raise _BuildFailure(f"Function {self.function_name!r} not found")

        module_name = f"aero_forge_{self.source_path.stem}"
        crate_name = _rust_identifier(module_name)

        uast = python_source_to_uast(source)
        graph = UASTToHINTranslator().translate(uast)

        shield_config: Dict[str, Any] = {}
        traits = Shield(config=shield_config).analyze(
            graph, func_name=self.function_name, source=source
        )
        traits["function_name"] = self.function_name
        graph.traits_by_name = {self.function_name: traits}
        graph.traits = graph.traits_by_name

        engine = Engine()
        crate_root = engine.generate(
            graph,
            sandbox_root,
            module_name=module_name,
            function_names=[self.function_name],
            source=source,
        )

        try:
            fmt = subprocess.run(
                ["cargo", "fmt"],
                cwd=crate_root,
                capture_output=True,
                text=True,
            )
            if fmt.returncode != 0:
                raise _BuildFailure(
                    f"Generated Rust code could not be formatted:\n{fmt.stdout}"
                )

            self._cargo_target.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env["CARGO_TARGET_DIR"] = str(self._cargo_target)

            build = subprocess.run(
                ["cargo", "build", "--release"],
                cwd=crate_root,
                env=env,
                capture_output=True,
                text=True,
            )
            if build.returncode != 0:
                raise _BuildFailure(classify_cargo_error(build.stdout))

            artifact = _find_artifact(self._cargo_target, crate_name)
            if artifact is None:
                raise _BuildFailure("No compiled shared library found after cargo build.")
            return artifact
        finally:
            shutil.rmtree(crate_root, ignore_errors=True)

    def _install_native_module(self, sandbox: Sandbox, artifact: Path) -> None:
        """Place the compiled extension next to a Python loader in the sandbox."""
        crate_name = _rust_identifier(f"aero_forge_{self.source_path.stem}")
        so_path = sandbox.root / artifact.name
        shutil.copy(artifact, so_path)

        loader = sandbox.root / self.source_path.name
        loader.write_text(
            self._loader_source(so_path, crate_name, [self.function_name]),
            encoding="utf-8",
        )

    def _loader_source(
        self, so_path: Path, module_name: str, function_names: List[str]
    ) -> str:
        lines = [
            "import importlib.util",
            "import pathlib",
            "_HERE = pathlib.Path(__file__).parent",
            f'_SO = _HERE / "{so_path.name}"',
            f'_SPEC = importlib.util.spec_from_file_location("{module_name}", _SO)',
            '_MOD = importlib.util.module_from_spec(_SPEC)',
            '_SPEC.loader.exec_module(_MOD)',
        ]
        for name in function_names:
            lines.append(f"{name} = _MOD.{name}")
        lines.append("\n__all__ = [" + ", ".join(f'"{n}"' for n in function_names) + "]")
        return "\n".join(lines) + "\n"

    def _merge_back(self, sandbox: Sandbox, artifact: Path) -> None:
        """Copy the loader and compiled extension to the original directory."""
        dest_dir = self.source_path.parent
        so_dest = dest_dir / artifact.name
        loader_dest = dest_dir / self.source_path.name
        shutil.copy(sandbox.root / self.source_path.name, loader_dest)
        shutil.copy(artifact, so_dest)

    def _attempt_fix(self, source: str, error_log: str) -> Optional[str]:
        """Try a pattern-based fix, then optionally ask the LLM."""
        fixed = try_auto_fix(error_log, source)
        if fixed is not None:
            return fixed
        if not self.use_llm:
            return None
        return self._llm_fix(source, error_log)

    def _llm_fix(self, source: str, error_log: str) -> Optional[str]:
        """Prompt the LLM for a corrected function body."""
        client = self._llm()

        func_source = self._extract_function_source(source)
        prompt = (
            f"Fix the following Python function so that it compiles and passes its tests.\n\n"
            f"Function `{self.function_name}`:\n{func_source}\n\n"
            f"Failure context:\n{error_log}\n\n"
            "Return ONLY the corrected function definition (no markdown fences, no explanation)."
        )

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert Python and Rust programmer.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
        except Exception as exc:
            raise ForgeError(f"LLM request failed: {exc}") from exc

        answer = response.choices[0].message.content or ""
        new_func = _extract_function_body(answer, self.function_name)
        return _replace_function(source, self.function_name, new_func)

    def _extract_function_source(self, source: str) -> str:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return source
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == self.function_name:
                return ast.unparse(node)
        return source


def _extract_function_body(text: str, name: str) -> str:
    """Strip markdown fences and return a function definition or body block."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        tree = ast.parse(text)
        if len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef):
            return ast.unparse(tree.body[0])
    except SyntaxError:
        pass

    return textwrap.dedent(text)


def _replace_function(source: str, name: str, new_body: str) -> str:
    """Replace the body of ``name`` in ``source`` with ``new_body``."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            try:
                body_tree = ast.parse(new_body)
            except SyntaxError:
                body_tree = ast.parse(f"def {name}():\n{textwrap.indent(new_body, '    ')}")

            if len(body_tree.body) == 1 and isinstance(
                body_tree.body[0], ast.FunctionDef
            ):
                new_func = body_tree.body[0]
                node.args = new_func.args
                node.body = new_func.body
                node.decorator_list = new_func.decorator_list
                node.returns = new_func.returns
            else:
                node.body = body_tree.body
            return ast.unparse(tree)

    return source


class _BuildFailure(UserError):
    """Internal exception used to signal a compilation failure with logs."""

    def __init__(self, message: str):
        super().__init__(message)
        self.log = message


def _extension_suffix() -> str:
    suffixes = importlib.machinery.EXTENSION_SUFFIXES
    return suffixes[0] if suffixes else ".so"


def _find_artifact(cargo_target_dir: Path, crate_name: str) -> Optional[Path]:
    candidates: List[Path] = []
    for root in (cargo_target_dir, cargo_target_dir / "release"):
        if root.is_dir():
            candidates.extend(root.rglob(f"lib{crate_name}.so"))
            candidates.extend(root.rglob(f"{crate_name}.dll"))
            candidates.extend(root.rglob(f"lib{crate_name}.dylib"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


__all__ = ["Orchestrator", "ForgeError"]
