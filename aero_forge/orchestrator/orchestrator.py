"""Orchestrate the LLM → transpile → compile → test → heal loop."""

from __future__ import annotations

import ast
import importlib.machinery
import importlib.util
import logging
import os
import shutil
import subprocess
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

from aero_forge.cache.fix_cache import FixCache
from aero_forge.config import load_config, resolve_settings
from aero_forge.errors import (
    UserError,
    check_toolchain,
    classify_cargo_error,
)
from aero_forge.healing.router import try_auto_fix
from aero_forge.orchestrator.error_classifier import (
    ErrorClass,
    classify_exception,
    is_fatal,
)
from aero_forge.llm import get_llm_client
from aero_forge.orchestrator.prompt_builder import PromptBuilder
from aero_forge.precision_shield.shield import Shield
from aero_forge.sandbox.manager import Sandbox
from aero_forge.scaffold.engine import (
    Engine,
    _find_function,
    _rust_identifier,
    ensure_init_files,
    ensure_sys_path,
    find_project_root,
)
from aero_forge.translator import UASTToHINTranslator, python_source_to_uast

logger = logging.getLogger("aero_forge.orchestrator")


class ForgeError(Exception):
    """Raised when the forge loop cannot produce a passing function."""


class Orchestrator:
    """Drive the generate/transpile/build/test loop."""

    def __init__(
        self,
        source_path: str | Path,
        function_name: str,
        function_names: Optional[List[str]] = None,
        test_path: Optional[str | Path] = None,
        test_paths: Optional[List[str | Path]] = None,
        max_iterations: Optional[int] = None,
        use_llm: Optional[bool] = None,
        llm_provider: Optional[str] = None,
        model: Optional[str] = None,
        model_priority: Optional[List[str]] = None,
        max_retries: Optional[int] = None,
        cache_enabled: Optional[bool] = None,
        fallback_model: Optional[str] = None,
        compiler_flags: Optional[List[str]] = None,
        output_dir: Optional[str | Path] = None,
    ):
        overrides: Dict[str, Any] = {}
        if max_iterations is not None:
            overrides["MAX_ITERATIONS"] = max_iterations
        if max_retries is not None:
            overrides["MAX_RETRIES"] = max_retries
        if cache_enabled is not None:
            overrides["CACHE_ENABLED"] = cache_enabled
        if llm_provider is not None:
            overrides["LLM_PROVIDER"] = llm_provider
        if model is not None:
            overrides["MODEL"] = model

        # Backward compat: --model sets the model name; --model-priority uses its first entry.
        if model_priority is not None:
            if isinstance(model_priority, list) and model_priority:
                overrides["MODEL"] = model_priority[0]
            elif isinstance(model_priority, str):
                overrides["MODEL"] = model_priority.split(",")[0].strip()

        # Backward compat: use_llm=False forces provider to none.
        if use_llm is False:
            overrides["LLM_PROVIDER"] = "none"

        file_config = load_config()
        self.settings = resolve_settings(file_config, **overrides)

        self.source_path = Path(source_path)
        self.function_name = function_name
        self.function_names = list(function_names) if function_names else [function_name]
        if test_paths:
            self.test_paths = [Path(p) for p in test_paths]
        elif test_path:
            self.test_paths = [Path(test_path)]
        else:
            self.test_paths = []
        self.test_path = self.test_paths[0] if self.test_paths else None
        self.output_dir = Path(output_dir) if output_dir else self.source_path.parent
        self._project_root: Optional[Path] = None
        self.max_iterations = self.settings["MAX_ITERATIONS"]
        self.use_llm = self.settings.get("LLM_PROVIDER", "none") != "none"
        self.compiler_flags = compiler_flags or []

        self.cache = FixCache(enabled=self.settings["CACHE_ENABLED"])
        self.prompt_builder = PromptBuilder()
        self.llm_client: Optional[Any] = None
        if self.use_llm:
            self.llm_client = get_llm_client(
                self.settings.get("LLM_PROVIDER"),
                model=self.settings.get("MODEL"),
                max_retries=self.settings["MAX_RETRIES"],
            )
            if self.llm_client is None:
                logger.warning(
                    "LLM provider %s could not be configured; falling back to router-only mode",
                    self.settings.get("LLM_PROVIDER"),
                )
                self.use_llm = False
        self._cargo_target = Path.home() / ".cache" / "aero-forge" / "target"

    def run(self) -> Dict[str, Any]:
        """Run the build/heal loop and return the final result."""
        logger.info(
            "Starting forge for %s::%s",
            self.source_path,
            ", ".join(self.function_names),
        )
        self._project_root = find_project_root(self.source_path)
        ensure_sys_path(self._project_root)
        check_toolchain()
        if not self.source_path.is_file():
            raise UserError(f"Source file not found: {self.source_path}")

        original_source = self.source_path.read_text(encoding="utf-8")
        source = original_source

        last_working_source: Optional[str] = None
        last_working_artifact: Optional[Path] = None
        self.prompt_builder.clear()

        for iteration in range(1, self.max_iterations + 1):
            logger.info("Forge iteration %d/%d", iteration, self.max_iterations)

            with Sandbox(
                self.source_path,
                self.function_name,
                test_paths=self.test_paths,
                project_root=self._project_root,
            ) as sandbox:
                sandbox.source_in_sandbox.write_text(source, encoding="utf-8")

                try:
                    artifact = self._compile_to_native(source, sandbox.root)
                except _BuildFailure as exc:
                    error_log = exc.log
                    if is_fatal(error_log):
                        logger.error("Fatal build error: %s", error_log)
                        raise UserError(f"Fatal build error: {error_log}") from exc

                    self.prompt_builder.add_error(error_log)
                    fixed = self._attempt_fix(source, error_log)
                    if fixed is None:
                        reason = f"Build failed and could not be fixed: {error_log[:500]}"
                        return self._partial_result(
                            iteration,
                            last_working_artifact,
                            reason,
                            error_log,
                        )
                    source = fixed
                    continue
                except UserError:
                    raise
                except Exception as exc:
                    cls = classify_exception(exc)
                    if cls == ErrorClass.FATAL:
                        raise
                    logger.exception("Unexpected error during build")
                    return self._partial_result(
                        iteration, last_working_artifact, str(exc), ""
                    )

                last_working_source = source
                last_working_artifact = artifact

                self._install_native_module(sandbox, artifact)

                result = sandbox.run_tests()
                if result["passed"]:
                    self._merge_back(sandbox, artifact)
                    logger.info("Tests passed after %d iteration(s)", iteration)
                    return {
                        "success": True,
                        "iterations": iteration,
                        "artifact": str(artifact),
                        "logs": result["logs"],
                    }

                error_log = result["logs"]
                if is_fatal(error_log):
                    raise UserError(f"Fatal test error: {error_log}")

                self.prompt_builder.add_error(error_log)
                fixed = self._attempt_fix(source, error_log)
                if fixed is None:
                    reason = f"Tests failed and could not be fixed: {error_log[:500]}"
                    return self._partial_result(
                        iteration,
                        last_working_artifact,
                        reason,
                        error_log,
                    )
                source = fixed

        return self._partial_result(
            self.max_iterations,
            last_working_artifact,
            "Maximum iterations exceeded without a passing result.",
            "",
        )

    def _partial_result(
        self,
        iterations: int,
        artifact: Optional[Path],
        reason: str,
        logs: str,
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "success": False,
            "partial": True,
            "iterations": iterations,
            "error": reason,
            "logs": logs,
        }
        if artifact is not None:
            result["artifact"] = str(artifact)
        return result

    def _attempt_fix(self, source: str, error_log: str) -> Optional[str]:
        """Try router, cache, then LLM."""
        # 1. Router-first healing.
        fixed = try_auto_fix(error_log, source)
        if fixed is not None and fixed != source:
            logger.info("Self-healing router produced a fix")
            return fixed

        # 2. Cached fix.
        cached = self.cache.get(error_log, source)
        if cached is not None and cached != source:
            logger.info("Fix cache hit")
            return cached

        # 3. LLM fallback.
        if not self.use_llm or self.llm_client is None:
            logger.info("LLM disabled; no fix available")
            return None

        func_source = self._extract_function_source(source)
        messages = self.prompt_builder.build(self.function_name, func_source)

        logger.info("Requesting LLM fix from %s", self.settings.get("LLM_PROVIDER"))
        answer = self.llm_client.generate(messages)
        if answer is None:
            logger.error("LLM failed to produce a fix")
            return None

        new_func = _extract_function_body(answer, self.function_name)
        fixed = _replace_function(source, self.function_name, new_func)
        if fixed == source:
            logger.warning("LLM returned a fix identical to current source; ignoring")
            return None

        self.cache.set(error_log, source, fixed)
        logger.info("LLM produced a fix")
        return fixed

    def _compile_to_native(self, source: str, sandbox_root: Path) -> Path:
        """Transpile ``source`` to Rust, build it, and return the compiled .so path."""
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            raise _BuildFailure(
                f"Syntax error in source: {exc} (line {exc.lineno})"
            ) from exc

        for name in self.function_names:
            if _find_function(tree, name) is None:
                raise _BuildFailure(f"Function {name!r} not found")

        # Use the source stem for the module name so multiple functions from the
        # same file are compiled into a single extension.
        module_name = f"aero_forge_{self.source_path.stem}"
        crate_name = _rust_identifier(module_name)

        uast = python_source_to_uast(source)
        graph = UASTToHINTranslator().translate(uast)

        shield_config: Dict[str, Any] = {}
        traits_by_name: Dict[str, Any] = {}
        for name in self.function_names:
            traits = Shield(config=shield_config).analyze(
                graph, func_name=name, source=source
            )
            traits["function_name"] = name
            traits_by_name[name] = traits
        graph.traits_by_name = traits_by_name
        graph.traits = graph.traits_by_name

        engine = Engine()
        crate_root = engine.generate(
            graph,
            sandbox_root,
            module_name=module_name,
            function_names=self.function_names,
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
            if self.compiler_flags:
                env["RUSTFLAGS"] = " ".join(
                    [os.environ.get("RUSTFLAGS", "")] + self.compiler_flags
                ).strip()

            build = subprocess.run(
                ["cargo", "build", "--release"],
                cwd=crate_root,
                env=env,
                capture_output=True,
                text=True,
            )
            if build.returncode != 0:
                full_output = f"{build.stdout}\n{build.stderr}".strip()
                logger.debug("Cargo build output:\n%s", full_output)
                raise _BuildFailure(
                    f"Cargo build failed:\n{full_output}\n{classify_cargo_error(full_output)}"
                )

            artifact = _find_artifact(self._cargo_target, crate_name)
            if artifact is None:
                raise _BuildFailure("No compiled shared library found after cargo build.")
            return artifact
        finally:
            shutil.rmtree(crate_root, ignore_errors=True)

    def _install_native_module(self, sandbox: Sandbox, artifact: Path) -> None:
        """Place the compiled extension next to a Python loader in the sandbox."""
        crate_name = _rust_identifier(f"aero_forge_{self.source_path.stem}")
        loader = sandbox.source_in_sandbox
        so_path = loader.parent / artifact.name
        loader.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(artifact, so_path)
        loader.write_text(
            self._loader_source(so_path, crate_name, self.function_names),
            encoding="utf-8",
        )
        ensure_init_files(loader, project_root=sandbox.root)

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
        """Copy the loader and compiled extension to the output directory."""
        dest_dir = self.output_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        so_dest = dest_dir / artifact.name
        loader_dest = dest_dir / self.source_path.name
        shutil.copy(sandbox.source_in_sandbox, loader_dest)
        shutil.copy(artifact, so_dest)
        if self._project_root is None:
            self._project_root = find_project_root(self.source_path)
        if dest_dir == self.source_path.parent:
            ensure_init_files(loader_dest, project_root=self._project_root)
        else:
            # Build outputs are isolated; do not turn them into packages.
            pass

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
        # Source is unparseable; if the LLM returned a clean function, use it.
        try:
            body_tree = ast.parse(new_body)
            if len(body_tree.body) == 1 and isinstance(body_tree.body[0], ast.FunctionDef):
                return ast.unparse(body_tree.body[0]) + "\n"
        except SyntaxError:
            pass
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
