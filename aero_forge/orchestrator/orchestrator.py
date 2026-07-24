"""Orchestrate the deterministic transpile → compile → test → heal loop.

The execution path is strictly deterministic: AST/UAST lowering, HIN graph
transformation, type inference, symbolic constraint verification, and code
healing are performed by static analysis, AST rewrites, and pattern matching.
LLMs are never invoked inside the build loop; they are confined to the
upstream intent-parsing and human-facing diagnostic layers.
"""

from __future__ import annotations

import ast
import importlib.machinery
import logging
import os
import shutil
import subprocess
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from aero_forge.builder import build_engine, spec_from_python
from aero_forge.cache.fix_cache import FixCache
from aero_forge.config import ConfigOverride, load_config, resolve_settings
from aero_forge.errors import (
    UserError,
    check_toolchain,
    classify_cargo_error,
)
from aero_forge.healing.router import try_auto_fix
from aero_forge.llm import get_llm_client
from aero_forge.orchestrator.error_classifier import (
    ErrorClass,
    classify_exception,
    format_transpiler_error_with_traceback,
    is_fatal,
)
from aero_forge.orchestrator.prompt_builder import PromptBuilder
from aero_forge.orchestrator.router import HIN_COMPUTE, classify
from aero_forge.precision_shield.shield import Shield
from aero_forge.sandbox.manager import Sandbox, ensure_cargo_in_path
from aero_forge.scaffold.engine import (
    Engine,
    _find_function,
    _find_top_level,
    _generate_pyi,
    _rust_identifier,
    ensure_init_files,
    ensure_sys_path,
    find_project_root,
)
from aero_forge.translator import (
    UASTToHINTranslator,
    python_source_to_uast,
    TargetMode,
)

logger = logging.getLogger("aero_forge.orchestrator")


def _is_main_guard(stmt: ast.stmt) -> bool:
    """Return True if ``stmt`` is ``if __name__ == '__main__':`` (any quote style)."""
    if not isinstance(stmt, ast.If):
        return False
    test = stmt.test
    if not isinstance(test, ast.Compare):
        return False
    if not isinstance(test.left, ast.Name) or test.left.id != "__name__":
        return False
    if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
        return False
    if len(test.comparators) != 1:
        return False
    comparator = test.comparators[0]
    if isinstance(comparator, ast.Constant) and isinstance(comparator.value, str):
        return comparator.value == "__main__"
    # Python < 3.8 compatibility for string literals in AST.
    if isinstance(comparator, getattr(ast, "Str", ())) and comparator.s == "__main__":
        return True
    return False


def _strip_main_guard(source: str) -> str:
    """Remove top-level ``if __name__ == '__main__':`` blocks from source.

    This keeps the transpiler from trying to lower entry-point code that may
    wrap function definitions or contain unsupported statements.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source
    lines = source.splitlines(keepends=True)
    removed = set()
    for stmt in tree.body:
        if _is_main_guard(stmt):
            for lineno in range(stmt.lineno, getattr(stmt, "end_lineno", stmt.lineno) + 1):
                removed.add(lineno - 1)
    if not removed:
        return source
    return "".join(line for i, line in enumerate(lines) if i not in removed)


class ForgeError(Exception):
    """Raised when the forge loop cannot produce a passing function."""


class Orchestrator:
    """Drive the deterministic transpile/build/test/heal loop.

    Healing is performed by the static ``try_auto_fix`` router and the
    deterministic fix cache. No LLM calls occur during compilation or test
    execution.
    """

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
        target: Optional[str] = None,
        target_mode: str = TargetMode.PYO3,
        config_override: Optional[ConfigOverride] = None,
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
        self.settings = resolve_settings(file_config, override=config_override, **overrides)

        self.source_path = Path(source_path)
        self.function_name = function_name
        self.function_names = (
            list(function_names) if function_names else [function_name]
        )
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
        self.target = target
        self.target_mode = target_mode

        self.cache = FixCache(enabled=self.settings["CACHE_ENABLED"])
        # prompt_builder and llm_client are retained for API compatibility but
        # are no longer used by the deterministic build loop.
        self.prompt_builder = PromptBuilder()
        self.llm_client: Optional[Any] = None
        if self.use_llm:
            self.llm_client = get_llm_client(
                self.settings.get("LLM_PROVIDER"),
                model=self.settings.get("MODEL"),
                max_retries=self.settings["MAX_RETRIES"],
                api_key=self.settings.get("API_KEY"),
            )
            if self.llm_client is None:
                logger.warning(
                    "LLM provider %s could not be configured; falling back to router-only mode",
                    self.settings.get("LLM_PROVIDER"),
                )
                self.use_llm = False
        self._cargo_target = Path.home() / ".cache" / "aero-forge" / "target"

    def run(self) -> Dict[str, Any]:
        """Run the deterministic transpile/compile/test/heal loop.

        All repair attempts are static AST/pattern-based. No LLM calls are made
        during execution.
        """
        logger.info(
            "Starting forge for %s::%s",
            self.source_path,
            ", ".join(self.function_names),
        )
        self._project_root = find_project_root(self.source_path)
        ensure_sys_path(self._project_root)
        ensure_cargo_in_path()
        if not self.source_path.is_file():
            raise UserError(f"Source file not found: {self.source_path}")

        original_source = self.source_path.read_text(encoding="utf-8")
        source = original_source

        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            return self._partial_result(
                0,
                None,
                f"Build failed and could not be fixed: Syntax error in source: {exc} "
                f"(line {exc.lineno})",
                "",
            )

        for name in self.function_names:
            if _find_top_level(tree, name)[0] is None:
                return self._partial_result(
                    0, None, f"Function or class {name!r} not found", ""
                )

        route_payload = classify(source, function_names=self.function_names)
        if route_payload["route"] != HIN_COMPUTE:
            return self._run_general_purpose(source, route_payload)

        check_toolchain()
        last_working_source: Optional[str] = None
        last_working_artifact: Optional[Path] = None

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
                        reason = (
                            f"Build failed and could not be fixed: {error_log[:500]}"
                        )
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
                        iteration,
                        last_working_artifact,
                        str(exc),
                        traceback.format_exc(),
                    )

                last_working_source = source
                last_working_artifact = artifact

                if self.target_mode == TargetMode.C_ABI:
                    self._install_c_abi_module(sandbox, artifact)
                else:
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

    def _package_general_purpose(self, source: str) -> Optional[Path]:
        """Package general-purpose Python code using the polyglot builder.

        This produces a Python source file and ``setup.py`` in a ``python_pkg/``
        subfolder of the output directory, demonstrating the pipeline's ability to
        emit language-specific artifacts from an abstract engine spec.
        """
        try:
            spec = spec_from_python(source, name=self.source_path.stem or "generated")
            output = build_engine(
                spec,
                target_language="python",
                template_names=["setup.py"],
            )
            pkg_dir = self.output_dir / "python_pkg"
            pkg_dir.mkdir(parents=True, exist_ok=True)
            main_file = pkg_dir / f"{spec.name}.py"
            main_file.write_text(output.source, encoding="utf-8")
            for artifact in output.artifacts.artifacts:
                dest = pkg_dir / artifact.path
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(artifact.content, encoding="utf-8")
            return pkg_dir
        except Exception as exc:
            logger.warning("Could not package general-purpose source: %s", exc)
            return None

    def _run_general_purpose(
        self, source: str, route_payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute dynamic/general-purpose code through the native Python runtime."""
        logger.info("Routing %s as general-purpose code", ", ".join(self.function_names))
        specific = next(
            (
                r
                for r in route_payload["reasons"]
                if any(k in r for k in ("uses ", "calls ", "contains ", "imports ", "not found"))
            ),
            None,
        )
        base_error = specific or (route_payload["reasons"][0] if route_payload["reasons"] else "General-purpose code")
        # Hard routing blocks (missing functions, unsupported constructs) fail
        # immediately rather than being silently treated as passing builds.
        if "not found" in base_error:
            return {
                "success": False,
                "iterations": 0,
                "route": route_payload["route"],
                "reasons": route_payload["reasons"],
                "target_functions": route_payload["target_functions"],
                "error": base_error,
                "logs": "",
            }
        if not self.test_paths or not any(p.is_file() for p in self.test_paths):
            return {
                "success": False,
                "iterations": 0,
                "route": route_payload["route"],
                "reasons": route_payload["reasons"],
                "target_functions": route_payload["target_functions"],
                "error": f"{base_error}: general-purpose code requires tests to verify",
                "logs": "",
            }
        with Sandbox(
            self.source_path,
            self.function_name,
            test_paths=self.test_paths,
            project_root=self._project_root,
        ) as sandbox:
            sandbox.source_in_sandbox.write_text(source, encoding="utf-8")
            result = sandbox.run_tests()
            if result["passed"]:
                package_path = self._package_general_purpose(source)
                output: Dict[str, Any] = {
                    "success": True,
                    "iterations": 0,
                    "route": route_payload["route"],
                    "reasons": route_payload["reasons"],
                    "target_functions": route_payload["target_functions"],
                    "logs": result["logs"],
                }
                if package_path is not None:
                    output["package"] = str(package_path)
                return output
            return {
                "success": False,
                "iterations": 0,
                "route": route_payload["route"],
                "reasons": route_payload["reasons"],
                "target_functions": route_payload["target_functions"],
                "error": f"{base_error}: general-purpose tests failed",
                "logs": result["logs"],
            }

    def _attempt_fix(self, source: str, error_log: str) -> Optional[str]:
        """Try deterministic router and cached fixes.

        The orchestrator never invokes an LLM during the build loop. All
        repairs are static AST rewrites or pattern-based patches produced by
        ``aero_forge.healing.router``.
        """
        fixed = try_auto_fix(error_log, source)
        if fixed is not None and fixed != source:
            logger.info("Self-healing router produced a fix")
            return fixed

        cached = self.cache.get(error_log, source)
        if cached is not None and cached != source:
            logger.info("Fix cache hit")
            return cached

        return None

    def _validate_return_tuple_sizes(self, tree: ast.AST) -> None:
        """Reject functions whose return statements return different tuple sizes.

        A bare ``return`` (no value) is ignored when other returns exist; the
        engine will emit it as ``return <zero>;`` for the function's return type.
        """

        def _returns(func: ast.AST) -> List[ast.Return]:
            """Yield Return nodes that belong to ``func``, not nested functions/classes."""
            returns: List[ast.Return] = []

            def _visit(n: Any) -> None:
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    return
                if isinstance(n, ast.Return):
                    returns.append(n)
                if isinstance(n, ast.AST):
                    for child in ast.iter_child_nodes(n):
                        _visit(child)
                elif isinstance(n, list):
                    for child in n:
                        _visit(child)

            _visit(func.body)
            return returns

        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name not in self.function_names:
                continue
            sizes: Dict[int, int] = {}
            for ret in _returns(node):
                if ret.value is None or (
                    isinstance(ret.value, ast.Constant) and ret.value.value is None
                ):
                    continue
                if isinstance(ret.value, ast.Tuple):
                    size = len(ret.value.elts)
                else:
                    size = 1
                sizes[size] = sizes.get(size, 0) + 1
            if len(sizes) > 1:
                counts = ", ".join(
                    f"{size} value(s) {count} time(s)"
                    for size, count in sorted(sizes.items())
                )
                raise _BuildFailure(
                    f"All return statements in '{node.name}' must return the same number of values. "
                    f"Found: {counts}. "
                    "Rewrite so every return has the same tuple size."
                )

    def _compile_to_native(self, source: str, sandbox_root: Path) -> Path:
        """Transpile ``source`` to Rust, build it, and return the compiled .so path."""
        # Isolate ``if __name__ == '__main__':`` blocks from function definitions
        # and the transpiler so entry-point code cannot wrap DSL functions.
        source = _strip_main_guard(source)
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            raise _BuildFailure(
                f"Syntax error in source: {exc} (line {exc.lineno})"
            ) from exc

        self._validate_return_tuple_sizes(tree)

        for name in self.function_names:
            if _find_top_level(tree, name)[0] is None:
                raise _BuildFailure(f"Function or class {name!r} not found")

        # Use the source stem for the module name so multiple functions from the
        # same file are compiled into a single extension.
        module_name = f"aero_forge_{self.source_path.stem}"
        crate_name = _rust_identifier(module_name)
        native_rust_dir = self.output_dir / "native_rust" / crate_name

        try:
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
                target_mode=self.target_mode,
            )
        except Exception as exc:
            raise _BuildFailure(
                format_transpiler_error_with_traceback(
                    exc, source_path=self.source_path, source=source
                )
            ) from exc

        try:
            fmt = subprocess.run(
                ["cargo", "fmt"],
                cwd=crate_root,
                capture_output=True,
                text=True,
                timeout=60,
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

            build_cmd = ["cargo", "build", "--release"]
            if self.target:
                build_cmd.extend(["--target", self.target])

            try:
                build = subprocess.run(
                    build_cmd,
                    cwd=crate_root,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
            except subprocess.TimeoutExpired as exc:
                raise _BuildFailure(
                    f"Cargo build timed out after {exc.timeout}s. Try a smaller prompt or reduce optimization flags."
                ) from exc

            if build.returncode != 0:
                full_output = f"{build.stdout}\n{build.stderr}".strip()
                if build.returncode < 0:
                    full_output = (
                        f"{full_output}\nProcess terminated by signal {-build.returncode} "
                        f"(possible OOM crash or external kill)."
                    )
                logger.debug("Cargo build output:\n%s", full_output)
                raise _BuildFailure(
                    f"Cargo build failed:\n{full_output}\n{classify_cargo_error(full_output)}"
                )

            artifact = _find_artifact(self._cargo_target, crate_name, self.target)
            if artifact is None:
                raise _BuildFailure(
                    "No compiled shared library found after cargo build."
                )
            pyi_path = self.output_dir / f"{self.source_path.stem}.pyi"
            pyi_path.parent.mkdir(parents=True, exist_ok=True)
            _generate_pyi(source, self.function_names, pyi_path)
            return artifact
        finally:
            try:
                if crate_root.exists():
                    native_rust_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(
                        crate_root,
                        native_rust_dir,
                        dirs_exist_ok=True,
                    )
            except Exception as exc:
                logger.warning("Could not persist generated Rust crate to %s: %s", native_rust_dir, exc)
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
            "_MOD = importlib.util.module_from_spec(_SPEC)",
            "_SPEC.loader.exec_module(_MOD)",
        ]
        for name in function_names:
            lines.append(f"{name} = _MOD.{name}")
        lines.append(
            "\n__all__ = [" + ", ".join(f'"{n}"' for n in function_names) + "]"
        )
        return "\n".join(lines) + "\n"

    def _install_c_abi_module(self, sandbox: Sandbox, artifact: Path) -> None:
        """Place a ctypes-based loader for the C-ABI shared library in the sandbox."""
        loader = sandbox.source_in_sandbox
        so_path = loader.parent / artifact.name
        loader.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(artifact, so_path)
        source = self.source_path.read_text(encoding="utf-8")
        loader.write_text(
            self._c_abi_loader_source(so_path, source, self.function_names),
            encoding="utf-8",
        )
        ensure_init_files(loader, project_root=sandbox.root)

    def _c_abi_loader_source(
        self, so_path: Path, source: str, function_names: List[str]
    ) -> str:
        """Generate a ctypes loader that mirrors the compiled C-ABI symbols."""
        tree = ast.parse(source)

        def _py_ann(node: Optional[ast.expr]) -> str:
            if node is None:
                return "Any"
            try:
                return ast.unparse(node)
            except AttributeError:
                return "Any"

        type_map = {"int": "ctypes.c_int64", "float": "ctypes.c_double", "bool": "ctypes.c_bool"}
        free_map = {"int": "free_buffer_i64", "float": "free_buffer_f64", "bool": "free_buffer_bool"}

        lines: List[str] = [
            "import ctypes",
            "import pathlib",
            "",
            "_HERE = pathlib.Path(__file__).parent",
            f'_SO = _HERE / "{so_path.name}"',
            "_LIB = ctypes.CDLL(str(_SO))",
            "",
            "_LIB.free_buffer_i64.argtypes = [ctypes.POINTER(ctypes.c_int64), ctypes.c_size_t]",
            "_LIB.free_buffer_f64.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]",
            "_LIB.free_buffer_bool.argtypes = [ctypes.POINTER(ctypes.c_bool), ctypes.c_size_t]",
            "",
        ]
        all_names: List[str] = []
        for func in tree.body:
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if func.name not in function_names:
                continue
            all_names.append(func.name)
            arg_info = []
            for arg in func.args.args:
                ann = _py_ann(arg.annotation)
                if ann.startswith(("list[", "List[")):
                    elem = ann.split("[", 1)[1].split("]", 1)[0]
                    ctype = type_map.get(elem, "ctypes.c_void_p")
                    arg_info.append(("array", elem, ctype, arg.arg))
                else:
                    ctype = type_map.get(ann, "ctypes.c_void_p")
                    arg_info.append(("scalar", ann, ctype, arg.arg))

            ret_ann = _py_ann(func.returns)
            ret_array = ret_ann.startswith(("list[", "List["))
            ret_elem = ret_ann.split("[", 1)[1].split("]", 1)[0] if ret_array else ret_ann
            ret_ctype = type_map.get(ret_elem, "ctypes.c_void_p")

            # Configure argument and return types on the C symbol.
            c_args = []
            for kind, _, ctype, name in arg_info:
                if kind == "scalar":
                    c_args.append(ctype)
                else:
                    c_args.append(f"ctypes.POINTER({ctype})")
                    c_args.append("ctypes.c_size_t")
            if ret_array:
                c_args.append("ctypes.POINTER(ctypes.c_size_t)")
            lines.append(f"_LIB.{func.name}.argtypes = [{', '.join(c_args)}]")
            if ret_array:
                lines.append(f"_LIB.{func.name}.restype = ctypes.POINTER({ret_ctype})")
            else:
                lines.append(f"_LIB.{func.name}.restype = {ret_ctype}")
            lines.append("")

            # Build the Python wrapper.
            py_args = ", ".join(name for _, _, _, name in arg_info)
            body_lines = []
            call_args = []
            for kind, elem, ctype, name in arg_info:
                if kind == "scalar":
                    call_args.append(name)
                else:
                    body_lines.append(
                        f"    _{name}_arr = ({ctype} * len({name}))(*{name})"
                    )
                    body_lines.append(
                        f"    _{name}_ptr = ctypes.cast(_{name}_arr, ctypes.POINTER({ctype}))"
                    )
                    call_args.append(f"_{name}_ptr")
                    call_args.append(f"len({name})")

            if ret_array:
                body_lines.append("    _out_len = ctypes.c_size_t()")
                call_args.append("ctypes.byref(_out_len)")
                body_lines.append(
                    f"    _ptr = _LIB.{func.name}({', '.join(call_args)})"
                )
                body_lines.append(
                    f"    _result = [_ptr[i] for i in range(_out_len.value)]"
                )
                free_name = free_map.get(ret_elem, "free_buffer_i64")
                body_lines.append(
                    f"    _LIB.{free_name}(_ptr, _out_len.value)"
                )
                body_lines.append("    return _result")
            else:
                body_lines.append(
                    f"    return _LIB.{func.name}({', '.join(call_args)})"
                )

            lines.append(f"def {func.name}({py_args}):")
            lines.extend(body_lines)
            lines.append("")

        if all_names:
            lines.append(f"__all__ = {all_names!r}")
            lines.append("")
        return "\n".join(lines)

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

class _BuildFailure(UserError):
    """Internal exception used to signal a compilation failure with logs."""

    def __init__(self, message: str):
        super().__init__(message)
        self.log = message


def _extension_suffix() -> str:
    suffixes = importlib.machinery.EXTENSION_SUFFIXES
    return suffixes[0] if suffixes else ".so"


def _find_artifact(
    cargo_target_dir: Path, crate_name: str, target: Optional[str] = None
) -> Optional[Path]:
    candidates: List[Path] = []
    roots = [cargo_target_dir]
    if target:
        roots.append(cargo_target_dir / target / "release")
    else:
        roots.append(cargo_target_dir / "release")
    for root in roots:
        if root.is_dir():
            candidates.extend(root.rglob(f"lib{crate_name}.so"))
            candidates.extend(root.rglob(f"{crate_name}.dll"))
            candidates.extend(root.rglob(f"lib{crate_name}.dylib"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


__all__ = ["Orchestrator", "ForgeError"]
