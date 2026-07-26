"""Direct Python-to-native accelerator bridge.

This module lets a Python function be transparently compiled to a Rust native
extension (via the existing transpiler and cargo) or a C++ shared library (via the
polyglot C++ emitter) and then executed from Python without spawning a new
subprocess for every call.  Compilation is keyed by the UAST of the decorated
function, so repeated calls hit a content-addressable node cache.
"""

from __future__ import annotations

import ast
import functools
import importlib.util
import inspect
import logging
import re
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Tuple

from aero_forge.blueprint import Blueprint, FunctionSpec
from aero_forge.builder import spec_from_python
from aero_forge.builder.emitters.cpp_emitter import CppEmitter
from aero_forge.builder import language_router
from aero_forge.build_runner import BuildRunner
from aero_forge.cache.build_cache import BuildCache
from aero_forge.errors import UnsupportedError
from aero_forge.translator import TargetMode, python_source_to_uast

logger = logging.getLogger("aero_forge.native_bridge")


def _original_function(func: Callable[..., Any]) -> Callable[..., Any]:
    """Return the original, undecorated function for source extraction."""
    original = getattr(func, "__aero_original__", None)
    if original is not None:
        return original
    if hasattr(func, "__wrapped__"):
        return func.__wrapped__
    return func


def _strip_decorators(source: str) -> str:
    """Remove decorators from source so transpilers see a plain function."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
    try:
        return ast.unparse(tree)
    except Exception:
        return source


def _source_for_function(func: Callable[..., Any]) -> Tuple[str, str]:
    original = _original_function(func)
    try:
        source = inspect.getsource(original)
    except (OSError, TypeError) as exc:
        raise UnsupportedError(f"Cannot get source for {original}: {exc}") from exc
    return _strip_decorators(textwrap.dedent(source)), original.__name__


def _extract_function_uast(source: str, function_name: str) -> Optional[Dict[str, Any]]:
    try:
        uast = python_source_to_uast(source)
    except Exception:
        return None
    for child in uast.get("children", []):
        if child.get("type") == "function_declaration" and child.get("name") == function_name:
            return child
    return None


def _sanitize(name: str) -> str:
    return "".join(c if c.isalnum() or c == "_" else "_" for c in name).strip("_") or "fn"


def _module_name_from_so(so_path: Path) -> Optional[str]:
    """Infer the PyO3 module name from a ``lib<name>.so`` artifact path."""
    match = re.search(r"lib([^/]+?)\.so$", so_path.name)
    if match:
        return match.group(1)
    return None


def _load_function_from_so(so_path: Path, function_name: str) -> Optional[Callable[..., Any]]:
    try:
        module_name = _module_name_from_so(so_path)
        if module_name is None:
            module_name = f"aero_forge_native_{_sanitize(function_name)}_{so_path.stat().st_ino}"
        spec = importlib.util.spec_from_file_location(module_name, so_path)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fn = getattr(mod, function_name, None)
        if fn is None:
            logger.warning("Function %s not found in compiled module %s", function_name, so_path)
            return None
        return fn
    except Exception as exc:
        logger.warning("Could not load native module %s: %s", so_path, exc)
        return None


def _find_cpp_compiler() -> Optional[str]:
    for name in ["g++", "clang++", "c++"]:
        if shutil.which(name):
            return name
    return None


def _ctypes_loader_source(source: str, so_path: Path, function_names: List[str]) -> str:
    """Generate a ``ctypes`` Python loader for the C-ABI ``.so`` at *so_path*."""
    tree = ast.parse(source)

    def _py_ann(node: Optional[ast.expr]) -> str:
        if node is None:
            return "Any"
        try:
            return ast.unparse(node)
        except AttributeError:
            return "Any"

    type_map = {
        "int": "ctypes.c_int64",
        "float": "ctypes.c_double",
        "bool": "ctypes.c_bool",
    }
    free_map = {
        "int": "free_buffer_i64",
        "float": "free_buffer_f64",
        "bool": "free_buffer_bool",
    }

    lines: List[str] = [
        "import ctypes",
        "import pathlib",
        "",
        "_HERE = pathlib.Path(__file__).parent",
        f'_SO = pathlib.Path({str(so_path)!r})',
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

        arg_info: List[Tuple[str, str, str, str]] = []
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

        c_args: List[str] = []
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

        py_args = ", ".join(name for _, _, _, name in arg_info)
        body_lines: List[str] = []
        call_args: List[str] = []
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
            body_lines.append(f"    _ptr = _LIB.{func.name}({', '.join(call_args)})")
            body_lines.append(f"    _result = [_ptr[i] for i in range(_out_len.value)]")
            free_name = free_map.get(ret_elem, "free_buffer_i64")
            body_lines.append(f"    _LIB.{free_name}(_ptr, _out_len.value)")
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


class NativeAccelerator:
    """Compile and call a Python function through a Rust or C++ native extension."""

    def __init__(
        self,
        func: Callable[..., Any],
        *,
        target: str = "rust_hin",
        compiler_flags: Optional[List[str]] = None,
        cache: Optional[BuildCache] = None,
    ):
        self.func = func
        self.target = target
        self.compiler_flags = list(compiler_flags or [])
        self.cache = cache or BuildCache()
        self._native: Optional[Callable[..., Any]] = None
        self._native_loaded = False

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if not self._native_loaded:
            self._native = self._compile()
            self._native_loaded = True
        if self._native is not None:
            try:
                return self._native(*args, **kwargs)
            except Exception as exc:
                logger.warning("Native call failed for %s: %s; falling back to Python", self.func.__name__, exc)
        return self.func(*args, **kwargs)

    def _compile(self) -> Optional[Callable[..., Any]]:
        try:
            source, name = _source_for_function(self.func)
        except Exception as exc:
            logger.warning("Native acceleration unavailable for %s: %s", self.func.__name__, exc)
            return None

        uast_node = _extract_function_uast(source, name)
        if uast_node is None:
            logger.warning("Could not lower %s to UAST; using Python fallback", name)
            return None

        effective_target = language_router.select_native_backend(source, self.target)
        if effective_target == "cpp":
            return self._compile_cpp(source, name, uast_node)

        # Default Rust/PyO3 path.
        target = "native"
        mode = TargetMode.PYO3
        cached = self.cache.get_node(
            uast_node, name, self.compiler_flags, target=target, target_mode=mode
        )
        if cached is not None and cached.is_file():
            return _load_function_from_so(cached, name)

        with tempfile.TemporaryDirectory(prefix="aero_forge_native_") as tmp:
            tmp_path = Path(tmp)
            src_dir = tmp_path / "src"
            src_dir.mkdir()
            source_path = src_dir / f"{_sanitize(name)}.py"
            source_path.write_text(source, encoding="utf-8")
            output_dir = tmp_path / "dist"
            output_dir.mkdir()

            blueprint = Blueprint(
                project=f"accelerated_{_sanitize(name)}",
                functions=[FunctionSpec(file=source_path, name=name)],
                output_dir=output_dir,
            )
            runner = BuildRunner(
                blueprint,
                max_workers=1,
                cache_enabled=False,
                target=target,
                target_mode=mode,
            )
            try:
                result = runner.build()
            except Exception as exc:
                logger.warning("Build failed for %s: %s", name, exc)
                return None

            if not result.get("success"):
                logger.warning("Build did not succeed for %s: %s", name, result.get("logs"))
                return None

            artifacts = [r for r in result.get("results", []) if r.get("artifact")]
            if not artifacts:
                logger.warning("No artifact produced for %s", name)
                return None

            artifact = Path(artifacts[0]["artifact"])
            if not artifact.is_file():
                logger.warning("Artifact missing for %s: %s", name, artifact)
                return None

            cached = self.cache.put_node(
                uast_node, name, artifact, self.compiler_flags, target=target, target_mode=mode
            )
            return _load_function_from_so(cached, name)

    def _compile_cpp(self, source: str, name: str, uast_node: Dict[str, Any]) -> Optional[Callable[..., Any]]:
        """Compile *source* to a C++ shared library and return a ctypes callable."""
        compiler = _find_cpp_compiler()
        if compiler is None:
            logger.warning("No C++ compiler found for %s; falling back to Python", name)
            return None

        cached = self.cache.get_node(
            uast_node, name, self.compiler_flags, target="cpp", target_mode=TargetMode.C_ABI
        )
        if cached is not None and cached.is_file():
            so_path = cached
        else:
            try:
                spec = spec_from_python(source, name=name)
                cpp_source = CppEmitter(c_abi=True).emit(spec)
            except Exception as exc:
                logger.warning("Could not generate C++ for %s: %s; falling back to Python", name, exc)
                return None

            with tempfile.TemporaryDirectory(prefix="aero_forge_cpp_") as tmp:
                tmp_path = Path(tmp)
                cpp_path = tmp_path / "native.cpp"
                cpp_path.write_text(cpp_source, encoding="utf-8")
                so_path = tmp_path / f"lib{name}.so"
                cmd = [
                    compiler,
                    "-shared",
                    "-fPIC",
                    "-O2",
                    "-std=c++17",
                    "-o",
                    str(so_path),
                    str(cpp_path),
                    *self.compiler_flags,
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.returncode != 0:
                    logger.warning(
                        "C++ compilation failed for %s:\nstdout:\n%s\nstderr:\n%s",
                        name,
                        proc.stdout,
                        proc.stderr,
                    )
                    return None

                cached = self.cache.put_node(
                    uast_node, name, so_path, self.compiler_flags, target="cpp", target_mode=TargetMode.C_ABI
                )
                so_path = cached

        try:
            loader_source = _ctypes_loader_source(source, so_path, [name])
            loader_path = so_path.parent / f"_{name}_ctypes_loader.py"
            loader_path.write_text(loader_source, encoding="utf-8")
            module_name = f"aero_cpp_loader_{_sanitize(name)}_{so_path.stat().st_ino}"
            spec = importlib.util.spec_from_file_location(module_name, loader_path)
            if spec is None or spec.loader is None:
                return None
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return getattr(mod, name, None)
        except Exception as exc:
            logger.warning("Could not load C++ shared library for %s: %s", name, exc)
            return None


def accelerate(
    target: str = "rust_hin",
    compiler_flags: Optional[List[str]] = None,
    cache: Optional[BuildCache] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that compiles the wrapped function to a native extension on first call.

    ``target`` may be ``"rust_hin"`` (default), ``"cpp"``/``"c_abi"``, or
    ``"auto"`` to let the language router choose between C++ and Rust based on
    the function's numeric workload.
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        accelerator = NativeAccelerator(
            func, target=target, compiler_flags=compiler_flags, cache=cache
        )

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return accelerator(*args, **kwargs)

        wrapper.__aero_original__ = func
        wrapper.__aero_accelerator__ = accelerator
        return wrapper
    return decorator
