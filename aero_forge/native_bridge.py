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
import os
import re
import shutil
import subprocess
import tempfile
import textwrap
import time
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


def _accel_log(level: str, message: str) -> None:
    """Append a structured line to the per-session accelerator log if set."""
    log_path = os.environ.get("AERO_FORGE_ACCEL_LOG")
    if not log_path:
        return
    try:
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{timestamp} [{level}] {message}\n")
    except Exception:
        pass


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


def _ctypes_loader_source(
    source: str,
    so_path: Path,
    function_names: List[str],
    workspace_root: Optional[Path] = None,
    loader_path: Optional[Path] = None,
) -> str:
    """Generate a ``ctypes`` Python loader for the C-ABI ``.so`` at *so_path*.

    When *workspace_root* and *loader_path* are supplied, the generated loader
    resolves the shared library relative to the project root and searches a
    small set of candidate build directories, avoiding brittle hardcoded paths.
    """
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

    so_path_obj = Path(so_path)
    so_name = so_path_obj.name
    stem = so_path_obj.stem
    bare_stem = stem[3:] if stem.startswith("lib") else stem

    if workspace_root is not None and loader_path is not None:
        root = Path(workspace_root).resolve()
        loader = Path(loader_path).resolve()
        try:
            rel = loader.parent.relative_to(root)
            n_parents = len(rel.parts)
        except ValueError:
            n_parents = 0
        try:
            so_rel = so_path_obj.resolve().relative_to(root)
            so_candidates = [f'_ROOT / {str(so_rel)!r}']
        except ValueError:
            so_candidates = [f'pathlib.Path({str(so_path_obj.resolve())!r})']
        search_block = (
            "    for _d in _SO_CANDIDATES:\n"
            "        if _d.is_file():\n"
            "            return _d\n"
            "    for _ext in (\".so\", \".dylib\", \".dll\"):\n"
            "        for _d in (\"dist\", \"target/release\", \"cpp_core\", \"cpp_engine\", \"src\"):\n"
            "            _dp = _ROOT / _d\n"
            "            if not _dp.is_dir():\n"
            "                continue\n"
            "            for _f in _dp.iterdir():\n"
            f"                if _f.suffix == _ext and ({bare_stem!r} in _f.name or {stem!r} in _f.name):\n"
            "                    return _f\n"
            f"    raise FileNotFoundError(f\"Could not find native library {so_name!r} under {{_ROOT}}\")"
        )
        prelude = [
            "import ctypes",
            "import pathlib",
            "",
            f"_ROOT = pathlib.Path(__file__).resolve().parents[{n_parents}]",
            "",
            f"_SO_NAME = {so_name!r}",
            f"_SO_CANDIDATES = [{', '.join(so_candidates)}]",
            "",
            "def _find_library():",
            search_block,
            "",
            "_SO = _find_library()",
            "_LIB = ctypes.CDLL(str(_SO))",
            "",
        ]
    else:
        prelude = [
            "import ctypes",
            "import pathlib",
            "",
            "_HERE = pathlib.Path(__file__).parent",
            f'_SO = pathlib.Path({str(so_path_obj)!r})',
            "_LIB = ctypes.CDLL(str(_SO))",
            "",
        ]

    lines: List[str] = prelude
    lines.extend([
        "_LIB.free_buffer_i64.argtypes = [ctypes.POINTER(ctypes.c_int64), ctypes.c_size_t]",
        "_LIB.free_buffer_f64.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]",
        "_LIB.free_buffer_bool.argtypes = [ctypes.POINTER(ctypes.c_bool), ctypes.c_size_t]",
        "",
    ])

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
        ret_tuple = False
        ret_elem = ""
        if ret_ann.startswith(("tuple[", "Tuple[")) and ret_ann.endswith("]"):
            inner = ret_ann[6:-1] if ret_ann.startswith("tuple[") else ret_ann[6:-1]
            parts = [p.strip() for p in inner.split(",")]
            if len(parts) == 2 and (parts[1].startswith(("list[", "List[")) or parts[1] in ("list", "List")):
                ret_tuple = True
                ret_array = True
                ret_elem = (
                    parts[1][5:-1].strip()
                    if parts[1].startswith(("list[", "List["))
                    else "float"
                )
        if not ret_elem:
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
        list_arg_names = [name for kind, _, _, name in arg_info if kind == "array"]
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
            if ret_tuple:
                body_lines.append("    return (_out_len.value, _result)")
            else:
                body_lines.append("    return _result")
        else:
            body_lines.append(
                f"    _result = _LIB.{func.name}({', '.join(call_args)})"
            )
            # Copy mutated output arrays back into the original Python list arguments.
            if list_arg_names:
                out_name = list_arg_names[-1]
                body_lines.append(f"    for _i, _v in enumerate(_{out_name}_arr):")
                body_lines.append(f"        {out_name}[_i] = _v")
            body_lines.append("    return _result")

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
        _accel_log("info", f"Executing {self.func.__name__} with target={self.target}")
        if not self._native_loaded:
            _accel_log("info", "AST parsing: initiated")
            self._native = self._compile()
            self._native_loaded = True
        if self._native is not None:
            _accel_log("success", f"ACCELERATED: {self.func.__name__} bound to native shared library")
            start = time.perf_counter()
            try:
                result = self._native(*args, **kwargs)
                _accel_log("info", f"Native execution completed in {(time.perf_counter() - start) * 1000:.3f} ms")
                return result
            except Exception as exc:
                _accel_log("error", f"Native call failed for {self.func.__name__}: {exc}; falling back to Python")
        else:
            _accel_log("info", f"PASSTHROUGH: {self.func.__name__} executed in standard Python")
        return self.func(*args, **kwargs)

    def _compile(self) -> Optional[Callable[..., Any]]:
        try:
            source, name = _source_for_function(self.func)
        except Exception as exc:
            _accel_log("error", f"Native acceleration unavailable for {self.func.__name__}: {exc}")
            return None

        uast_node = _extract_function_uast(source, name)
        if uast_node is None:
            _accel_log("error", f"Could not lower {name} to UAST; using Python fallback")
            return None

        _accel_log("success", "AST parsing: success")
        effective_target = language_router.select_native_backend(source, self.target)
        _accel_log("info", f"Target language selected: {effective_target.upper()}")
        cpp_friendly = language_router.is_cpp_friendly(source)
        should_accel = language_router.should_accelerate_with_native(source)
        _accel_log("info", f"Acceleration heuristic verdict: cpp_friendly={cpp_friendly}, should_accelerate={should_accel}")
        if effective_target == "cpp":
            _accel_log("success", "ACCELERATED: function routed to C++ shared library")
            return self._compile_cpp(source, name, uast_node)

        # Default Rust/PyO3 path.
        _accel_log("info", "Compiling to Rust (cdylib / PyO3)")
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
                _accel_log("error", f"Rust build failed for {name}: {exc}")
                return None

            if not result.get("success"):
                _accel_log("error", f"Rust build did not succeed for {name}: {result.get('logs')}")
                return None

            _accel_log("success", f"Rust compilation succeeded for {name}")

            artifacts = [r for r in result.get("results", []) if r.get("artifact")]
            if not artifacts:
                _accel_log("error", f"No artifact produced for {name}")
                return None

            artifact = Path(artifacts[0]["artifact"])
            if not artifact.is_file():
                _accel_log("error", f"Artifact missing for {name}: {artifact}")
                return None

            cached = self.cache.put_node(
                uast_node, name, artifact, self.compiler_flags, target=target, target_mode=mode
            )
            _accel_log("success", f"Loaded Rust shared library for {name}: {cached}")
            return _load_function_from_so(cached, name)

    def _compile_cpp(self, source: str, name: str, uast_node: Dict[str, Any]) -> Optional[Callable[..., Any]]:
        """Compile *source* to a C++ shared library and return a ctypes callable."""
        compiler = _find_cpp_compiler()
        if compiler is None:
            _accel_log("error", f"No C++ compiler found for {name}; falling back to Python")
            return None

        _accel_log("info", f"C++ compilation: using {compiler}")
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
                    _accel_log("error", f"C++ compilation failed for {name}:\n{proc.stderr}")
                    return None

                _accel_log("success", f"C++ compilation succeeded: {so_path}")
                cached = self.cache.put_node(
                    uast_node, name, so_path, self.compiler_flags, target="cpp", target_mode=TargetMode.C_ABI
                )
                so_path = cached

        try:
            loader_path = so_path.parent / f"_{name}_ctypes_loader.py"
            loader_source = _ctypes_loader_source(
                source,
                so_path,
                [name],
                workspace_root=so_path.parent,
                loader_path=loader_path,
            )
            loader_path.write_text(loader_source, encoding="utf-8")
            module_name = f"aero_cpp_loader_{_sanitize(name)}_{so_path.stat().st_ino}"
            spec = importlib.util.spec_from_file_location(module_name, loader_path)
            if spec is None or spec.loader is None:
                return None
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            _accel_log("success", f"Loaded C++ shared library for {name}: {so_path}")
            return getattr(mod, name, None)
        except Exception as exc:
            _accel_log("error", f"Could not load C++ shared library for {name}: {exc}")
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
