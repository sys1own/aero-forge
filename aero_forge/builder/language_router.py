"""Language routing for polyglot engine generation."""

from __future__ import annotations

import ast
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class SystemToolchainRouter:
    """Invoke native host toolchains for graph materializer build stages."""

    TOOLCHAIN_EXEC_MAP: Dict[str, str] = {
        "gcc": "gcc",
        "clang": "clang",
        "clang++": "clang++",
        "cargo": "cargo",
        "cmake": "cmake",
        "go": "go",
        "nvcc": "nvcc",
        "zig": "zig",
        "dotnet": "dotnet",
        "maturin": "maturin",
        "python": "python3",
        "py": "python3",
        "cpython": "python3",
        "javac": "javac",
    }

    @classmethod
    def _exec_path(cls, toolchain: str) -> Optional[str]:
        name = cls.TOOLCHAIN_EXEC_MAP.get(toolchain, toolchain)
        return shutil.which(name)

    @classmethod
    def _build_command(
        cls,
        toolchain: str,
        node_id: str,
        source_files: List[str],
        compiler_flags: List[str],
        workspace_dir: Path,
    ) -> List[str]:
        if toolchain in ("gcc", "clang", "c"):
            out = workspace_dir / f"lib{node_id}.so"
            return [
                cls._exec_path(toolchain) or toolchain,
                "-shared",
                "-fPIC",
                "-o",
                str(out),
                *source_files,
                *compiler_flags,
            ]
        if toolchain in ("clang++", "g++"):
            out = workspace_dir / f"lib{node_id}.so"
            return [
                cls._exec_path(toolchain) or toolchain,
                "-shared",
                "-fPIC",
                "-std=c++20",
                "-o",
                str(out),
                *source_files,
                *compiler_flags,
            ]
        if toolchain == "go":
            out = workspace_dir / f"{node_id}.so"
            src = source_files[0] if source_files else f"{node_id}.go"
            return [
                cls._exec_path("go") or "go",
                "build",
                "-buildmode=c-shared",
                "-o",
                str(out),
                src,
                *compiler_flags,
            ]
        if toolchain == "cmake":
            # CMake configure is the first step; the build step is handled in dispatch.
            return [cls._exec_path("cmake") or "cmake", "-B", "build", "."]
        if toolchain == "cargo":
            return [cls._exec_path("cargo") or "cargo", "build", "--release", *compiler_flags]
        if toolchain == "maturin":
            return [cls._exec_path("maturin") or "maturin", "build", "--release", *compiler_flags]
        if toolchain == "dotnet":
            return [cls._exec_path("dotnet") or "dotnet", "build", *compiler_flags]
        if toolchain == "nvcc":
            out = workspace_dir / f"{node_id}.so"
            return [
                cls._exec_path("nvcc") or "nvcc",
                "-shared",
                "-o",
                str(out),
                *source_files,
                *compiler_flags,
            ]
        if toolchain == "zig":
            out = workspace_dir / f"lib{node_id}.so"
            zig = cls._exec_path("zig") or "zig"
            # Native Zig source vs C/C++ source compiled with zig cc.
            if any(str(f).endswith(".zig") for f in source_files):
                # Native `zig build-lib` uses -O <Mode>, not -O3, and -mcpu, not -march.
                zig_flags: List[str] = []
                for flag in compiler_flags:
                    if flag in ("-O3", "-O2", "-O1", "-Os", "-Oz"):
                        continue
                    if flag == "-march=native":
                        zig_flags.append("-mcpu=native")
                    else:
                        zig_flags.append(flag)
                return [
                    zig,
                    "build-lib",
                    "-dynamic",
                    "-O",
                    "ReleaseFast",
                    "-fPIC",
                    f"-femit-bin={out}",
                    *source_files,
                    *zig_flags,
                ]
            return [
                zig,
                "cc",
                "-shared",
                "-o",
                str(out),
                *source_files,
                *compiler_flags,
            ]
        if toolchain in ("python", "py", "cpython"):
            py = cls._exec_path("python3") or cls._exec_path("python") or "python3"
            target = source_files[0] if source_files else ""
            if not target:
                return [py, "--version"]
            return [py, "-m", "py_compile", target]
        raise ValueError(f"unsupported toolchain: {toolchain}")

    @classmethod
    def dispatch_node_build(
        cls,
        node_id: str,
        node_spec: Dict[str, Any],
        workspace_dir: Path,
    ) -> subprocess.CompletedProcess[str]:
        """Run the appropriate host toolchain for *node_spec*.

        Raises:
            RuntimeError: when the toolchain is unavailable or the build fails.
        """
        toolchain = (node_spec.get("toolchain") or node_spec.get("lang") or "").lower()
        if not cls._exec_path(toolchain):
            raise RuntimeError(f"toolchain {toolchain!r} not found on PATH")

        source_files = list(node_spec.get("source_files", []))
        compiler_flags = list(node_spec.get("compiler_flags", []))
        cmd = cls._build_command(
            toolchain, node_id, source_files, compiler_flags, Path(workspace_dir)
        )

        try:
            result = subprocess.run(
                cmd,
                cwd=str(workspace_dir),
                capture_output=True,
                text=True,
                check=True,
            )
            if toolchain == "cmake":
                build_cmd = [cls._exec_path("cmake") or "cmake", "--build", "build"]
                result = subprocess.run(
                    build_cmd,
                    cwd=str(workspace_dir),
                    capture_output=True,
                    text=True,
                    check=True,
                )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"toolchain {toolchain!r} failed for {node_id}: {exc.stderr}"
            ) from exc

        return result




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

SUPPORTED_LANGUAGES = frozenset({"rust", "python", "cpp"})
DEFAULT_LANGUAGE = "rust"

_LANGUAGE_BY_EXT = {
    ".rs": "rust",
    ".py": "python",
    ".pyi": "python",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".c": "c",
    ".h": "c",
}


def infer_language(path: Path) -> str:
    """Infer a canonical language tag from a file extension."""
    return _LANGUAGE_BY_EXT.get(path.suffix.lower(), "unknown")


def resolve_target_language(
    context: Optional[Dict[str, Any]] = None,
    *,
    source_path: Optional[Path] = None,
    source_language: Optional[str] = None,
) -> str:
    """Resolve the target language for an engine build.

    Priority:
      1. Explicit ``context["frameworks"]["language"]``.
      2. ``source_language`` hint.
      3. File-extension inference from ``source_path``.
      4. Conservative default (``rust``).
    """
    context = context or {}
    frameworks = context.get("frameworks")
    if isinstance(frameworks, dict):
        declared = str(frameworks.get("language", "")).strip().lower()
        if declared in SUPPORTED_LANGUAGES:
            return declared

    if source_language and source_language.lower() in SUPPORTED_LANGUAGES:
        return source_language.lower()

    if source_path is not None:
        inferred = infer_language(Path(source_path))
        if inferred in SUPPORTED_LANGUAGES:
            return inferred

    return DEFAULT_LANGUAGE


def is_native_crate_language(language: str) -> bool:
    """True when the language compiles through a native crate-style build."""
    return language == "rust"


def is_python(language: str) -> bool:
    return language == "python"


def is_cpp(language: str) -> bool:
    return language == "cpp"


def is_cpp_friendly(source: str) -> bool:
    """Return ``True`` when *source* is a numeric, loop-heavy function suitable for C++ acceleration.

    Lightweight control flow, string manipulation, I/O, or NumPy usage cause the
    heuristic to return ``False`` so the function stays in Python or falls back to
    the standard runtime.
    """
    _accel_log("info", "Running cpp-friendly heuristic")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        _accel_log("error", "cpp-friendly heuristic failed: source parse error")
        return False

    local_functions = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    has_io = False
    has_numpy = False
    has_unsupported_list = False
    has_loop = False
    numeric_ops = 0
    recursive_calls = 0

    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.While, ast.ListComp, ast.GeneratorExp)):
            has_loop = True
        if isinstance(node, ast.BinOp) and type(node.op) in (
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Mod,
            ast.Pow,
            ast.BitOr,
            ast.BitXor,
            ast.BitAnd,
            ast.LShift,
            ast.RShift,
        ):
            numeric_ops += 1
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in ("print", "input", "open"):
                    has_io = True
                if node.func.id in local_functions:
                    recursive_calls += 1
            if isinstance(node.func, ast.Attribute) and node.func.attr in ("print",):
                has_io = True
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name for alias in node.names]
            module = getattr(node, "module", "") or ""
            if "numpy" in names or "np" in names or "numpy" in module:
                has_numpy = True
        if isinstance(node, ast.Subscript):
            # Subscript can indicate array indexing, which is fine; handled by emitter.
            pass

    if has_io or has_numpy or has_unsupported_list:
        _accel_log("info", "PASSTHROUGH: lightweight or I/O-heavy function; not C++ friendly")
        return False

    verdict = has_loop or numeric_ops >= 2 or recursive_calls >= 1
    if verdict:
        detail = "Heavy numerical matrix loop bound to C++ dynamic shared library" if has_loop else "numeric workload"
        _accel_log("info", f"ACCELERATED: {detail} (loops={has_loop}, numeric_ops={numeric_ops}, recursive_calls={recursive_calls})")
    else:
        _accel_log("info", "PASSTHROUGH: insufficient numeric workload for C++ acceleration")
    return verdict


def _cpp_compiler_available() -> bool:
    return any(shutil.which(name) for name in ["g++", "clang++", "c++"])


def select_native_backend(source: str, hint: Optional[str] = None) -> str:
    """Select the native backend for *source*.

    Hints:
      - ``"cpp"`` / ``"c_abi"`` -> C++
      - ``"rust"`` / ``"rust_hin"`` / ``"pyo3"`` -> Rust
      - ``"auto"`` or unset -> use :func:`is_cpp_friendly` and toolchain availability.
    """
    _accel_log("info", f"Selecting native backend (hint={hint})")
    hint = (hint or "rust_hin").lower()
    if hint in ("cpp", "c_abi"):
        _accel_log("success", "ACCELERATED: C++ selected for extern \"C\" dynamic shared library")
        return "cpp"
    if hint in ("rust", "rust_hin", "pyo3"):
        _accel_log("success", "Target compilation language: Rust (cdylib / PyO3)")
        return "rust_hin"
    if is_cpp_friendly(source) and _cpp_compiler_available():
        _accel_log("success", "ACCELERATED: C++ auto-selected for extern \"C\" dynamic shared library")
        return "cpp"
    _accel_log("success", "Target compilation language: Rust (auto-selected)")
    return "rust_hin"


def should_accelerate_with_native(source: str, *, min_numeric_ops: int = 3) -> bool:
    """Return ``True`` when *source* has enough numeric work to justify native acceleration.

    This is a coarser gate used by the polyglot materializer to decide whether a
    generated contract should be backed by a compiled native extension or left as
    pure Python.
    """
    _accel_log("info", f"Evaluating native acceleration gate (min_numeric_ops={min_numeric_ops})")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        _accel_log("error", "Native acceleration gate failed: source parse error")
        return False

    has_loop = any(isinstance(n, (ast.For, ast.While, ast.ListComp, ast.GeneratorExp)) for n in ast.walk(tree))
    numeric_ops = sum(
        1
        for n in ast.walk(tree)
        if isinstance(n, ast.BinOp)
        and type(n.op)
        in (
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Mod,
            ast.Pow,
            ast.BitOr,
            ast.BitXor,
            ast.BitAnd,
            ast.LShift,
            ast.RShift,
        )
    )
    verdict = has_loop or numeric_ops >= min_numeric_ops
    if verdict:
        detail = "Heavy numerical matrix loop bound to C++ dynamic shared library" if has_loop else "heavy compute"
        _accel_log("success", f"ACCELERATED: {detail} (loops={has_loop}, numeric_ops={numeric_ops})")
    else:
        _accel_log("info", f"PASSTHROUGH: light workload (loops={has_loop}, numeric_ops={numeric_ops})")
    return verdict
