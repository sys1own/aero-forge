"""Universal prompt-to-stack classification for blueprint generation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


def _lower_tokens(prompt: str) -> Set[str]:
    """Return a set of normalized lowercase tokens from *prompt*."""
    normalized = (
        prompt.lower()
        .replace("-", " ")
        .replace("_", " ")
        .replace("/", " ")
        .replace(",", " ")
        .replace(".", " ")
    )
    return set(normalized.split())


# Language/tool markers. Keep these broad and domain-neutral; the classifier
# should only infer *what build tools are implied*, not encode business logic.
_LANGUAGE_MARKERS: Dict[str, List[str]] = {
    "python": ["python", "py", "maturin", "pyo3", "pybind11", "cython"],
    "rust": ["rust", "cargo", "crates", "rustc"],
    "cpp": ["c++", "cpp", "cplusplus", "cxx", "pybind11", "cffi", "cmake"],
    "go": ["go", "golang"],
    "javascript": ["javascript", "js", "node", "nodejs", "typescript", "ts"],
    "java": ["java", "jni", "jvm"],
    "csharp": ["csharp", "c#", "dotnet", "pinvoke", "mono"],
    "zig": ["zig"],
    "mojo": ["mojo"],
    "nim": ["nim"],
    "d": ["d lang", "d language"],
}

_TOOL_MARKERS: Dict[str, List[str]] = {
    "cargo": ["cargo", "rust", "crates"],
    "maturin": ["maturin"],
    "cmake": ["cmake"],
    "npm": ["npm", "yarn", "pnpm", "node"],
    "go": ["go", "golang"],
    "python": ["python", "py"],
}

_FEATURE_MARKERS: Dict[str, List[str]] = {
    "web": ["web", "http", "rest", "api", "service", "server", "fastapi", "flask", "axum", "express"],
    "async": ["async", "await", "asyncio", "tokio"],
    "data_pipeline": ["pipeline", "stream", "batch", "etl", "queue", "kafka"],
    "wasm": ["wasm", "webassembly", "wasm32"],
    "gpu": ["gpu", "cuda", "roc"],
    "multi_crate": ["workspace", "monorepo", "multi crate", "crates"],
    "cli": ["cli", "command line", "console"],
    "math": ["math", "numeric", "linear algebra", "matrix", "fft"],
}

# Intent aliases map to existing intent constants used by the rest of the pipeline.
INTENT_PURE_PYTHON = "pure_python"
INTENT_PURE_RUST = "pure_rust"
INTENT_HYBRID_RUST_PYTHON = "hybrid_rust_python"
INTENT_HYBRID_CPP_PYTHON = "hybrid_cpp_python"
INTENT_HYBRID_CPP_RUST = "hybrid_cpp_rust"
INTENT_TRI_POLYGLOT_RUST_CPP_PYTHON = "tri_polyglot_rust_cpp_python"
INTENT_GRAPH_POLYGLOT = "graph_polyglot"
INTENT_GENERIC = "generic"

# Advanced cross-language boundaries that force the graph_polyglot path.
_ADVANCED_BOUNDARY_MARKERS: Dict[str, List[str]] = {
    "wasm_wasi": ["wasm", "webassembly", "wasm32", "wasi"],
    "cgo": ["cgo", "go binding", "go ffi"],
    "jni": ["jni", "java native", "jvm"],
    "pinvoke": ["pinvoke", "p/invoke", "csharp", "dotnet", "mono"],
    "cuda_hip_c": ["cuda", "hip", "gpu kernel"],
}


@dataclass(frozen=True)
class StackClassification:
    architecture: str
    toolchains: List[str]
    languages: List[str]
    features: List[str]


def _has_any(tokens: Set[str], markers: List[str]) -> bool:
    return any(m in tokens for m in markers)


def classify_stack(prompt: str) -> StackClassification:
    """Return a language/tool/feature classification for *prompt*."""
    tokens = _lower_tokens(prompt)

    languages = sorted(
        {lang for lang, markers in _LANGUAGE_MARKERS.items() if _has_any(tokens, markers)}
    )
    features = sorted(
        {feat for feat, markers in _FEATURE_MARKERS.items() if _has_any(tokens, markers)}
    )

    # Toolchains are language/tool markers with overlap deduplicated.
    raw_toolchains: Set[str] = set()
    for tool, markers in _TOOL_MARKERS.items():
        if _has_any(tokens, markers):
            raw_toolchains.add(tool)
    if "rust" in languages:
        raw_toolchains.add("rust")
        raw_toolchains.add("cargo")
    if "cpp" in languages:
        raw_toolchains.add("cpp")
        if _has_any(tokens, ["cmake"]):
            raw_toolchains.add("cmake")
    if "python" in languages:
        raw_toolchains.add("python")
        if _has_any(tokens, ["maturin"]):
            raw_toolchains.add("maturin")
    if "go" in languages:
        raw_toolchains.add("go")
    if "javascript" in languages:
        raw_toolchains.add("npm")

    # Architecture inference: prefer hybrid when multiple languages are present,
    # unless the prompt explicitly constrains itself to a single stack.
    has_python = "python" in languages
    has_rust = "rust" in languages
    has_cpp = "cpp" in languages
    has_hybrid_word = _has_any(tokens, ["hybrid", "polyglot", "mixed", "mixed language", "multi language"])

    lower = prompt.lower()
    has_python_only = (
        "pure_python" in lower
        or "pure python" in lower
        or "python only" in lower
        or _has_any(tokens, ["pure python", "python only"])
    )
    has_rust_only = _has_any(tokens, ["pure rust", "rust only", "cargo only"])
    # A prompt like "pure_python ... use @accelerate(target='rust_hin')" is a
    # single-language Python function that should be compiled through the HIN
    # pipeline, not materialized as a multi-language graph.
    has_accelerate = "@accelerate" in lower
    if has_accelerate and (has_python or "python" in lower) and not has_hybrid_word:
        has_python_only = True

    if has_python and has_rust and has_cpp:
        architecture = INTENT_TRI_POLYGLOT_RUST_CPP_PYTHON
        raw_toolchains.update(["python", "rust", "cpp", "cargo"])
    elif has_python and has_rust:
        architecture = INTENT_HYBRID_RUST_PYTHON
        raw_toolchains.update(["python", "rust", "cargo"])
    elif has_python and has_cpp:
        architecture = INTENT_HYBRID_CPP_PYTHON
        raw_toolchains.update(["python", "cpp", "cmake"])
    elif has_rust and has_cpp:
        architecture = INTENT_HYBRID_CPP_RUST
        raw_toolchains.update(["rust", "cpp", "cargo"])
    elif has_hybrid_word and (has_python or has_rust):
        # Generic "hybrid" / "polyglot" with no explicit C++ defaults to Rust/Python.
        architecture = INTENT_HYBRID_RUST_PYTHON
        raw_toolchains.update(["python", "rust", "cargo"])
    elif has_rust or has_rust_only:
        architecture = INTENT_PURE_RUST
        raw_toolchains.update(["rust", "cargo"])
    elif has_cpp:
        architecture = INTENT_GENERIC
    else:
        architecture = INTENT_PURE_PYTHON

    if has_python_only:
        architecture = INTENT_PURE_PYTHON
        # Keep only the Python toolchain for a single-language accelerated function.
        raw_toolchains = {"python"} | {t for t in raw_toolchains if t == "python"}
    if has_rust_only:
        architecture = INTENT_PURE_RUST

    advanced_boundaries = sorted(
        {b for b, markers in _ADVANCED_BOUNDARY_MARKERS.items() if _has_any(tokens, markers)}
    )
    # Any language outside the built-in {python, rust, cpp} trio (e.g. Go,
    # JavaScript, Java, C#, Zig, Mojo, Nim, D) forces the generic graph_polyglot
    # path so the engine can JIT-synthesize the required emitter plugin.
    builtin_langs = {"python", "rust", "cpp"}
    extra_languages = [lang for lang in languages if lang not in builtin_langs]
    if not has_python_only and not has_rust_only:
        if len(languages) > 2 or extra_languages or advanced_boundaries:
            architecture = INTENT_GRAPH_POLYGLOT
            raw_toolchains.update(languages)
            if advanced_boundaries:
                features = sorted(set(features) | set(advanced_boundaries))

    toolchains = sorted(raw_toolchains)

    return StackClassification(
        architecture=architecture,
        toolchains=toolchains,
        languages=languages,
        features=features,
    )


def architecture_intent(classification: StackClassification) -> str:
    """Return the legacy intent string used by existing router helpers."""
    return classification.architecture


def _explicit_source_paths(prompt: str) -> Dict[str, List[str]]:
    """Find explicit source file paths mentioned in a prompt.

    Returns a mapping: ``python``, ``cpp``, ``header``, ``rust``.
    """
    pattern = re.compile(r"\b([A-Za-z_][\w/]*\.(?:py|cpp|cc|cxx|h|hpp|rs))\b")
    raw = pattern.findall(prompt)
    result: Dict[str, List[str]] = {
        "python": [],
        "cpp": [],
        "header": [],
        "rust": [],
    }
    for p in raw:
        suffix = p.rsplit(".", 1)[-1].lower()
        if suffix == "py":
            result["python"].append(p)
        elif suffix in ("cpp", "cc", "cxx"):
            result["cpp"].append(p)
        elif suffix in ("h", "hpp"):
            result["header"].append(p)
        elif suffix == "rs":
            result["rust"].append(p)
    return result


def extract_source_directories(prompt: str) -> Dict[str, Optional[str]]:
    """Extract canonical directory names from explicit paths in *prompt*.

    Keys:
      - ``python_package``: first non-test package directory from a ``.py`` path.
      - ``cpp_source``: first ``.cpp``/``.cc``/``.cxx`` path.
      - ``cpp_header``: first ``.h``/``.hpp`` path.
      - ``rust_crate_dir``: directory containing ``Cargo.toml`` or the first ``.rs`` crate.
    """
    paths = _explicit_source_paths(prompt)
    result: Dict[str, Optional[str]] = {
        "python_package": None,
        "cpp_source": None,
        "cpp_header": None,
        "rust_crate_dir": None,
    }
    non_package = {"tests", "src", "scripts", "examples", "docs"}
    for p in paths["python"]:
        parts = Path(p).parts
        if len(parts) > 1 and parts[0] not in non_package:
            if "test" not in Path(p).name.lower():
                result["python_package"] = parts[0]
                break

    if paths["cpp"]:
        result["cpp_source"] = paths["cpp"][0]

    if paths["header"]:
        result["cpp_header"] = paths["header"][0]

    # Prefer an explicit Cargo.toml directory, then infer from .rs layout.
    for p in paths["rust"]:
        if Path(p).name == "Cargo.toml":
            result["rust_crate_dir"] = str(Path(p).parent)
            break
    if not result["rust_crate_dir"] and paths["rust"]:
        p = Path(paths["rust"][0])
        if p.name == "lib.rs" and p.parent.name == "src":
            result["rust_crate_dir"] = str(p.parent.parent)
        else:
            result["rust_crate_dir"] = str(p.parent)

    return result


def default_manifest_for_architecture(
    architecture: str,
    project_name: str,
    main_function: str = "compute",
    prompt: str = "",
) -> List[Dict[str, str]]:
    """Return a generic manifest for an architecture when the LLM omits one.

    When *prompt* contains explicit file paths (e.g. ``cpp_engine/src/kernels.cpp``),
    the generated manifest uses those locations instead of hardcoded defaults.
    """
    pkg = _sanitize_name(project_name)
    dirs = extract_source_directories(prompt)
    python_package = dirs["python_package"] or pkg
    cpp_source = dirs["cpp_source"] or None
    rust_crate_dir = dirs["rust_crate_dir"] or "rust_core"
    rust_cargo = f"{rust_crate_dir}/Cargo.toml"
    rust_lib = f"{rust_crate_dir}/src/lib.rs"

    if architecture in (INTENT_TRI_POLYGLOT_RUST_CPP_PYTHON, INTENT_GRAPH_POLYGLOT):
        cpp_entry = cpp_source or "cpp_core/native.cpp"
        manifest = [
            {"path": "Cargo.toml", "lang": "toml", "purpose": "Rust workspace manifest"},
            {"path": rust_cargo, "lang": "toml", "purpose": "PyO3 crate manifest"},
            {"path": rust_lib, "lang": "rust", "purpose": "Rust native core"},
            {"path": cpp_entry, "lang": "cpp", "purpose": "C-ABI dynamic shared library source"},
            {"path": "pyproject.toml", "lang": "toml", "purpose": "Python package manifest"},
            {"path": f"{python_package}/__init__.py", "lang": "python", "purpose": "Python driver package"},
            {"path": f"{python_package}/main.py", "lang": "python", "purpose": "Python CLI / REPL entrypoint"},
            {"path": "run_shell.py", "lang": "python", "purpose": "Headless launcher"},
            {"path": "tests/test_graph.py", "lang": "python", "purpose": "pytest tests"},
            {"path": "README.md", "lang": "markdown", "purpose": "Project README"},
        ]
        if dirs["cpp_header"]:
            manifest.insert(3, {"path": dirs["cpp_header"], "lang": "cpp", "purpose": "C-ABI header"})
        if _has_any(_lower_tokens(prompt), ["go", "golang"]):
            manifest.insert(
                4,
                {"path": "go_core/main.go", "lang": "go", "purpose": "Go module entrypoint"},
            )
        return manifest
    if architecture == INTENT_HYBRID_CPP_RUST:
        cpp_entry = cpp_source or "src/cpp_core/native.cpp"
        return [
            {"path": "Cargo.toml", "lang": "toml", "purpose": "Rust package manifest"},
            {"path": "build.rs", "lang": "rust", "purpose": "C++ build and link script"},
            {"path": "src/main.rs", "lang": "rust", "purpose": "Rust CLI binary"},
            {"path": cpp_entry, "lang": "cpp", "purpose": "C-ABI math source"},
            {"path": "tests/test_hybrid_cpp_rust.rs", "lang": "rust", "purpose": "Rust integration test"},
            {"path": "README.md", "lang": "markdown", "purpose": "Project README"},
        ]
    if architecture == INTENT_HYBRID_RUST_PYTHON:
        return [
            {"path": "Cargo.toml", "lang": "toml", "purpose": "Rust workspace manifest"},
            {"path": rust_cargo, "lang": "toml", "purpose": "PyO3 crate manifest"},
            {"path": rust_lib, "lang": "rust", "purpose": "Rust native core"},
            {"path": "pyproject.toml", "lang": "toml", "purpose": "Python package manifest"},
            {"path": f"{python_package}/__init__.py", "lang": "python", "purpose": "Python driver package"},
            {"path": f"{python_package}/core.py", "lang": "python", "purpose": "Python wrapper"},
            {"path": f"{python_package}/native.py", "lang": "python", "purpose": "native loader"},
            {"path": "tests/test_core.py", "lang": "python", "purpose": "pytest tests"},
            {"path": "README.md", "lang": "markdown", "purpose": "Project README"},
        ]
    if architecture == INTENT_PURE_RUST:
        return [
            {"path": "Cargo.toml", "lang": "toml", "purpose": "Rust workspace manifest"},
            {"path": "src/lib.rs", "lang": "rust", "purpose": "Rust core library"},
            {"path": "README.md", "lang": "markdown", "purpose": "Project README"},
        ]
    if architecture == INTENT_HYBRID_CPP_PYTHON:
        cpp_entry = cpp_source or "src/cpp_core/extension.cpp"
        return [
            {"path": "CMakeLists.txt", "lang": "cmake", "purpose": "CMake build"},
            {"path": cpp_entry, "lang": "cpp", "purpose": "C++ extension"},
            {"path": "pyproject.toml", "lang": "toml", "purpose": "Python package manifest"},
            {"path": f"src/{pkg}/__init__.py", "lang": "python", "purpose": "Python package"},
            {"path": f"src/{pkg}/core.py", "lang": "python", "purpose": "Python wrapper"},
            {"path": "tests/test_core.py", "lang": "python", "purpose": "pytest tests"},
            {"path": "README.md", "lang": "markdown", "purpose": "Project README"},
        ]
    # pure_python / generic fallback
    return [
        {"path": "pyproject.toml", "lang": "toml", "purpose": "Python package manifest"},
        {"path": f"src/{pkg}/__init__.py", "lang": "python", "purpose": "Python package"},
        {"path": f"src/{pkg}/core.py", "lang": "python", "purpose": "Pure Python implementation"},
        {"path": "tests/test_core.py", "lang": "python", "purpose": "pytest tests"},
        {"path": "README.md", "lang": "markdown", "purpose": "Project README"},
    ]


def _sanitize_name(name: str) -> str:
    sanitized = "".join(c if c.isalnum() or c == "_" else "_" for c in name.lower())
    sanitized = sanitized.strip("_")
    if not sanitized or sanitized[0].isdigit():
        sanitized = "engine"
    return sanitized


def suggested_blueprint_template(architecture: str) -> str:
    """Return the name of the closest reference blueprint template."""
    mapping = {
        INTENT_PURE_PYTHON: "pure_python",
        INTENT_PURE_RUST: "pure_rust",
        INTENT_HYBRID_RUST_PYTHON: "hybrid_rust_python",
        INTENT_HYBRID_CPP_PYTHON: "hybrid_cpp_python",
        INTENT_HYBRID_CPP_RUST: "hybrid_cpp_rust",
        INTENT_TRI_POLYGLOT_RUST_CPP_PYTHON: "tri_polyglot_rust_cpp_python",
        INTENT_GRAPH_POLYGLOT: "tri_polyglot_rust_cpp_python",
    }
    return mapping.get(architecture, "pure_python")
