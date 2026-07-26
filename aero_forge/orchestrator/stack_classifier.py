"""Universal prompt-to-stack classification for blueprint generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple


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
    "python": ["python", "py", "maturin", "pyo3", "cython"],
    "rust": ["rust", "cargo", "crates", "rustc"],
    "cpp": ["c++", "cpp", "cplusplus", "cxx", "pybind11", "cffi", "cmake"],
    "go": ["go", "golang"],
    "javascript": ["javascript", "js", "node", "nodejs", "typescript", "ts"],
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
INTENT_GENERIC = "generic"


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
    has_python_only = _has_any(tokens, ["pure python", "python only"])
    has_rust_only = _has_any(tokens, ["pure rust", "rust only", "cargo only"])

    if (has_python and has_rust) or (has_hybrid_word and (has_python or has_rust)):
        architecture = INTENT_HYBRID_RUST_PYTHON
        raw_toolchains.update(["python", "rust", "cargo"])
    elif has_python and has_cpp:
        architecture = INTENT_HYBRID_CPP_PYTHON
        raw_toolchains.update(["python", "cpp", "cmake"])
    elif has_rust or has_rust_only:
        architecture = INTENT_PURE_RUST
        raw_toolchains.update(["rust", "cargo"])
    elif has_cpp:
        architecture = INTENT_GENERIC
    else:
        architecture = INTENT_PURE_PYTHON

    if has_python_only:
        architecture = INTENT_PURE_PYTHON
    if has_rust_only:
        architecture = INTENT_PURE_RUST

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


def default_manifest_for_architecture(
    architecture: str, project_name: str, main_function: str = "compute"
) -> List[Dict[str, str]]:
    """Return a generic manifest for an architecture when the LLM omits one."""
    pkg = _sanitize_name(project_name)
    if architecture == INTENT_HYBRID_RUST_PYTHON:
        python_package = pkg
        return [
            {"path": "Cargo.toml", "lang": "toml", "purpose": "Rust workspace manifest"},
            {"path": "rust_core/Cargo.toml", "lang": "toml", "purpose": "PyO3 crate manifest"},
            {"path": "rust_core/src/lib.rs", "lang": "rust", "purpose": "Rust native core"},
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
        return [
            {"path": "CMakeLists.txt", "lang": "cmake", "purpose": "CMake build"},
            {"path": "src/cpp_core/extension.cpp", "lang": "cpp", "purpose": "C++ extension"},
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
    }
    return mapping.get(architecture, "pure_python")
