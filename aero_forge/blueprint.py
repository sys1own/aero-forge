"""Blueprint parser and validation for multi-function builds."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator

logger = logging.getLogger("aero_forge.blueprint")


class FunctionSpec(BaseModel):
    """A single function or a wildcard entry that compiles every public function in ``file``."""

    file: Path
    name: Optional[str] = None
    compile_all: bool = False
    tests: List[Path] = Field(default_factory=list)
    output_name: Optional[str] = None
    compiler_flags: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _resolve(self) -> "FunctionSpec":
        if self.name == "*":
            self.compile_all = True
        if self.compile_all:
            self.name = self.name or "*"
        if not self.compile_all and not self.name:
            raise ValueError("FunctionSpec requires 'name' unless 'compile_all' is true")
        if self.output_name is None:
            self.output_name = self.name
        return self


class LLMConfig(BaseModel):
    """LLM settings inside a blueprint."""

    provider: str = "none"
    model: Optional[str] = None


class Blueprint(BaseModel):
    """Normalized build blueprint."""

    project: str = "aero_forge_project"
    functions: List[FunctionSpec]
    compiler_flags: List[str] = Field(default_factory=list)
    output_dir: Path = Path("./dist")
    llm: LLMConfig = Field(default_factory=LLMConfig)

    @model_validator(mode="after")
    def _validate_files(self) -> "Blueprint":
        missing: List[str] = []
        for func in self.functions:
            if not func.file.is_file():
                missing.append(str(func.file))
            for test in func.tests:
                if not test.is_file():
                    missing.append(str(test))
        if missing:
            raise ValueError(f"Blueprint references missing file(s): {', '.join(missing)}")
        return self


def _is_yaml_content(text: str) -> bool:
    """Heuristic: if the first non-empty character is one of YAML structural markers."""
    first = ""
    for char in text.lstrip():
        if char and not char.isspace():
            first = char
            break
    return first in {"-", "[", "{", "p", "f", "c", "o", "l", "#"}


def parse_aero(text: str) -> Dict[str, Any]:
    """Parse a ``.aero`` blueprint.

    Aero-forge ``.aero`` files are YAML-compatible for the build command. If
    parsing fails, fall back to the legacy INI-style parser for compatibility
    with older accelerator blueprints.
    """
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        data = None

    if data is not None:
        return data

    # Legacy INI/TOML-like fallback.
    data = {}
    current: Optional[str] = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1].strip()
            data.setdefault(current, {})
            continue
        if current is None or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        data[current][key] = _parse_ini_value(value)
    return data


def _parse_ini_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    if value.startswith("[") and value.endswith("]"):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    try:
        if "." in value or "e" in value.lower():
            return float(value)
        return int(value)
    except ValueError:
        pass
    return value


def parse_blueprint(path: Path) -> Blueprint:
    """Parse ``.aero`` or ``.yaml`` blueprint into a normalized model."""
    path = Path(path).resolve()
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()

    if suffix in {".yaml", ".yml"}:
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ValueError(f"Failed to parse YAML blueprint {path}: {exc}") from exc
    elif suffix == ".aero":
        data = parse_aero(text)
    else:
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError:
            data = None
        if data is None:
            data = parse_aero(text)

    if not isinstance(data, dict):
        raise ValueError(f"Blueprint {path} did not parse to a mapping")

    # Normalize functions to absolute paths relative to the blueprint directory.
    base = path.parent
    for func in data.get("functions", []):
        if not isinstance(func, dict):
            continue
        func["file"] = str(base / Path(func["file"]))
        if "tests" in func:
            func["tests"] = [str(base / Path(t)) for t in func["tests"]]

    if "output_dir" in data:
        data["output_dir"] = str(base / Path(data["output_dir"]))

    try:
        return Blueprint.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"Invalid blueprint {path}: {exc}") from exc


def discover_functions(path: Path) -> List[FunctionSpec]:
    """Discover all top-level public functions in a Python file."""
    import ast

    source_path = Path(path)
    if not source_path.is_file():
        raise ValueError(f"Source file not found: {source_path}")

    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    functions: List[FunctionSpec] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            test_candidates = [
                source_path.parent / f"test_{node.name}.py",
                source_path.parent / f"test_{source_path.stem}.py",
            ]
            tests = [t for t in test_candidates if t.is_file()]
            functions.append(
                FunctionSpec(
                    file=source_path,
                    name=node.name,
                    tests=tests,
                )
            )
    return functions


def generate_blueprint(
    project: str,
    functions: List[FunctionSpec],
    output_dir: Path = Path("./dist"),
    compiler_flags: Optional[List[str]] = None,
) -> Blueprint:
    """Create a Blueprint from discovered or supplied function specs."""
    return Blueprint(
        project=project,
        functions=functions,
        output_dir=output_dir,
        compiler_flags=compiler_flags or [],
    )


def write_blueprint(blueprint: Blueprint, path: Path) -> None:
    """Serialize a Blueprint to a YAML ``.aero`` file."""
    path.write_text(
        yaml.safe_dump(
            blueprint.model_dump(mode="json"),
            sort_keys=False,
            default_flow_style=False,
        ),
        encoding="utf-8",
    )
