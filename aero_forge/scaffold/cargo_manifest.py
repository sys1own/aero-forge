"""Cargo manifest handling and synthesis for aero-forge."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger("aero_forge.cargo")

MANIFEST_NAME = "Cargo.toml"
DEFAULT_EDITION = "2021"
DEFAULT_VERSION = "0.1.0"
PYO3_REQUIRED_FEATURES = ("extension-module", "experimental-declarative-modules")


def _load_toml(path: Path) -> Optional[Dict[str, Any]]:
    """Load a TOML file using the best available parser."""
    try:
        import tomllib

        with open(path, "rb") as fh:
            return tomllib.load(fh)
    except Exception:
        pass
    try:
        import tomli

        with open(path, "rb") as fh:
            return tomli.load(fh)
    except Exception:
        pass
    try:
        import toml

        return toml.load(str(path))
    except Exception:
        return None


def ensure_pyo3_features(dependencies: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Guarantee a ``pyo3`` dependency declares the declarative-modules features."""
    if not dependencies or "pyo3" not in dependencies:
        return dict(dependencies or {})
    deps = dict(dependencies)
    spec = deps["pyo3"]
    if isinstance(spec, dict):
        new_spec = dict(spec)
        features = list(new_spec.get("features", []) or [])
    else:
        new_spec = {"version": str(spec)}
        features = []
    for feature in PYO3_REQUIRED_FEATURES:
        if feature not in features:
            features.append(feature)
    new_spec["features"] = features
    deps["pyo3"] = new_spec
    return deps


@dataclass
class CargoPlan:
    """How (and where) a Rust target should be built."""

    crate_root: Path
    manifest_path: Path
    crate_name: str
    used_existing: bool
    synthesized: bool
    dependencies: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    @property
    def target_dir(self) -> Path:
        return self.crate_root / "target"

    def profile_dir(self, release: bool = False) -> Path:
        return self.target_dir / ("release" if release else "debug")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "crate_root": str(self.crate_root),
            "manifest_path": str(self.manifest_path),
            "crate_name": self.crate_name,
            "used_existing": self.used_existing,
            "synthesized": self.synthesized,
            "dependencies": dict(self.dependencies),
            "target_dir": str(self.target_dir),
            "notes": list(self.notes),
        }


def _walk_up_for_manifest(start: Path, ceiling: Path) -> Optional[Path]:
    start = start.resolve()
    ceiling = ceiling.resolve()
    current = start if start.is_dir() else start.parent
    while True:
        candidate = current / MANIFEST_NAME
        if candidate.is_file():
            return candidate
        if current == ceiling or current.parent == current:
            return None
        if ceiling not in current.parents and current != ceiling:
            return None
        current = current.parent


def resolve_crate_root(
    workspace: Path,
    sources: Sequence[str] = (),
    manifest_path: Optional[str] = None,
    root: Optional[str] = None,
) -> Path:
    workspace = Path(workspace).resolve()
    if manifest_path:
        candidate = (workspace / manifest_path).resolve()
        if candidate.name == MANIFEST_NAME or candidate.suffix == ".toml":
            return candidate.parent
        return candidate
    if root:
        return (workspace / root).resolve()
    for source in sources:
        source_path = (workspace / source).resolve()
        found = _walk_up_for_manifest(source_path, workspace)
        if found is not None:
            return found.parent
    source_dirs = {(workspace / s).resolve().parent for s in sources if s}
    if len(source_dirs) == 1:
        only = next(iter(source_dirs))
        if only.name == "src":
            return only.parent
        return only
    return workspace


def find_existing_manifest(crate_root: Path) -> Optional[Path]:
    candidate = Path(crate_root) / MANIFEST_NAME
    return candidate if candidate.is_file() else None


def read_manifest_package_name(manifest_path: Path) -> Optional[str]:
    data = _load_toml(manifest_path)
    if not data:
        return None
    try:
        name = data.get("package", {}).get("name")
        return str(name) if name else None
    except Exception as exc:
        logger.debug("Could not read package name from %s: %s", manifest_path, exc)
        return None


def sanitize_crate_name(name: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z_]+", "_", name.strip()).strip("_")
    if not cleaned:
        cleaned = "aero_crate"
    if cleaned[0].isdigit():
        cleaned = f"crate_{cleaned}"
    return cleaned.lower()


def _render_toml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_render_toml_scalar(item) for item in value) + "]"
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _render_dependency(name: str, spec: Any) -> str:
    if isinstance(spec, dict):
        inner = ", ".join(f"{key} = {_render_toml_scalar(val)}" for key, val in spec.items())
        return f"{name} = {{ {inner} }}"
    return f"{name} = {_render_toml_scalar(str(spec))}"


def render_manifest(
    crate_name: str,
    dependencies: Optional[Dict[str, Any]] = None,
    edition: str = DEFAULT_EDITION,
    version: str = DEFAULT_VERSION,
    crate_type: Optional[Sequence[str]] = None,
    header: Optional[str] = None,
) -> str:
    """Render a minimal, valid ``Cargo.toml`` as text."""
    dependencies = ensure_pyo3_features(dependencies or {})
    if header is None:
        header_lines = [
            "# Synthesised by aero-forge. Commit a Cargo.toml to take full control;",
            "# aero-forge will then use it verbatim and never overwrite it.",
        ]
    else:
        header_lines = [line if line.startswith("#") or not line else f"# {line}" for line in header.splitlines()]
    lines = [
        *header_lines,
        "[package]",
        f'name = "{crate_name}"',
        f'version = "{version}"',
        f'edition = "{edition}"',
        "",
    ]
    if crate_type:
        lines.extend(["[lib]", f"crate-type = {_render_toml_scalar(list(crate_type))}", ""])
    lines.append("[dependencies]")
    for name, spec in dependencies.items():
        lines.append(_render_dependency(name, spec))
    return "\n".join(lines).rstrip() + "\n"


def prepare_crate(
    workspace: Path,
    target_name: str,
    sources: Sequence[str] = (),
    cargo_options: Optional[Dict[str, Any]] = None,
    manifest_path: Optional[str] = None,
    root: Optional[str] = None,
    write: bool = True,
) -> CargoPlan:
    """Resolve crate root and either respect or synthesise its manifest."""
    workspace = Path(workspace).resolve()
    cargo_options = cargo_options or {}

    crate_root = resolve_crate_root(workspace, sources, manifest_path, root)

    explicit_manifest: Optional[Path] = None
    if manifest_path:
        candidate = (workspace / manifest_path).resolve()
        if candidate.is_file():
            explicit_manifest = candidate
            crate_root = candidate.parent

    existing = explicit_manifest or find_existing_manifest(crate_root)
    if existing is not None:
        crate_name = read_manifest_package_name(existing) or sanitize_crate_name(target_name)
        return CargoPlan(
            crate_root=crate_root,
            manifest_path=existing,
            crate_name=crate_name,
            used_existing=True,
            synthesized=False,
            dependencies={},
            notes=[f"using existing manifest at {existing}"],
        )

    crate_name = sanitize_crate_name(cargo_options.get("package_name", target_name))
    dependencies = dict(cargo_options.get("dependencies", {}) or {})
    edition = str(cargo_options.get("edition", DEFAULT_EDITION))
    version = str(cargo_options.get("version", DEFAULT_VERSION))
    crate_type = cargo_options.get("crate_type")

    manifest = crate_root / MANIFEST_NAME
    notes = [f"synthesised manifest for crate '{crate_name}'"]
    if dependencies:
        notes.append("pinned dependencies: " + ", ".join(f"{k}={v}" for k, v in dependencies.items()))

    if write:
        crate_root.mkdir(parents=True, exist_ok=True)
        manifest.write_text(
            render_manifest(crate_name, dependencies, edition, version, crate_type),
            encoding="utf-8",
        )

    return CargoPlan(
        crate_root=crate_root,
        manifest_path=manifest,
        crate_name=crate_name,
        used_existing=False,
        synthesized=True,
        dependencies=dependencies,
        notes=notes,
    )
