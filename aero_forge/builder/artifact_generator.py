"""Template-driven artifact generation for aero-forge engine specs."""

from __future__ import annotations

import json
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from aero_forge.builder.spec import EngineSpec


class TemplateNotFoundError(Exception):
    """Raised when a requested template cannot be found."""


class ArtifactGenerationError(Exception):
    """Raised when artifact rendering fails."""


@dataclass
class Artifact:
    """A rendered artifact file."""

    path: str
    content: str


@dataclass
class ArtifactBundle:
    """A collection of artifacts generated for an engine spec."""

    artifacts: List[Artifact] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifacts": [{"path": a.path, "content": a.content} for a in self.artifacts],
            "metadata": dict(self.metadata),
        }


class ArtifactGenerator:
    """Render domain-agnostic artifacts from templates and an engine spec.

    Templates are loaded from the package ``templates/`` directory and any
    user-supplied ``template_dirs``. They use :class:`string.Template` syntax:
    ``$project_name``, ``$struct_name``, ``$key_type``, ``$value_type``, etc.
    """

    def __init__(
        self,
        template_dirs: Optional[List[Path]] = None,
        package_template_dir: Optional[Path] = None,
    ) -> None:
        self.template_dirs: List[Path] = []
        if template_dirs:
            self.template_dirs.extend(template_dirs)
        if package_template_dir is None:
            package_template_dir = Path(__file__).parent / "templates"
        if package_template_dir.is_dir():
            self.template_dirs.append(package_template_dir)

    def _find_template(self, name: str) -> Path:
        for directory in self.template_dirs:
            candidate = directory / f"{name}.tmpl"
            if candidate.is_file():
                return candidate
            candidate = directory / name
            if candidate.is_file():
                return candidate
        raise TemplateNotFoundError(f"Template {name!r} not found in {self.template_dirs}")

    def _substitute(self, template: string.Template, spec: EngineSpec, **extra: Any) -> str:
        values: Dict[str, str] = {
            "project_name": spec.name,
            "language": spec.metadata.get("language", "rust"),
            "safe_name": spec.name.replace("-", "_").replace(" ", "_"),
        }
        values.update({k: str(v) for k, v in spec.metadata.items()})
        values.update({k: str(v) for k, v in extra.items()})
        try:
            return template.substitute(values)
        except KeyError as exc:
            raise ArtifactGenerationError(f"Missing template placeholder: {exc}") from exc

    def render(self, template_name: str, spec: EngineSpec, *, output_path: str, **extra: Any) -> Artifact:
        """Render a single template into an :class:`Artifact`."""
        path = self._find_template(template_name)
        text = path.read_text(encoding="utf-8")
        template = string.Template(text)
        content = self._substitute(template, spec, **extra)
        return Artifact(path=output_path, content=content)

    def generate(
        self,
        spec: EngineSpec,
        template_names: List[str],
        *,
        output_paths: Optional[Dict[str, str]] = None,
        **extra: Any,
    ) -> ArtifactBundle:
        """Render a bundle of artifacts for *spec*.

        ``output_paths`` maps template name to output path; missing entries use
        the template name with the ``.tmpl`` extension stripped.
        """
        bundle = ArtifactBundle(metadata={"project_name": spec.name, "language": spec.metadata.get("language", "rust")})
        output_paths = output_paths or {}
        for name in template_names:
            out = output_paths.get(name, name.replace(".tmpl", ""))
            artifact = self.render(name, spec, output_path=out, **extra)
            bundle.artifacts.append(artifact)
        return bundle

    def list_templates(self) -> List[str]:
        """Return available template names."""
        names: List[str] = []
        seen: set[str] = set()
        for directory in self.template_dirs:
            if not directory.is_dir():
                continue
            for path in directory.iterdir():
                if path.is_file() and path.suffix == ".tmpl":
                    if path.stem not in seen:
                        seen.add(path.stem)
                        names.append(path.stem)
        return sorted(names)
