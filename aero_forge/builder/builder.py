"""High-level polyglot builder for aero-forge engine specs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from aero_forge.builder.artifact_generator import ArtifactBundle, ArtifactGenerator
from aero_forge.builder.emitters import get_emitter
from aero_forge.builder.language_router import resolve_target_language
from aero_forge.builder.spec import EngineSpec


@dataclass
class BuildOutput:
    """Result of building an engine spec for a target language."""

    language: str
    source: str
    spec: EngineSpec
    artifacts: ArtifactBundle = field(default_factory=ArtifactBundle)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "language": self.language,
            "source": self.source,
            "artifacts": self.artifacts.to_dict(),
            "metadata": dict(self.metadata),
        }


def build_engine(
    spec: EngineSpec,
    target_language: Optional[str] = None,
    *,
    context: Optional[Dict[str, Any]] = None,
    template_names: Optional[List[str]] = None,
    template_dirs: Optional[List[Path]] = None,
    output_paths: Optional[Dict[str, str]] = None,
) -> BuildOutput:
    """Render *spec* to *target_language* source and optional artifacts.

    If *target_language* is not provided, it is resolved from *context* or the
    spec's ``metadata.language`` hint.
    """
    context = context or {}
    language = target_language or resolve_target_language(
        context,
        source_language=spec.metadata.get("language"),
    )
    emitter = get_emitter(language)
    source = emitter.emit(spec)

    artifacts = ArtifactBundle()
    if template_names:
        generator = ArtifactGenerator(template_dirs=template_dirs)
        artifacts = generator.generate(
            spec,
            template_names,
            output_paths=output_paths,
        )

    return BuildOutput(
        language=language,
        source=source,
        spec=spec,
        artifacts=artifacts,
        metadata={"language": language, **spec.metadata},
    )
