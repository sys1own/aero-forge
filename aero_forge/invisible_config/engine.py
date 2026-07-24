"""Invisible configuration engine for aero-forge."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from aero_forge.builder import build_engine, spec_from_python
from aero_forge.builder.language_router import resolve_target_language
from aero_forge.environment import VerifyDependencies
from aero_forge.invisible_config.lean_parser import LeanBlueprint, parse_lean_blueprint
from aero_forge.scaffold.universal_generator import UniversalRepoGenerator

_OPTIMIZE_PROFILES = {
    "maximum_hardware": {"optimization_level": "O3", "gpu": True, "vectorize": True},
    "balanced": {"optimization_level": "O2", "gpu": False, "vectorize": True},
    "size": {"optimization_level": "Os", "gpu": False, "vectorize": False},
    "debug": {"optimization_level": "O0", "gpu": False, "vectorize": False},
}


@dataclass
class InferredTarget:
    """A target inferred from lean blueprint intent."""

    name: str
    language: str
    role: str
    source: str = ""


class InvisibleConfigEngine:
    """Parse + infer + emit an executable build context from lean intent."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = Path(project_root)
        self._generator: Optional[UniversalRepoGenerator] = None

    def infer_from_source(self, content: str) -> Dict[str, Any]:
        """Parse a lean blueprint and return a build context."""
        blueprint = parse_lean_blueprint(content)
        return self.infer(blueprint)

    def infer(self, blueprint: LeanBlueprint) -> Dict[str, Any]:
        """Turn a :class:`LeanBlueprint` into a normalised build context."""
        profile = _OPTIMIZE_PROFILES.get(blueprint.optimize, _OPTIMIZE_PROFILES["balanced"])
        targets: List[Dict[str, Any]] = []

        # Verify environment contract for every requested target language.
        languages = {resolve_target_language(source_language=t) for t in blueprint.targets}
        for language in languages:
            VerifyDependencies.verify_language(language)

        # The optional `source` extra lets a lean blueprint carry the Python seed inline.
        source_seed = str(blueprint.extras.get("source", ""))

        for target in blueprint.targets:
            lang = resolve_target_language(source_language=target)
            targets.append(
                {
                    "name": target,
                    "language": lang,
                    "role": "executable" if lang == "python" else "library",
                    "source_seed": source_seed,
                }
            )

        return {
            "workspace_status": "inferred_active",
            "config_layer": "invisible",
            "timestamp": time.time(),
            "project": blueprint.project,
            "optimize": blueprint.optimize,
            "profile": profile,
            "ingest": list(blueprint.ingest),
            "targets": targets,
            "source_seed": source_seed,
        }

    def generate_repo(self, blueprint: LeanBlueprint, output_dir: Path) -> Dict[str, Any]:
        """Generate a standalone repository for the first requested target."""
        context = self.infer(blueprint)
        if not context["targets"]:
            raise ValueError("No targets in blueprint")
        target = context["targets"][0]
        language = target["language"]
        source_seed = context["source_seed"]
        if not source_seed:
            raise ValueError("lean blueprint 'source' extra is required for repo generation")

        self._generator = UniversalRepoGenerator(output_dir)
        result = self._generator.generate(
            source_seed,
            target_language=language,
            project_name=blueprint.project,
        )
        return result.to_dict()

    def build_context_from_source(self, content: str, *, output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Parse + infer + optionally materialise a project from lean intent."""
        blueprint = parse_lean_blueprint(content)
        context = self.infer(blueprint)
        if output_dir is not None:
            context["repo"] = self.generate_repo(blueprint, output_dir)
        return context
