"""Universal repository generator: from Python source to Rust/Python project."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from aero_forge.builder import BuildOutput, build_engine, spec_from_python
from aero_forge.overlay import OverlayManager
from aero_forge.scaffold.cargo_manifest import sanitize_crate_name
from aero_forge.scaffold.python_repo_generator import (
    PythonRepoSpec,
    build_python_spec,
    generate_python_repo,
    sanitize_project_name,
)
from aero_forge.scaffold.repo_generator import (
    RepoSpec,
    build_spec,
    generate_repo,
)


@dataclass
class UniversalGenerationResult:
    """Result of generating a project from a Python source prompt."""

    language: str
    root: Path
    files: List[str]
    spec: Union[RepoSpec, PythonRepoSpec]
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "language": self.language,
            "root": str(self.root),
            "files": list(self.files),
            "spec": self.spec.to_dict(),
            "source": self.source,
            "metadata": dict(self.metadata),
        }


def _extract_source_from_prompt(prompt_or_source: str) -> str:
    """If the prompt contains a fenced code block, return its contents; else return the whole text."""
    fenced = re.search(r"```(?:python)?\n(.*?)```", prompt_or_source, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return prompt_or_source.strip()


class UniversalRepoGenerator:
    """Generate a standalone Rust or Python project from Python source intent."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._overlay_managers: Dict[Path, OverlayManager] = {}

    def _build(
        self,
        prompt: str,
        *,
        target_language: str = "rust",
        project_name: Optional[str] = None,
        entry_filename: str = "",
    ) -> tuple[BuildOutput, Union[RepoSpec, PythonRepoSpec], Path, Path]:
        """Return emitted source, repo spec, destination root, and main source file path."""
        source = _extract_source_from_prompt(prompt)
        name = project_name or "aero_project"

        spec = spec_from_python(source, name=name)
        build_output = build_engine(spec, target_language=target_language)

        if target_language == "rust":
            crate_name = sanitize_crate_name(name)
            dest = self.output_dir / crate_name
            repo_spec = build_spec(
                name=crate_name,
                source=build_output.source,
                description="Standalone Rust crate generated from prompt by aero-forge.",
            )
            source_file = dest / "src" / "lib.rs"
            return build_output, repo_spec, dest, source_file

        if target_language == "python":
            dest = self.output_dir / sanitize_project_name(name)
            repo_spec = build_python_spec(
                name=name,
                source=build_output.source,
                entry_filename=entry_filename,
                description="Standalone Python project generated from prompt by aero-forge.",
            )
            source_file = dest / repo_spec.entry_filename
            return build_output, repo_spec, dest, source_file

        raise ValueError(f"Unsupported target language for universal generation: {target_language!r}")

    def generate(
        self,
        prompt: str,
        *,
        target_language: str = "rust",
        project_name: Optional[str] = None,
        entry_filename: str = "",
    ) -> UniversalGenerationResult:
        """Generate a project for *target_language* from a Python source *prompt*."""
        build_output, repo_spec, dest, source_file = self._build(
            prompt,
            target_language=target_language,
            project_name=project_name,
            entry_filename=entry_filename,
        )

        if target_language == "rust":
            generated = generate_repo(repo_spec, dest)
        else:
            generated = generate_python_repo(repo_spec, dest)

        manager = OverlayManager(dest)
        if source_file.is_file():
            manager.record_generated(source_file)
        self._overlay_managers[dest] = manager

        return UniversalGenerationResult(
            language=target_language,
            root=generated.root,
            files=generated.files,
            spec=repo_spec,
            source=build_output.source,
        )

    def commit_overlay(self, file: Path) -> Optional[str]:
        """Commit the current on-disk contents of *file* as a user overlay."""
        path = Path(file).resolve()
        # Find an overlay manager whose workspace is an ancestor of the file.
        manager: Optional[OverlayManager] = None
        for ws, m in self._overlay_managers.items():
            try:
                path.relative_to(ws)
                manager = m
                break
            except ValueError:
                continue
        if manager is None:
            manager = OverlayManager(path.parent)
            self._overlay_managers[path.parent] = manager
            if path.is_file():
                manager.record_generated(path)
        return manager.commit_overlay(path)

    def generate_with_overlay(
        self,
        prompt: str,
        *,
        target_language: str = "rust",
        project_name: Optional[str] = None,
        entry_filename: str = "",
    ) -> UniversalGenerationResult:
        """Generate while re-applying any committed overlays for incremental updates."""
        build_output, repo_spec, dest, source_file = self._build(
            prompt,
            target_language=target_language,
            project_name=project_name,
            entry_filename=entry_filename,
        )

        manager = self._overlay_managers.get(dest)
        if manager is None:
            manager = OverlayManager(dest)
            self._overlay_managers[dest] = manager
            if source_file.is_file():
                manager.record_generated(source_file)

        if source_file.is_file():
            manager.structural_reapply(source_file, build_output.source, language=target_language)
            # Synchronise the repo spec with the merged source so generation writes it out.
            repo_spec.source = source_file.read_text(encoding="utf-8")

        if target_language == "rust":
            generated = generate_repo(repo_spec, dest)
        else:
            generated = generate_python_repo(repo_spec, dest)

        return UniversalGenerationResult(
            language=target_language,
            root=generated.root,
            files=generated.files,
            spec=repo_spec,
            source=build_output.source,
        )
