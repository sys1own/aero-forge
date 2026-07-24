"""Project-level build, zip upload, and bundle generation for web integration."""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from aero_forge.blueprint import (
    Blueprint,
    discover_functions,
    discover_project,
    generate_blueprint,
)
from aero_forge.build_runner import BuildRunner
from aero_forge.build_summary import format_build_summary
from aero_forge.config import ConfigOverride
from aero_forge.generate import generate_project
from aero_forge.scaffold.engine import ProjectScaffolder

logger = logging.getLogger("aero_forge.project_builder")


def _safe_extract(zip_path: Path, dest: Path) -> None:
    """Extract a zip file to ``dest``, guarding against path traversal."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            member_path = zf.getinfo(member).filename
            target = dest / member_path
            try:
                target.relative_to(dest.resolve())
            except ValueError as exc:
                raise ValueError(
                    f"Zip member escapes extraction directory: {member_path}"
                ) from exc
        zf.extractall(dest)


def _zip_directory(src: Path, zip_path: Path, arc_root: Optional[str] = None) -> None:
    """Zip the contents of ``src`` under an optional archive root directory."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    root = arc_root or src.name
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in src.rglob("*"):
            if path.is_file():
                zf.write(path, f"{root}/{path.relative_to(src)}")


class ProjectBuilder:
    """Scan a project directory, compile public functions, and bundle the result."""

    def __init__(
        self,
        project_root: Path,
        *,
        output_zip: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        llm_provider: Optional[str] = None,
        model: Optional[str] = None,
        max_workers: int = 4,
        cache_enabled: bool = True,
        target: str = "native",
        template: Optional[str] = None,
        config_override: Optional[ConfigOverride] = None,
    ):
        self.project_root = Path(project_root).resolve()
        self.output_dir = Path(output_dir) if output_dir else self.project_root / "dist"
        if output_zip:
            self.output_zip = Path(output_zip).resolve()
        else:
            self.output_zip = (
                self.project_root.parent / f"{self.project_root.name}_bundle.zip"
            )
        self.llm_provider = llm_provider
        self.model = model
        self.max_workers = max_workers
        self.cache_enabled = cache_enabled
        self.target = target
        self.template = template
        self.config_override = config_override

    def _discover(self) -> List[Any]:
        """Return discovered ``FunctionSpec`` objects for the project."""
        return discover_project(self.project_root)

    def scaffold(self, template: Optional[str] = None) -> Dict[str, Any]:
        """Generate a project shell (axum, clap, or python_hybrid) for the project."""
        chosen = template or self.template
        if not chosen:
            raise ValueError("No template specified for scaffold()")
        functions = [f.name for f in self._discover()] or None
        project_name = self.project_root.name
        if chosen == "axum":
            root = ProjectScaffolder.scaffold_axum(
                self.project_root, project_name, functions
            )
        elif chosen == "clap":
            root = ProjectScaffolder.scaffold_clap(
                self.project_root, project_name, functions
            )
        elif chosen == "python_hybrid":
            root = ProjectScaffolder.scaffold_python_hybrid(
                self.project_root, project_name, functions
            )
        else:
            raise ValueError(f"Unknown template: {chosen}")
        return {
            "status": "scaffolded",
            "template": chosen,
            "project_root": str(root),
            "files": sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()),
        }

    def _context_summary(self, functions: List[Any]) -> str:
        """Build a short text summary of existing project functions for the LLM."""
        by_file: Dict[str, List[str]] = {}
        for func in functions:
            by_file.setdefault(
                str(func.file.relative_to(self.project_root)), []
            ).append(func.name)
        lines = ["Existing project files and functions:"]
        for file, names in sorted(by_file.items()):
            lines.append(f"  {file}: {', '.join(sorted(names))}")
        return "\n".join(lines)

    def build(self) -> Dict[str, Any]:
        """Compile all public functions in the project and bundle the result."""
        start = time.perf_counter()
        functions = self._discover()
        if not functions:
            return {
                "status": "error",
                "error": f"No public Python functions found in {self.project_root}",
                "output_zip": None,
            }

        self.output_dir.mkdir(parents=True, exist_ok=True)
        blueprint = generate_blueprint(
            project=self.project_root.name,
            functions=functions,
            output_dir=self.output_dir,
        )

        runner = BuildRunner(
            blueprint,
            max_workers=self.max_workers,
            llm_provider=self.llm_provider,
            model=self.model,
            cache_enabled=self.cache_enabled,
            target=self.target,
            config_override=self.config_override,
        )
        build_result = runner.build()

        elapsed = time.perf_counter() - start
        manifest = self._write_manifest(build_result, functions, elapsed)
        self._write_package_init(build_result)

        _zip_directory(
            self.project_root, self.output_zip, arc_root=self.project_root.name
        )

        summary = ""
        if build_result.get("success"):
            summary = format_build_summary(
                build_result,
                output_dir=self.output_dir,
                llm_provider=self.llm_provider,
                model=self.model,
                config_override=self.config_override,
            )

        return {
            "status": "success" if build_result.get("success") else "partial",
            "success": build_result.get("success", False),
            "project_root": str(self.project_root),
            "output_zip": str(self.output_zip),
            "output_dir": str(self.output_dir),
            "functions_compiled": sorted(
                {
                    name
                    for r in build_result.get("results", [])
                    for name in r.get("functions", [])
                }
            ),
            "total": build_result.get("total", 0),
            "passed": build_result.get("passed", 0),
            "failed": build_result.get("failed", 0),
            "build_time_seconds": round(elapsed, 3),
            "manifest": manifest,
            "summary": summary,
            "build": build_result,
        }

    def generate_and_build(
        self,
        prompt: str,
        *,
        constraints: Optional[str] = None,
        prompt_template: Optional[str] = None,
        output_name: str = "generated",
    ) -> Dict[str, Any]:
        """Generate a new function in the project context and rebuild the project."""
        functions = self._discover()
        context = (
            self._context_summary(functions) if functions else "The project is empty."
        )

        src_dir = self.project_root / "src"
        src_dir.mkdir(parents=True, exist_ok=True)

        enriched_prompt = (
            f"{prompt}\n\n"
            f"Project context:\n{context}\n\n"
            "Place the new function in a file named "
            f"`src/{output_name}.py` and add matching tests in `tests/test_{output_name}.py`. "
            "Use compatible types with the existing functions when possible."
        )

        generate_project(
            enriched_prompt,
            constraints=constraints,
            output_dir=self.project_root,
            project_name=output_name,
            llm_provider=self.llm_provider,
            model=self.model,
            prompt_template=prompt_template,
            config_override=self.config_override,
        )

        return self.build()

    def _write_manifest(
        self, build_result: Dict[str, Any], functions: List[Any], elapsed: float
    ) -> Dict[str, Any]:
        """Create ``build_manifest.json`` in the output directory."""
        manifest = {
            "project": self.project_root.name,
            "functions_compiled": sorted(
                {
                    name
                    for r in build_result.get("results", [])
                    for name in r.get("functions", [])
                }
            ),
            "files": [str(r.get("source")) for r in build_result.get("results", [])],
            "artifacts": [
                str(self.output_dir / Path(r.get("artifact")).name)
                for r in build_result.get("results", [])
                if r.get("artifact")
            ],
            "tests_passed": build_result.get("passed", 0),
            "tests_failed": build_result.get("failed", 0),
            "build_time_seconds": round(elapsed, 3),
            "status": "success" if build_result.get("success") else "partial",
        }
        manifest_path = self.output_dir / "build_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, default=str), encoding="utf-8"
        )
        return manifest

    def _write_package_init(self, build_result: Dict[str, Any]) -> None:
        """Write ``__init__.py`` files so the compiled modules can be imported as a package."""
        dist_init = self.output_dir / "__init__.py"
        dist_init.write_text("# Compiled Aero-Forge modules\n", encoding="utf-8")

        root_init = self.project_root / "__init__.py"
        imports: List[str] = []
        all_names: List[str] = []
        for r in build_result.get("results", []):
            if not r.get("success"):
                continue
            source = Path(r["source"])
            module = source.stem
            for name in r.get("functions", []):
                imports.append(f"from .dist.{module} import {name}")
                all_names.append(name)
        if imports:
            root_init.write_text(
                "\n".join(imports)
                + "\n\n__all__ = ["
                + ", ".join(f'"{n}"' for n in all_names)
                + "]\n",
                encoding="utf-8",
            )
        elif root_init.is_file():
            root_init.write_text("# Aero-Forge project package\n", encoding="utf-8")


def build_from_upload(
    zip_path: Path,
    *,
    output_zip: Optional[Path] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Extract a zip, build the project, and re-zip the result."""
    zip_path = Path(zip_path).resolve()
    if not zip_path.is_file():
        return {
            "status": "error",
            "error": f"Upload zip not found: {zip_path}",
            "output_zip": None,
        }

    with tempfile.TemporaryDirectory(prefix="aero_forge_upload_") as tmp:
        extract_root = Path(tmp) / "project"
        extract_root.mkdir(parents=True, exist_ok=True)
        _safe_extract(zip_path, extract_root)

        # If the zip contained a single top-level directory, use it as the project root.
        top_dirs = [p for p in extract_root.iterdir() if p.is_dir()]
        top_files = [p for p in extract_root.iterdir() if p.is_file()]
        if len(top_dirs) == 1 and not top_files:
            project_root = top_dirs[0]
        else:
            project_root = extract_root

        builder = ProjectBuilder(
            project_root,
            output_zip=output_zip,
            **kwargs,
        )
        return builder.build()
