"""LLM-driven synthesis of a complete Blueprint v3.0.0 from a draft or raw project."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from jinja2 import Template

from aero_forge.blueprint.schema import (
    ABIContractV3,
    ArtifactType,
    BindingFramework,
    BlueprintStatus,
    BlueprintV3,
    BuildArtifact,
    ExecutionStrategyV3,
    GenerationMethod,
    MemoryModel,
    Metadata,
    ToolchainSpec,
    VerificationNode,
    write_v3_blueprint,
)
from aero_forge.llm.clients import get_llm_client

logger = logging.getLogger("aero_forge.blueprint.synthesizer")


class LLMBlueprintSynthesizer:
    """Synthesize a finalized Blueprint v3 from a draft, project tree, or text spec."""

    DEFAULT_MODEL = os.getenv("AERO_FORGE_MODEL")

    def __init__(
        self,
        provider: str = "deepseek",
        model: Optional[str] = None,
    ) -> None:
        self.provider = provider
        self.model = model or self.DEFAULT_MODEL

    def _load_prompt(self) -> Template:
        prompt_path = Path(__file__).with_name("prompts") / "blueprint_synthesis.j2"
        # The template is stored next to the package, not inside the blueprint subpackage.
        alt_path = Path(__file__).parents[1] / "prompts" / "blueprint_synthesis.j2"
        path = prompt_path if prompt_path.is_file() else alt_path
        if path.is_file():
            return Template(path.read_text(encoding="utf-8"))
        return Template(_DEFAULT_PROMPT_TEMPLATE)

    def _client(self):
        return get_llm_client(self.provider, model=self.model)

    def synthesize(
        self,
        workspace: Path,
        draft: Optional[BlueprintV3] = None,
        spec: Optional[str] = None,
        output_path: Optional[Path] = None,
    ) -> BlueprintV3:
        """Create a finalized, transferable Blueprint v3 from the inputs in *workspace*."""
        workspace = Path(workspace).resolve()
        context = self._gather_context(workspace, draft, spec)
        prompt = self._load_prompt().render(**context)

        client = self._client()
        raw = client.generate(
            prompt,
            temperature=0.1,
            response_format={"type": "json_object"},
        )

        blueprint = self._parse_and_normalize(raw, workspace)
        if output_path:
            write_v3_blueprint(blueprint, output_path)
        return blueprint

    def synthesize_from_text(
        self,
        text: str,
        workspace: Path,
        output_path: Optional[Path] = None,
    ) -> BlueprintV3:
        """Synthesize a finalized blueprint from a raw textual description."""
        return self.synthesize(workspace, draft=None, spec=text, output_path=output_path)

    def _gather_context(
        self,
        workspace: Path,
        draft: Optional[BlueprintV3],
        spec: Optional[str],
    ) -> Dict[str, str]:
        project_name = workspace.name or "synthesized_project"
        files_summary: List[str] = []
        manifests: List[str] = []

        for pattern in ("Cargo.toml", "pyproject.toml", "setup.py", "CMakeLists.txt", "Makefile", "package.json", "go.mod"):
            for manifest in workspace.rglob(pattern):
                rel = manifest.relative_to(workspace)
                manifests.append(str(rel))
                if manifest.name in ("Cargo.toml", "pyproject.toml"):
                    try:
                        content = manifest.read_text(encoding="utf-8")
                        # Trim large manifests for context window management.
                        files_summary.append(f"--- {rel} ---\n{content[:2000]}")
                    except OSError:
                        pass

        source_files: List[str] = []
        for ext in ("*.py", "*.rs", "*.cpp", "*.c", "*.h", "*.hpp", "*.toml"):
            for src in workspace.rglob(ext):
                rel = src.relative_to(workspace)
                if "/target/" in str(rel) or "/node_modules/" in str(rel):
                    continue
                source_files.append(str(rel))

        draft_yaml = ""
        if draft:
            draft_yaml = yaml.safe_dump(draft.model_dump(mode="json"), sort_keys=False)

        return {
            "project_name": project_name,
            "workspace_root": str(workspace),
            "manifests": "\n".join(manifests) or "none",
            "manifest_contents": "\n".join(files_summary) or "none",
            "source_files": "\n".join(source_files) or "none",
            "draft_blueprint": draft_yaml or "none",
            "spec": spec or "none",
        }

    def _parse_and_normalize(self, raw: str, workspace: Path) -> BlueprintV3:
        if not raw or not raw.strip():
            raise ValueError("LLM returned an empty blueprint response")

        # Extract JSON from markdown code fences if needed.
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text.rsplit("\n", 1)[0]
        text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Fallback: try YAML parsing.
            try:
                data = yaml.safe_load(text) or {}
            except yaml.YAMLError as exc:
                raise ValueError(f"Could not parse LLM blueprint response: {exc}") from exc

        if not isinstance(data, dict):
            raise ValueError("LLM blueprint response is not a JSON object")

        # Normalize metadata to finalized/transferable/synthesized.
        metadata = data.setdefault("metadata", {})
        metadata["schema_version"] = "3.0.0"
        metadata["status"] = "finalized"
        metadata["generation_method"] = "llm_synthesized"
        metadata["transferable"] = True

        # Normalize all paths to be workspace-relative and reject any absolute paths.
        data = self._normalize_paths(data, workspace)

        return BlueprintV3.model_validate(data)

    def _normalize_paths(self, data: Dict[str, Any], workspace: Path) -> Dict[str, Any]:
        def _norm(value: Any) -> Any:
            if isinstance(value, str):
                p = Path(value)
                if p.is_absolute():
                    try:
                        return str(p.relative_to(workspace))
                    except ValueError:
                        # Keep absolute-looking placeholders such as ${WORKSPACE_ROOT}/...
                        if value.startswith("${WORKSPACE_ROOT}"):
                            return value
                        # Otherwise force a relative basename (best-effort).
                        return value.lstrip("/")
                return value
            if isinstance(value, dict):
                return {k: _norm(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_norm(v) for v in value]
            return value

        return _norm(data)


def synthesize_v3_blueprint(
    workspace: Path,
    output_path: Path,
    provider: str = "deepseek",
    model: Optional[str] = None,
    draft_path: Optional[Path] = None,
) -> BlueprintV3:
    """Convenience function: synthesize a finalized v3 blueprint and write it to disk."""
    draft: Optional[BlueprintV3] = None
    if draft_path and draft_path.is_file():
        draft = BlueprintV3.load(draft_path)
    synthesizer = LLMBlueprintSynthesizer(provider=provider, model=model)
    return synthesizer.synthesize(workspace, draft=draft, output_path=output_path)


_DEFAULT_PROMPT_TEMPLATE = """\
You are an expert build-system architect. Inspect the project described below and produce a complete, valid Blueprint v3.0.0 (JSON) that can be used to deterministically build and run the project.

Project context:
- Project name: {project_name}
- Workspace root: {workspace_root}
- Detected manifests: {manifests}
- Manifest contents:
{manifest_contents}
- Source files:
{source_files}
- Draft blueprint (if any):
{draft_blueprint}
- Additional specification:
{spec}

Your response MUST be a single JSON object matching the BlueprintV3 Pydantic schema with these exact rules:
- metadata.schema_version = "3.0.0"
- metadata.status = "finalized"
- metadata.generation_method = "llm_synthesized"
- metadata.transferable = true
- metadata.project_name = "{project_name}"
- toolchains: list the concrete compilers, runtimes, and build tools required (e.g. CPython, Cargo/PyO3, GCC, Clang, CMake, Node/NAPI, Go C-Archive). Include version/channel if known.
- build_pipeline: a DAG of artifacts. Each artifact has id, type (one of: shared_library, static_library, binary, cargo_cdylib, python_extension, custom_cmd), source_files (workspace-relative paths), output_path (workspace-relative), compiler_flags, linker_flags, dependencies (artifact ids that must be built first), and commands for custom_cmd artifacts.
- abi_contracts: cross-language symbol contracts with binding_framework (ctypes, c_abi, cffi, pyo3, cxx), memory_model (caller_allocates, callee_allocates, shared_pyo3), inputs/outputs as name/type pairs.
- execution_strategy: primary_entrypoint (workspace-relative path), runtime (e.g. python3, cargo, node, ./binary), args, env map using ${WORKSPACE_ROOT} placeholders, working_dir using ${WORKSPACE_ROOT}, timeout in seconds.
- verification_nodes: list of integration tests with command, expected_exit_code, stdout_match_patterns, stderr_prohibited_patterns.
- All file paths must be relative to the workspace root. Do NOT include absolute paths. Use ${WORKSPACE_ROOT} for workspace-relative placeholders if needed.

Output only the JSON object, no prose.
"""
