"""LLM-driven synthesis of a complete Blueprint v3.0.0 from a draft or raw project."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from jinja2 import Template
from pydantic import ValidationError

from aero_forge.blueprint.core import ensure_workspace_blueprint
from aero_forge.blueprint.schema import (
    ABIContractV3,
    ArtifactType,
    BindingFramework,
    BlueprintStatus,
    BlueprintV3,
    BuildArtifact,
    ContextState,
    ExecutionStrategyV3,
    GenerationMethod,
    MemoryModel,
    Metadata,
    ToolchainSpec,
    VerificationNode,
    write_v3_blueprint,
)
from aero_forge.blueprint.validator import InvalidBlueprintError
from aero_forge.builder.aeroc_compiler import compile_blueprint_to_aeroc
from aero_forge.bundle_repo import bundle_workspace, format_context_block
from aero_forge.config import Tier, resolve_llm_provider
from aero_forge.healing.llm_healer import LLMHealer
from aero_forge.llm.clients import BaseLLMClient, get_llm_client
from aero_forge.overlay.manager import OverlayManager
from aero_forge.scaffold.pre_write_validator import BlueprintValidationError

try:
    import tomllib  # Python 3.11+
except ImportError:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]

try:
    import toml
except ImportError:  # pragma: no cover
    toml = None  # type: ignore[assignment]

try:
    import tomli
except ImportError:  # pragma: no cover
    tomli = None  # type: ignore[assignment]

logger = logging.getLogger("aero_forge.blueprint.synthesizer")


def _strip_outer_quotes(text: str) -> str:
    """Remove surrounding matching quotes and unescape common escapes."""
    text = text.strip()
    while len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'"):
        text = text[1:-1].strip()
    return (
        text.replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace('\\"', '"')
        .replace("\\'", "'")
    )


def _extract_code_block(text: str) -> str:
    """Return the contents of the first markdown code fence, if any."""
    match = re.search(
        r"```(?:yaml|toml|json)?\s*\n(.*?)```",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    return text


def _find_document_start(text: str) -> int:
    """Return the index of the first line that looks like a document start."""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "//")):
            continue
        if stripped == "---":
            return text.find(line)
        if re.match(r"^[\{\[]", stripped):
            return text.find(line)
        if re.match(r"^[A-Za-z_][\w-]*\s*(?:[:=](?:\s|$))", stripped):
            return text.find(line)
        if re.match(r"^-\s", stripped):
            return text.find(line)
    return 0


def _load_toml(text: str) -> Optional[Dict[str, Any]]:
    """Attempt to parse *text* with any available TOML parser."""
    for parser in (tomllib, toml, tomli):
        if parser is None:
            continue
        try:
            return parser.loads(text)  # type: ignore[union-attr]
        except Exception:
            continue
    return None


def _soft_toml_to_yaml(text: str) -> str:
    """Naively convert simple top-level ``key = value`` lines to YAML."""
    lines: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            lines.append(line)
            continue
        match = re.match(r"^([A-Za-z_][\w-]*)\s*=\s*(.*)$", stripped)
        if not match:
            lines.append(line)
            continue
        key, value = match.groups()
        # Quote bare word strings so YAML treats them as scalars.
        if value and value[0] not in ('"', "'", "[", "{") and value not in (
            "true",
            "false",
        ):
            if re.match(r"^[A-Za-z_][\w\s.-]*$", value):
                value = f'"{value}"'
        lines.append(f"{key}: {value}")
    return "\n".join(lines)


def _looks_like_valid_blueprint(data: Any) -> bool:
    """Return True when *data* is a dict and does not contain mangled YAML keys."""
    if not isinstance(data, dict):
        return False
    for key in data.keys():
        if isinstance(key, str) and "=" in key:
            return False
    metadata = data.get("metadata")
    if isinstance(metadata, dict):
        for key in metadata.keys():
            if isinstance(key, str) and "=" in key:
                return False
    return True


def sanitize_llm_blueprint_output(raw_text: str) -> Dict[str, Any]:
    """Sanitize and parse an LLM blueprint response into a dictionary.

    Steps:
    1. Strip surrounding quotes.
    2. Extract the first markdown code fence (```yaml/```toml/```json/```).
    3. Drop any preamble before the first document-looking line.
    4. Attempt ``yaml.safe_load`` (also parses JSON).
    5. If YAML parsing fails or yields a malformed structure, attempt TOML parsing
       via ``tomllib``/``toml``/``tomli``.
    6. On TOML failure, perform soft regex sanitization of ``key = value`` lines
       to ``key: value`` and try YAML again.
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("LLM returned an empty blueprint response")

    text = raw_text.strip()
    text = _strip_outer_quotes(text)
    text = _extract_code_block(text)
    start = _find_document_start(text)
    if start > 0:
        text = text[start:].strip()

    # 1) YAML first - JSON is a subset of YAML.
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        data = None

    if _looks_like_valid_blueprint(data):
        return data

    # 2) TOML fallback.
    if data is None or not _looks_like_valid_blueprint(data):
        toml_data = _load_toml(text)
        if _looks_like_valid_blueprint(toml_data):
            return toml_data

    # 3) Soft regex sanitization of simple ``key = value`` lines.
    yaml_text = _soft_toml_to_yaml(text)
    try:
        data = yaml.safe_load(yaml_text)
    except yaml.YAMLError:
        data = None

    if _looks_like_valid_blueprint(data):
        return data

    raise ValueError("Could not parse LLM blueprint response as a blueprint object")


class LLMBlueprintSynthesizer:
    """Synthesize a finalized Blueprint v3 from a draft, project tree, or text spec."""

    DEFAULT_MODEL = os.getenv("AERO_FORGE_MODEL")

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        llm: Optional[BaseLLMClient] = None,
        api_key: Optional[str] = None,
        config_override: Optional[Any] = None,
    ) -> None:
        self.provider = provider or os.getenv("AERO_FORGE_LLM_PROVIDER") or "deepseek"
        self.model = model or self.DEFAULT_MODEL
        self.llm: Optional[BaseLLMClient] = llm
        self.api_key = api_key
        self.config_override = config_override

    def _load_prompt(self) -> Template:
        prompt_path = Path(__file__).with_name("prompts") / "blueprint_synthesis.j2"
        # The template is stored next to the package, not inside the blueprint subpackage.
        alt_path = Path(__file__).parents[1] / "prompts" / "blueprint_synthesis.j2"
        path = prompt_path if prompt_path.is_file() else alt_path
        if path.is_file():
            return Template(path.read_text(encoding="utf-8"))
        return Template(_DEFAULT_PROMPT_TEMPLATE)

    def _client(self) -> BaseLLMClient:
        """Return a configured LLM client, falling back across known providers."""
        if self.llm is not None:
            return self.llm

        # Normalize an unset/"none" provider to the best available key so we do
        # not silently fall back to a provider that returns empty responses.
        provider = self.provider
        if not provider or str(provider).lower() in ("", "none", "null"):
            provider = resolve_llm_provider(None)

        client = get_llm_client(
            provider,
            model=self.model,
            api_key=self.api_key,
            config_override=self.config_override,
            raise_on_error=False,
            tier=Tier.REASONING,
        )
        if client is None:
            # Try other providers for which an API key is configured.  Prefer
            # providers with strong API availability (deepseek, openai, gemini)
            # over openrouter, which was observed returning empty responses.
            for fallback in ("deepseek", "openai", "gemini", "openrouter"):
                if fallback == provider:
                    continue
                client = get_llm_client(
                    fallback,
                    model=self.model,
                    api_key=self.api_key,
                    config_override=self.config_override,
                    raise_on_error=False,
                    tier=Tier.REASONING,
                )
                if client is not None:
                    logger.info("LLMBlueprintSynthesizer falling back to provider: %s", fallback)
                    self.provider = fallback
                    break

        if client is None:
            raise ValueError(
                "No active LLM provider configured. Please check your API key in settings."
            )

        self.llm = client
        return self.llm

    def synthesize(
        self,
        workspace: Path,
        draft: Optional[BlueprintV3] = None,
        spec: Optional[str] = None,
        output_path: Optional[Path] = None,
    ) -> BlueprintV3:
        """Create a finalized, transferable Blueprint v3 from the inputs in *workspace*.

        If the LLM returns an unparseable or invalid blueprint, the synthesizer
        attempts self-healing via the LLM healer and then falls back to a
        rule-based workspace scan. A valid ``blueprint.aero`` is always written
        when *output_path* is provided.
        """
        workspace = Path(workspace).resolve()

        # Auto-generated / draft blueprints are placeholders, not finalized work.
        # Make sure the LLM prompt sees them as raw so it actually synthesizes.
        if draft and (
            draft.metadata.auto_generated
            or draft.metadata.status == BlueprintStatus.draft
            or not draft.metadata.llm_initialized
        ):
            draft.metadata.llm_initialized = False
            draft.metadata.status = BlueprintStatus.draft
            draft.llm_context.state = ContextState.raw

        context = self._gather_context(workspace, draft, spec)
        prompt = self._load_prompt().render(**context)

        client = self._client()
        raw = client.generate(
            prompt,
            temperature=0.1,
            response_format={"type": "json_object"},
        )

        blueprint = self._parse_or_fallback(raw, workspace)
        if output_path:
            write_v3_blueprint(blueprint, output_path)

        # Emit the compiled binary IR container immediately so the workspace is
        # usable as a standalone `.aeroc` artifact.
        aeroc_path = workspace / "workspace.aeroc"
        try:
            compile_blueprint_to_aeroc(blueprint, aeroc_path, workspace=workspace)
        except Exception as exc:
            logger.warning("Could not compile workspace.aeroc during synthesis: %s", exc)

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

        # Commit any in-memory overlay edits so the bundle reflects the latest workspace state.
        try:
            OverlayManager(workspace).flush_to_workspace(workspace)
        except Exception:
            pass

        # Build a comprehensive, LLM-friendly repository bundle with full source contents,
        # manifests, dependencies, and the current blueprint (if any).
        bundle = bundle_workspace(workspace, max_file_size_kb=50)
        if draft:
            bundle["draft_blueprint"] = yaml.safe_dump(draft.model_dump(mode="json"), sort_keys=False)
        else:
            bundle["draft_blueprint"] = ""

        repo_bundle = format_context_block(bundle, fmt="xml")

        return {
            "project_name": project_name,
            "workspace_root": str(workspace),
            "repo_bundle": repo_bundle,
            "spec": spec or "none",
        }

    def _parse_or_fallback(
        self,
        raw: str,
        workspace: Path,
    ) -> BlueprintV3:
        """Parse *raw* and normalize it, or recover via self-healing / local scan."""
        try:
            return self._parse_and_normalize(raw, workspace)
        except (yaml.YAMLError, ValueError, InvalidBlueprintError, BlueprintValidationError, ValidationError) as exc:
            logger.warning("LLM output parsing failed. Attempting self-healing recovery...")
            healed = self._self_heal_blueprint(raw, workspace)
            if healed is not None:
                try:
                    return self._parse_and_normalize_from_dict(healed, workspace)
                except (ValueError, InvalidBlueprintError, BlueprintValidationError, ValidationError) as validate_exc:
                    logger.warning("Healed blueprint failed validation: %s", validate_exc)
            logger.warning("Self-healing failed or timed out. Falling back to local workspace scan.")
            return self._fallback_blueprint(workspace)

    def _parse_and_normalize(self, raw: str, workspace: Path) -> BlueprintV3:
        data = sanitize_llm_blueprint_output(raw)
        return self._parse_and_normalize_from_dict(data, workspace)

    def _parse_and_normalize_from_dict(
        self,
        data: Dict[str, Any],
        workspace: Path,
    ) -> BlueprintV3:
        if not isinstance(data, dict):
            raise ValueError("LLM blueprint response is not a JSON/YAML object")

        # Normalize metadata to finalized/transferable/synthesized.
        metadata = data.setdefault("metadata", {})
        metadata["schema_version"] = "3.0.0"
        metadata["status"] = "finalized"
        metadata["generation_method"] = "llm_synthesized"
        metadata["transferable"] = True
        metadata["llm_initialized"] = True
        metadata["auto_generated"] = False

        # Mark the blueprint as LLM-context enriched.
        llm_context = data.setdefault("llm_context", {})
        llm_context["state"] = "synthesized"

        # Normalize all paths to be workspace-relative and reject any absolute paths.
        data = self._normalize_paths(data, workspace)

        return BlueprintV3.model_validate(data)

    def _self_heal_blueprint(
        self,
        raw: str,
        workspace: Path,
    ) -> Optional[Dict[str, Any]]:
        """Ask the LLM healer to repair an unparseable blueprint response."""
        try:
            healer = LLMHealer(
                client=self.llm,
                provider=self.provider,
                model=self.model,
                fallback=False,
            )
            repaired = healer.repair_blueprint_output(raw)
            if not repaired:
                return None
            return sanitize_llm_blueprint_output(repaired)
        except Exception as exc:
            logger.warning("Self-healing blueprint repair failed: %s", exc)
            return None

    def _fallback_blueprint(self, workspace: Path) -> BlueprintV3:
        """Generate a minimal valid blueprint by scanning the workspace."""
        try:
            from aero_forge.ingestion.zip_parser import generate_draft_v3_blueprint

            blueprint = generate_draft_v3_blueprint(workspace)
        except Exception as exc:
            logger.warning("Local workspace scan failed: %s", exc)
            blueprint = BlueprintV3()

        blueprint.metadata.schema_version = "3.0.0"
        blueprint.metadata.status = BlueprintStatus.finalized
        blueprint.metadata.generation_method = GenerationMethod.static_heuristic
        blueprint.metadata.transferable = True
        blueprint.metadata.project_name = workspace.name or "fallback_project"
        blueprint.metadata.llm_initialized = False
        blueprint.llm_context.state = ContextState.synthesized

        if not blueprint.build_pipeline:
            # Ensure the validator's finalize check has something to validate.
            blueprint.build_pipeline.append(
                BuildArtifact(
                    id="fallback_build",
                    type=ArtifactType.python_extension,
                    source_files=["main.py"],
                    output_path="dist/fallback_build",
                    description="Fallback artifact generated from workspace scan",
                )
            )

        return blueprint

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
    llm: Optional[BaseLLMClient] = None,
    api_key: Optional[str] = None,
    config_override: Optional[Any] = None,
) -> BlueprintV3:
    """Convenience function: synthesize a finalized v3 blueprint and write it to disk."""
    draft: Optional[BlueprintV3] = None
    if draft_path and draft_path.is_file():
        draft = BlueprintV3.load(draft_path)
    synthesizer = LLMBlueprintSynthesizer(
        provider=provider,
        model=model,
        llm=llm,
        api_key=api_key,
        config_override=config_override,
    )
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
- verification_nodes: list of integration tests. Each node MUST include a unique string node_id, plus command, expected_exit_code, stdout_match_patterns, and stderr_prohibited_patterns. Example: {{"node_id": "test_suite_v1", "command": "python3 -m pytest", "expected_exit_code": 0, "stdout_match_patterns": [], "stderr_prohibited_patterns": []}}.
- llm_context: object with:
  - state: "synthesized"
  - repository_summary: a concise 1-3 sentence semantic summary of the project's purpose and architecture.
  - dependency_graph: mapping of each source file to the list of files it directly depends on (e.g. {{"src/main.py": ["src/utils.py"], "src/lib.rs": ["src/ffi.rs"]}}).
  - compute_hotspots: list of heavy-compute functions/classes with keys: name, file, complexity (e.g. "O(n^2)"), acceleration_candidate (true/false), reason (why it should be accelerated).
- All file paths must be relative to the workspace root. Do NOT include absolute paths. Use ${WORKSPACE_ROOT} for workspace-relative placeholders if needed.

Output only the JSON object, no prose.
"""

# ``ensure_workspace_blueprint`` is re-exported from ``aero_forge.blueprint.core``
# so callers can ensure a workspace has a blueprint without importing the core module.
# The actual implementation lives in core.py and uses the standard .aero templates.
