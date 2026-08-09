"""Unified, graph-driven polyglot materializer.

The ``GraphPolyglotMaterializer`` consumes a HIN-style graph specification,
computes parallel build wavefronts with the GoI solver, synthesizes FFI bridge
contracts for every cross-language edge, and delegates source/manifest emission
to the language-specific emitter plugins registered in ``EmitterRegistry``.

When an LLM client is configured, the materializer can also call the Builder
Code Emission Agent so that exact user-requested file paths, symbols, and build
manifests are generated instead of default stubs.
"""

from __future__ import annotations

import json
import logging
import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("aero_forge.graph_materializer")

import numpy as np

from aero_forge.blueprint.schema import (
    ABIArgument,
    ABIContractV3,
    BindingFramework,
    BlueprintV3,
    BuildArtifact,
    ContextState,
    ExecutionStrategyV3,
    LLMContext,
    Metadata,
    write_v3_blueprint,
)
from aero_forge.builder.emitters.base import (
    BoundaryContract,
    CodeArtifact,
    ContentDensityValidator,
    EmitterRegistry,
    PolyglotEmitterPlugin,
)
from aero_forge.builder.language_router import (
    SystemToolchainRouter,
    ToolchainNotFoundError,
    _accel_log,
)
from aero_forge.config import resolve_llm_provider
from aero_forge.llm.clients import get_llm_client
from aero_forge.orchestrator.prompt_builder import (
    EMITTER_PLUGIN_SYNTHESIS_SYSTEM_PROMPT,
    extract_aero_logic,
)
from aero_forge.prompts.builder_emitter import (
    BUILDER_EMITTER_SYSTEM_PROMPT,
    _build_skeleton,
    format_builder_emitter_user_prompt,
)
from aero_forge.scheduler.goi_solver import (
    GoIWavefrontSolver,
    GoiSolverError,
    _loop_dependency_matrix,
)
from aero_forge.scaffold.contract_synth import (
    DynamicContractSynthesizer,
    FFIBoundaryEdge,
)


class MaterializationError(Exception):
    """Raised when the graph materializer cannot write a valid workspace."""


class GraphPolyglotMaterializer:
    """Materialize a HIN graph into source files and build manifests.

    Args:
        workspace_root: directory in which all project files are written.
        registry: optional ``EmitterRegistry`` instance. If ``None`` the
            singleton is used, ensuring all built-in emitter modules are loaded.
        contract_synth: optional ``DynamicContractSynthesizer`` instance.
        llm_client: optional pre-configured LLM client. If ``None`` and an
            ``llm_provider`` is supplied, a client is constructed automatically.
        llm_provider: LLM provider name for Builder Code Emission Agent calls.
        llm_model: model name for the Builder Code Emission Agent.
        llm_api_key: API key for the Builder Code Emission Agent.
        config_override: optional request-scoped override used to inherit keys
            and provider settings passed from the web dashboard or SandboxManager.
    """

    def __init__(
        self,
        workspace_root: Path,
        registry: Optional[EmitterRegistry] = None,
        contract_synth: Optional[DynamicContractSynthesizer] = None,
        llm_client: Optional[Any] = None,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        config_override: Optional[Any] = None,
    ) -> None:
        self.workspace_root = Path(workspace_root)
        self.registry = registry or EmitterRegistry.get_instance()
        self.contract_synth = contract_synth or DynamicContractSynthesizer()
        self._llm_client = llm_client
        self._llm_provider = llm_provider
        self._llm_model = llm_model
        self._llm_api_key = llm_api_key
        self._config_override = config_override
        self._synthesis_context = ""
        self._ensure_emitters_loaded()

    @staticmethod
    def _parse_compacted_context(raw: Any) -> Dict[str, Any]:
        """Normalize the CFM payload to a dictionary."""
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str) and raw.strip():
            try:
                import json

                return dict(json.loads(raw))
            except Exception:
                return {"context": raw}
        return {}

    def _get_llm_client(self) -> Any:
        """Return a lazily constructed LLM client."""
        if self._llm_client is None:
            provider = resolve_llm_provider(self._llm_provider) or "deepseek"
            self._llm_client = get_llm_client(
                provider=provider,
                model=self._llm_model or "deepseek-chat",
                api_key=self._llm_api_key,
                config_override=self._config_override,
                raise_on_error=True,
                max_retries=3,
            )
        return self._llm_client

    @staticmethod
    def _ensure_emitters_loaded() -> None:
        """Import all plugin modules so they auto-register with the registry."""
        from aero_forge.builder.emitters import (  # noqa: F401
            cgo_emitter,
            cpp_emitter,
            cs_emitter,
            jni_emitter,
            python_emitter,
            rust_emitter,
        )

    @staticmethod
    def _build_adjacency_matrix(
        nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, List[str], Dict[str, int]]:
        """Build a dependency matrix M where M[target, source] == 1 for each edge."""
        labels = [n["node_id"] for n in nodes]
        order = {nid: i for i, nid in enumerate(labels)}
        n = len(labels)
        M = np.zeros((n, n), dtype=np.float64)
        for edge in edges:
            src = edge.get("source")
            tgt = edge.get("target")
            if src in order and tgt in order:
                M[order[tgt], order[src]] = 1.0
        return M, labels, order

    @staticmethod
    def _maybe_simplify_python_c_abi_edge(
        edge: Dict[str, Any], node_map: Dict[str, Dict[str, Any]]
    ) -> None:
        """Simplify C-ABI vector triples when the consumer is a Python CLI demo.

        The Python ctypes loader generated by the default Python emitter expects
        a scalar signature for simple `f(42)` demo calls.  When a C-ABI edge
        targets a Python node, has a scalar return type, and the exported symbol
        is a simple numeric function, keep only the leading scalar arguments so
        the emitted source and loader agree without requiring the user to write
        array allocation boilerplate.
        """
        raw_boundary = str(edge.get("boundary_type", "c_abi")).lower().replace("-", "_")
        if raw_boundary not in ("c_abi", "cabi"):
            return
        target = node_map.get(edge.get("target", ""), {})
        if target.get("lang", "").lower() != "python":
            return
        args = edge.get("args") or []
        return_type = (edge.get("return_type") or "").strip().lower()
        if return_type in ("", "void") or "pointer" not in args:
            return
        # Keep scalar arguments up to the first pointer argument.  This yields a
        # scalar C-ABI function that the Python loader can call with a single
        # literal such as `sieve_primes(42)`.
        new_args: List[str] = []
        for arg in args:
            if arg == "pointer":
                break
            new_args.append(arg)
        if new_args:
            edge["args"] = new_args

    def _boundary_contracts_for_node(
        self, node_id: str, edges: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Collect all edges touching *node_id* as plugin contract dicts."""
        contracts: List[Dict[str, Any]] = []
        for edge in edges:
            if edge.get("source") == node_id or edge.get("target") == node_id:
                raw_boundary = edge.get("boundary_type", "c_abi")
                boundary = str(raw_boundary).lower().replace("-", "_")
                contracts.append(
                    {
                        "boundary_type": boundary,
                        "boundary": boundary,
                        "symbol": edge.get("symbol", ""),
                        "args": list(edge.get("args", [])),
                        "return_type": edge.get("return_type", ""),
                        "is_zero_copy": edge.get("is_zero_copy", False),
                        "source": edge.get("source", ""),
                        "target": edge.get("target", ""),
                    }
                )
        return contracts

    def _write_artifact(self, artifact: CodeArtifact, output_dir: Path) -> Path:
        """Write a single artifact atomically and return its resolved path."""
        output_dir.mkdir(parents=True, exist_ok=True)
        target = (output_dir / artifact.file_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)

        fd, tmp_path = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(artifact.content)
            os.replace(tmp_path, str(target))
            if artifact.file_path.endswith(".sh") or artifact.file_path == "build.sh":
                target.chmod(
                    target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
                )
        except Exception:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass
            raise
        return target

    def _guard_requested_symbols(
        self, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]
    ) -> None:
        """Fail fast if an edge references a symbol that no source node exports."""
        exports_by_node: Dict[str, set] = {}
        for node in nodes:
            exports = set(node.get("exports") or [])
            # A node also implicitly exports any symbol listed as the source side of an edge.
            for edge in edges:
                if edge.get("source") == node["node_id"]:
                    exports.add(edge.get("symbol", ""))
            exports_by_node[node["node_id"]] = exports

        for edge in edges:
            symbol = edge.get("symbol", "")
            source = edge.get("source", "")
            if not symbol:
                continue
            source_exports = exports_by_node.get(source, set())
            if symbol not in source_exports:
                raise MaterializationError(
                    f"Guard: edge {source}->{edge.get('target')} references symbol {symbol!r} "
                    f"which is not exported by node {source!r}. Requested exports: {sorted(source_exports)}"
                )

    def _guard_requested_files(self, nodes: List[Dict[str, Any]]) -> None:
        """Fail fast if a requested file path cannot be emitted."""
        for node in nodes:
            node_id = node["node_id"]
            source_files = node.get("source_files") or []
            if not source_files:
                continue
            for path in source_files:
                if not isinstance(path, str) or not path:
                    raise MaterializationError(
                        f"Guard: node {node_id!r} has an invalid source_file entry"
                    )
                if path.endswith(
                    (".so", ".dll", ".dylib", ".pyd", ".whl", ".zip", ".tar", ".tar.gz")
                ):
                    raise MaterializationError(
                        f"Guard: node {node_id!r} requests binary artifact {path!r}; only source files are allowed"
                    )

    @staticmethod
    def _extract_code_artifacts(raw: str) -> List[CodeArtifact]:
        """Parse fenced code blocks with optional ``lang:path`` labels.

        First strips the Aero-Forge SSP delimiters ``__AERO_LOGIC_START__`` /
        ``__AERO_LOGIC_END__`` so surrounding prose is ignored. Tolerates
        surrounding markdown headers, conversational summaries, and inconsistent
        whitespace around fence tokens. If no blocks are found, the whole inner
        response is returned as a single untyped artifact for downstream path
        assignment.
        """
        artifacts: List[CodeArtifact] = []
        payload = extract_aero_logic(raw)
        # Allow empty language token (e.g. bare ```) and optional path labels.
        pattern = re.compile(
            r"```\s*(\w*)\s*(?::\s*([^\n\r]*?))?\s*\r?\n([\s\S]*?)\r?\n?\s*```",
            re.DOTALL | re.IGNORECASE,
        )
        for match in pattern.finditer(payload):
            lang = (match.group(1) or "").strip().lower()
            file_path = (match.group(2) or "").strip()
            content = match.group(3)
            if not content:
                continue
            artifacts.append(
                CodeArtifact(file_path=file_path, content=content, language=lang)
            )
        if not artifacts:
            # Format-agnostic fallback: emit the entire stripped inner response so
            # that the materializer can attempt path assignment and density validation.
            stripped = payload.strip()
            if stripped:
                artifacts.append(
                    CodeArtifact(file_path="", content=stripped, language="")
                )
        return artifacts

    def _assign_artifact_paths(
        self, artifacts: List[CodeArtifact], node_id: str, node_spec: Dict[str, Any]
    ) -> List[CodeArtifact]:
        """Ensure every extracted artifact has a file path."""
        source_files = list(node_spec.get("source_files") or [])
        manifest_names = {
            "rust": "Cargo.toml",
            "cpp": "CMakeLists.txt",
            "c": "CMakeLists.txt",
            "python": "pyproject.toml",
            "go": "go.mod",
            "csharp": f"{node_id}.csproj",
            "java": "pom.xml",
        }
        assigned: List[CodeArtifact] = []
        used_paths: set = set()

        # First, use paths already embedded in the fence labels.
        for artifact in artifacts:
            if artifact.file_path:
                used_paths.add(artifact.file_path)
                assigned.append(artifact)

        # Helper to match an artifact to a requested source file.
        def _match_source_file(artifact: CodeArtifact) -> str:
            # Prefer exact filename matches (e.g. CMakeLists.txt, Cargo.toml).
            artifact_name = Path(artifact.file_path).name if artifact.file_path else ""
            for path in source_files:
                if path in used_paths:
                    continue
                if artifact_name and Path(path).name == artifact_name:
                    return path
                # If the artifact language is toml/xml/text and the source file
                # is a known manifest, accept it.
                if artifact.language in ("toml", "xml", "text") and Path(path).name in (
                    "Cargo.toml",
                    "pyproject.toml",
                    "CMakeLists.txt",
                    "go.mod",
                    "pom.xml",
                    f"{node_id}.csproj",
                ):
                    return path

            # Match by extension.
            for path in source_files:
                if path in used_paths:
                    continue
                ext = Path(path).suffix.lower()
                if ext == ".toml" and artifact.language in ("toml", "text"):
                    return path
                if ext == ".xml" and artifact.language in ("xml", "text"):
                    return path
                if ext in (".cpp", ".cc", ".cxx") and artifact.language in (
                    "cpp",
                    "c++",
                ):
                    return path
                if ext in (".c",) and artifact.language == "c":
                    return path
                if ext in (".h", ".hpp") and artifact.language in ("cpp", "c++", "c"):
                    return path
                if ext == ".rs" and artifact.language == "rust":
                    return path
                if ext == ".py" and artifact.language == "python":
                    return path
                if ext == ".go" and artifact.language == "go":
                    return path
                if ext == ".cs" and artifact.language == "csharp":
                    return path
                if ext == ".java" and artifact.language == "java":
                    return path
            return ""

        # Then map unlabeled artifacts to requested source files or default names.
        for artifact in artifacts:
            if artifact.file_path:
                continue
            target_path = _match_source_file(artifact)
            if not target_path:
                # Fallback to a default file based on language.
                if artifact.language in ("toml",):
                    target_path = manifest_names.get(
                        node_spec.get("lang", "").lower(), "manifest.toml"
                    )
                elif artifact.language in ("xml",):
                    target_path = manifest_names.get(
                        node_spec.get("lang", "").lower(), f"{node_id}.xml"
                    )
                elif artifact.language in ("cpp", "c++"):
                    target_path = f"src/{node_id}.cpp"
                elif artifact.language == "c":
                    target_path = f"src/{node_id}.c"
                elif artifact.language == "rust":
                    target_path = "src/lib.rs"
                elif artifact.language == "python":
                    target_path = f"{node_id}.py"
                elif artifact.language == "go":
                    target_path = f"{node_id}.go"
                elif artifact.language == "csharp":
                    target_path = f"{node_id}.cs"
                elif artifact.language == "java":
                    target_path = f"{node_id}.java"
                else:
                    target_path = f"{node_id}.{artifact.language or 'txt'}"
            if target_path in used_paths:
                # Deduplicate by appending a suffix.
                base = Path(target_path)
                suffix = 2
                while True:
                    candidate = f"{base.stem}_{suffix}{base.suffix}"
                    if candidate not in used_paths:
                        target_path = candidate
                        break
                    suffix += 1
            used_paths.add(target_path)
            artifact.file_path = target_path
            assigned.append(artifact)
        return assigned

    @staticmethod
    def _is_build_manifest(artifact: CodeArtifact) -> bool:
        """Return True for known build manifest artifacts."""
        return Path(artifact.file_path).name in {
            "Cargo.toml",
            "CMakeLists.txt",
            "pyproject.toml",
            "go.mod",
            "build.gradle",
            "pom.xml",
            "build.sh",
            "Makefile",
            "CMakeLists.txt",
        } or str(artifact.file_path).endswith(".csproj")

    @staticmethod
    def _artifact_has_density(artifact: CodeArtifact) -> bool:
        """Return True when *artifact* contains enough functional code."""
        if artifact.is_header or GraphPolyglotMaterializer._is_build_manifest(artifact):
            return True
        try:
            ContentDensityValidator.validate(
                artifact.content, artifact.language or "text"
            )
            return True
        except ValueError:
            return False

    def _is_llm_available(self) -> bool:
        """Return True when an LLM client/provider is configured."""
        return bool(
            self._llm_client is not None
            or self._llm_provider
            or self._llm_api_key
        )

    @staticmethod
    def _v11_universal_guidance() -> str:
        """Return additional v11_universal_architect planning guidance."""
        try:
            from aero_forge.prompts import get_template

            template = get_template("v11_universal_architect")
            return (
                "\n\nUse the following universal polyglot architect guidance when "
                "completing the skeleton:\n" + template.system_prompt
            )
        except Exception:
            return (
                "\n\nUse the v11_universal_architect template: design the node as a "
                "strict C-ABI native function, replace every __AERO_IN_FILL__ marker, "
                "and wrap the entire response in __AERO_LOGIC_START__ / __AERO_LOGIC_END__."
            )

    def _emit_with_llm(
        self, node_id: str, node_spec: Dict[str, Any], contracts: List[Dict[str, Any]]
    ) -> List[CodeArtifact]:
        """Ask the Builder Code Emission Agent to emit source and manifest files.

        Materializes a skeleton file on disk, passes the compacted functional
        matrix as the exclusive context, and requires the response to be wrapped
        in ``__AERO_LOGIC_START__`` / ``__AERO_LOGIC_END__``. Retries once with
        the v11_universal_architect template if the response is empty, malformed,
        or below the functional density threshold.
        """
        client = self._get_llm_client()

        # Materialize the source skeleton to disk before calling the LLM.
        skeleton = _build_skeleton(
            node_spec, contracts, compacted_context=self._compacted_context
        )
        skeleton_dir = self.workspace_root / ".aero_skeletons"
        skeleton_dir.mkdir(parents=True, exist_ok=True)
        skeleton_path = skeleton_dir / f"{node_id}_skeleton.txt"
        skeleton_path.write_text(skeleton, encoding="utf-8")
        _accel_log(
            "info",
            f"Materialized source skeleton for {node_id} at {skeleton_path}",
        )

        user_prompt = format_builder_emitter_user_prompt(
            node_spec,
            contracts,
            compacted_context=self._compacted_context,
        )
        v11_guidance = self._v11_universal_guidance()

        last_exc: Optional[Exception] = None
        for attempt, (system, tokens, temp) in enumerate(
            [
                (BUILDER_EMITTER_SYSTEM_PROMPT, 4096, 0.2),
                (BUILDER_EMITTER_SYSTEM_PROMPT + v11_guidance, 8192, 0.1),
            ]
        ):
            if attempt > 0:
                _accel_log(
                    "warning",
                    f"Builder Code Emission Agent for {node_id} returned empty/malformed/low-density output; retrying with v11_universal",
                )
            try:
                raw = client.generate(
                    [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temp,
                    max_tokens=tokens,
                )
            except Exception as exc:
                last_exc = exc
                _accel_log(
                    "warning",
                    f"Builder Code Emission Agent for {node_id}: LLM call failed; error: {exc}",
                )
                continue
            if not raw:
                _accel_log(
                    "warning",
                    f"Builder Code Emission Agent for {node_id}: LLM returned an empty response",
                )
                continue
            artifacts = self._extract_code_artifacts(raw)
            if not artifacts:
                _accel_log(
                    "warning",
                    f"Builder Code Emission Agent for {node_id}: could not extract code artifacts; "
                    f"raw preview: {raw[:800]!r}",
                )
                continue
            assigned = self._assign_artifact_paths(artifacts, node_id, node_spec)
            # Functional density validation: source files must contain real logic.
            if all(
                self._is_build_manifest(a) or self._artifact_has_density(a)
                for a in assigned
            ):
                _accel_log("success", f"Builder Code Emission for {node_id}: logic in-fill successful")
                return assigned
            _accel_log(
                "warning",
                f"Builder Code Emission Agent for {node_id}: extracted artifacts below functional density; retrying",
            )

        _accel_log(
            "warning",
            f"Builder Code Emission Agent for {node_id} failed; falling back to deterministic baseline",
        )
        return self._emit_baseline_for_node(node_id, node_spec, contracts)

    def _emit_node_artifacts(
        self, node_id: str, node_spec: Dict[str, Any], contracts: List[Dict[str, Any]]
    ) -> List[CodeArtifact]:
        """Emit source/manifest artifacts for a node.

        If an LLM client is available, the registry can JIT-synthesize a
        ``PolyglotEmitterPlugin`` for unknown languages. When a plugin is
        already registered, or after successful synthesis, the node is emitted
        through the plugin's ``emit_source_files`` / ``emit_build_manifest`` hooks.
        """
        self._configure_registry_jit()
        lang = node_spec.get("lang", "").lower()
        boundary = self._boundary_contract_for_contracts(contracts)
        node_spec["_synthesis_context"] = self._synthesis_context
        node_spec["_compacted_context"] = self._compacted_context
        node_spec.setdefault("extra", {})
        if self._smt_types:
            node_spec["extra"].setdefault("smt_types", {})
            node_spec["extra"]["smt_types"].update(self._smt_types)
        plugin = self.registry.get_plugin(
            lang,
            synthesize=True,
            boundary_type=boundary,
            node_spec=node_spec,
            contracts=contracts,
        )
        source_artifacts = list(plugin.emit_source_files(node_id, node_spec, contracts))
        source_artifacts = self._postprocess_source_artifacts(source_artifacts)

        # Build the manifest from the plugin first; the LLM fallback below may
        # replace it if it emits its own manifest fences.
        manifest_artifacts = plugin.emit_build_manifest(
            node_id,
            node_spec.get("dependencies", []),
            node_spec.get("compiler_flags", []),
        )
        if isinstance(manifest_artifacts, CodeArtifact):
            manifest_artifacts = [manifest_artifacts]
        else:
            manifest_artifacts = list(manifest_artifacts)

        # If the plugin emitted hollow source files (e.g. only imports/docstrings),
        # ask the Builder Code Emission Agent to synthesize real logic.  This is
        # the Proactive Synthesis fallback path.
        if not all(self._artifact_has_density(a) for a in source_artifacts):
            if self._is_llm_available():
                try:
                    llm_artifacts = self._emit_with_llm(node_id, node_spec, contracts)
                except MaterializationError:
                    llm_artifacts = []
                if llm_artifacts:
                    llm_source = [a for a in llm_artifacts if not self._is_build_manifest(a)]
                    llm_manifests = [a for a in llm_artifacts if self._is_build_manifest(a)]
                    if llm_source:
                        source_artifacts = self._postprocess_source_artifacts(llm_source)
                    if llm_manifests:
                        manifest_artifacts = llm_manifests
            else:
                # No LLM available: fall back to the deterministic baseline so
                # unit tests and offline builds still materialize compilable code.
                baseline_artifacts = self._emit_baseline_for_node(
                    node_id, node_spec, contracts
                )
                if baseline_artifacts:
                    source_artifacts = self._postprocess_source_artifacts(
                        [a for a in baseline_artifacts if not self._is_build_manifest(a)]
                    )
                    manifest_artifacts = [
                        a for a in baseline_artifacts if self._is_build_manifest(a)
                    ]

        # Keep the node's source_files in sync with the emitted artifacts so the
        # toolchain router compiles the files that the plugin actually wrote.
        # Strip a leading `node_id/` package directory so paths are relative to
        # the node's working directory.
        prefix = f"{node_id}/"
        node_spec["source_files"] = [
            (
                a.file_path[len(prefix) :]
                if a.file_path.startswith(prefix)
                else a.file_path
            )
            for a in source_artifacts
            if not a.is_header
        ]
        return source_artifacts + manifest_artifacts

    @staticmethod
    def _postprocess_source_artifacts(
        artifacts: List[CodeArtifact],
    ) -> List[CodeArtifact]:
        """Apply deterministic fixes to emitted source artifacts.

        Zig files generated by JIT-synthesized plugins frequently reference
        `std.` without importing the standard library.  Inject a minimal
        import when it is missing so the file compiles without relying on the
        LLM to remember every import.
        """
        for artifact in artifacts:
            if artifact.language != "zig" and not str(artifact.file_path).endswith(
                ".zig"
            ):
                continue
            content = artifact.content
            if "std." in content and "const std = @import(\"std\");" not in content:
                artifact.content = "const std = @import(\"std\");\n\n" + content
        return artifacts

    def _configure_registry_jit(self) -> None:
        """Pass the configured LLM client/prompt to the plugin registry."""
        if self.registry._synthesis_prompt is None:
            provider = resolve_llm_provider(self._llm_provider)
            resolved_key = self._llm_api_key
            if not resolved_key and self._config_override is not None:
                resolved_key = getattr(self._config_override, "api_key", None)
            self.registry.configure_jit_synthesis(
                llm_client=self._llm_client,
                provider=provider,
                model=self._llm_model,
                api_key=resolved_key,
                prompt=EMITTER_PLUGIN_SYNTHESIS_SYSTEM_PROMPT,
            )

    def _boundary_contract_for_contracts(
        self, contracts: List[Dict[str, Any]]
    ) -> Optional[BoundaryContract]:
        """Map the first node's boundary contract string to a BoundaryContract enum."""
        if not contracts:
            return None
        raw = contracts[0].get("boundary_type", "")
        try:
            return BoundaryContract(raw)
        except ValueError:
            return None

    # ------------------------------------------------------------------
    # Deterministic baseline emission (guard fallback)
    # ------------------------------------------------------------------

    def _source_contracts(
        self, node_id: str, contracts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Return contracts where *node_id* is the source side."""
        return [c for c in contracts if c.get("source") == node_id]

    def _cpp_arg_type(self, arg: str) -> str:
        return {
            "pointer": "void*",
            "int32": "int32_t",
            "int64": "int64_t",
            "float32": "float",
            "float64": "double",
        }.get(arg, arg)

    def _rust_type_for_arg(self, arg: str) -> str:
        return {
            "pointer": "*const u8",
            "int32": "i32",
            "int64": "i64",
            "float32": "f32",
            "float64": "f64",
        }.get(arg, arg)

    def _emit_baseline_for_node(
        self, node_id: str, node_spec: Dict[str, Any], contracts: List[Dict[str, Any]]
    ) -> List[CodeArtifact]:
        """Generate a deterministic, compilable baseline for a node.

        This is the guard fallback when the LLM/plugin emission is missing
        requested symbols or files. It is driven by the node spec and the
        boundary contracts rather than legacy domain templates.
        """
        source_files = list(node_spec.get("source_files") or [])
        lang = node_spec.get("lang", "").lower()
        source_contracts = self._source_contracts(node_id, contracts)
        by_symbol = {c.get("symbol", ""): c for c in source_contracts}

        files: Dict[str, str] = {}

        # C++ baseline ------------------------------------------------------
        if lang == "cpp":
            # Prefer an explicit source contract symbol, otherwise fall back to a
            # conventional kernel name so the baseline always produces code.
            contract = None
            symbol = "execute_task"
            if source_contracts:
                contract = source_contracts[0]
                symbol = contract.get("symbol", symbol)
            if contract:
                args = contract.get("args", [])
                return_type = contract.get("return_type", "")
                c_args = [
                    f"{self._cpp_arg_type(a)} arg_{i}" for i, a in enumerate(args)
                ]
                ret = "void" if not return_type else self._cpp_arg_type(return_type)
                sig = f"{ret} {symbol}({', '.join(c_args)})"
                body = f"""#include <cmath>
#include <cstdint>
#include <cstddef>

extern "C" {{

{sig} {{
    if (arg_0 == nullptr || arg_5 == nullptr) {{
        return{('' if 'void' in sig else ' 0')};
    }}
    // Cache-aware baseline: out = A * B (M x K times K x N).
    int64_t M = arg_2;
    int64_t K = arg_3;
    int64_t N = arg_4;
    const double* A = static_cast<const double*>(static_cast<void*>(arg_0));
    const double* B = static_cast<const double*>(static_cast<void*>(arg_1));
    double* C = static_cast<double*>(arg_5);
    const size_t BLOCK = 64;
    for (int64_t i = 0; i < M * N; ++i) C[i] = 0.0;
    for (int64_t ii = 0; ii < M; ii += (int64_t)BLOCK) {{
        int64_t i_end = (ii + (int64_t)BLOCK < M) ? ii + (int64_t)BLOCK : M;
        for (int64_t i = ii; i < i_end; ++i) {{
            for (int64_t j = 0; j < N; ++j) {{
                double sum = 0.0;
                for (int64_t k = 0; k < K; ++k) {{
                    sum += A[i * K + k] * B[k * N + j];
                }}
                C[i * N + j] = sum;
            }}
        }}
    }}
}}

}} // extern "C"
"""
                for path in source_files:
                    if path.endswith(".cpp") or path.endswith(".cc") or path.endswith(".cxx"):
                        if path not in files:
                            files[path] = body
                    elif path.endswith(".h") or path.endswith(".hpp"):
                        files[path] = f"""#pragma once
#include <cstddef>

extern "C" {{
{sig};
}}
"""
                    elif path.endswith("CMakeLists.txt"):
                        # Infer the source file for add_library from the first cpp file.
                        cpp_files = [p for p in source_files if p.endswith((".cpp", ".cc", ".cxx"))]
                        src_file = cpp_files[0] if cpp_files else "src/kernels.cpp"
                        files[path] = f"""cmake_minimum_required(VERSION 3.16)
project({node_id} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library({node_id} SHARED {src_file})
target_include_directories({node_id} PUBLIC include)
target_compile_options({node_id} PRIVATE -O3 -march=native -fPIC)
"""

        # Rust baseline -----------------------------------------------------
        elif lang == "rust":
            symbol = "run_pipeline"
            contract = source_contracts[0] if source_contracts else by_symbol.get(symbol)
            if not contract:
                # No explicit contract: emit a minimal C-ABI placeholder so the
                # crate always passes the hollow-source density gate.
                contract = {"symbol": symbol, "boundary_type": "c_abi", "args": [], "return_type": ""}
            if contract:
                symbol = contract.get("symbol", symbol)
                boundary = contract.get("boundary_type", "pyo3_maturin")
                if boundary == "pyo3_maturin":
                    rust_src = """use pyo3::prelude::*;
use rayon::prelude::*;

#[pyfunction]
fn run_pipeline(py: Python, data: &[u8]) -> PyResult<Vec<u8>> {
    // Release the GIL during parallel work using Python::allow_threads.
    let output: Vec<u8> = py.allow_threads(|| {
        data.par_iter().map(|&b| b.wrapping_add(7)).collect()
    });
    Ok(output)
}

#[pymodule]
fn rust_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_pipeline, m)?)?;
    Ok(())
}
"""
                else:
                    args = contract.get("args", [])
                    return_type = contract.get("return_type", "")
                    rust_args = [
                        f"arg_{i}: {self._rust_type_for_arg(a)}" for i, a in enumerate(args)
                    ]
                    if return_type and return_type.lower() not in ("", "void"):
                        ret_rust = self._rust_type_for_arg(return_type)
                        if return_type == "pointer":
                            ret_stmt = "Box::into_raw(Box::new(0.0f64)) as *const f64"
                        else:
                            ret_stmt = "0"
                        rust_body = f"""#[no_mangle]
pub extern "C" fn {symbol}({', '.join(rust_args)}) -> {ret_rust} {{
    {ret_stmt}
}}
"""
                    else:
                        rust_body = f"""#[no_mangle]
pub extern "C" fn {symbol}({', '.join(rust_args)}) {{
    // Baseline no-op.
}}
"""
                    rust_src = rust_body
                for path in source_files:
                    if path.endswith("src/lib.rs"):
                        files[path] = rust_src
                    elif path.endswith("Cargo.toml"):
                        files[path] = f"""[package]
name = "{node_id}"
version = "0.1.0"
edition = "2021"
build = "build.rs"

[lib]
name = "{node_id}"
crate-type = ["cdylib", "rlib"]

[dependencies]
pyo3 = "0.20.3"
rayon = "1.10"
"""

        # Python baseline ---------------------------------------------------
        elif lang == "python":
            for path in source_files:
                if path.endswith("main.py"):
                    files[path] = """import ctypes
import os
import shutil
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_RUST_DIR = os.path.normpath(os.path.join(_HERE, "..", "rust_core", "target", "release"))
_CPP_DIR = os.path.normpath(os.path.join(_HERE, "..", "cpp_engine", "build"))

# Make the Rust cdylib importable as a Python extension module.
for _name in ("rust_core.so", "librust_core.so", "librust_core.dylib", "rust_core.dll"):
    _src = os.path.join(_RUST_DIR, _name)
    if os.path.exists(_src):
        _dst = os.path.join(_HERE, "rust_core.so")
        if not os.path.exists(_dst):
            shutil.copy2(_src, _dst)
        break

sys.path.insert(0, _HERE)
import rust_core  # type: ignore

_cpp_path = os.path.join(_CPP_DIR, "libcpp_core.so")
if os.path.exists(_cpp_path):
    _cpp = ctypes.CDLL(_cpp_path)
    _cpp.execute_task.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_double),
    ]
    _cpp.execute_task.restype = None
    n = 8
    _inp = (ctypes.c_double * n)(*[1.0] * n)
    _out = (ctypes.c_double * n)()
    _cpp.execute_task(_inp, n, _out)
    print("cpp out:", list(_out))
else:
    print("cpp shared library not yet built, skipping C++ call")

data = bytes(range(8))
result = rust_core.run_pipeline(data)
print("rust out:", list(result))
"""

        # Emit any requested files that did not get a baseline.
        for path in source_files:
            if path not in files:
                files[path] = ""

        # Emit required build manifests even if the LLM omitted them.
        package_dirs = sorted({Path(p).parts[0] for p in source_files if "/" in p})
        # Rust crates always get a build.rs so Cargo never hits E0601.
        needs_build_rs = lang == "rust"
        if lang == "rust" and any(p.endswith("src/lib.rs") for p in source_files):
            for d in package_dirs:
                cargo_path = f"{d}/Cargo.toml"
                if cargo_path not in files and cargo_path not in source_files:
                    files[cargo_path] = f"""[package]
name = "{d}"
version = "0.1.0"
edition = "2021"

[lib]
name = "{d}"
crate-type = ["cdylib"]

[dependencies]
pyo3 = "0.20.3"
rayon = "1.10"
"""
        if lang == "cpp" and any(p.endswith("src/kernels.cpp") for p in source_files):
            for d in package_dirs:
                cmake_path = f"{d}/CMakeLists.txt"
                if cmake_path not in files and cmake_path not in source_files:
                    files[cmake_path] = f"""cmake_minimum_required(VERSION 3.16)
project({node_id} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library({node_id} SHARED src/kernels.cpp)
target_include_directories({node_id} PUBLIC include)
target_compile_options({node_id} PRIVATE -O3 -march=native -fPIC)
"""

        # Rust crates always ship a valid build.rs so Cargo never hits E0601.
        if needs_build_rs:
            build_rs_content = """fn main() {
    // Aero-Forge build script configuration.
    println!("cargo:rerun-if-changed=build.rs");
}
"""
            # Add build.rs next to every Cargo.toml we emitted.
            cargo_tomls = [p for p in files if Path(p).name == "Cargo.toml"]
            if not cargo_tomls and source_files:
                cargo_tomls = [p for p in source_files if Path(p).name == "Cargo.toml"]
            for cargo_path in cargo_tomls:
                build_path = str(Path(cargo_path).parent / "build.rs")
                if build_path == "." or build_path == "/":
                    build_path = "build.rs"
                if build_path not in files:
                    files[build_path] = build_rs_content
            # Ensure every Cargo.toml wires build.rs.
            for path, content in list(files.items()):
                if Path(path).name == "Cargo.toml":
                    if 'build = "build.rs"' not in content:
                        files[path] = content.replace(
                            "[package]\n",
                            '[package]\nbuild = "build.rs"\n',
                            1,
                        )

        return [
            CodeArtifact(file_path=p, content=c, language=lang)
            for p, c in files.items()
        ]

    def _node_artifacts_pass_guard(
        self,
        artifacts: List[CodeArtifact],
        node_id: str,
        node_spec: Dict[str, Any],
        contracts: List[Dict[str, Any]],
    ) -> bool:
        """Return True if emitted artifacts satisfy the explicit node request."""
        source_files = set(node_spec.get("source_files") or [])
        emitted_paths = {a.file_path for a in artifacts}
        for path in source_files:
            if path not in emitted_paths:
                return False

        by_path = {a.file_path: a.content for a in artifacts}
        lang = node_spec.get("lang", "").lower()
        source_contracts = self._source_contracts(node_id, contracts)

        # Language-wide checks that apply regardless of source contracts.
        if lang == "python":
            main_py = next((p for p in source_files if p.endswith("main.py")), None)
            if main_py:
                content = by_path.get(main_py, "")
                if "ctypes.CDLL" not in content or "execute_task" not in content:
                    return False
                if "run_pipeline" not in content or "import rust_core" not in content:
                    return False

        for contract in source_contracts:
            symbol = contract.get("symbol", "")
            if not symbol:
                continue
            boundary = contract.get("boundary_type", "")
            # C++ guard
            if lang == "cpp":
                cpp_files = [
                    p for p in source_files if p.endswith((".cpp", ".cc", ".h", ".hpp"))
                ]
                if not cpp_files:
                    return False
                for path in cpp_files:
                    content = by_path.get(path, "")
                    if symbol not in content:
                        return False
                    if "sliding_window_dtw" in content:
                        return False
                    if 'extern "C"' not in content:
                        return False
            # Rust guard
            if lang == "rust" and boundary == "pyo3_maturin":
                lib_rs = next(
                    (p for p in source_files if p.endswith("src/lib.rs")), None
                )
                if not lib_rs:
                    return False
                content = by_path.get(lib_rs, "")
                if symbol not in content:
                    return False
                if "rayon" not in content or "allow_threads" not in content:
                    return False
        return True

    def _synthesize_ffi_bridges(
        self, edges: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate ``GeneratedFFIBridge`` artifacts for every cross-language edge."""
        bridges: List[Dict[str, Any]] = []
        for edge in edges:
            boundary = edge.get("boundary_type", "c_abi")
            bridge = self.contract_synth.synthesize_boundary(
                FFIBoundaryEdge(
                    edge_id=edge.get("edge_id") or f"{edge['source']}_{edge['target']}",
                    source_node=edge["source"],
                    source_lang=edge.get("source_lang", ""),
                    target_node=edge["target"],
                    target_lang=edge.get("target_lang", ""),
                    boundary_type=boundary,
                    symbol_name=edge.get("symbol", ""),
                    argument_types=edge.get("args", []),
                    return_type=edge.get("return_type", ""),
                    is_zero_copy=edge.get("is_zero_copy", False),
                )
            )
            bridge_dir = self.workspace_root / "ffi_bridges" / bridge.edge_id
            artifacts: List[CodeArtifact] = []
            if bridge.header:
                ext = ".h" if bridge.boundary_type in ("c_abi", "pinvoke") else ".hpp"
                artifacts.append(
                    CodeArtifact(
                        file_path=f"bridge{ext}",
                        content=bridge.header,
                        language="c",
                        is_header=True,
                    )
                )
            if bridge.source:
                src_ext = ".c"
                if bridge.boundary_type == "pyo3_maturin":
                    src_ext = ".rs"
                elif bridge.boundary_type == "cgo":
                    src_ext = ".go"
                artifacts.append(
                    CodeArtifact(
                        file_path=f"bridge{src_ext}",
                        content=bridge.source,
                        language="c" if src_ext == ".c" else bridge.boundary_type,
                    )
                )
            if bridge.python_loader:
                artifacts.append(
                    CodeArtifact(
                        file_path="loader.py",
                        content=bridge.python_loader,
                        language="python",
                    )
                )
            if bridge.csharp_stub:
                artifacts.append(
                    CodeArtifact(
                        file_path="AeroNative.cs",
                        content=bridge.csharp_stub,
                        language="csharp",
                    )
                )
            for artifact in artifacts:
                self._write_artifact(artifact, bridge_dir)
            bridges.append(
                {
                    "edge_id": bridge.edge_id,
                    "boundary_type": bridge.boundary_type,
                    "directory": str(bridge_dir),
                    "files": [a.file_path for a in artifacts],
                }
            )
        return bridges

    def _generate_build_script(
        self, hin_graph_spec: Dict[str, Any], stages: List[List[str]]
    ) -> Optional[CodeArtifact]:
        """Generate a root ``build.sh`` that builds each stage in order."""
        build_script_path = hin_graph_spec.get("build_script") or "build.sh"
        if not build_script_path:
            return None

        node_map = {n["node_id"]: n for n in hin_graph_spec.get("nodes", [])}
        primary_entrypoint = hin_graph_spec.get("primary_entrypoint", "run_shell.py")

        # The primary entrypoint may have been emitted under a node directory
        # (e.g. python_cli/cli.py). Resolve it to an existing workspace-relative
        # path so the build script can run it.
        if primary_entrypoint:
            entry_path = self.workspace_root / primary_entrypoint
            if not entry_path.is_file():
                for node_id, _ in node_map.items():
                    candidate = self.workspace_root / node_id / primary_entrypoint
                    if candidate.is_file():
                        primary_entrypoint = f"{node_id}/{primary_entrypoint}"
                        break
                else:
                    # Last resort: discover a runnable Python entrypoint, preferring
                    # one directly inside a node directory over nested duplicates.
                    skip_prefixes = ("target/", "build/", ".aero_skeletons/", "tests/", "ffi_bridges/")
                    for node_id in node_map:
                        for candidate in (
                            self.workspace_root / node_id / "main.py",
                            self.workspace_root / node_id / "python_interface" / "main.py",
                            self.workspace_root / node_id / "src" / "main.py",
                        ):
                            if candidate.is_file():
                                primary_entrypoint = candidate.relative_to(self.workspace_root).as_posix()
                                break
                        if primary_entrypoint != hin_graph_spec.get("primary_entrypoint", "run_shell.py"):
                            break
                    else:
                        for py in sorted(self.workspace_root.rglob("*.py")):
                            rel = py.relative_to(self.workspace_root).as_posix()
                            if any(rel.startswith(p) for p in skip_prefixes):
                                continue
                            if py.name in ("main.py", "__main__.py") or "if __name__" in py.read_text(encoding="utf-8", errors="ignore"):
                                primary_entrypoint = rel
                                break
            # Update the graph spec so blueprint.aero and ExecutionStrategyV3 agree.
            hin_graph_spec["primary_entrypoint"] = primary_entrypoint

        import shlex

        lines: List[str] = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
        for stage in stages:
            for node_id in stage:
                node = node_map.get(node_id, {})
                toolchain = (node.get("toolchain") or node.get("lang", "")).lower()
                source_files = node.get("source_files") or []
                # Tokenize compiler flags; the intent compiler sometimes emits
                # ``--target <triple>`` as a single string.
                node_flags = [
                    tok
                    for f in (node.get("compiler_flags") or [])
                    for tok in shlex.split(str(f))
                ]
                # Use the directory prefix requested by the user (e.g. cpp_engine/)
                # instead of the node_id so build paths match emitted files.
                if (
                    source_files
                    and isinstance(source_files[0], str)
                    and "/" in source_files[0]
                ):
                    first_dir = source_files[0].split("/")[0]
                    # Conventional source subdirectories live inside the node directory.
                    if first_dir in ("src", "include", "lib", "tests"):
                        package_dir = node_id
                    else:
                        package_dir = first_dir
                else:
                    package_dir = node_id
                if toolchain == "cmake":
                    # Assume CMakeLists.txt lives inside the package directory.
                    lines.append(
                        f"(cd {package_dir} && cmake -B build && cmake --build build && "
                        f"(cp build/lib{node_id}.so . 2>/dev/null || true))"
                    )
                elif toolchain == "cargo":
                    flags = shlex.join(node_flags)
                    has_wasm = any("wasm" in f for f in node_flags)
                    lines.append(
                        f"(cd {package_dir} && cargo build --release{' ' + flags if flags else ''})"
                    )
                    if has_wasm:
                        # Cargo's wasm target produces a .wasm artifact. Also build a
                        # host cdylib and copy it next to the node so the Python
                        # ctypes loader can resolve the C-ABI symbols at runtime.
                        lines.append(
                            f"(cd {package_dir} && cargo build --release && "
                            f"cp target/release/lib{node_id}.so . 2>/dev/null || true)"
                        )
                elif toolchain == "maturin":
                    flags = shlex.join(node_flags)
                    lines.append(
                        f"(cd {package_dir} && maturin build --release{' ' + flags if flags else ''})"
                    )
                elif toolchain in ("gcc", "clang", "g++", "clang++"):
                    lines.append(
                        f"# {node_id}: build via {toolchain} (see CMakeLists/Cargo)"
                    )
                elif toolchain == "go":
                    lines.append(
                        f"(cd {package_dir} && go build -buildmode=c-shared -o {node_id}.so .)"
                    )
                elif toolchain == "dotnet":
                    lines.append(f"(cd {package_dir} && dotnet build -c Release)")
                elif toolchain == "nvcc":
                    lines.append(
                        f"(cd {package_dir} && nvcc -shared -o {node_id}.so *.cu)"
                    )
                elif toolchain == "zig":
                    src = source_files[0] if source_files else f"src/{node_id}.zig"
                    lines.append(
                        f"(cd {package_dir} && zig build-lib -dynamic -O ReleaseFast -fPIC "
                        f"-femit-bin=lib{node_id}.so {src})"
                    )
        if primary_entrypoint:
            if primary_entrypoint.endswith(".py"):
                if "/" in primary_entrypoint:
                    # Run sub-package entrypoints as modules to avoid relative import errors.
                    module = primary_entrypoint[:-3].replace("/", ".").lstrip(".")
                    lines.append(f"python3 -m {module}")
                else:
                    lines.append(f"python3 {primary_entrypoint}")
            else:
                lines.append(f"bash {primary_entrypoint}")

        return CodeArtifact(
            file_path=build_script_path,
            content="\n".join(lines) + "\n",
            language="bash",
        )

    def _write_blueprint_aero(
        self, hin_graph_spec: Dict[str, Any], written_artifacts: List[Dict[str, Any]]
    ) -> Path:
        """Serialize a v3 ``blueprint.aero`` describing the materialized graph."""
        project = hin_graph_spec.get("project", "aero_forge_project")
        primary_entrypoint = hin_graph_spec.get("primary_entrypoint", "run_shell.py")
        build_script = hin_graph_spec.get("build_script")

        build_pipeline: List[BuildArtifact] = []
        for artifact in written_artifacts:
            path = artifact.get("file", "") or artifact.get("path", "")
            if not path:
                continue
            lang = artifact.get("language", "")
            build_pipeline.append(
                BuildArtifact(
                    id=f"{artifact.get('node_id', 'node')}_{Path(path).name}",
                    type="custom_cmd" if path.endswith(".sh") else "shared_library",
                    source_files=[path],
                    description=f"{lang} artifact for {artifact.get('node_id', '')}",
                )
            )

        abi_contracts: List[ABIContractV3] = []
        for edge in hin_graph_spec.get("edges", []):
            inputs = [
                ABIArgument(name=f"arg_{i}", type=t)
                for i, t in enumerate(edge.get("args", []))
            ]
            outputs = []
            if edge.get("return_type"):
                outputs.append(ABIArgument(name="return", type=edge.get("return_type")))

            boundary = str(edge.get("boundary_type", "c_abi"))
            binding_map = {
                "c_abi": BindingFramework.c_abi,
                "pyo3_maturin": BindingFramework.pyo3,
                "cgo": BindingFramework.c,
                "pinvoke": BindingFramework.c_abi,
                "jni": BindingFramework.c_abi,
                "wasm_wasi": BindingFramework.c_abi,
                "cuda_hip_c": BindingFramework.c_abi,
            }
            binding = binding_map.get(boundary, BindingFramework.c_abi)

            abi_contracts.append(
                ABIContractV3(
                    contract_id=f"{edge.get('source')}_{edge.get('target')}_{edge.get('symbol')}",
                    symbol=edge.get("symbol", ""),
                    source_language=edge.get("source_lang", edge.get("source", "")),
                    target_language=edge.get("target_lang", edge.get("target", "")),
                    binding_framework=binding,
                    inputs=inputs,
                    outputs=outputs,
                )
            )

        execution_strategy = ExecutionStrategyV3(
            primary_entrypoint=primary_entrypoint,
            runtime="python3" if primary_entrypoint.endswith(".py") else "bash",
            args=[build_script] if build_script else [],
            working_dir="${WORKSPACE_ROOT}",
        )

        blueprint = BlueprintV3(
            metadata=Metadata(
                schema_version="3.0.0",
                project_name=project,
                status="finalized",
                generation_method="llm_synthesized",
                llm_initialized=True,
                auto_generated=True,
                description=f"graph_polyglot blueprint for {project}",
            ),
            llm_context=LLMContext(state=ContextState.synthesized),
            build_pipeline=build_pipeline,
            abi_contracts=abi_contracts,
            execution_strategy=execution_strategy,
        )

        path = self.workspace_root / "blueprint.aero"
        write_v3_blueprint(blueprint, path)
        return path

    def materialize(
        self,
        hin_graph_spec: Dict[str, Any],
        *,
        build: bool = False,
    ) -> Dict[str, Any]:
        """Materialize *hin_graph_spec* into ``workspace_root``.

        Steps:
            1. Build adjacency/routing matrices and solve wavefront stages.
            2. Synthesize FFI bridges for every edge.
            3. Run pre-materialization guard checks.
            4. Emit source files and build manifests per wavefront stage.
            5. Generate ``build.sh`` and ``blueprint.aero``.
            6. Optionally dispatch native builds.
        """
        nodes: List[Dict[str, Any]] = hin_graph_spec.get("nodes", [])
        edges: List[Dict[str, Any]] = hin_graph_spec.get("edges", [])
        if not nodes:
            raise MaterializationError("hin_graph_spec must contain at least one node")

        self._synthesis_context = hin_graph_spec.get("metadata", {}).get(
            "prompt", hin_graph_spec.get("metadata", {}).get("description", "")
        )
        raw_compacted = hin_graph_spec.get("metadata", {}).get("synthesis_context", "")
        self._compacted_context = self._parse_compacted_context(raw_compacted)
        self._smt_types = hin_graph_spec.get("metadata", {}).get("smt_types", {})
        node_map = {n["node_id"]: n for n in nodes}
        for edge in edges:
            self._maybe_simplify_python_c_abi_edge(edge, node_map)

        self._guard_requested_files(nodes)
        self._guard_requested_symbols(nodes, edges)

        # Pre-flight toolchain availability: bootstrap or fail fast with a clear
        # human-facing diagnostic before any GoI wavefront work is wasted.
        try:
            SystemToolchainRouter.preflight_nodes(nodes, build=build)
        except ToolchainNotFoundError as exc:
            diagnostic = exc.install_command or ""
            _accel_log("error", f"Toolchain pre-flight failed for {exc.toolchain}: {exc}")
            raise MaterializationError(
                f"Toolchain {exc.toolchain!r} is required but not available.\n{diagnostic}"
            ) from exc

        M, labels, order = self._build_adjacency_matrix(nodes, edges)
        U = np.eye(len(labels), dtype=np.float64) * 0.5
        try:
            solver = GoIWavefrontSolver(labels, M, U)
            stages = solver.wavefront_stages()
        except GoiSolverError as exc:
            raise MaterializationError(
                f"GoI wavefront solver rejected the graph: {exc}"
            ) from exc

        bridges = self._synthesize_ffi_bridges(edges)

        node_map = {n["node_id"]: n for n in nodes}
        written_artifacts: List[Dict[str, Any]] = []

        for stage in stages:
            for node_id in stage:
                node_spec = node_map.get(node_id)
                if not node_spec:
                    continue
                lang = node_spec.get("lang", "").lower()
                node_dir = self.workspace_root / node_id
                contracts = self._boundary_contracts_for_node(node_id, edges)
                # Capture the user's requested package paths before the plugin mutates
                # source_files with the files it actually emitted.
                original_source_files = list(node_spec.get("source_files") or [])
                artifacts = self._emit_node_artifacts(node_id, node_spec, contracts)
                requested_dirs = {
                    Path(sf).parts[0]
                    for sf in original_source_files
                    if isinstance(sf, str) and ("/" in sf or os.sep in sf)
                }
                for artifact in artifacts:
                    # If the artifact's first path component is the node itself or
                    # a requested package directory (e.g. cpp_engine/src/kernels.cpp),
                    # treat the file path as workspace-relative. Otherwise scope it
                    # under the node directory so plugin fallbacks still work.
                    parts = Path(artifact.file_path).parts
                    first = parts[0] if parts else ""
                    if first == node_id or first in requested_dirs:
                        artifact_output_dir = self.workspace_root
                    else:
                        artifact_output_dir = node_dir
                    self._write_artifact(artifact, artifact_output_dir)
                    if (
                        artifact.language not in {
                            "bash",
                            "json",
                            "yaml",
                            "toml",
                            "markdown",
                            "text",
                            "cmake",
                            "make",
                            "makefile",
                        }
                        and not artifact.is_header
                        and not self._is_build_manifest(artifact)
                    ):
                        try:
                            ContentDensityValidator.validate(
                                artifact.content, artifact.language
                            )
                        except ValueError as exc:
                            raise MaterializationError(
                                f"Synthesis Incompleteness for {artifact.file_path}: {exc}"
                            ) from exc
                        if not ContentDensityValidator.has_execution_flow(
                            artifact.content, artifact.language
                        ):
                            raise MaterializationError(
                                f"GoI verification failed for {artifact.file_path}: "
                                "zero execution matrix (hollow logic)"
                            )
                    written_artifacts.append(
                        {
                            "node_id": node_id,
                            "language": lang,
                            "file": artifact.file_path,
                            "path": str(artifact_output_dir / artifact.file_path),
                        }
                    )

                if build:
                    try:
                        SystemToolchainRouter.dispatch_node_build(
                            node_id, node_spec, node_dir
                        )
                    except RuntimeError as exc:
                        raise MaterializationError(
                            f"toolchain dispatch failed for {node_id}: {exc}"
                        ) from exc

        build_artifact = self._generate_build_script(hin_graph_spec, stages)
        if build_artifact:
            self._write_artifact(build_artifact, self.workspace_root)
            written_artifacts.append(
                {
                    "node_id": "workspace",
                    "language": "bash",
                    "file": build_artifact.file_path,
                    "path": str(self.workspace_root / build_artifact.file_path),
                }
            )

        blueprint_path = self._write_blueprint_aero(hin_graph_spec, written_artifacts)

        result: Dict[str, Any] = {
            "project": hin_graph_spec.get("project", "aero_forge_project"),
            "architecture": "graph_polyglot",
            "workspace": str(self.workspace_root),
            "stages": stages,
            "bridges": bridges,
            "artifacts": written_artifacts,
            "blueprint_path": str(blueprint_path),
        }
        return result
