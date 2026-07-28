"""Validation and transferability safety checks for Blueprint v3.0.0."""

from __future__ import annotations

import json
import re
from pathlib import Path, PurePath
from typing import Any, Dict, List, Optional

import yaml
from pydantic import ValidationError

from aero_forge.blueprint.schema import (
    ArtifactType,
    BlueprintStatus,
    BlueprintV3,
    BuildArtifact,
)


class InvalidBlueprintError(ValueError):
    """Raised when a v3 blueprint fails structural validation."""


class DraftBlueprintExportError(InvalidBlueprintError):
    """Raised when a draft or non-transferable blueprint is exported or cached for remote execution."""


class BlueprintV3Validator:
    """Validate a ``blueprint.aero`` against the v3.0.0 contract."""

    _ABSOLUTE_PATH_RE = re.compile(r"^(?:/|[A-Za-z]:\\|\\\\)")

    def __init__(self, blueprint_path_or_dict: Any, workspace: Optional[Path] = None):
        data: Dict[str, Any]
        if isinstance(blueprint_path_or_dict, dict):
            data = blueprint_path_or_dict
        else:
            path = Path(blueprint_path_or_dict)
            text = path.read_text(encoding="utf-8")
            suffix = path.suffix.lower()
            if suffix == ".json":
                data = json.loads(text)
            else:
                data = yaml.safe_load(text) or {}

        self.workspace = Path(workspace or ".").resolve()
        self._raw = data
        try:
            self.blueprint = BlueprintV3.model_validate(data)
        except ValidationError as exc:
            raise InvalidBlueprintError(f"Invalid v3 blueprint: {exc}") from exc

    @staticmethod
    def _is_placeholder(value: str) -> bool:
        return value.startswith("${") and value.endswith("}")

    def _is_valid_path(self, value: str, field_path: str) -> None:
        if not value or self._is_placeholder(value):
            return
        # Windows-style absolute paths are also rejected.
        if self._ABSOLUTE_PATH_RE.match(value):
            raise InvalidBlueprintError(
                f"Absolute path not allowed in {field_path}: {value!r}"
            )

    def _scan_string_value(self, value: Any, field_path: str) -> None:
        if isinstance(value, str):
            # Some strings are allowed to be empty or contain placeholder text.
            if not value or self._is_placeholder(value):
                return
            if self._ABSOLUTE_PATH_RE.match(value):
                raise InvalidBlueprintError(
                    f"Absolute path not allowed in {field_path}: {value!r}"
                )
        elif isinstance(value, dict):
            for k, v in value.items():
                self._scan_string_value(v, f"{field_path}.{k}")
        elif isinstance(value, list):
            for idx, item in enumerate(value):
                self._scan_string_value(item, f"{field_path}[{idx}]")

    def _validate_paths(self) -> None:
        self._is_valid_path(self.blueprint.metadata.project_name, "metadata.project_name")
        for idx, art in enumerate(self.blueprint.build_pipeline):
            prefix = f"build_pipeline[{idx}]"
            self._is_valid_path(art.id, f"{prefix}.id")
            self._is_valid_path(art.output_path, f"{prefix}.output_path")
            for s_idx, src in enumerate(art.source_files):
                self._is_valid_path(src, f"{prefix}.source_files[{s_idx}]")
            for c_idx, cmd in enumerate(art.commands):
                self._scan_string_value(cmd, f"{prefix}.commands[{c_idx}]")

        for idx, contract in enumerate(self.blueprint.abi_contracts):
            if contract.header_path:
                self._is_valid_path(contract.header_path, f"abi_contracts[{idx}].header_path")

        self._is_valid_path(self.blueprint.execution_strategy.primary_entrypoint, "execution_strategy.primary_entrypoint")
        self._is_valid_path(self.blueprint.execution_strategy.working_dir, "execution_strategy.working_dir")
        for idx, node in enumerate(self.blueprint.verification_nodes):
            self._is_valid_path(node.command, f"verification_nodes[{idx}].command")

    def _validate_finalized(self) -> None:
        if self.blueprint.metadata.status == BlueprintStatus.finalized and self.blueprint.metadata.transferable:
            if not self.blueprint.build_pipeline:
                raise InvalidBlueprintError(
                    "Finalized transferable blueprint requires a non-empty build_pipeline"
                )
            for idx, art in enumerate(self.blueprint.build_pipeline):
                if art.type == ArtifactType.custom_cmd and not art.commands:
                    raise InvalidBlueprintError(
                        f"build_pipeline[{idx}] custom_cmd artifact requires a 'commands' list"
                    )

    def validate(self, *, allow_draft: bool = True) -> "BlueprintV3":
        """Run all v3 validation checks and return the parsed blueprint.

        If ``allow_draft`` is False, the blueprint must be finalized and transferable.
        """
        if self.blueprint.metadata.schema_version != "3.0.0":
            raise InvalidBlueprintError(
                f"Unsupported schema version {self.blueprint.metadata.schema_version!r}; expected '3.0.0'"
            )
        self._validate_paths()
        self._validate_finalized()

        if not allow_draft:
            if self.blueprint.metadata.status == BlueprintStatus.draft:
                raise DraftBlueprintExportError(
                    "Draft blueprint cannot be exported or remotely executed"
                )
            if not self.blueprint.metadata.transferable:
                raise DraftBlueprintExportError(
                    "Non-transferable blueprint cannot be exported or remotely executed"
                )
        return self.blueprint

    def check_exportable(self) -> "BlueprintV3":
        """Shorthand for ``validate(allow_draft=False)`` used by exporters."""
        return self.validate(allow_draft=False)

    def validate_for_cache(self) -> "BlueprintV3":
        """Draft/non-transferable blueprints may not be cached deterministically."""
        bp = self.validate()
        if bp.metadata.status == BlueprintStatus.draft or not bp.metadata.transferable:
            raise DraftBlueprintExportError(
                "Draft or non-transferable blueprint cannot be deterministically cached"
            )
        return bp
