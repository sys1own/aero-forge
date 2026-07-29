"""Aero-Forge blueprint package.

This package exposes the legacy v2 Blueprint model and helpers alongside the
new v3.0.0 schema, validator, synthesizer, and conversion utilities.
"""

from aero_forge.blueprint.blueprint_parser import (
    is_blueprint_ready,
    load_blueprint,
)
from aero_forge.blueprint.core import (
    ABIContract,
    Blueprint,
    BlueprintSchemaV2,
    BlueprintValidator,
    CLIContract,
    CLIContractFlag,
    ContractEntry,
    ExecutionStrategy,
    FunctionSpec,
    LLMConfig,
    ManifestEntry,
    _contracts_to_abi_contracts,
    discover_functions,
    discover_project,
    generate_blueprint,
    generate_blueprint_from_uploaded_repo,
    parse_aero,
    parse_blueprint,
    write_blueprint,
)
from aero_forge.blueprint.schema import BlueprintV3, write_v3_blueprint
from aero_forge.blueprint.synthesizer import LLMBlueprintSynthesizer
from aero_forge.blueprint.validator import (
    BlueprintV3Validator,
    DraftBlueprintExportError,
    InvalidBlueprintError,
)

__all__ = [
    "ABIContract",
    "Blueprint",
    "BlueprintSchemaV2",
    "BlueprintValidator",
    "BlueprintV3",
    "BlueprintV3Validator",
    "CLIContract",
    "CLIContractFlag",
    "ContractEntry",
    "DraftBlueprintExportError",
    "ExecutionStrategy",
    "FunctionSpec",
    "InvalidBlueprintError",
    "is_blueprint_ready",
    "LLMBlueprintSynthesizer",
    "LLMConfig",
    "load_blueprint",
    "ManifestEntry",
    "_contracts_to_abi_contracts",
    "discover_functions",
    "discover_project",
    "generate_blueprint",
    "generate_blueprint_from_uploaded_repo",
    "parse_aero",
    "parse_blueprint",
    "write_blueprint",
    "write_v3_blueprint",
]
