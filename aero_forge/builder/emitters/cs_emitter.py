"""C# / .NET target emitter plugin for aero-forge engine specs."""

from __future__ import annotations

from typing import Any, Dict, List

from aero_forge.builder.emitters.base import (
    BoundaryContract,
    CapabilityDescriptor,
    CodeArtifact,
    EmitterRegistry,
    PolyglotEmitterPlugin,
)


class CsEmitterPlugin(PolyglotEmitterPlugin):
    """Emit C# source files with P/Invoke and NativeAOT ``[UnmanagedCallersOnly]`` stubs."""

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return CapabilityDescriptor(
            language_id="csharp",
            supported_boundaries={BoundaryContract.PINVOKE, BoundaryContract.C_ABI},
            toolchains=["dotnet", "csc"],
            file_extensions=[".cs"],
            supports_zero_copy=False,
            supports_async_ffi=False,
        )

    def emit_source_files(
        self,
        node_id: str,
        node_spec: Dict[str, Any],
        boundary_contracts: List[Dict[str, Any]],
    ) -> List[CodeArtifact]:
        namespace = _cs_namespace(node_id)
        contracts = boundary_contracts or node_spec.get("contracts", [])
        lines = [
            "using System;",
            "using System.Runtime.InteropServices;",
            "",
            f"namespace {namespace}",
            "{",
            "    /// <summary>P/Invoke declarations for C-ABI shared libraries.</summary>",
            "    public static partial class NativeLib",
            "    {",
        ]

        for contract in contracts:
            symbol = contract.get("export_symbol") or contract.get("name") or "aero_stub"
            sig = contract.get("signature", {})
            inputs = sig.get("inputs", [])
            outputs = sig.get("outputs", [])
            ret = outputs[0].get("type", "void") if outputs else "void"
            arg_strs = []
            for i, arg in enumerate(inputs):
                arg_name = arg.get("name") or f"arg{i}"
                arg_type = _cs_type(arg.get("type", "int"))
                arg_strs.append(f"{arg_type} {arg_name}")
            ret_type = _cs_type(ret)
            ret_attr = "" if ret_type == "void" else f"[return: MarshalAs(UnmanagedType.{_cs_marshal(ret)})]\n        "
            lines.extend([
                f'        [LibraryImport("{node_id or "aero_native"}", EntryPoint = "{symbol}")]',
                f"        {ret_attr}[UnmanagedCallersOnly]",
                f"        public static partial {ret_type} {symbol}({', '.join(arg_strs)});",
                "",
            ])

        if not contracts:
            lines.extend([
                '        [LibraryImport("aero_native", EntryPoint = "aero_stub")]',
                "        [UnmanagedCallersOnly]",
                "        public static partial void AeroStub();",
                "",
            ])

        lines.extend([
            "    }",
            "",
            "    /// <summary>NativeAOT callable exports for reverse P/Invoke scenarios.</summary>",
            "    public static class NativeExports",
            "    {",
        ])

        for contract in contracts:
            symbol = contract.get("export_symbol") or contract.get("name") or "aero_stub"
            sig = contract.get("signature", {})
            inputs = sig.get("inputs", [])
            outputs = sig.get("outputs", [])
            ret = outputs[0].get("type", "void") if outputs else "void"
            arg_strs = []
            for i, arg in enumerate(inputs):
                arg_name = arg.get("name") or f"arg{i}"
                arg_type = _cs_type(arg.get("type", "int"))
                arg_strs.append(f"{arg_type} {arg_name}")
            ret_type = _cs_type(ret)
            lines.extend([
                "        [UnmanagedCallersOnly]",
                f"        public static {ret_type} Export{symbol.title()}({', '.join(arg_strs)})",
                "        {",
            ])
            if ret_type != "void":
                lines.append(f"            return {_cs_zero(ret)};")
            lines.extend([
                "        }",
                "",
            ])

        if not contracts:
            lines.extend([
                "        [UnmanagedCallersOnly]",
                "        public static void ExportAeroStub() { }",
                "",
            ])

        lines.extend([
            "    }",
            "}",
            "",
        ])

        return [CodeArtifact(file_path=f"{namespace}.cs", content="\n".join(lines), language="csharp")]

    def emit_build_manifest(
        self,
        node_id: str,
        dependencies: List[str],
        compiler_flags: List[str],
    ) -> CodeArtifact:
        assembly = _cs_namespace(node_id)
        deps = "\n".join(
            f'    <PackageReference Include="{d}" Version="*" />' for d in dependencies
        )
        flags = " ".join(compiler_flags)
        content = (
            "<Project Sdk=\"Microsoft.NET.Sdk\">\n"
            "  <PropertyGroup>\n"
            f"    <AssemblyName>{assembly}</AssemblyName>\n"
            "    <TargetFramework>net8.0</TargetFramework>\n"
            "    <AllowUnsafeBlocks>true</AllowUnsafeBlocks>\n"
            "    <PublishAot>true</PublishAot>\n"
            f"    <OtherFlags>{flags}</OtherFlags>\n"
            "  </PropertyGroup>\n"
            "  <ItemGroup>\n"
            f"{deps}\n"
            "  </ItemGroup>\n"
            "</Project>\n"
        )
        return CodeArtifact(
            file_path=f"{assembly}.csproj",
            content=content,
            language="xml",
            metadata={"build_command": f"dotnet build {assembly}.csproj"},
        )


def _cs_namespace(node_id: str) -> str:
    """Return a C#-safe namespace/assembly name."""
    name = (node_id or "AeroForge").replace("-", "_").replace(" ", "_")
    if not name:
        name = "AeroForge"
    parts = [p for p in name.split("_") if p]
    return ".".join(p.capitalize() for p in parts) if parts else "AeroForge"


def _cs_type(type_hint: str) -> str:
    """Map an abstract type hint to a C# type."""
    if not type_hint:
        return "int"
    t = str(type_hint).lower().strip()
    if t in ("int", "i32"):
        return "int"
    if t in ("i64", "long", "longlong", "long long"):
        return "long"
    if t in ("float", "f32"):
        return "float"
    if t in ("double", "f64"):
        return "double"
    if t in ("bool",):
        return "bool"
    if t in ("str", "string", "const char*", "cstring"):
        return "string"
    if t in ("void", "none"):
        return "void"
    if t in ("usize", "size_t"):
        return "UIntPtr"
    return "int"


def _cs_marshal(type_hint: str) -> str:
    """Return a MarshalAs UnmanagedType for a C# return type."""
    t = str(type_hint).lower().strip()
    if t in ("str", "string", "const char*", "cstring"):
        return "LPStr"
    if t in ("bool",):
        return "I1"
    return "SysInt"


def _cs_zero(type_hint: str) -> str:
    """Return the C# zero value for a type."""
    t = str(type_hint).lower().strip()
    if t in ("str", "string", "const char*", "cstring"):
        return 'string.Empty'
    if t in ("bool",):
        return "false"
    if t in ("float", "f32"):
        return "0.0f"
    if t in ("double", "f64"):
        return "0.0"
    return "0"


EmitterRegistry.get_instance().register(CsEmitterPlugin())
