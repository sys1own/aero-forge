"""Go/CGO target emitter plugin for aero-forge engine specs."""

from __future__ import annotations

from typing import Any, Dict, List

from aero_forge.builder.emitters.base import (
    BoundaryContract,
    CapabilityDescriptor,
    CodeArtifact,
    EmitterRegistry,
    PolyglotEmitterPlugin,
)


class CgoEmitterPlugin(PolyglotEmitterPlugin):
    """Emit Go source files with CGO ``//export`` C-callable stubs."""

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return CapabilityDescriptor(
            language_id="go",
            supported_boundaries={BoundaryContract.CGO, BoundaryContract.C_ABI},
            toolchains=["go", "cgo"],
            file_extensions=[".go"],
            supports_zero_copy=False,
            supports_async_ffi=False,
        )

    def emit_source_files(
        self,
        node_id: str,
        node_spec: Dict[str, Any],
        boundary_contracts: List[Dict[str, Any]],
    ) -> List[CodeArtifact]:
        package = _go_package_name(node_id)
        contracts = boundary_contracts or node_spec.get("contracts", [])
        cgo_flags = "\n".join(f"// #cgo CFLAGS: {f}" for f in node_spec.get("compiler_flags", []))
        lines = [
            "//go:build cgo",
            "",
            f"package {package}",
            "",
            '/*',
        ]
        if cgo_flags:
            lines.append(cgo_flags)
        lines.extend([
            '#include <stdint.h>',
            '#include <stdlib.h>',
            '*/',
            'import "C"',
            "",
            "import (",
            '    "fmt"',
            ")",
            "",
        ])

        for contract in contracts:
            symbol = contract.get("export_symbol") or contract.get("name") or "aero_stub"
            sig = contract.get("signature", {})
            inputs = sig.get("inputs", [])
            outputs = sig.get("outputs", [])
            ret_type = outputs[0].get("type", "void") if outputs else "void"
            arg_strs = []
            for i, arg in enumerate(inputs):
                arg_name = arg.get("name") or f"arg{i}"
                arg_type = _go_c_type(arg.get("type", "int"))
                arg_strs.append(f"{arg_name} {arg_type}")
            ret = "" if ret_type == "void" else f" {_go_c_type(ret_type)}"
            zero = _go_zero(ret_type)
            body = (
                f'    fmt.Println("CGO stub called:", "{symbol}")\n'
                f"    return {zero}"
            ) if ret != "" else (
                f'    fmt.Println("CGO stub called:", "{symbol}")'
            )
            lines.extend([
                f"//export {symbol}",
                f"func {symbol}({', '.join(arg_strs)}){ret} {{",
                body,
                "}",
                "",
            ])

        if not contracts:
            lines.extend([
                "//export aero_stub",
                "func aero_stub() {",
                '    fmt.Println("CGO stub")',
                "}",
                "",
            ])

        lines.extend([
            "// main is required for building a c-shared library.",
            "func main() {}",
            "",
        ])

        return [CodeArtifact(file_path=f"{package}.go", content="\n".join(lines), language="go")]

    def emit_build_manifest(
        self,
        node_id: str,
        dependencies: List[str],
        compiler_flags: List[str],
    ) -> CodeArtifact:
        module = _go_package_name(node_id)
        require = "\n".join(f"require {d} v0.0.0" for d in dependencies)
        flags = " ".join(compiler_flags)
        content = (
            f"module {module}\n\n"
            f"go 1.21\n\n"
            f"{require}\n"
        )
        return CodeArtifact(
            file_path="go.mod",
            content=content,
            language="go.mod",
            metadata={"build_command": f"go build -buildmode=c-shared -o {module}.so {module}.go", "cgo_cflags": flags},
        )


def _go_package_name(node_id: str) -> str:
    """Return a Go-safe package name."""
    name = (node_id or "aero_forge_go").replace("-", "_").replace(" ", "_")
    if not name:
        name = "aero_forge_go"
    return name


def _go_c_type(type_hint: str) -> str:
    """Map an abstract type hint to a CGO C type."""
    if not type_hint:
        return "C.int"
    t = str(type_hint).lower().strip()
    if t in ("int", "i32"):
        return "C.int"
    if t in ("i64", "long", "longlong", "long long"):
        return "C.longlong"
    if t in ("float", "f32"):
        return "C.float"
    if t in ("double", "f64"):
        return "C.double"
    if t in ("str", "string", "const char*", "cstring"):
        return "*C.char"
    if t in ("bool",):
        return "C.int"
    if t in ("usize", "size_t"):
        return "C.size_t"
    if t in ("void", "none"):
        return ""
    return "C.int"


def _go_zero(type_hint: str) -> str:
    """Return the CGO zero value for a C type."""
    t = str(type_hint).lower().strip()
    if t in ("str", "string", "const char*", "cstring"):
        return 'nil'
    if t in ("float", "f32"):
        return "C.float(0)"
    if t in ("double", "f64"):
        return "C.double(0)"
    if t in ("i64", "long", "longlong", "long long"):
        return "C.longlong(0)"
    if t in ("usize", "size_t"):
        return "C.size_t(0)"
    return "C.int(0)"


EmitterRegistry.get_instance().register(CgoEmitterPlugin())
