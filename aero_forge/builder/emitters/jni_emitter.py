"""Java/JNI target emitter plugin for aero-forge engine specs."""

from __future__ import annotations

from typing import Any, Dict, List

from aero_forge.builder.emitters.base import (
    BoundaryContract,
    CapabilityDescriptor,
    CodeArtifact,
    EmitterRegistry,
    PolyglotEmitterPlugin,
)


class JniEmitterPlugin(PolyglotEmitterPlugin):
    """Emit JNI dynamic native stubs and Java bridge signatures."""

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return CapabilityDescriptor(
            language_id="java",
            supported_boundaries={BoundaryContract.JNI},
            toolchains=["javac", "gcc", "clang++"],
            file_extensions=[".java", ".c", ".cpp", ".h"],
            supports_zero_copy=False,
            supports_async_ffi=False,
        )

    def emit_source_files(
        self,
        node_id: str,
        node_spec: Dict[str, Any],
        boundary_contracts: List[Dict[str, Any]],
    ) -> List[CodeArtifact]:
        java_class = _jni_class_name(node_id)
        package = _jni_package(node_id)
        contracts = boundary_contracts or node_spec.get("contracts", [])

        java_lines = [
            f"package {package};",
            "",
            f"public class {java_class} {{",
            "    static {",
            '        System.loadLibrary("aero_forge_native");',
            "    }",
            "",
        ]

        native_lines = [
            '#include <jni.h>',
            '#include <stdint.h>',
            '#include <stdlib.h>',
            f'// JNI native stubs for {java_class}',
            "",
        ]

        for contract in contracts:
            symbol = contract.get("export_symbol") or contract.get("name") or "aeroStub"
            sig = contract.get("signature", {})
            inputs = sig.get("inputs", [])
            outputs = sig.get("outputs", [])
            ret = outputs[0].get("type", "void") if outputs else "void"
            arg_types = [_jni_java_type(arg.get("type", "int")) for arg in inputs]
            ret_type = _jni_java_type(ret)
            java_sig = f"({''.join(arg_types)}){ret_type}"
            java_lines.append(
                f"    public native {ret_type} {symbol}({', '.join(arg_types)});"
            )

            native_sig = _jni_native_sig(
                package,
                java_class,
                symbol,
                java_sig,
                [_jni_c_type(arg.get("type", "int")) for arg in inputs],
                _jni_c_type(ret),
                ret,
            )
            native_lines.extend([native_sig, ""])

        if not contracts:
            java_lines.append("    public native void aeroStub();")
            native_lines.extend([
                _jni_native_sig(package, java_class, "aeroStub", "()V", [], "void", "void"),
                "",
            ])

        java_lines.extend(["}", ""])
        return [
            CodeArtifact(
                file_path=f"{java_class}.java",
                content="\n".join(java_lines),
                language="java",
            ),
            CodeArtifact(
                file_path=f"{java_class}_jni.cpp",
                content="\n".join(native_lines),
                language="cpp",
            ),
        ]

    def emit_build_manifest(
        self,
        node_id: str,
        dependencies: List[str],
        compiler_flags: List[str],
    ) -> CodeArtifact:
        java_class = _jni_class_name(node_id)
        content = (
            "plugins {\n"
            "    id 'java'\n"
            "    id 'cpp-application'\n"
            "}\n\n"
            "repositories { mavenCentral() }\n\n"
            f"group = '{_jni_package(node_id)}'\n"
            "version = '0.1.0'\n\n"
            "dependencies {\n"
        )
        for dep in dependencies:
            content += f"    implementation '{dep}'\n"
        content += "}\n"
        return CodeArtifact(
            file_path="build.gradle",
            content=content,
            language="gradle",
            metadata={"build_command": f"javac {java_class}.java"},
        )


def _jni_class_name(node_id: str) -> str:
    """Return a Java-safe class name."""
    name = (node_id or "AeroForgeNative").replace("-", "_").replace(" ", "_")
    parts = [p for p in name.split("_") if p]
    return "".join(p.capitalize() for p in parts) if parts else "AeroForgeNative"


def _jni_package(node_id: str) -> str:
    """Return a Java-safe package name."""
    base = (node_id or "aero.forge").lower().replace("-", ".").replace(" ", ".").replace("_", ".")
    return ".".join(p for p in base.split(".") if p) or "aero.forge"


def _jni_java_type(type_hint: str) -> str:
    """Map an abstract type to a Java JNI signature character or Java type."""
    if not type_hint:
        return "I"
    t = str(type_hint).lower().strip()
    if t in ("int", "i32"):
        return "int"
    if t in ("i64", "long"):
        return "long"
    if t in ("float", "f32"):
        return "float"
    if t in ("double", "f64"):
        return "double"
    if t in ("bool",):
        return "boolean"
    if t in ("str", "string", "const char*", "cstring"):
        return "String"
    if t in ("void", "none"):
        return "void"
    return "int"


def _jni_c_type(type_hint: str) -> str:
    """Map an abstract type to a JNI C type."""
    if not type_hint:
        return "jint"
    t = str(type_hint).lower().strip()
    if t in ("int", "i32"):
        return "jint"
    if t in ("i64", "long"):
        return "jlong"
    if t in ("float", "f32"):
        return "jfloat"
    if t in ("double", "f64"):
        return "jdouble"
    if t in ("bool",):
        return "jboolean"
    if t in ("str", "string", "const char*", "cstring"):
        return "jstring"
    if t in ("void", "none"):
        return "void"
    return "jint"


def _jni_native_sig(
    package: str,
    java_class: str,
    symbol: str,
    java_sig: str,
    arg_types: List[str],
    ret_type: str,
    ret: str,
) -> str:
    """Return a JNI function stub implementation."""
    mangled = f"Java_{package.replace('.', '_')}_{java_class}_{symbol}"
    arg_names = [f"arg{i}" for i in range(len(arg_types))]
    params = ["JNIEnv* env", "jclass cls"]
    for i, at in enumerate(arg_types):
        params.append(f"{at} {arg_names[i]}")
    ret_clause = "return 0;" if ret_type != "void" else ""
    zero = _jni_zero(ret)
    if ret_clause:
        ret_clause = f"    return {zero};"
    return (
        f'JNIEXPORT {ret_type} JNICALL {mangled}\n'
        f'({", ".join(params)})\n'
        "{\n"
        f"    // JNI stub for {package}.{java_class}.{symbol}{java_sig}\n"
        f"    {ret_clause}\n"
        "}"
    )


def _jni_zero(type_hint: str) -> str:
    """Return the JNI zero value for a type."""
    t = str(type_hint).lower().strip()
    if t in ("str", "string", "const char*", "cstring"):
        return 'NULL'
    if t in ("bool",):
        return "JNI_FALSE"
    if t in ("float", "f32"):
        return "0.0f"
    if t in ("double", "f64"):
        return "0.0"
    return "0"


EmitterRegistry.get_instance().register(JniEmitterPlugin())
