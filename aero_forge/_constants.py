"""Shared constants for the transpiler."""

from __future__ import annotations

IO_MODULES = {
    "requests",
    "urllib",
    "socket",
    "os",
    "subprocess",
    "pathlib",
    "io",
    "ftplib",
    "http",
    "sys",
}

IO_NAMES = {
    "open",
    "print",
    "input",
    "exec",
    "eval",
    "compile",
    "exit",
    "quit",
    "__import__",
}

# Standard library modules and builtins that are safe to ignore or stub during
# transpilation (e.g., logging, timing, and math utilities).
SAFE_STD_MODULES = {"io", "sys", "time", "math"}
SAFE_BUILTINS = {"print"}

MATH_ATTRS = {
    "sqrt",
    "sin",
    "cos",
    "tan",
    "exp",
    "log",
    "log10",
    "ceil",
    "floor",
    "trunc",
    "pow",
    "radians",
    "degrees",
}

MATH_CONSTANTS = {
    "pi": 3.141592653589793,
    "e": 2.718281828459045,
    "tau": 6.283185307179586,
}
