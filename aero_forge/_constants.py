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
}

MATH_CONSTANTS = {
    "pi": 3.141592653589793,
    "e": 2.718281828459045,
    "tau": 6.283185307179586,
}
