"""Semantic shields and auto-correction for Rust sources using ``rug`` / ``pyo3``."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

RUST_ANCHORS = ("rug", "pyo3")

_TRAIT_SENTINEL = "AeroNegMutExt"

EXTENSION_TRAITS = """\
// --- Aero compatibility shims (auto-injected for rug/pyo3) ---
trait AeroNegMutExt { fn neg_mut(&mut self); }
impl AeroNegMutExt for rug::Float {
    #[inline] fn neg_mut(&mut self) { let c = -self.clone(); <rug::Float as rug::Assign>::assign(self, c); }
}
impl AeroNegMutExt for rug::Complex {
    #[inline] fn neg_mut(&mut self) { let c = -self.clone(); <rug::Complex as rug::Assign>::assign(self, c); }
}
trait AeroNthRootExt { fn nth_root(&self, n: u32) -> rug::Float; }
impl AeroNthRootExt for rug::Float {
    #[inline] fn nth_root(&self, n: u32) -> rug::Float { rug::Float::with_val(self.prec(), self.clone().root(n)) }
}
// --- end Aero compatibility shims ---
"""

_MATCH_ASSIGN_RE = re.compile(
    r"(?P<indent>[ \t]*)let[ \t]+(?P<name>[A-Za-z_]\w*)[ \t]*=[ \t]*match\b",
)
_BORROW_MUT_RE = re.compile(r"cannot borrow `(?P<name>[A-Za-z_]\w*)` as mutable")


@dataclass
class ShieldReport:
    """The result of shielding a Rust source."""

    source: str
    anchors: Set[str] = field(default_factory=set)
    applied: List[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.applied)

    def to_dict(self) -> dict:
        return {
            "anchors": sorted(self.anchors),
            "applied": list(self.applied),
            "changed": self.changed,
        }


class RustSemanticShield:
    """Apply codified rug/pyo3 compatibility fixes to Rust source."""

    def detect_anchors(self, source: str) -> Set[str]:
        found: Set[str] = set()
        for anchor in RUST_ANCHORS:
            if re.search(rf"\b{re.escape(anchor)}\b\s*(::|;)", source) or re.search(
                rf"\b(?:use|extern crate)\s+{re.escape(anchor)}\b", source
            ):
                found.add(anchor)
        return found

    def needs_shielding(self, source: str) -> bool:
        return bool(self.detect_anchors(source))

    def apply(
        self,
        source: str,
        compatibility_shims: Optional[List[str]] = None,
    ) -> ShieldReport:
        anchors = self.detect_anchors(source)
        report = ShieldReport(source=source, anchors=anchors)

        if compatibility_shims is not None:
            if not compatibility_shims:
                return report
            if "rug_v1_30_patch" in compatibility_shims:
                source, injected = self.inject_extension_traits(source)
                if injected:
                    report.applied.append("extension-traits(neg_mut,nth_root)")
            if "pyo3_usize_alignment" in compatibility_shims:
                source, aligned = self.align_match_types(source)
                if aligned:
                    report.applied.append(f"type-cascade-alignment(usize x{aligned})")
            report.source = source
            return report

        if not anchors:
            return report

        if "rug" in anchors:
            source, injected = self.inject_extension_traits(source)
            if injected:
                report.applied.append("extension-traits(neg_mut,nth_root)")

        source, aligned = self.align_match_types(source)
        if aligned:
            report.applied.append(f"type-cascade-alignment(usize x{aligned})")

        report.source = source
        return report

    def inject_extension_traits(self, source: str) -> Tuple[str, bool]:
        if _TRAIT_SENTINEL in source:
            return source, False

        lines = source.splitlines(keepends=True)
        insert_at = 0
        for index, raw in enumerate(lines):
            stripped = raw.strip()
            if (
                stripped == ""
                or stripped.startswith("#![")
                or stripped.startswith("//!")
                or stripped.startswith("//")
            ):
                insert_at = index + 1
                continue
            break

        block = ("\n" if insert_at > 0 else "") + EXTENSION_TRAITS + "\n"
        new_source = "".join(lines[:insert_at]) + block + "".join(lines[insert_at:])
        return new_source, True

    def align_match_types(self, source: str) -> Tuple[str, int]:
        count = 0
        out: List[str] = []
        pos = 0
        for match in _MATCH_ASSIGN_RE.finditer(source):
            block_end = _find_block_end(source, match.end())
            arms = source[match.end() : block_end] if block_end > match.end() else ""
            if not _arms_are_integer_like(arms):
                continue
            out.append(source[pos : match.start()])
            out.append(f"{match.group('indent')}let {match.group('name')}: usize = match")
            pos = match.end()
            count += 1
        out.append(source[pos:])
        return ("".join(out), count) if count else (source, 0)

    def fix_mutability(self, source: str, diagnostics: str) -> Tuple[str, List[str]]:
        applied: List[str] = []
        names = {m.group("name") for m in _BORROW_MUT_RE.finditer(diagnostics)}
        for name in sorted(names):
            pattern = re.compile(rf"\blet[ \t]+({re.escape(name)})\b(?![ \t]+mut\b)")
            new_source, n = pattern.subn(rf"let mut \1", source)
            if n:
                source = new_source
                applied.append(f"mut({name})")
        return source, applied

    def correct_from_diagnostics(self, source: str, diagnostics: str) -> Tuple[str, List[str]]:
        applied: List[str] = []
        source, mut_fixes = self.fix_mutability(source, diagnostics)
        applied.extend(mut_fixes)
        if re.search(r"expected `?usize`?|mismatched types", diagnostics):
            source, aligned = self.align_match_types(source)
            if aligned:
                applied.append(f"type-cascade-alignment(usize x{aligned})")
        return source, applied


def _find_block_end(source: str, open_search_from: int) -> int:
    brace = source.find("{", open_search_from)
    if brace == -1:
        return -1
    depth = 0
    for i in range(brace, len(source)):
        ch = source[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _arms_are_integer_like(arms: str) -> bool:
    results = re.findall(r"=>\s*([^,\n}]+)", arms)
    if not results:
        return False
    integer_like = 0
    for result in results:
        token = result.strip().rstrip(",").strip()
        if re.fullmatch(r"-?\d+(usize|u32|u64|i32|i64)?", token) or token.endswith("as usize"):
            integer_like += 1
    return integer_like >= len(results) // 2
