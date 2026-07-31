"""Python runtime loader for ``.hinb`` HIN native bundles.

A ``.hinb`` bundle is a zip archive that contains a ``manifest.json``
metadata manifest and per-source graph payloads.  This module lets
applications inspect the bundle and execute entrypoint functions using
the original source as a portable fallback.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class HINBundle:
    """In-memory handle for a ``.hinb`` HIN native bundle.

    The bundle is loaded lazily: only ``manifest.json`` is parsed when the
    object is constructed.  Graph payloads and source files are read on
    demand so the loader can be used for inspection without allocating
    compute graphs or GPU buffers.
    """

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path)
        self._data = self.path.read_bytes()
        self._zip = zipfile.ZipFile(io.BytesIO(self._data))
        self.manifest: Dict[str, Any] = json.loads(self._zip.read("manifest.json"))

    @property
    def project(self) -> str:
        return str(self.manifest.get("project", "unknown"))

    @property
    def hin_version(self) -> str:
        return str(self.manifest.get("hin_version", "1.0"))

    @property
    def default_backend(self) -> str:
        return str(self.manifest.get("default_backend", "hin_cpu"))

    @property
    def precision_mode(self) -> str:
        return str(self.manifest.get("precision_mode", "ieee"))

    @property
    def entrypoints(self) -> List[Dict[str, Any]]:
        return self.manifest.get("entrypoints", [])

    @property
    def input_schema(self) -> List[Dict[str, Any]]:
        return self.manifest.get("input_schema", [])

    @property
    def output_schema(self) -> List[Dict[str, Any]]:
        return self.manifest.get("output_schema", [])

    def graph_payload(self, source: str) -> Dict[str, Any]:
        """Return the stored graph/UAST payload for a given source path."""
        arcname = f"graphs/{source}.json"
        return json.loads(self._zip.read(arcname))

    def source(self, source: str) -> str:
        """Return the original Python source text for a given source path."""
        return self.graph_payload(source).get("source_text", "")

    def entrypoint(self, name: str) -> Optional[Dict[str, Any]]:
        """Look up a manifest entrypoint by function name."""
        for ep in self.entrypoints:
            if ep.get("name") == name:
                return ep
        return None

    def run(
        self,
        entrypoint: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute an entrypoint from the bundle.

        The source containing the entrypoint is loaded and ``exec``-ed in a
        fresh namespace, then the entrypoint is called with the provided
        positional or keyword arguments.  This is the portable Python
        execution path; a native HIN engine may later replace this fallback.
        """
        ep = self.entrypoint(entrypoint)
        if ep is None:
            raise ValueError(f"Entrypoint {entrypoint!r} not found in manifest")

        source_path = ep.get("source", "")
        payload = self.graph_payload(source_path)
        source = payload.get("source_text", payload.get("source", ""))
        if not source:
            raise ValueError(f"No source text available for {source_path!r}")

        namespace: Dict[str, Any] = {}
        exec(compile(source, f"<hinb:{source_path}>", "exec"), namespace)

        func = namespace.get(entrypoint)
        if not callable(func):
            raise ValueError(f"Function {entrypoint!r} is not callable in {source_path!r}")

        # Build kwargs from manifest input order when positional args are used.
        if args:
            inputs = ep.get("inputs", [])
            bound_kwargs = {}
            for i, arg in enumerate(args):
                if i < len(inputs):
                    bound_kwargs[inputs[i]["name"]] = arg
            bound_kwargs.update(kwargs)
            kwargs = bound_kwargs

        return func(**kwargs)


def load_hin_bundle(path: Union[str, Path]) -> HINBundle:
    """Parse ``manifest.json`` from a ``.hinb`` bundle and return a handle."""
    return HINBundle(path)


def inspect_hin_bundle(path: Union[str, Path]) -> Dict[str, Any]:
    """Return the ``manifest.json`` contents of a ``.hinb`` bundle."""
    return load_hin_bundle(path).manifest
