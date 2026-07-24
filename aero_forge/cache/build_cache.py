"""Cache for compiled native artifacts keyed by source + compiler flags hash."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger("aero_forge.cache.build")


class BuildCache:
    """Persistent cache mapping (source_hash + flags_hash) to a compiled artifact."""

    def __init__(self, root: Optional[Path] = None, enabled: bool = True):
        self.root = (root or _default_cache_root()).resolve()
        self.enabled = enabled
        self.root.mkdir(parents=True, exist_ok=True)
        self._index_path = self.root / "index.json"
        self._index: Dict[str, str] = self._load_index()
        self._task_index_path = self.root / "task_index.json"
        self._task_index: Dict[str, Any] = self._load_task_index()
        self._rustc_version = _rustc_version()

    def _load_index(self) -> Dict[str, str]:
        if self._index_path.is_file():
            try:
                data = json.loads(self._index_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save_index(self) -> None:
        self._index_path.write_text(
            json.dumps(self._index, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _key(
        self,
        source: str,
        compiler_flags: list,
        function_name: str,
        target: Optional[str] = None,
        target_mode: Optional[str] = None,
    ) -> str:
        source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
        target_token = target or "native"
        mode_token = target_mode or "pyo3"
        material = (
            f"{source_hash}::"
            f"{','.join(sorted(compiler_flags))}::"
            f"{function_name}::"
            f"{self._rustc_version}::"
            f"{target_token}::"
            f"{mode_token}"
        ).encode("utf-8")
        return hashlib.sha256(material).hexdigest()

    def get(
        self,
        source: str,
        compiler_flags: list,
        function_name: str,
        target: Optional[str] = None,
        target_mode: Optional[str] = None,
    ) -> Optional[Path]:
        if not self.enabled:
            return None
        key = self._key(source, compiler_flags, function_name, target, target_mode)
        artifact_name = self._index.get(key)
        if not artifact_name:
            return None
        artifact_path = self.root / artifact_name
        if artifact_path.is_file():
            logger.info("Build cache hit for %s", function_name)
            return artifact_path
        # Stale entry; remove.
        self._index.pop(key, None)
        self._save_index()
        return None

    def put(
        self,
        source: str,
        compiler_flags: list,
        function_name: str,
        artifact: Path,
        target: Optional[str] = None,
        target_mode: Optional[str] = None,
    ) -> Path:
        if not self.enabled:
            return Path(artifact)
        key = self._key(source, compiler_flags, function_name, target, target_mode)
        dest = self.root / f"{key}_{artifact.name}"
        shutil.copy(artifact, dest)
        self._index[key] = dest.name
        self._save_index()
        logger.info("Build cache stored for %s", function_name)
        return dest

    def clear(self) -> None:
        for child in self.root.iterdir():
            if child.is_file():
                child.unlink()
        self._index = {}
        self._save_index()
        self._task_index = {}
        self._save_task_index()

    def _load_task_index(self) -> Dict[str, Any]:
        if self._task_index_path.is_file():
            try:
                data = json.loads(self._task_index_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save_task_index(self) -> None:
        self._task_index_path.write_text(
            json.dumps(self._task_index, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def get_task(
        self, task_name: str, inputs: List[Any]
    ) -> Optional[Dict[str, Any]]:
        """Return a cached task result if inputs and recorded outputs are unchanged."""
        if not self.enabled:
            return None
        input_hash = _hash_inputs(inputs)
        entry = self._task_index.get(task_name)
        if not entry or entry.get("input_hash") != input_hash:
            return None
        for path_str, file_hash in entry.get("outputs", {}).items():
            path = Path(path_str)
            if not path.is_file() or _hash_file(path) != file_hash:
                return None
        return entry.get("result")

    def put_task(
        self,
        task_name: str,
        inputs: List[Any],
        outputs: List[Path],
        result: Dict[str, Any],
    ) -> None:
        """Store a task result keyed by the SHA-256 hash of its inputs."""
        if not self.enabled:
            return
        input_hash = _hash_inputs(inputs)
        output_hashes = {str(p): _hash_file(p) for p in outputs if p.is_file()}
        self._task_index[task_name] = {
            "input_hash": input_hash,
            "outputs": output_hashes,
            "result": result,
        }
        self._save_task_index()


def _default_cache_root() -> Path:
    env_dir = os.getenv("AERO_FORGE_CACHE_DIR")
    if env_dir:
        return Path(env_dir)
    return Path.home() / ".cache" / "aero-forge" / "build_cache"


def _hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _hash_inputs(inputs: List[Any]) -> str:
    hasher = hashlib.sha256()
    for item in sorted(inputs, key=lambda x: x if isinstance(x, str) else str(x)):
        if isinstance(item, str):
            hasher.update(item.encode("utf-8"))
        else:
            path = Path(item)
            if path.is_file():
                hasher.update(path.read_bytes())
            else:
                hasher.update(str(path).encode("utf-8"))
    return hasher.hexdigest()


def _rustc_version() -> str:
    try:
        result = subprocess.run(
            ["rustc", "-Vv"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else "rustc-unknown"
    except (OSError, subprocess.TimeoutExpired):
        return "rustc-unknown"
