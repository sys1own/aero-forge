"""Multi-function build orchestrator driven by a Blueprint."""

from __future__ import annotations

import concurrent.futures
import logging
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from aero_forge.blueprint import Blueprint, FunctionSpec
from aero_forge.cache.build_cache import BuildCache
from aero_forge.orchestrator.orchestrator import Orchestrator

logger = logging.getLogger("aero_forge.build")


class BuildResult:
    """Result of a single source-file build."""

    def __init__(
        self,
        source: Path,
        function_names: List[str],
        success: bool,
        artifact: Optional[Path] = None,
        logs: str = "",
        iterations: int = 0,
    ):
        self.source = source
        self.function_names = function_names
        self.success = success
        self.artifact = artifact
        self.logs = logs
        self.iterations = iterations


class BuildRunner:
    """Compile and test all functions described by a Blueprint."""

    def __init__(
        self,
        blueprint: Blueprint,
        max_workers: int = 4,
        llm_provider: Optional[str] = None,
        model: Optional[str] = None,
        max_iterations: Optional[int] = None,
        max_retries: Optional[int] = None,
        cache_enabled: bool = True,
    ):
        self.blueprint = blueprint
        self.max_workers = max(1, max_workers)
        self.llm_provider = llm_provider or blueprint.llm.provider
        self.model = model or blueprint.llm.model
        self.max_iterations = max_iterations
        self.max_retries = max_retries
        self.cache = BuildCache(enabled=cache_enabled)

    def build(self) -> Dict[str, Any]:
        """Run the build for every source file in the blueprint."""
        output_dir = self.blueprint.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        grouped = self._group_by_source()
        source_specs = list(grouped.items())

        results: List[BuildResult] = []
        if self.max_workers == 1 or len(source_specs) == 1:
            for source, specs in source_specs:
                results.append(self._build_source(output_dir, source, specs))
        else:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers
            ) as executor:
                futures = {
                    executor.submit(
                        self._build_source, output_dir, source, specs
                    ): source
                    for source, specs in source_specs
                }
                for future in concurrent.futures.as_completed(futures):
                    results.append(future.result())

        return self._summarize(results)

    def _group_by_source(
        self,
    ) -> Dict[Path, List[FunctionSpec]]:
        groups: Dict[Path, List[FunctionSpec]] = defaultdict(list)
        for spec in self.blueprint.functions:
            groups[spec.file.resolve()].append(spec)
        return groups

    def _build_source(
        self,
        output_dir: Path,
        source: Path,
        specs: List[FunctionSpec],
    ) -> BuildResult:
        function_names = [spec.name for spec in specs]
        primary = function_names[0]
        all_tests = sorted(
            {str(t.resolve()) for spec in specs for t in spec.tests if t.is_file()}
        )
        flags = self._combined_flags(specs)

        source_text = source.read_text(encoding="utf-8")
        cache_key_name = f"{source.stem}_{'_'.join(function_names)}"
        cached = self.cache.get(source_text, flags, cache_key_name)

        source_output = output_dir
        source_output.mkdir(parents=True, exist_ok=True)

        if cached is not None:
            try:
                so_dest = source_output / cached.name
                shutil.copy(cached, so_dest)
                module_name = f"aero_forge_{source.stem}"
                self._write_loader(
                    source_output, cached.name, function_names, source.name, module_name
                )
                return BuildResult(
                    source=source,
                    function_names=function_names,
                    success=True,
                    artifact=so_dest,
                    logs="Build cache hit",
                    iterations=0,
                )
            except OSError as exc:
                logger.warning("Cache copy failed for %s: %s", source, exc)

        logger.info(
            "[%s/%s] Compiling %s -> %s",
            list(self._group_by_source().keys()).index(source) + 1,
            len(self._group_by_source()),
            source,
            ", ".join(function_names),
        )

        orchestrator = Orchestrator(
            source_path=source,
            function_name=primary,
            function_names=function_names,
            test_paths=all_tests,
            max_iterations=self.max_iterations,
            llm_provider=self.llm_provider,
            model=self.model,
            max_retries=self.max_retries,
            cache_enabled=False,
            compiler_flags=flags,
            output_dir=source_output,
        )
        result = orchestrator.run()
        success = result.get("success", False)
        artifact_path: Optional[Path] = None
        if success and result.get("artifact"):
            artifact = Path(result["artifact"])
            if artifact.is_file():
                self.cache.put(source_text, flags, cache_key_name, artifact)
                artifact_path = source_output / artifact.name

        return BuildResult(
            source=source,
            function_names=function_names,
            success=success,
            artifact=artifact_path,
            logs=result.get("logs", ""),
            iterations=result.get("iterations", 0),
        )

    def _combined_flags(self, specs: List[FunctionSpec]) -> List[str]:
        flags: List[str] = []
        flags.extend(self.blueprint.compiler_flags)
        for spec in specs:
            flags.extend(spec.compiler_flags)
        return flags

    def _write_loader(
        self,
        source_output: Path,
        so_name: str,
        function_names: List[str],
        loader_name: str,
        module_name: str,
    ) -> Path:
        """Write a minimal loader when the artifact came from the build cache."""
        loader = source_output / loader_name
        loader.write_text(
            "import importlib.util\n"
            "import pathlib\n"
            f'_SO = pathlib.Path(__file__).parent / "{so_name}"\n'
            f'_SPEC = importlib.util.spec_from_file_location("{module_name}", _SO)\n'
            '_MOD = importlib.util.module_from_spec(_SPEC)\n'
            '_SPEC.loader.exec_module(_MOD)\n'
            + "".join(f"{name} = _MOD.{name}\n" for name in function_names)
            + '\n__all__ = [' + ', '.join(f'"{n}"' for n in function_names) + ']\n',
            encoding="utf-8",
        )
        return loader

    def _summarize(self, results: List[BuildResult]) -> Dict[str, Any]:
        total_functions = sum(len(r.function_names) for r in results)
        passed_functions = sum(
            len(r.function_names) for r in results if r.success
        )
        failed_functions = total_functions - passed_functions

        for r in results:
            status = "OK" if r.success else "FAIL"
            logger.info(
                "[%s] %s -> %s",
                ", ".join(r.function_names),
                r.source,
                status,
            )
            if not r.success:
                logger.error("%s failed:\n%s", r.source, r.logs)

        logger.info(
            "Build summary: %d succeeded, %d failed out of %d",
            passed_functions,
            failed_functions,
            total_functions,
        )

        return {
            "success": failed_functions == 0,
            "project": self.blueprint.project,
            "output_dir": str(self.blueprint.output_dir),
            "total": total_functions,
            "passed": passed_functions,
            "failed": failed_functions,
            "results": [
                {
                    "source": str(r.source),
                    "functions": r.function_names,
                    "success": r.success,
                    "artifact": str(r.artifact) if r.artifact else None,
                    "iterations": r.iterations,
                }
                for r in results
            ],
        }
