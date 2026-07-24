"""Multi-function build orchestrator driven by a Blueprint."""

from __future__ import annotations

import concurrent.futures
import logging
import os
import shutil
import subprocess
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import click

from aero_forge.blueprint import Blueprint, FunctionSpec, discover_functions
from aero_forge.cache.build_cache import BuildCache
from aero_forge.config import ConfigOverride
from aero_forge.error_explainer import explain_error
from aero_forge.gpu import compile_gpu_kernel, find_gpu_functions
from aero_forge.orchestrator.orchestrator import Orchestrator
from aero_forge.scaffold.engine import _generate_pyi
from aero_forge.translator import TargetMode
from aero_forge.wasm import build_wasm_module

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
        explanation: str = "",
    ):
        self.source = source
        self.function_names = function_names
        self.success = success
        self.artifact = artifact
        self.logs = logs
        self.iterations = iterations
        self.explanation = explanation


class BuildTaskDAG:
    """Directed acyclic graph of build tasks with SHA-256 input hashing.

    Tasks are run in topological order. A task is skipped when its input file
    hashes match a cached entry and all recorded output files still exist and
    are unchanged.
    """

    def __init__(self, cache: BuildCache, max_workers: int = 1):
        self.cache = cache
        self.max_workers = max(1, max_workers)
        self._tasks: Dict[str, Dict[str, Any]] = {}

    def add_task(
        self,
        name: str,
        func: Any,
        inputs: List[Any],
        outputs: Optional[List[Path]] = None,
        deps: Optional[List[str]] = None,
    ) -> None:
        """Register a task that produces ``outputs`` from ``inputs``.

        ``func`` must return either a ``BuildResult`` or a dict with keys
        ``result`` and optionally ``outputs``.
        """
        self._tasks[name] = {
            "func": func,
            "inputs": list(inputs),
            "outputs": [Path(p) for p in (outputs or [])],
            "deps": list(deps or []),
        }

    def run(self) -> Dict[str, Any]:
        """Execute tasks in dependency order and return their results."""
        order = self._topological_order()
        results: Dict[str, Any] = {}

        if self.max_workers == 1 or len(order) == 1:
            for name in order:
                results[name] = self._run_task(name)
        else:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers
            ) as executor:
                futures = {
                    executor.submit(self._run_task, name): name for name in order
                }
                for future in concurrent.futures.as_completed(futures):
                    name = futures[future]
                    results[name] = future.result()

        return results

    def _topological_order(self) -> List[str]:
        """Return a valid execution order using Kahn's algorithm."""
        in_degree: Dict[str, int] = {name: 0 for name in self._tasks}
        dependents: Dict[str, List[str]] = defaultdict(list)
        for name, task in self._tasks.items():
            for dep in task["deps"]:
                if dep in self._tasks:
                    in_degree[name] += 1
                    dependents[dep].append(name)

        queue = [name for name, deg in in_degree.items() if deg == 0]
        order: List[str] = []
        while queue:
            name = queue.pop(0)
            order.append(name)
            for dependent in dependents[name]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(order) != len(self._tasks):
            raise RuntimeError("Build task graph contains a cycle")
        return order

    def _run_task(self, name: str) -> Any:
        task = self._tasks[name]
        inputs = task["inputs"]
        cached = self.cache.get_task(name, inputs)
        if cached is not None:
            logger.info("DAG cache hit for task %s", name)
            return self._build_result_from_cache(cached)

        result = task["func"]()
        outputs = list(task["outputs"])
        if isinstance(result, dict) and "result" in result:
            outputs.extend(Path(p) for p in result.get("outputs", []))
            result = result["result"]

        if isinstance(result, BuildResult):
            self.cache.put_task(name, inputs, outputs, self._result_to_dict(result))
        return result

    @staticmethod
    def _result_to_dict(result: BuildResult) -> Dict[str, Any]:
        return {
            "source": str(result.source),
            "function_names": result.function_names,
            "success": result.success,
            "artifact": str(result.artifact) if result.artifact else None,
            "logs": result.logs,
            "iterations": result.iterations,
            "explanation": result.explanation,
        }

    @staticmethod
    def _build_result_from_cache(cached: Dict[str, Any]) -> BuildResult:
        return BuildResult(
            source=Path(cached["source"]),
            function_names=list(cached.get("function_names", [])),
            success=cached.get("success", False),
            artifact=Path(cached["artifact"]) if cached.get("artifact") else None,
            logs="DAG cache hit",
            iterations=0,
            explanation=cached.get("explanation", ""),
        )


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
        cache_dir: Optional[Path] = None,
        force: bool = False,
        gpu: bool = False,
        target: str = "native",
        target_mode: str = TargetMode.PYO3,
        distributed: bool = False,
        dry_run: bool = False,
        progress: bool = False,
        config_override: Optional[ConfigOverride] = None,
    ):
        self.blueprint = blueprint
        self.max_workers = max(1, max_workers)
        self.llm_provider = llm_provider or blueprint.llm.provider
        self.model = model or blueprint.llm.model
        self.max_iterations = max_iterations
        self.max_retries = max_retries
        self.force = force
        self.gpu = gpu
        self.target = target
        self.target_mode = target_mode
        self.distributed = distributed
        self.progress = progress and sys.stderr.isatty()
        self.config_override = config_override
        self._host_target = _host_target()
        env_cache = os.getenv("AERO_FORGE_CACHE_ENABLED", "true").lower() not in (
            "0",
            "false",
            "no",
        )
        effective_cache_enabled = (
            cache_enabled and env_cache and not force and target_mode == TargetMode.PYO3
        )
        cache_root = cache_dir or _cache_dir_from_env()
        self.cache = BuildCache(root=cache_root, enabled=effective_cache_enabled)
        self.dry_run = dry_run

    def build(self) -> Dict[str, Any]:
        """Run the build for every source file in the blueprint."""
        output_dir = self.blueprint.output_dir.resolve()
        if not self.dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)

        expanded = self._expand_specs()
        grouped = self._group_by_source(expanded)
        source_specs = list(grouped.items())

        if self.dry_run:
            return self._summarize(
                [
                    BuildResult(
                        source=source,
                        function_names=[spec.name for spec in specs],
                        success=True,
                        logs="dry run",
                        iterations=0,
                    )
                    for source, specs in source_specs
                ],
                dry_run=True,
            )

        dag = BuildTaskDAG(self.cache, max_workers=self.max_workers)
        for source, specs in source_specs:
            inputs = [source]
            for spec in specs:
                for test in spec.tests:
                    if test.is_file():
                        inputs.append(test)
            inputs = sorted({p.resolve() for p in inputs})
            function_names = [spec.name for spec in specs]
            flags = self._combined_flags(specs)
            metadata = (
                f"functions:{','.join(sorted(function_names))};"
                f"flags:{','.join(sorted(flags))};"
                f"target:{self.target};"
                f"target_mode:{self.target_mode}"
            )
            # Include the source path so identical files in different projects do
            # not share a task cache entry.
            inputs.append(str(source.resolve()))
            inputs.append(metadata)
            task_outputs = [output_dir / source.name]
            dag.add_task(
                name=f"compile:{source.name}",
                func=lambda s=source, sp=specs: self._safe_build_source(
                    output_dir, s, sp
                ),
                inputs=inputs,
                outputs=task_outputs,
            )

        dag_results = dag.run()
        results: List[BuildResult] = list(dag_results.values())
        return self._summarize(results)

    def _safe_build_source(
        self,
        output_dir: Path,
        source: Path,
        specs: List[FunctionSpec],
    ) -> BuildResult:
        """Wrap ``_build_source`` so one broken source file cannot crash the whole build."""
        try:
            return self._build_source(output_dir, source, specs)
        except Exception as exc:
            logger.error("Unexpected error building %s: %s", source, exc)
            return BuildResult(
                source=source,
                function_names=[spec.name for spec in specs],
                success=False,
                logs=f"{exc}\n{traceback.format_exc()}",
            )

    def _expand_specs(self) -> List[FunctionSpec]:
        """Expand any compile-all specs into individual function specs."""
        expanded: List[FunctionSpec] = []
        for spec in self.blueprint.functions:
            if spec.compile_all:
                discovered = discover_functions(spec.file)
                if not discovered:
                    logger.warning("No public functions found in %s", spec.file)
                for func in discovered:
                    expanded.append(
                        FunctionSpec(
                            file=spec.file,
                            name=func.name,
                            tests=spec.tests or func.tests,
                            output_name=func.name,
                            compiler_flags=list(spec.compiler_flags),
                        )
                    )
            else:
                expanded.append(spec)
        return expanded

    def _group_by_source(
        self,
        specs: List[FunctionSpec],
    ) -> Dict[Path, List[FunctionSpec]]:
        groups: Dict[Path, List[FunctionSpec]] = defaultdict(list)
        for spec in specs:
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
        # Skip running tests when cross-compiling to a different host.
        if self.target not in ("native", self._host_target):
            all_tests = []
        flags = self._combined_flags(specs)

        source_text = source.read_text(encoding="utf-8")

        if self.gpu:
            gpu_functions = find_gpu_functions(source_text)
            if gpu_functions:
                try:
                    gpu_artifact = compile_gpu_kernel(source, gpu_functions)
                    if gpu_artifact is not None:
                        return BuildResult(
                            source=source,
                            function_names=gpu_functions,
                            success=True,
                            artifact=gpu_artifact,
                            logs="GPU kernel compiled",
                            iterations=0,
                        )
                except UnsupportedError as exc:
                    return BuildResult(
                        source=source,
                        function_names=gpu_functions,
                        success=False,
                        logs=str(exc),
                        iterations=0,
                    )

        source_output = output_dir
        source_output.mkdir(parents=True, exist_ok=True)

        if self.target == "wasm32-unknown-unknown":
            try:
                wasm_artifact = build_wasm_module(
                    source, function_names, source_output, module_name=source.stem
                )
                return BuildResult(
                    source=source,
                    function_names=function_names,
                    success=True,
                    artifact=wasm_artifact,
                    logs="WASM module built",
                    iterations=0,
                )
            except UnsupportedError as exc:
                return BuildResult(
                    source=source,
                    function_names=function_names,
                    success=False,
                    logs=str(exc),
                    iterations=0,
                )

        cache_key_name = f"{source.stem}_{'_'.join(function_names)}"
        cached = self.cache.get(
            source_text, flags, cache_key_name, target=self.target, target_mode=self.target_mode
        )

        if cached is not None:
            try:
                so_dest = source_output / cached.name
                shutil.copy(cached, so_dest)
                module_name = f"aero_forge_{source.stem}"
                self._write_loader(
                    source_output, cached.name, function_names, source.name, module_name
                )
                _generate_pyi(
                    source_text, function_names, source_output / f"{source.name}i"
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
            list(self._group_by_source(self._expand_specs()).keys()).index(source) + 1,
            len(self._group_by_source(self._expand_specs())),
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
            target=self.target if self.target != "native" else None,
            target_mode=self.target_mode,
            config_override=self.config_override,
        )
        result = orchestrator.run()
        success = result.get("success", False)
        artifact_path: Optional[Path] = None
        explanation = ""
        if not success:
            explanation = explain_error(
                result.get("error", result.get("logs", "")),
                source=source_text,
                llm_provider=self.llm_provider,
                model=self.model,
                config_override=self.config_override,
            )
            logger.info("\n%s", explanation)
        if success and result.get("artifact"):
            artifact = Path(result["artifact"])
            if artifact.is_file():
                self.cache.put(
                    source_text,
                    flags,
                    cache_key_name,
                    artifact,
                    target=self.target,
                    target_mode=self.target_mode,
                )
                artifact_path = source_output / artifact.name
            elif artifact.is_dir():
                artifact_path = artifact

        logs = result.get("logs", "")
        if not success:
            error = result.get("error")
            if error:
                logs = f"{error}\n{logs}".strip()

        return BuildResult(
            source=source,
            function_names=function_names,
            success=success,
            artifact=artifact_path,
            logs=logs,
            iterations=result.get("iterations", 0),
            explanation=explanation,
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
            "_MOD = importlib.util.module_from_spec(_SPEC)\n"
            "_SPEC.loader.exec_module(_MOD)\n"
            + "".join(f"{name} = _MOD.{name}\n" for name in function_names)
            + "\n__all__ = ["
            + ", ".join(f'"{n}"' for n in function_names)
            + "]\n",
            encoding="utf-8",
        )
        return loader

    def _summarize(
        self, results: List[BuildResult], dry_run: bool = False
    ) -> Dict[str, Any]:
        total_functions = sum(len(r.function_names) for r in results)
        passed_functions = sum(len(r.function_names) for r in results if r.success)
        failed_functions = total_functions - passed_functions

        for r in results:
            status = "OK" if r.success else "FAIL"
            if dry_run:
                logger.info("[DRY-RUN] %s -> %s", r.source, ", ".join(r.function_names))
            else:
                logger.info(
                    "[%s] %s -> %s",
                    ", ".join(r.function_names),
                    r.source,
                    status,
                )
            if not r.success and not dry_run:
                logger.error("%s failed:\n%s", r.source, r.logs)
                if r.explanation:
                    logger.error("Suggestion:\n%s", r.explanation)

        if dry_run:
            logger.info(
                "Dry-run summary: %d function(s) across %d source file(s) would be built",
                total_functions,
                len(results),
            )
        else:
            logger.info(
                "Build summary: %d succeeded, %d failed out of %d",
                passed_functions,
                failed_functions,
                total_functions,
            )

        # Surface the first concrete failure for the web UI; when everything
        # passes, still surface the first non-empty log so bypass messages show.
        first_error = ""
        first_logs = ""
        for r in results:
            if r.logs and not first_logs:
                first_logs = r.logs
            if not r.success:
                first_error = first_logs.splitlines()[0] if first_logs else "Build failed"
                break

        return {
            "success": failed_functions == 0 and not dry_run,
            "dry_run": dry_run,
            "project": self.blueprint.project,
            "output_dir": str(self.blueprint.output_dir),
            "total": total_functions,
            "passed": passed_functions,
            "failed": failed_functions,
            "error": first_error,
            "logs": first_logs,
            "results": [
                {
                    "source": str(r.source),
                    "functions": r.function_names,
                    "success": r.success,
                    "artifact": str(r.artifact) if r.artifact else None,
                    "iterations": r.iterations,
                    "logs": r.logs,
                    "explanation": r.explanation,
                }
                for r in results
            ],
        }


def _host_target() -> str:
    try:
        result = subprocess.run(
            ["rustc", "-vV"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        for line in result.stdout.splitlines():
            if line.startswith("host:"):
                return line.split(":", 1)[1].strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def _cache_dir_from_env() -> Optional[Path]:
    env_dir = os.getenv("AERO_FORGE_CACHE_DIR")
    return Path(env_dir) if env_dir else None
