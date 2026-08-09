"""Topological wavefront scheduler for build and test tasks.

A *wavefront* is a set of DAG nodes that share the same dependency level.
Nodes within a wave have no intra-dependencies and may execute concurrently;
waves are ordered by topological dependency and a strict join barrier is enforced
($W_{i+1}$ starts only after $W_i$ completes).
"""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import shutil
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger("aero_forge.scheduler.wavefront")


class SchedulerError(Exception):
    """Base exception for scheduling failures."""


class CycleError(SchedulerError):
    """Raised when the graph contains a cycle and cannot be levelised."""


class MutationKind(str, Enum):
    """Kinds of graph mutation used by incremental schedule repair."""

    ADD_NODE = "add_node"
    REMOVE_NODE = "remove_node"
    ADD_EDGE = "add_edge"
    REMOVE_EDGE = "remove_edge"
    UPDATE_NODE = "update_node"


@dataclass
class GraphMutation:
    """A single structural change to a task DAG."""

    kind: MutationKind
    node_id: Optional[str] = None
    edge: Optional[Tuple[str, str]] = None


@dataclass
class Task:
    """A single executable unit in a wavefront schedule."""

    name: str
    command: str
    cwd: Optional[Path] = None
    env: Optional[Dict[str, str]] = None
    timeout: Optional[float] = None


@dataclass
class WavefrontScheduler:
    """Compute topological wavefronts and execute tasks with join barriers."""

    thread_limit: int = field(default_factory=lambda: int(os.environ.get("AERO_MAX_THREADS", "0")) or max(1, os.cpu_count() or 1))
    memory_limit_mb: int = field(default_factory=lambda: int(os.environ.get("AERO_MAX_MEMORY_MB", "0")) or 2048)
    log_callback: Optional[Callable[[str, str, str], None]] = None
    progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None

    def _log(self, level: str, prefix: str, message: str) -> None:
        if self.log_callback:
            self.log_callback(level, prefix, message)
        logger.log(getattr(logging, level.upper(), logging.INFO), "[%s] %s", prefix, message)

    def _emit(self, event: str, payload: Dict[str, Any]) -> None:
        if self.progress_callback:
            try:
                self.progress_callback(event, payload)
            except Exception as exc:  # pragma: no cover - callers must not crash scheduler
                logger.debug("progress callback failed: %s", exc)

    def compute_wavefronts(
        self,
        adj_list: Dict[str, List[str]],
        roots: Optional[Sequence[str]] = None,
        *,
        use_goi: bool = False,
        goi_weights: Optional[Dict[Tuple[str, str], float]] = None,
        goi_damping: float = 0.15,
    ) -> List[List[str]]:
        """Return level-ordered waves from a DAG adjacency list.

        Uses Kahn's algorithm grouped into breadth-first layers.  When
        ``use_goi`` is True and ``numpy`` is available, nodes inside each wave
        are ordered by a GoI-derived precedence score so that tasks on the
        most critical dependency paths run first.
        """
        from aero_forge.scheduler import goi_solver

        nodes = set(adj_list.keys()) | {n for deps in adj_list.values() for n in deps}
        reverse: Dict[str, List[str]] = {n: [] for n in nodes}
        for node, deps in adj_list.items():
            for dep in deps:
                reverse[dep].append(node)

        in_degree = {n: len(adj_list.get(n, [])) for n in nodes}
        queue: deque = deque(roots if roots is not None else [n for n in nodes if in_degree[n] == 0])
        waves: List[List[str]] = []

        scores: Dict[str, float] = {}
        if use_goi:
            try:
                scores = goi_solver.precedence_scores(
                    adj_list, damping=goi_damping, weights=goi_weights
                )
            except Exception as exc:
                self._log("warning", "SCHEDULER", f"GoI precedence failed ({exc}); using lexical order")
                scores = {}

        while queue:
            wave: List[str] = []
            next_queue: deque = deque()
            # Order within a wave: GoI score desc, then name asc for determinism.
            ordered_queue = sorted(
                queue,
                key=lambda n: (-scores.get(n, 0.0), n),
            )
            for node in ordered_queue:
                wave.append(node)
                for nbr in reverse.get(node, []):
                    in_degree[nbr] -= 1
                    if in_degree[nbr] == 0:
                        next_queue.append(nbr)
            waves.append(sorted(wave))
            queue = next_queue

        if any(d > 0 for d in in_degree.values()):
            raise CycleError("Graph contains a cycle; cannot compute wavefront schedule")

        self._emit("goi_wave_state", {"waves": waves, "goi": use_goi, "damping": goi_damping})
        return waves

    def _z3_available(self) -> bool:
        """Return True when the Z3 solver is importable."""
        return shutil.which("z3") is not None or self._import_z3() is not None

    def _import_z3(self):
        try:
            import z3
            return z3
        except Exception:
            return None

    def _verify_resource_limits(self, tasks_in_wave: Sequence[Task]) -> bool:
        """Use Z3 to check thread and memory limits for a wave, or fall back."""
        if not self._z3_available():
            return self._heuristic_resource_check(tasks_in_wave)

        z3 = self._import_z3()
        if z3 is None:
            return self._heuristic_resource_check(tasks_in_wave)

        solver = z3.Solver()
        total_threads = z3.Int("total_threads")
        total_memory = z3.Int("total_memory")

        # Each task is assumed to consume at least 1 thread and an estimated
        # memory footprint; concurrent execution must not exceed limits.
        thread_terms = [z3.Int(f"threads_{i}") for i in range(len(tasks_in_wave))]
        memory_terms = [z3.Int(f"memory_{i}") for i in range(len(tasks_in_wave))]

        for i, _ in enumerate(tasks_in_wave):
            solver.add(thread_terms[i] >= 1, thread_terms[i] <= self.thread_limit)
            solver.add(memory_terms[i] >= 0, memory_terms[i] <= self.memory_limit_mb)

        solver.add(total_threads == sum(thread_terms))
        solver.add(total_memory == sum(memory_terms))
        solver.add(total_threads <= self.thread_limit)
        solver.add(total_memory <= self.memory_limit_mb)

        if solver.check() == z3.sat:
            return True
        self._log("warning", "SCHEDULER", "Z3 resource check unsatisfiable; falling back to heuristic")
        return self._heuristic_resource_check(tasks_in_wave)

    def _heuristic_resource_check(self, tasks_in_wave: Sequence[Task]) -> bool:
        """Fallback check: accept the wave if task count is within thread limit."""
        return len(tasks_in_wave) <= self.thread_limit

    def _split_command(self, command: str) -> List[str]:
        """Parse a command string into an argv list for direct execution.

        Falls back to a simple split if ``shlex`` cannot tokenize the string.
        """
        try:
            return shlex.split(command)
        except ValueError:
            return command.split()

    async def _run_task(self, task: Task) -> Tuple[Task, Dict[str, Any]]:
        """Run a single task and return its result dictionary."""
        env = task.env or os.environ.copy()
        cwd = str(task.cwd) if task.cwd else None
        timeout = task.timeout or 120
        self._log("info", "WAVE", f"Starting task: {task.name} (`{task.command}`)")

        argv = self._split_command(task.command)
        if not argv:
            return task, {
                "name": task.name,
                "command": task.command,
                "returncode": -1,
                "stdout": "",
                "stderr": "Empty command",
                "timed_out": False,
            }

        proc = await asyncio.create_subprocess_exec(
            argv[0],
            *argv[1:],
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=env,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return task, {
                "name": task.name,
                "command": task.command,
                "returncode": -1,
                "stdout": "",
                "stderr": f"Timed out after {timeout}s",
                "timed_out": True,
            }

        return task, {
            "name": task.name,
            "command": task.command,
            "returncode": proc.returncode,
            "stdout": stdout.decode("utf-8", "replace"),
            "stderr": stderr.decode("utf-8", "replace"),
            "timed_out": False,
        }

    def update_schedule(
        self,
        adj_list: Dict[str, List[str]],
        old_schedule: List[List[str]],
        mutations: Sequence[GraphMutation],
    ) -> List[List[str]]:
        """Repair a schedule after local mutations without a full rebuild.

        Only the *influence zone* (mutated nodes plus all ancestors and
        descendants) is re-levelised; untouched nodes keep their old levels.
        """
        if not mutations:
            return old_schedule

        nodes = set(adj_list.keys()) | {n for deps in adj_list.values() for n in deps}
        adj = {n: list(adj_list.get(n, [])) for n in nodes}
        reverse: Dict[str, List[str]] = {n: [] for n in nodes}
        for n, deps in adj.items():
            for d in deps:
                reverse[d].append(n)

        old_levels: Dict[str, int] = {}
        for level, wave in enumerate(old_schedule):
            for node in wave:
                old_levels[node] = level

        affected: set[str] = set()
        for m in mutations:
            if m.node_id:
                affected.add(m.node_id)
            if m.edge:
                affected.update(m.edge)

        # Expand the influence zone both up- and downstream.
        stack = list(affected)
        while stack:
            node = stack.pop()
            for nbr in adj.get(node, []):
                if nbr not in affected:
                    affected.add(nbr)
                    stack.append(nbr)
            for pred in reverse.get(node, []):
                if pred not in affected:
                    affected.add(pred)
                    stack.append(pred)

        # Base level for an affected node is constrained by its unaffected preds.
        base_level: Dict[str, int] = {n: 0 for n in affected}
        for node in affected:
            for pred in adj.get(node, []):
                if pred not in affected:
                    base_level[node] = max(base_level[node], old_levels.get(pred, 0) + 1)

        sub_indeg: Dict[str, int] = {}
        for node in affected:
            count = 0
            for pred in adj.get(node, []):
                if pred in affected:
                    count += 1
            sub_indeg[node] = count

        new_level: Dict[str, int] = {}
        queue: deque = deque()
        for node in affected:
            new_level[node] = base_level[node]
            if sub_indeg[node] == 0:
                queue.append(node)

        processed = 0
        while queue:
            node = queue.popleft()
            processed += 1
            for nbr in reverse.get(node, []):
                if nbr not in affected:
                    continue
                candidate = new_level[node] + 1
                if candidate > new_level.get(nbr, base_level[nbr]):
                    new_level[nbr] = candidate
                sub_indeg[nbr] -= 1
                if sub_indeg[nbr] == 0:
                    queue.append(nbr)

        if processed != len(affected):
            raise CycleError("Affected subgraph contains a cycle")

        levels: Dict[int, List[str]] = defaultdict(list)
        max_level = 0
        for node, level in old_levels.items():
            if node not in affected:
                levels[level].append(node)
                max_level = max(max_level, level)
        for node, level in new_level.items():
            levels[level].append(node)
            max_level = max(max_level, level)

        return [sorted(levels[i]) for i in range(max_level + 1)]

    async def execute(
        self,
        tasks: Dict[str, Task],
        adj_list: Optional[Dict[str, List[str]]] = None,
        roots: Optional[Sequence[str]] = None,
        *,
        use_goi: bool = False,
        goi_weights: Optional[Dict[Tuple[str, str], float]] = None,
    ) -> List[Dict[str, Any]]:
        """Execute ``tasks`` in topological waves.

        ``adj_list`` maps task name -> list of dependency task names. If no
        adjacency list is provided, all tasks are assumed independent and run
        in a single wave.
        """
        if not tasks:
            return []

        adj = adj_list or {name: [] for name in tasks}
        waves = self.compute_wavefronts(adj, roots=roots, use_goi=use_goi, goi_weights=goi_weights)

        results: List[Dict[str, Any]] = []
        total_waves = len(waves)
        for wave_idx, wave in enumerate(waves):
            wave_tasks = [tasks[n] for n in wave if n in tasks]
            if not wave_tasks:
                continue

            self._log("info", "WAVE", f"Wave {wave_idx}: {len(wave_tasks)} task(s) ({', '.join(t.name for t in wave_tasks)})")
            self._emit(
                "goi_wave_state",
                {
                    "wave": wave_idx,
                    "total_waves": total_waves,
                    "tasks": [t.name for t in wave_tasks],
                    "status": "running",
                },
            )

            if not self._verify_resource_limits(wave_tasks):
                self._log("error", "WAVE", f"Wave {wave_idx} exceeds resource limits")
                for task in wave_tasks:
                    results.append({
                        "name": task.name,
                        "command": task.command,
                        "returncode": -1,
                        "stdout": "",
                        "stderr": "Resource limit check failed",
                        "timed_out": False,
                    })
                self._emit(
                    "goi_wave_state",
                    {"wave": wave_idx, "total_waves": total_waves, "tasks": [t.name for t in wave_tasks], "status": "resource_limited"},
                )
                continue

            for t in wave_tasks:
                if wave_idx == 0:
                    task_env = t.env or os.environ.copy()
                    pythonpath_parts = [str(t.cwd or os.getcwd())]
                    src_dir = Path(t.cwd or os.getcwd()) / "src"
                    if src_dir.is_dir():
                        pythonpath_parts.append(str(src_dir))
                    pythonpath_parts.append(task_env.get("PYTHONPATH", ""))
                    task_env["PYTHONPATH"] = os.pathsep.join(
                        p for p in pythonpath_parts if p
                    ).strip(os.pathsep)
                    t.env = task_env

            wave_results = await asyncio.gather(*[self._run_task(t) for t in wave_tasks])
            for _task, result in wave_results:
                results.append(result)
                status = "ok" if result["returncode"] == 0 else "failed"
                self._log(
                    "info" if result["returncode"] == 0 else "error",
                    "WAVE",
                    f"Task {result['name']} finished with exit code {result['returncode']} ({status})",
                )

            failed = any(r["returncode"] != 0 for _t, r in wave_results)
            self._emit(
                "goi_wave_state",
                {
                    "wave": wave_idx,
                    "total_waves": total_waves,
                    "tasks": [t.name for t in wave_tasks],
                    "status": "failed" if failed else "completed",
                    "results": [r["returncode"] for _t, r in wave_results],
                },
            )
            if failed:
                self._log("warning", "WAVE", f"Wave {wave_idx} had failures; continuing to next wave")

        return results

    def execute_sync(
        self,
        tasks: Dict[str, Task],
        adj_list: Optional[Dict[str, List[str]]] = None,
        roots: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Synchronous entry point that wraps ``execute`` in an event loop."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.execute(tasks, adj_list, roots))

        # Already inside an event loop; schedule and run via a helper coroutine.
        async def _run():
            return await self.execute(tasks, adj_list, roots)

        return loop.run_until_complete(_run())
