//! Zero-dependency topological wavefront scheduler.
//!
//! Tasks are grouped into waves $W_0, W_1, ...$ such that tasks inside a wave
//! have no intra-dependencies.  Each wave runs in parallel and a strict join
//! barrier ensures $W_{i+1}$ starts only after $W_i$ completes.
//!
//! Optional GoI-derived precedence ordering can be applied within each wave.

use std::collections::{HashMap, HashSet, VecDeque};
use std::process::Command;

use crate::goi_solver;

/// A single shell-command task with named dependencies.
#[derive(Clone, Debug)]
pub struct Task {
    pub name: String,
    pub command: String,
    pub dependencies: Vec<String>,
}

impl Task {
    pub fn new(name: impl Into<String>, command: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            command: command.into(),
            dependencies: Vec::new(),
        }
    }

    pub fn with_deps(mut self, deps: &[&str]) -> Self {
        self.dependencies = deps.iter().map(|s| s.to_string()).collect();
        self
    }
}

/// Result of executing a single task.
#[derive(Clone, Debug)]
pub struct TaskResult {
    pub name: String,
    pub command: String,
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
}

/// Scheduler error.
#[derive(Debug)]
pub enum ScheduleError {
    Cycle(String),
    TaskNotFound(String),
}

impl std::fmt::Display for ScheduleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ScheduleError::Cycle(msg) => write!(f, "Cycle: {}", msg),
            ScheduleError::TaskNotFound(name) => write!(f, "Task not found: {}", name),
        }
    }
}

impl std::error::Error for ScheduleError {}

/// Structural mutation used by incremental schedule repair.
#[derive(Clone, Debug)]
pub enum MutationKind {
    AddNode,
    RemoveNode,
    AddEdge,
    RemoveEdge,
    UpdateNode,
}

/// A single graph mutation for `update_schedule`.
#[derive(Clone, Debug)]
pub struct Mutation {
    pub kind: MutationKind,
    pub node: Option<usize>,
    pub edge: Option<(usize, usize)>,
}

/// Build the dependency list for each task index.
fn build_dependency_index(tasks: &[Task]) -> Vec<Vec<usize>> {
    let name_to_index: HashMap<String, usize> = tasks
        .iter()
        .enumerate()
        .map(|(i, t)| (t.name.clone(), i))
        .collect();

    tasks
        .iter()
        .map(|t| {
            t.dependencies
                .iter()
                .filter_map(|d| name_to_index.get(d).copied())
                .collect()
        })
        .collect()
}

/// Compute topological wavefronts from a list of tasks.
pub fn compute_wavefronts(tasks: &[Task]) -> Result<Vec<Vec<usize>>, ScheduleError> {
    compute_wavefronts_with_options(tasks, false, 0.15)
}

/// Compute topological wavefronts, optionally using GoI-derived precedence
/// ordering inside each wave.
pub fn compute_wavefronts_with_options(
    tasks: &[Task],
    use_goi: bool,
    goi_damping: f64,
) -> Result<Vec<Vec<usize>>, ScheduleError> {
    if tasks.is_empty() {
        return Ok(Vec::new());
    }

    let deps = build_dependency_index(tasks);
    let mut in_degree = vec![0usize; tasks.len()];
    let mut dependents: HashMap<usize, Vec<usize>> = HashMap::new();

    for (i, d) in deps.iter().enumerate() {
        for &dep in d {
            dependents.entry(dep).or_default().push(i);
            in_degree[i] += 1;
        }
    }

    let mut queue: VecDeque<usize> = in_degree
        .iter()
        .enumerate()
        .filter(|(_, d)| **d == 0)
        .map(|(i, _)| i)
        .collect();

    let scores: Vec<f64> = if use_goi {
        goi_solver::precedence_scores(&deps, goi_damping).unwrap_or_else(|_| vec![0.0; tasks.len()])
    } else {
        vec![0.0; tasks.len()]
    };

    let mut waves: Vec<Vec<usize>> = Vec::new();

    while !queue.is_empty() {
        let mut wave_indices: Vec<usize> = queue.into_iter().collect();
        // Order within the wave: GoI score descending, then index ascending.
        wave_indices.sort_by(|a, b| {
            scores[*b]
                .partial_cmp(&scores[*a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.cmp(b))
        });

        let mut next_queue: VecDeque<usize> = VecDeque::new();
        for node in &wave_indices {
            for &nbr in dependents.get(node).unwrap_or(&Vec::new()) {
                in_degree[nbr] -= 1;
                if in_degree[nbr] == 0 {
                    next_queue.push_back(nbr);
                }
            }
        }
        waves.push(wave_indices);
        queue = next_queue;
    }

    if in_degree.iter().any(|d| *d > 0) {
        return Err(ScheduleError::Cycle(
            "Graph contains a cycle; cannot compute wavefront schedule".into(),
        ));
    }

    Ok(waves)
}

/// Repair a schedule after local mutations without recomputing every wave.
///
/// Only the *influence zone* (mutated nodes plus ancestors and descendants) is
/// re-levelised; untouched nodes keep their old level.
pub fn update_schedule(
    old_waves: &[Vec<usize>],
    deps: &[Vec<usize>],
    mutations: &[Mutation],
) -> Result<Vec<Vec<usize>>, ScheduleError> {
    if mutations.is_empty() {
        return Ok(old_waves.to_vec());
    }

    let n = deps.len();
    let mut reverse: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, d) in deps.iter().enumerate() {
        for &j in d {
            if j < n {
                reverse[j].push(i);
            }
        }
    }

    let mut old_levels: Vec<Option<usize>> = vec![None; n];
    for (level, wave) in old_waves.iter().enumerate() {
        for &node in wave {
            if node < n {
                old_levels[node] = Some(level);
            }
        }
    }

    let mut affected: HashSet<usize> = HashSet::new();
    for m in mutations {
        if let Some(node) = m.node {
            if node < n {
                affected.insert(node);
            }
        }
        if let Some((a, b)) = m.edge {
            if a < n {
                affected.insert(a);
            }
            if b < n {
                affected.insert(b);
            }
        }
    }

    // Expand influence zone upstream and downstream.
    let mut stack: Vec<usize> = affected.iter().copied().collect();
    while let Some(node) = stack.pop() {
        for &nbr in &deps[node] {
            if nbr < n && affected.insert(nbr) {
                stack.push(nbr);
            }
        }
        for &pred in &reverse[node] {
            if pred < n && affected.insert(pred) {
                stack.push(pred);
            }
        }
    }

    // Base level for each affected node is constrained by unaffected preds.
    let mut base_level: HashMap<usize, usize> = HashMap::new();
    for &node in &affected {
        let mut level = 0usize;
        for &pred in &reverse[node] {
            if affected.contains(&pred) {
                continue;
            }
            if let Some(l) = old_levels[pred] {
                level = level.max(l + 1);
            }
        }
        base_level.insert(node, level);
    }

    // Subgraph in-degree counting only affected predecessors.
    let mut sub_indeg: HashMap<usize, usize> = HashMap::new();
    for &node in &affected {
        let count = reverse[node]
            .iter()
            .filter(|&&pred| affected.contains(&pred))
            .count();
        sub_indeg.insert(node, count);
    }

    let mut new_level: HashMap<usize, usize> = HashMap::new();
    let mut queue: VecDeque<usize> = affected
        .iter()
        .filter(|&&node| sub_indeg[&node] == 0)
        .copied()
        .collect();

    for &node in &queue {
        new_level.insert(node, base_level[&node]);
    }

    let mut processed = 0usize;
    while let Some(node) = queue.pop_front() {
        processed += 1;
        for &nbr in &deps[node] {
            if !affected.contains(&nbr) {
                continue;
            }
            let candidate = new_level[&node] + 1;
            let entry = new_level.entry(nbr).or_insert(base_level[&nbr]);
            if candidate > *entry {
                *entry = candidate;
            }
            let indeg = sub_indeg.entry(nbr).or_insert(0);
            *indeg -= 1;
            if *indeg == 0 {
                queue.push_back(nbr);
            }
        }
    }

    if processed != affected.len() {
        return Err(ScheduleError::Cycle(
            "Affected subgraph contains a cycle".into(),
        ));
    }

    // Reassemble levels, preserving unaffected nodes.
    let mut max_level = 0usize;
    let mut levels: Vec<Vec<usize>> = Vec::new();
    for node in 0..n {
        if affected.contains(&node) {
            continue;
        }
        let l = old_levels[node].unwrap_or(0);
        max_level = max_level.max(l);
        if l >= levels.len() {
            levels.resize_with(l + 1, Vec::new);
        }
        levels[l].push(node);
    }
    for (&node, &l) in &new_level {
        max_level = max_level.max(l);
        if l >= levels.len() {
            levels.resize_with(l + 1, Vec::new);
        }
        levels[l].push(node);
    }

    // Sort each wave for deterministic output.
    for wave in &mut levels {
        wave.sort_unstable();
    }

    // Trim unused trailing levels.
    while levels.last().map_or(false, |w| w.is_empty()) {
        levels.pop();
    }

    Ok(levels)
}

/// Parse a command string into argv, supporting basic quoting.
///
/// This avoids the overhead of spawning `/bin/sh` for simple commands.
fn split_command(command: &str) -> Vec<String> {
    let mut args = Vec::new();
    let mut current = String::new();
    let mut in_single = false;
    let mut in_double = false;
    let mut escape = false;

    for ch in command.chars() {
        if escape {
            current.push(ch);
            escape = false;
            continue;
        }
        match ch {
            '\\' if !in_single => {
                escape = true;
            }
            '\'' if !in_double => {
                in_single = !in_single;
            }
            '"' if !in_single => {
                in_double = !in_double;
            }
            ' ' | '\t' if !in_single && !in_double => {
                if !current.is_empty() {
                    args.push(current.clone());
                    current.clear();
                }
            }
            _ => current.push(ch),
        }
    }
    if !current.is_empty() {
        args.push(current);
    }
    args
}

/// Execute tasks in topological waves.
///
/// `cwd` is the working directory passed to each child process.
pub fn execute(
    tasks: &[Task],
    cwd: Option<&str>,
    timeout_seconds: Option<u64>,
) -> Result<Vec<TaskResult>, ScheduleError> {
    execute_with_options(tasks, cwd, timeout_seconds, false, 0.15)
}

/// Execute tasks with optional GoI precedence ordering.
pub fn execute_with_options(
    tasks: &[Task],
    cwd: Option<&str>,
    timeout_seconds: Option<u64>,
    use_goi: bool,
    goi_damping: f64,
) -> Result<Vec<TaskResult>, ScheduleError> {
    let waves = compute_wavefronts_with_options(tasks, use_goi, goi_damping)?;
    let mut results: Vec<TaskResult> = Vec::with_capacity(tasks.len());

    for (wave_idx, wave) in waves.iter().enumerate() {
        eprintln!("[WAVE] Wave {}: {} task(s)", wave_idx, wave.len());

        let wave_results: Vec<(usize, TaskResult)> = std::thread::scope(|scope| {
            let handles: Vec<_> = wave
                .iter()
                .map(|&idx| {
                    scope.spawn(move || {
                        let task = &tasks[idx];
                        eprintln!("[WAVE] Starting task: {} (`{}`)", task.name, task.command);

                        let argv = split_command(&task.command);
                        let mut cmd = if argv.is_empty() {
                            // Degenerate: produce a failing placeholder.
                            Command::new("false")
                        } else {
                            let mut c = Command::new(&argv[0]);
                            c.args(&argv[1..]);
                            c
                        };

                        if let Some(dir) = cwd {
                            cmd.current_dir(dir);
                        }

                        let output = if let Some(_sec) = timeout_seconds {
                            match cmd.output() {
                                Ok(o) if !o.status.success() => o,
                                Ok(o) => o,
                                Err(_) => std::process::Output {
                                    status: std::process::ExitStatus::default(),
                                    stdout: Vec::new(),
                                    stderr: b"timed out or failed to start".to_vec(),
                                },
                            }
                        } else {
                            cmd.output().ok().unwrap_or_else(|| std::process::Output {
                                status: std::process::ExitStatus::default(),
                                stdout: Vec::new(),
                                stderr: b"failed to start".to_vec(),
                            })
                        };

                        (
                            idx,
                            TaskResult {
                                name: task.name.clone(),
                                command: task.command.clone(),
                                exit_code: output.status.code().unwrap_or(-1),
                                stdout: String::from_utf8_lossy(&output.stdout).to_string(),
                                stderr: String::from_utf8_lossy(&output.stderr).to_string(),
                            },
                        )
                    })
                })
                .collect();

            handles
                .into_iter()
                .map(|h| h.join().unwrap())
                .collect()
        });

        for (_idx, result) in wave_results {
            eprintln!(
                "[WAVE] Task {} finished with exit code {}",
                result.name, result.exit_code
            );
            results.push(result);
        }
    }

    Ok(results)
}
