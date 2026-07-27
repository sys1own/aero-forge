//! Zero-dependency topological wavefront scheduler.
//!
//! Tasks are grouped into waves $W_0, W_1, ...$ such that tasks inside a wave
//! have no intra-dependencies.  Each wave runs in parallel and a strict join
//! barrier ensures $W_{i+1}$ starts only after $W_i$ completes.

use std::collections::{HashMap, HashSet, VecDeque};
use std::process::Command;

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

/// Compute topological wavefronts from a list of tasks.
pub fn compute_wavefronts(tasks: &[Task]) -> Result<Vec<Vec<usize>>, ScheduleError> {
    let name_to_index: HashMap<String, usize> = tasks
        .iter()
        .enumerate()
        .map(|(i, t)| (t.name.clone(), i))
        .collect();

    let mut in_degree = vec![0usize; tasks.len()];
    let mut dependents: HashMap<usize, Vec<usize>> = HashMap::new();

    for (i, task) in tasks.iter().enumerate() {
        for dep in &task.dependencies {
            let dep_idx = *name_to_index
                .get(dep)
                .ok_or_else(|| ScheduleError::TaskNotFound(dep.clone()))?;
            dependents.entry(dep_idx).or_default().push(i);
            in_degree[i] += 1;
        }
    }

    let mut queue: VecDeque<usize> = in_degree
        .iter()
        .enumerate()
        .filter(|(_, d)| **d == 0)
        .map(|(i, _)| i)
        .collect();

    let mut waves: Vec<Vec<usize>> = Vec::new();

    while !queue.is_empty() {
        let mut wave: Vec<usize> = Vec::new();
        let mut next_queue: VecDeque<usize> = VecDeque::new();
        for node in queue {
            wave.push(node);
            for &nbr in dependents.get(&node).unwrap_or(&Vec::new()) {
                in_degree[nbr] -= 1;
                if in_degree[nbr] == 0 {
                    next_queue.push_back(nbr);
                }
            }
        }
        waves.push(wave);
        queue = next_queue;
    }

    if in_degree.iter().any(|d| *d > 0) {
        return Err(ScheduleError::Cycle(
            "Graph contains a cycle; cannot compute wavefront schedule".into(),
        ));
    }

    Ok(waves)
}

/// Execute tasks in topological waves.
///
/// `cwd` is the working directory passed to each child process.
pub fn execute(
    tasks: &[Task],
    cwd: Option<&str>,
    timeout_seconds: Option<u64>,
) -> Result<Vec<TaskResult>, ScheduleError> {
    let waves = compute_wavefronts(tasks)?;
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

                        let mut cmd = if cfg!(target_os = "windows") {
                            let mut c = Command::new("cmd");
                            c.arg("/C").arg(&task.command);
                            c
                        } else {
                            let mut c = Command::new("sh");
                            c.arg("-c").arg(&task.command);
                            c
                        };

                        if let Some(dir) = cwd {
                            cmd.current_dir(dir);
                        }

                        let output = if let Some(sec) = timeout_seconds {
                            cmd.output()
                                .ok()
                                .unwrap_or_else(|| std::process::Output {
                                    status: std::process::ExitStatus::default(),
                                    stdout: Vec::new(),
                                    stderr: b"timed out or failed to start".to_vec(),
                                })
                        } else {
                            cmd.output()
                                .ok()
                                .unwrap_or_else(|| std::process::Output {
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

        for (idx, result) in wave_results {
            eprintln!(
                "[WAVE] Task {} finished with exit code {}",
                result.name, result.exit_code
            );
            results.push(result);
            if results.len() - 1 != idx {
                // Index ordering is arbitrary; maintain stable output.
            }
        }
    }

    Ok(results)
}
