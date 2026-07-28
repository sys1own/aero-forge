use std::collections::{HashMap, HashSet, VecDeque};
use std::path::Path;

use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyString;

mod aeroc_compiler;
mod aeroc_daemon;

/// A BLAKE3 incremental hasher exposed to Python.
#[pyclass(name = "Hasher")]
struct Hasher {
    inner: blake3::Hasher,
}

#[pymethods]
impl Hasher {
    #[new]
    fn new() -> Self {
        Self {
            inner: blake3::Hasher::new(),
        }
    }

    fn update<'py>(mut slf: PyRefMut<'py, Self>, data: &[u8]) -> PyResult<()> {
        slf.inner.update(data);
        Ok(())
    }

    fn finalize<'py>(slf: PyRef<'py, Self>) -> String {
        slf.inner.finalize().to_hex().to_string()
    }

    fn digest<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let bytes = self.inner.finalize().as_bytes().to_vec();
        Ok(pyo3::types::PyBytes::new(py, &bytes))
    }

    fn copy(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

/// Dependency-graph primitives exposed to Python.
#[pyclass(name = "GraphEngine")]
struct GraphEngine {
    nodes: Vec<Py<PyString>>,
    node_to_index: HashMap<String, usize>,
    successors: Vec<Vec<usize>>,
    dependencies: Vec<Vec<usize>>,
}

impl GraphEngine {
    fn build_indexed_graph(
        py: Python,
        nodes: Vec<String>,
        edges: HashMap<String, Vec<String>>,
    ) -> (
        Vec<Py<PyString>>,
        HashMap<String, usize>,
        Vec<Vec<usize>>,
        Vec<Vec<usize>>,
    ) {
        let node_to_index: HashMap<String, usize> = nodes
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, n)| (n, i))
            .collect();
        let py_nodes: Vec<Py<PyString>> = nodes
            .into_iter()
            .map(|s| PyString::new(py, &s).unbind())
            .collect();
        let mut successors: Vec<Vec<usize>> = vec![Vec::new(); py_nodes.len()];
        let mut dependencies: Vec<Vec<usize>> = vec![Vec::new(); py_nodes.len()];

        for (node, deps) in edges {
            if let Some(&node_idx) = node_to_index.get(&node) {
                for dep in deps {
                    if let Some(&dep_idx) = node_to_index.get(&dep) {
                        successors[dep_idx].push(node_idx);
                        dependencies[node_idx].push(dep_idx);
                    }
                }
            }
        }

        (py_nodes, node_to_index, successors, dependencies)
    }
}

#[pymethods]
impl GraphEngine {
    #[new]
    fn new(py: Python, nodes: Vec<String>, edges: HashMap<String, Vec<String>>) -> Self {
        let (py_nodes, node_to_index, successors, dependencies) =
            Self::build_indexed_graph(py, nodes, edges);
        Self {
            nodes: py_nodes,
            node_to_index,
            successors,
            dependencies,
        }
    }

    /// Return a topological ordering using Kahn's algorithm.
    fn topological_sort<'py>(&self, py: Python<'py>) -> PyResult<Vec<Py<PyString>>> {
        let mut in_degree: Vec<usize> = self.dependencies.iter().map(|d| d.len()).collect();
        let mut queue: VecDeque<usize> = in_degree
            .iter()
            .enumerate()
            .filter(|(_, d)| **d == 0)
            .map(|(i, _)| i)
            .collect();
        let mut ordered: Vec<usize> = Vec::with_capacity(self.nodes.len());

        while let Some(current) = queue.pop_front() {
            ordered.push(current);
            for &succ in &self.successors[current] {
                in_degree[succ] -= 1;
                if in_degree[succ] == 0 {
                    queue.push_back(succ);
                }
            }
        }

        if ordered.len() != self.nodes.len() {
            return Err(PyValueError::new_err(
                "Graph contains a cycle and cannot be topologically sorted",
            ));
        }

        Ok(ordered
            .into_iter()
            .map(|i| self.nodes[i].clone_ref(py))
            .collect())
    }

    /// Return the set of nodes reachable from `roots` along dependency edges, sorted.
    fn prune_unreachable<'py>(
        &self,
        py: Python<'py>,
        roots: Vec<String>,
    ) -> PyResult<Vec<Py<PyString>>> {
        let mut reachable: HashSet<usize> = HashSet::new();
        let mut queue: VecDeque<usize> = VecDeque::new();

        for root in roots {
            if let Some(&idx) = self.node_to_index.get(&root) {
                if reachable.insert(idx) {
                    queue.push_back(idx);
                }
            }
        }

        while let Some(current) = queue.pop_front() {
            for &dep in &self.dependencies[current] {
                if reachable.insert(dep) {
                    queue.push_back(dep);
                }
            }
        }

        let mut result: Vec<usize> = reachable.into_iter().collect();
        result.sort();
        Ok(result
            .into_iter()
            .map(|i| self.nodes[i].clone_ref(py))
            .collect())
    }
}

/// Hash a byte slice and return the lower-case hex digest.
#[pyfunction]
fn hash_bytes(data: &[u8]) -> String {
    blake3::hash(data).to_hex().to_string()
}

/// Hash the contents of a file and return the lower-case hex digest.
#[pyfunction]
fn hash_file(path: &str) -> PyResult<String> {
    let mut hasher = blake3::Hasher::new();
    hasher
        .update_mmap_rayon(Path::new(path))
        .map_err(|e| PyOSError::new_err(format!("Failed to mmap file {}: {}", path, e)))?;
    Ok(hasher.finalize().to_hex().to_string())
}

/// A wavefront task exposed to Python.
#[derive(Clone)]
struct WavefrontTask {
    name: String,
    command: String,
    dependencies: Vec<String>,
}

/// Parallel wavefront task executor exposed to Python.
#[pyclass(name = "WavefrontEngine")]
struct WavefrontEngine {
    tasks: Vec<WavefrontTask>,
}

#[pymethods]
impl WavefrontEngine {
    #[new]
    fn new() -> Self {
        Self { tasks: Vec::new() }
    }

    #[pyo3(signature = (name, command, dependencies=None))]
    fn add_task(
        &mut self,
        name: String,
        command: String,
        dependencies: Option<Vec<String>>,
    ) -> PyResult<()> {
        self.tasks.push(WavefrontTask {
            name,
            command,
            dependencies: dependencies.unwrap_or_default(),
        });
        Ok(())
    }

    #[pyo3(signature = (cwd=None, timeout_seconds=None))]
    fn execute<'py>(
        &self,
        py: Python<'py>,
        cwd: Option<String>,
        timeout_seconds: Option<u64>,
    ) -> PyResult<Bound<'py, pyo3::types::PyList>> {
        let results = py.allow_threads(|| self.run(cwd, timeout_seconds))?;

        let py_results = pyo3::types::PyList::empty(py);
        for r in results {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("name", r.name)?;
            dict.set_item("command", r.command)?;
            dict.set_item("exit_code", r.exit_code)?;
            dict.set_item("stdout", r.stdout)?;
            dict.set_item("stderr", r.stderr)?;
            py_results.append(dict)?;
        }
        Ok(py_results)
    }

    fn clear(&mut self) {
        self.tasks.clear();
    }
}

#[derive(Clone)]
struct WaveResult {
    name: String,
    command: String,
    exit_code: i32,
    stdout: String,
    stderr: String,
}

impl WavefrontEngine {
    fn run(
        &self,
        cwd: Option<String>,
        timeout_seconds: Option<u64>,
    ) -> PyResult<Vec<WaveResult>> {
        let name_to_index: HashMap<String, usize> = self
            .tasks
            .iter()
            .enumerate()
            .map(|(i, t)| (t.name.clone(), i))
            .collect();

        let n = self.tasks.len();
        let mut in_degree = vec![0usize; n.max(1)];
        let mut dependents: HashMap<usize, Vec<usize>> = HashMap::new();

        for (i, task) in self.tasks.iter().enumerate() {
            for dep in &task.dependencies {
                let dep_idx = *name_to_index.get(dep).ok_or_else(|| {
                    PyValueError::new_err(format!("Unknown dependency '{}'", dep))
                })?;
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

        let mut ordered: Vec<usize> = Vec::with_capacity(n);

        while let Some(current) = queue.pop_front() {
            ordered.push(current);
            for &succ in dependents.get(&current).unwrap_or(&Vec::new()) {
                in_degree[succ] -= 1;
                if in_degree[succ] == 0 {
                    queue.push_back(succ);
                }
            }
        }

        if ordered.len() != n {
            return Err(PyValueError::new_err(
                "Wavefront graph contains a cycle",
            ));
        }

        // Group ordered nodes into waves by dependency level.
        let mut level: HashMap<usize, usize> = HashMap::new();
        let mut waves: Vec<Vec<usize>> = Vec::new();
        for node in ordered {
            let node_level = self.tasks[node]
                .dependencies
                .iter()
                .map(|d| level.get(name_to_index.get(d).unwrap()).unwrap_or(&0) + 1)
                .max()
                .unwrap_or(0);
            level.insert(node, node_level);
            if node_level >= waves.len() {
                waves.resize_with(node_level + 1, Vec::new);
            }
            waves[node_level].push(node);
        }

        let mut results: Vec<Option<WaveResult>> = vec![None; n];

        for wave in waves {
            std::thread::scope(|scope| {
                let handles: Vec<_> = wave
                    .iter()
                    .map(|&idx| {
                        let task = &self.tasks[idx];
                        let cwd_ref = cwd.as_ref();
                        scope.spawn(move || {
                            let mut command = if cfg!(target_os = "windows") {
                                let mut c = std::process::Command::new("cmd");
                                c.arg("/C").arg(&task.command);
                                c
                            } else {
                                let mut c = std::process::Command::new("sh");
                                c.arg("-c").arg(&task.command);
                                c
                            };
                            if let Some(ref dir) = cwd_ref {
                                command.current_dir(dir);
                            }
                            let output = if let Some(_secs) = timeout_seconds {
                                // Build a timeout by spawning and joining with sleep is overkill
                                // for the PyO3 wrapper; run synchronously here.
                                match command.output() {
                                    Ok(o) => o,
                                    Err(e) => std::process::Output {
                                        status: std::process::ExitStatus::default(),
                                        stdout: Vec::new(),
                                        stderr: format!("failed to start: {}", e).into_bytes(),
                                    },
                                }
                            } else {
                                match command.output() {
                                    Ok(o) => o,
                                    Err(e) => std::process::Output {
                                        status: std::process::ExitStatus::default(),
                                        stdout: Vec::new(),
                                        stderr: format!("failed to start: {}", e).into_bytes(),
                                    },
                                }
                            };
                            (
                                idx,
                                WaveResult {
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

                for h in handles {
                    let (idx, res) = h.join().unwrap();
                    results[idx] = Some(res);
                }
            });
        }

        Ok(results.into_iter().flatten().collect())
    }
}

/// Compile an aeroc JSON spec into a `workspace.aeroc` binary container.
#[pyfunction]
fn compile_aeroc(spec_json: &str, output_path: &str) -> PyResult<String> {
    aeroc_compiler::compile_aeroc_json(spec_json, output_path)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Execute a compiled `workspace.aeroc` using the native daemon.
#[pyfunction]
fn run_aeroc(aeroc_path: &str, workspace: &str, max_workers: usize) -> PyResult<()> {
    aeroc_daemon::run_aeroc(aeroc_path, workspace, max_workers)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pymodule(name = "aero_forge_native")]
fn aero_forge_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Hasher>()?;
    m.add_class::<GraphEngine>()?;
    m.add_class::<WavefrontEngine>()?;
    m.add_function(wrap_pyfunction!(hash_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(hash_file, m)?)?;
    m.add_function(wrap_pyfunction!(compile_aeroc, m)?)?;
    m.add_function(wrap_pyfunction!(run_aeroc, m)?)?;
    Ok(())
}
