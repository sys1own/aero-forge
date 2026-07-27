use std::collections::{HashMap, HashSet, VecDeque};
use std::path::Path;

use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyString;

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

#[pymodule(name = "aero_forge_native")]
fn aero_forge_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Hasher>()?;
    m.add_class::<GraphEngine>()?;
    m.add_function(wrap_pyfunction!(hash_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(hash_file, m)?)?;
    Ok(())
}
