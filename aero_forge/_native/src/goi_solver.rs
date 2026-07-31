//! Geometry of Interaction (GoI) matrix repair-isolation solver.
//!
//! Encodes a DAG dependency graph as routing matrix ``U`` and adjacency matrix
//! ``M``.  Given a base matrix pair and a perturbation ``ΔM`` the module
//! evaluates whether the spectral radius of ``U (M + ΔM)`` stays strictly
//! below the unit boundary, and computes the support set of the perturbation.

use std::collections::HashSet;

use ndarray::{Array1, Array2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Bundle of GoI matrices for a single workspace snapshot.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GoIMatrixData {
    pub u: Vec<Vec<f64>>,
    pub m: Vec<Vec<f64>>,
    pub size: usize,
}

/// Result of a spectral-radius repair-isolation check.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RepairRadiusResult {
    pub radius: f64,
    pub bound: f64,
    pub isolated: bool,
    pub support_rows: Vec<usize>,
    pub support_cols: Vec<usize>,
}

fn matrix_from_json(v: &Value) -> PyResult<Array2<f64>> {
    let rows = v
        .as_array()
        .ok_or_else(|| PyValueError::new_err("matrix JSON is not an array"))?
        .iter()
        .map(|row| {
            row.as_array()
                .ok_or_else(|| PyValueError::new_err("matrix row is not an array"))?
                .iter()
                .map(|x| {
                    x.as_f64()
                        .ok_or_else(|| PyValueError::new_err("matrix entry is not a number"))
                })
                .collect::<PyResult<Vec<f64>>>()
        })
        .collect::<PyResult<Vec<Vec<f64>>>>()?;
    if rows.is_empty() {
        return Err(PyValueError::new_err("matrix is empty"));
    }
    let n = rows.len();
    let m = rows[0].len();
    let flat: Vec<f64> = rows.into_iter().flatten().collect();
    Array2::from_shape_vec((n, m), flat).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Estimate the dominant eigenvalue magnitude (spectral radius) of a square
/// matrix using power iteration.  Returns ``0.0`` for zero matrices.
fn spectral_radius(a: &Array2<f64>) -> f64 {
    let (n, m) = a.dim();
    if n == 0 || m == 0 {
        return 0.0;
    }
    let mut b = Array1::from_vec(vec![1.0; n]);
    let mut prev = 0.0;
    for _ in 0..80 {
        let mut v = a.dot(&b);
        let norm = v.iter().map(|x| x.abs()).fold(0.0, f64::max);
        if norm < 1e-15 {
            return 0.0;
        }
        v /= norm;
        b = v;
        if (norm - prev).abs() < 1e-12 {
            return norm;
        }
        prev = norm;
    }
    prev
}

/// Compute the row and column support sets of ``ΔM`` (non-zero entries).
fn perturbation_support(delta: &Array2<f64>) -> (Vec<usize>, Vec<usize>) {
    let mut rows = HashSet::new();
    let mut cols = HashSet::new();
    for ((r, c), v) in delta.indexed_iter() {
        if *v != 0.0 {
            rows.insert(r);
            cols.insert(c);
        }
    }
    let mut row_vec: Vec<usize> = rows.into_iter().collect();
    let mut col_vec: Vec<usize> = cols.into_iter().collect();
    row_vec.sort_unstable();
    col_vec.sort_unstable();
    (row_vec, col_vec)
}

/// Evaluate whether a perturbation ``ΔM`` keeps the GoI matrix ``U (M + ΔM)``
/// inside the unit spectral-radius boundary.
///
/// ``base_matrix_json`` may be a JSON object ``{"U": ..., "M": ...}`` or just
/// a matrix ``M`` (in which case ``U`` is taken as the identity).
/// ``delta_matrix_json`` is a matrix ``ΔM``.
#[pyfunction]
pub fn enforce_repair_isolation_py(
    base_matrix_json: &str,
    delta_matrix_json: &str,
) -> PyResult<String> {
    let base: Value = serde_json::from_str(base_matrix_json)
        .map_err(|e| PyValueError::new_err(format!("invalid base matrix JSON: {}", e)))?;
    let delta: Value = serde_json::from_str(delta_matrix_json)
        .map_err(|e| PyValueError::new_err(format!("invalid delta matrix JSON: {}", e)))?;

    let (u_mat, m_mat) = if let Some(u) = base.get("U").or_else(|| base.get("u")) {
        let m = base
            .get("M")
            .or_else(|| base.get("m"))
            .ok_or_else(|| PyValueError::new_err("base object missing M matrix"))?;
        (matrix_from_json(u)?, matrix_from_json(m)?)
    } else {
        let m = matrix_from_json(&base)?;
        let n = m.dim().0;
        let u = Array2::eye(n);
        (u, m)
    };

    let delta_mat = matrix_from_json(&delta)?;

    if u_mat.dim() != m_mat.dim() {
        return Err(PyValueError::new_err(
            "U and M must have the same dimensions",
        ));
    }
    if m_mat.dim() != delta_mat.dim() {
        return Err(PyValueError::new_err(
            "base M and delta M must have the same dimensions",
        ));
    }

    let m_new = &m_mat + &delta_mat;
    let product = u_mat.dot(&m_new);
    let radius = spectral_radius(&product);
    let bound = 1.0;
    let isolated = radius < bound;
    let (support_rows, support_cols) = perturbation_support(&delta_mat);

    let result = RepairRadiusResult {
        radius,
        bound,
        isolated,
        support_rows,
        support_cols,
    };
    serde_json::to_string(&result)
        .map_err(|e| PyValueError::new_err(format!("repair result serialization failed: {}", e)))
}
