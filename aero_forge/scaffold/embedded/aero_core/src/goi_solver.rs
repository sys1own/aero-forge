//! Geometry of Interaction (GoI) matrix solver.
//!
//! Computes Girard's formula
//!
//! ```text
//!     EX(M, U) = (I - U * M)^(-1) * U
//! ```
//!
//! and analytical gradients of the routing rule matrix U.  This implementation
//! is intentionally dependency-free so it can be embedded in standalone
//! `.aeroc` packages.

/// Row-major dense matrix.
#[derive(Clone, Debug)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    /// Zero matrix.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    /// Identity matrix of size n x n.
    pub fn eye(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.set(i, i, 1.0);
        }
        m
    }

    /// Build from a 2D slice (row-major).
    pub fn from_2d(values: &[Vec<f64>]) -> Self {
        if values.is_empty() {
            return Self::zeros(0, 0);
        }
        let rows = values.len();
        let cols = values[0].len();
        let mut data = Vec::with_capacity(rows * cols);
        for row in values {
            data.extend(row.iter().copied());
        }
        Self { rows, cols, data }
    }

    /// Build an n x n matrix from an adjacency list of task dependencies.
    /// `deps[i]` contains the indices of tasks that must complete before i.
    pub fn from_adjacency(n: usize, deps: &[Vec<usize>]) -> Self {
        let mut m = Self::zeros(n, n);
        for (i, d) in deps.iter().enumerate() {
            for &j in d {
                if j < n {
                    m.set(i, j, 1.0);
                }
            }
        }
        m
    }

    #[inline]
    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.data[row * self.cols + col]
    }

    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: f64) {
        self.data[row * self.cols + col] = value;
    }

    /// C = self * other
    pub fn dot(&self, other: &Self) -> Self {
        assert_eq!(self.cols, other.rows, "incompatible matrix dimensions");
        let mut out = Self::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for k in 0..self.cols {
                let a = self.get(i, k);
                if a == 0.0 {
                    continue;
                }
                for j in 0..other.cols {
                    out.data[i * out.cols + j] += a * other.get(k, j);
                }
            }
        }
        out
    }

    /// self - other
    pub fn sub(&self, other: &Self) -> Self {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let mut out = self.clone();
        for i in 0..out.data.len() {
            out.data[i] -= other.data[i];
        }
        out
    }

    /// self + other
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let mut out = self.clone();
        for i in 0..out.data.len() {
            out.data[i] += other.data[i];
        }
        out
    }

    /// Transpose.
    pub fn transpose(&self) -> Self {
        let mut out = Self::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                out.set(j, i, self.get(i, j));
            }
        }
        out
    }

    /// In-place Gauss-Jordan elimination with partial pivoting.
    pub fn inverse(&self) -> Result<Self, &'static str> {
        if self.rows != self.cols {
            return Err("matrix must be square");
        }
        let n = self.rows;
        let mut a = self.clone();
        let mut inv = Self::eye(n);

        for col in 0..n {
            // Partial pivoting.
            let mut pivot = col;
            let mut max = a.get(col, col).abs();
            for row in (col + 1)..n {
                let val = a.get(row, col).abs();
                if val > max {
                    max = val;
                    pivot = row;
                }
            }
            if max < 1e-15 {
                return Err("matrix is singular or near-singular");
            }
            if pivot != col {
                let start_a = col * n;
                let start_b = pivot * n;
                let (lo, hi) = if start_a < start_b {
                    (start_a, start_b)
                } else {
                    (start_b, start_a)
                };
                let (left, right) = a.data.split_at_mut(hi);
                left[lo..lo + n].swap_with_slice(&mut right[0..n]);
                let (left_inv, right_inv) = inv.data.split_at_mut(hi);
                left_inv[lo..lo + n].swap_with_slice(&mut right_inv[0..n]);
            }

            // Normalize pivot row.
            let pivot_val = a.get(col, col);
            let inv_pivot = 1.0 / pivot_val;
            for j in 0..n {
                a.set(col, j, a.get(col, j) * inv_pivot);
                inv.set(col, j, inv.get(col, j) * inv_pivot);
            }

            // Eliminate column.
            for row in 0..n {
                if row == col {
                    continue;
                }
                let factor = a.get(row, col);
                if factor == 0.0 {
                    continue;
                }
                for j in 0..n {
                    let a_val = a.get(row, j) - factor * a.get(col, j);
                    let i_val = inv.get(row, j) - factor * inv.get(col, j);
                    a.set(row, j, a_val);
                    inv.set(row, j, i_val);
                }
            }
        }

        Ok(inv)
    }

    /// Row-wise L2 norm for each row.
    pub fn row_norms(&self) -> Vec<f64> {
        let mut norms = Vec::with_capacity(self.rows);
        for i in 0..self.rows {
            let mut sum = 0.0;
            for j in 0..self.cols {
                sum += self.get(i, j).powi(2);
            }
            norms.push(sum.sqrt());
        }
        norms
    }
}

/// Compute EX(M, U) = (I - U * M)^(-1) * U.
pub fn execute_goi_wave(m: &Matrix, u: &Matrix) -> Result<Matrix, &'static str> {
    if m.rows != m.cols || u.rows != u.cols || m.rows != u.rows {
        return Err("M and U must be square matrices of the same size");
    }
    let n = m.rows;
    let i = Matrix::eye(n);
    let inv_term = i.sub(&u.dot(m)).inverse()?;
    Ok(inv_term.dot(u))
}

/// Compute the analytical gradient dL/dU for the GoI formula.
///
/// X = (I - U * M)^(-1)
/// EX_current = X * U
/// dL/dU = X.T * loss_grad_out * (I + M * EX_current).T
pub fn compute_metamorphic_gradients(
    m: &Matrix,
    u: &Matrix,
    loss_grad_out: &Matrix,
) -> Result<Matrix, &'static str> {
    if m.rows != m.cols || u.rows != u.cols || m.rows != u.rows {
        return Err("M, U and loss_grad_out must be square and same size");
    }
    let n = m.rows;
    let i = Matrix::eye(n);
    let x = i.sub(&u.dot(m)).inverse()?;
    let ex = x.dot(u);
    let right = i.add(&m.dot(&ex)).transpose();
    let left = x.transpose().dot(loss_grad_out);
    Ok(left.dot(&right))
}

/// Compute GoI-derived precedence scores from a task dependency list.
/// Returns a vector of scores parallel to the input tasks.
pub fn precedence_scores(deps: &[Vec<usize>], damping: f64) -> Result<Vec<f64>, &'static str> {
    let n = deps.len();
    if n == 0 {
        return Ok(vec![]);
    }
    let m = Matrix::from_adjacency(n, deps);
    let mut u = Matrix::eye(n);
    for i in 0..n {
        u.set(i, i, 1.0 - damping.clamp(0.0, 1.0));
    }
    let ex = execute_goi_wave(&m, &u)?;
    Ok(ex.row_norms())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_inverse_identity() {
        let m = Matrix::from_2d(&vec![
            vec![2.0, 0.0, 0.0],
            vec![0.0, 2.0, 0.0],
            vec![0.0, 0.0, 2.0],
        ]);
        let inv = m.inverse().unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 0.5 } else { 0.0 };
                assert!((inv.get(i, j) - expected).abs() < 1e-12, "inv[{},{}] = {} != {}", i, j, inv.get(i, j), expected);
            }
        }
    }

    #[test]
    fn goi_execute_identity() {
        let m = Matrix::eye(3);
        let u = Matrix::from_2d(&vec![
            vec![0.5, 0.0, 0.0],
            vec![0.0, 0.5, 0.0],
            vec![0.0, 0.0, 0.5],
        ]);
        let ex = execute_goi_wave(&m, &u).unwrap();
        // (I - 0.5 I)^-1 * 0.5 I = I
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((ex.get(i, j) - expected).abs() < 1e-12, "ex[{},{}] = {} != {}", i, j, ex.get(i, j), expected);
            }
        }
    }

    #[test]
    fn precedence_chain_highest_last() {
        // 0 -> 1 -> 2
        let deps = vec![vec![], vec![0], vec![1]];
        let scores = precedence_scores(&deps, 0.15).unwrap();
        assert!(scores[2] > scores[1]);
        assert!(scores[1] > scores[0]);
    }
}
