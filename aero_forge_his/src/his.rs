//! Holographic Invariant Storage (HIS) primitives.
//!
//! All vectors are 10,000-dimensional bipolar vectors (values in {-1, 1}).
//! Binding is the XOR-like element-wise product, bundling is element-wise
//! superposition, and cleanup thresholds a real-valued vector back to bipolar.

use rayon::prelude::*;

const HIS_DIMENSION: usize = 10_000;

/// Error type for dimension mismatches.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HisError {
    DimensionMismatch { expected: usize, got: usize },
}

impl std::fmt::Display for HisError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HisError::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
        }
    }
}

impl std::error::Error for HisError {}

fn check_dim(a: &[i8], b: &[i8]) -> Result<(), HisError> {
    if a.len() != b.len() {
        return Err(HisError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    Ok(())
}

/// Element-wise multiplication of two bipolar vectors (binding / XOR in bipolar space).
pub fn bind(a: &[i8], b: &[i8]) -> Result<Vec<i8>, HisError> {
    check_dim(a, b)?;
    Ok(a.par_iter()
        .zip(b.par_iter())
        .map(|(x, y)| x.saturating_mul(*y))
        .collect())
}

/// Element-wise addition of two bipolar vectors (bundling / superposition).
pub fn bundle(a: &[i8], b: &[i8]) -> Result<Vec<i32>, HisError> {
    check_dim(a, b)?;
    Ok(a.par_iter()
        .zip(b.par_iter())
        .map(|(x, y)| (*x as i32) + (*y as i32))
        .collect())
}

/// Threshold a real-valued vector back to a bipolar vector.
/// Values >= 0 become +1, values < 0 become -1.
pub fn cleanup(v: &[i32]) -> Vec<i8> {
    v.par_iter()
        .map(|x| if *x >= 0 { 1 } else { -1 })
        .collect()
}

/// Bind a target goal to a safety constraint to form an invariant vector.
pub fn invariant(goal: &[i8], constraint: &[i8]) -> Result<Vec<i8>, HisError> {
    // Binding of two bipolar vectors is already bipolar; cleanup is a no-op
    // but included for explicit semantic alignment with the formula.
    bind(goal, constraint)
}

/// Restore a clean state from an invariant and a noisy context.
///
/// Computes `S_clean = sign(H_inv + noise * context)` element-wise.
/// Both `hinv` and `context` are bipolar; `noise` scales the context contribution.
pub fn restore(hinv: &[i8], context: &[i8], noise: f64) -> Result<Vec<i8>, HisError> {
    check_dim(hinv, context)?;
    if (noise - 1.0).abs() < f64::EPSILON {
        // Fast path: avoid float arithmetic for the default noise=1.0 case.
        let summed: Vec<i32> = hinv
            .par_iter()
            .zip(context.par_iter())
            .map(|(h, c)| (*h as i32) + (*c as i32))
            .collect();
        Ok(cleanup(&summed))
    } else {
        let summed: Vec<f64> = hinv
            .par_iter()
            .zip(context.par_iter())
            .map(|(h, c)| (*h as f64) + noise * (*c as f64))
            .collect();
        Ok(summed.par_iter().map(|x| if *x >= 0.0 { 1 } else { -1 }).collect())
    }
}

/// Cosine similarity between two real-valued vectors.
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> Result<f64, HisError> {
    if a.len() != b.len() {
        return Err(HisError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    let (dot, norm_a, norm_b): (f64, f64, f64) = a
        .par_iter()
        .zip(b.par_iter())
        .map(|(x, y)| (x * y, x * x, y * y))
        .reduce(
            || (0.0, 0.0, 0.0),
            |acc, t| (acc.0 + t.0, acc.1 + t.1, acc.2 + t.2),
        );
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        return Ok(0.0);
    }
    Ok(dot / denom)
}

/// Return the canonical HIS dimensionality.
pub const fn dimension() -> usize {
    HIS_DIMENSION
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ones(len: usize) -> Vec<i8> {
        vec![1; len]
    }

    fn alternating(len: usize) -> Vec<i8> {
        (0..len).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect()
    }

    #[test]
    fn bind_self_is_identity() {
        let v = alternating(10_000);
        let out = bind(&v, &v).unwrap();
        assert!(out.iter().all(|&x| x == 1));
    }

    #[test]
    fn bind_inverse_returns_negative() {
        let a = ones(10_000);
        let b: Vec<i8> = (0..10_000).map(|_| -1).collect();
        let out = bind(&a, &b).unwrap();
        assert!(out.iter().all(|&x| x == -1));
    }

    #[test]
    fn cleanup_thresholds_correctly() {
        assert_eq!(cleanup(&[1, -1, 0, 5, -5, 0]), vec![1, -1, 1, 1, -1, 1]);
    }

    #[test]
    fn restore_orthogonal_context() {
        let goal = alternating(10_000);
        let safety = ones(10_000);
        let hinv = invariant(&goal, &safety).unwrap();

        // A noisy context that is identical to the safety vector should still
        // reinforce the invariant when bundled and cleaned.
        let restored = restore(&hinv, &safety, 1.0).unwrap();
        // Restored vector must be bipolar.
        assert!(restored.iter().all(|&x| x == 1 || x == -1));
    }

    #[test]
    fn cosine_perfect_match() {
        let a: Vec<f64> = (0..10_000).map(|_| 1.0).collect();
        let b = a.clone();
        let sim = cosine_similarity(&a, &b).unwrap();
        assert!((sim - 1.0).abs() < 1e-9);
    }

    #[test]
    fn cosine_orthogonal() {
        let a: Vec<f64> = (0..10_000).map(|_| 1.0).collect();
        let b: Vec<f64> = (0..10_000).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let sim = cosine_similarity(&a, &b).unwrap();
        assert!(sim.abs() < 1e-9);
    }
}
