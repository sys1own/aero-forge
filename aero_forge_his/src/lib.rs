use pyo3::prelude::*;

mod his;

/// Create a 10,000-dimensional bipolar vector where every element is +1.
#[pyfunction]
fn ones() -> Vec<i8> {
    vec![1; his::dimension()]
}

/// Create a 10,000-dimensional random bipolar vector (seeded for reproducibility).
#[pyfunction]
fn random_bipolar(seed: u64) -> Vec<i8> {
    let mut state = seed;
    (0..his::dimension())
        .map(|_| {
            // Simple xorshift64* PRNG.
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let rand = state.wrapping_mul(0x2545_f491_4f6c_dd1d);
            if rand >> 63 == 0 { 1 } else { -1 }
        })
        .collect()
}

/// Bind (⊗) two bipolar vectors: element-wise multiplication.
#[pyfunction]
fn bind(a: Vec<i8>, b: Vec<i8>) -> PyResult<Vec<i8>> {
    his::bind(&a, &b)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// Bundle (+) two bipolar vectors: element-wise addition (returns i32 counts).
#[pyfunction]
fn bundle(a: Vec<i8>, b: Vec<i8>) -> PyResult<Vec<i32>> {
    his::bundle(&a, &b)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// Cleanup (sign): threshold a real-valued vector back to {-1, 1}.
#[pyfunction]
fn cleanup(v: Vec<i32>) -> Vec<i8> {
    his::cleanup(&v)
}

/// Create an invariant vector by binding a target goal to a safety constraint.
#[pyfunction]
fn invariant(goal: Vec<i8>, constraint: Vec<i8>) -> PyResult<Vec<i8>> {
    his::invariant(&goal, &constraint)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// Restore a clean bipolar vector from an invariant and a noisy context.
/// Computes `sign(H_inv + noise * context)` element-wise.
#[pyfunction]
#[pyo3(signature = (hinv, context, noise=1.0))]
fn restore(hinv: Vec<i8>, context: Vec<i8>, noise: f64) -> PyResult<Vec<i8>> {
    his::restore(&hinv, &context, noise)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// Cosine similarity between two real-valued vectors.
#[pyfunction]
fn cosine_similarity(a: Vec<f64>, b: Vec<f64>) -> PyResult<f64> {
    his::cosine_similarity(&a, &b)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// Return the canonical HIS dimensionality (10,000).
#[pyfunction]
fn dimension() -> usize {
    his::dimension()
}

#[pymodule(name = "aero_forge_his")]
fn aero_forge_his(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ones, m)?)?;
    m.add_function(wrap_pyfunction!(random_bipolar, m)?)?;
    m.add_function(wrap_pyfunction!(bind, m)?)?;
    m.add_function(wrap_pyfunction!(bundle, m)?)?;
    m.add_function(wrap_pyfunction!(cleanup, m)?)?;
    m.add_function(wrap_pyfunction!(invariant, m)?)?;
    m.add_function(wrap_pyfunction!(restore, m)?)?;
    m.add_function(wrap_pyfunction!(cosine_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(dimension, m)?)?;
    Ok(())
}
