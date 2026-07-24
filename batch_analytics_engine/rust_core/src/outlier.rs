use crate::record::AggregateResult;
use pyo3::prelude::*;

/// Detect outlier windows by Z-score over the sequence of per-window means.
///
/// Returns the indices of windows whose mean is more than `threshold` standard
/// deviations away from the global mean of window means. Empty input or zero
/// variance yields an empty outlier list.
#[pyfunction]
pub fn detect_outliers(results: Vec<AggregateResult>, threshold: f64) -> PyResult<Vec<usize>> {
    if results.is_empty() {
        return Ok(Vec::new());
    }

    let n = results.len() as f64;
    let mean = results.iter().map(|r| r.mean).sum::<f64>() / n;
    let variance = results
        .iter()
        .map(|r| (r.mean - mean).powi(2))
        .sum::<f64>()
        / n;
    let std = variance.sqrt();

    if std == 0.0 || threshold <= 0.0 {
        return Ok(Vec::new());
    }

    let mut outliers = Vec::new();
    for (i, result) in results.iter().enumerate() {
        let z = ((result.mean - mean) / std).abs();
        if z > threshold {
            outliers.push(i);
        }
    }

    Ok(outliers)
}
