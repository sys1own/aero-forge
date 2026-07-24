use crate::errors::invalid_window;
use crate::record::{AggregateResult, Record};
use pyo3::prelude::*;

/// Compute per-window mean and population standard deviation for a batch of records.
///
/// Windows are non-overlapping and start at index 0. The last window may be smaller
/// than `window` if the record count is not an exact multiple.
#[pyfunction]
pub fn aggregate_batch(records: Vec<Record>, window: usize) -> PyResult<Vec<AggregateResult>> {
    if window == 0 {
        return Err(invalid_window(window));
    }
    if records.is_empty() {
        return Ok(Vec::new());
    }

    let mut results = Vec::new();
    for (idx, chunk) in records.chunks(window).enumerate() {
        let start = idx * window;
        let n = chunk.len() as f64;
        if n == 0.0 {
            continue;
        }

        let sum: f64 = chunk.iter().map(|r| r.value).sum();
        let mean = sum / n;
        let variance = chunk.iter().map(|r| (r.value - mean).powi(2)).sum::<f64>() / n;
        let std = variance.sqrt();

        results.push(AggregateResult::new(start, mean, std, Vec::new()));
    }

    Ok(results)
}

/// Validate that `window` is a positive integer.
///
/// This is a thin helper exposed so Python-side services can re-use the same
/// validation rule without duplicating logic.
#[pyfunction]
pub fn validate_window(window: usize) -> PyResult<()> {
    if window == 0 {
        return Err(invalid_window(window));
    }
    Ok(())
}
