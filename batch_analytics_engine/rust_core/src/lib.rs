mod aggregator;
mod errors;
mod outlier;
mod record;

use pyo3::prelude::*;

/// The native `_native` submodule inside the `batch_analytics` Python package.
#[pymodule(name = "_native")]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<record::Record>()?;
    m.add_class::<record::AggregateResult>()?;
    m.add_function(wrap_pyfunction!(aggregator::aggregate_batch, m)?)?;
    m.add_function(wrap_pyfunction!(aggregator::validate_window, m)?)?;
    m.add_function(wrap_pyfunction!(outlier::detect_outliers, m)?)?;
    Ok(())
}
