use pyo3::prelude::*;

/// A single numeric record in the stream.
#[pyclass]
#[derive(Clone, Debug)]
pub struct Record {
    #[pyo3(get, set)]
    pub id: String,
    #[pyo3(get, set)]
    pub value: f64,
    #[pyo3(get, set)]
    pub timestamp: f64,
}

#[pymethods]
impl Record {
    #[new]
    pub fn new(id: String, value: f64, timestamp: f64) -> Self {
        Self {
            id,
            value,
            timestamp,
        }
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "Record(id={id:?}, value={value}, timestamp={timestamp})",
            id = self.id,
            value = self.value,
            timestamp = self.timestamp
        ))
    }
}

/// Aggregation statistics for a fixed-size window of records.
#[pyclass]
#[derive(Clone, Debug)]
pub struct AggregateResult {
    #[pyo3(get, set)]
    pub window_start: usize,
    #[pyo3(get, set)]
    pub mean: f64,
    #[pyo3(get, set)]
    pub std: f64,
    #[pyo3(get, set)]
    pub outliers: Vec<usize>,
}

#[pymethods]
impl AggregateResult {
    #[new]
    pub fn new(window_start: usize, mean: f64, std: f64, outliers: Vec<usize>) -> Self {
        Self {
            window_start,
            mean,
            std,
            outliers,
        }
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "AggregateResult(window_start={ws}, mean={mean}, std={std}, outliers={outliers:?})",
            ws = self.window_start,
            mean = self.mean,
            std = self.std,
            outliers = self.outliers
        ))
    }
}
