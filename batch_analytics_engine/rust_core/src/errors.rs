use pyo3::exceptions::PyValueError;
use pyo3::PyErr;

/// Return a Python `ValueError` carrying a descriptive message.
pub fn invalid_input<T: Into<String>>(msg: T) -> PyErr {
    PyValueError::new_err(msg.into())
}

/// Return a Python `ValueError` for an invalid window size.
pub fn invalid_window(window: usize) -> PyErr {
    invalid_input(format!("window size must be positive, got {window}"))
}
