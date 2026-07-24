use pyo3::prelude::*;
#[allow(unused_imports)]
use pyo3::types::PyType;
#[allow(unused_imports)]
use std::collections::HashMap;
{shield_imports}

{functions}

#[pymodule]
fn {module_name}(_py: Python, m: &PyModule) -> PyResult<()> {{
{module_init}
    Ok(())
}}
