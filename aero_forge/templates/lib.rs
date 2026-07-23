use pyo3::prelude::*;
{shield_imports}

{functions}

#[pymodule]
fn {module_name}(_py: Python, m: &PyModule) -> PyResult<()> {{
{module_init}
    Ok(())
}}
