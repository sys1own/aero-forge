#![allow(unused_imports)]
use pyo3::prelude::*;
use pyo3::types::PyType;
use std::collections::BTreeMap;
use std::collections::HashSet;
{shield_imports}

{functions}

#[pymodule]
fn {module_name}(_py: Python, m: &PyModule) -> PyResult<()> {{
{module_init}
    Ok(())
}}
