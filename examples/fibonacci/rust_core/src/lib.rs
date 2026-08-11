#![allow(unused_imports)]
use pyo3::prelude::*;
use pyo3::types::PyType;
use std::collections::BTreeMap;
use std::collections::HashSet;


#[allow(unused_variables)]
#[pyfunction(name = "fibonacci")]
fn _accel_fibonacci(n: i64) -> i64 {
    let mut a;
    let mut b;
    if n <= 1_i64 {
        return n;
    }
    a = 0_i64;
    b = 1_i64;
    for _ in 2_i64..(n + 1_i64) {
        let _accel_tmp1 = (b, a + b);
        a = _accel_tmp1.0;
        b = _accel_tmp1.1;
    }
    return b;
}

#[pymodule]
fn aero_forge_generated(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(_accel_fibonacci))?;
    Ok(())
}
