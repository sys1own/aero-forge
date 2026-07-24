# batch-analytics-engine

A hybrid Python/Rust batch analytics engine.

- **Rust core** (`rust_core/`): high-throughput streaming aggregation, mean/std
  computation, and Z-score outlier detection, exposed through PyO3.
- **Python engine** (`python_engine/`): async HTTP service, client, validation,
  tests, and benchmarks.

## Build

```bash
cd batch_analytics_engine
maturin develop --release
```

## Test

```bash
cd batch_analytics_engine
cargo test --release
cargo clippy --release -- -D warnings
pytest -q
```

## Run the service

```bash
cd batch_analytics_engine
python scripts/run_service.py
```

POST to `http://localhost:8080/aggregate` with a JSON body:

```json
{
  "records": [
    {"id": "a", "value": 1.0, "timestamp": 0.0},
    {"id": "b", "value": 2.0, "timestamp": 1.0}
  ],
  "window": 2,
  "threshold": 1.5
}
```
