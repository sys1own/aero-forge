# Aero-Forge Examples Gallery

This directory contains curated, ready-to-build examples. Each subdirectory has a
`blueprint.aero` file, `src/generated.py`, and `tests/test_generated.py`.

The numeric examples are written in the high-performance style the engine
expects: explicit `for i in range(n)` loops, index-based access, simple type
hints, no list comprehensions, and no dynamic containers. Wrap any of them with
`@accelerate(target="rust_hin")` at runtime to compile to a native Rust
extension via the Holographic Interaction Net (HIN) pipeline.

## Available Examples

| Example | Description |
|---------|-------------|
| `fibonacci` | Iterative Fibonacci |
| `factorial` | Iterative factorial |
| `gcd` | Euclid's greatest common divisor |
| `is_prime` | Trial-division prime test |
| `power` | Float exponentiation by repeated multiplication |
| `mandelbrot` | Mandelbrot escape-time count |
| `sum_of_squares` | Sum of i^2 for i in [0, n] |
| `sum_even` | Sum of even numbers up to n |
| `counter` | Simple PyO3 class with methods |
| `vector_dot` | NumPy vector dot product |

## Run an Example

```bash
aero-forge examples list
aero-forge examples run fibonacci
aero-forge examples create new_example --prompt "Write a fast prime sieve"
```

To request a new language target, include it in the prompt; the Universal Plugin
Synthesis pipeline will synthesize a `PolyglotEmitterPlugin` and FFI contract
on demand.
