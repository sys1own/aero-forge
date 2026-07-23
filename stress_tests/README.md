# Aero-Forge Stress Test Campaign

This directory contains a growing suite of real-world Python patterns used to
push the boundaries of the Aero-Forge transpiler. Tests are run via
`pytest stress_tests/test_stress.py` (or `python -m pytest` from the repo root).

## Running the suite

```bash
python -m pytest stress_tests/test_stress.py -v
```

Each test invokes `aero-forge build` on a level-specific blueprint and checks the
result. Passing tests compile successfully; unsupported patterns are verified to
fail with a clear `UnsupportedError` message rather than crashing.

## Level results

| Level | Theme | Status | Notes |
|-------|-------|--------|-------|
| 1 | Mathematical & numerical | **Pass** | `factorial`, `power`, `is_prime`, `mandelbrot` compile and pass tests. `matrix_multiply` is unsupported (lists/append/len). |
| 2 | Data structures & collections | **Partial** | Tuple unpacking and `min`/`max` multi-arg work. Lists, dicts, slicing, `sum`, `enumerate`, `zip` raise `UnsupportedError`. |
| 3 | Object-oriented | **Partial** | Simple classes with `__init__`, instance methods, `@staticmethod`, and `@classmethod` compile. Dataclasses, inheritance, and properties are still unsupported. |
| 4 | Control flow | **Pass** | `break`/`continue` in `for`/`while` loops, recursion, nested `if`/`elif`/`else` work. |
| 5 | Standard library | **Partial** | `math` module functions and constants work. `random`, `datetime`, `re`, `json` are unsupported. |
| 6 | Cross-file & multi-module | **Pass** | Multiple source files build in parallel with per-source tests. |
| 7 | LLM healing | **Partial / API-dependent** | `test_no_llm_graceful_failure` passes. Real healing verified with `openrouter/free` for syntax, type, and multi-function broken files when a valid key is available. Rate-limit/backoff and API-fallback behavior is covered by mocked unit tests in `stress_tests/test_llm_rate_limits.py`. Gemini free-tier keys are often quota-exhausted and will xfail/skip. |
| 8 | Performance & scale | **Pass** | 50 functions from one file compile in ~1 second. |
| 9 | Blueprint edge cases | **Pass** | Missing files, missing function names, and name/compile_all combinations produce clear messages. |
| 10 | Classes & methods | **Pass** | `Counter` and `Calculator` classes compile, including instance methods, read-only methods, `@staticmethod`, and `@classmethod`. |
| 11 | Class `Vec` attributes | **Pass** | `Matrix` class with `rows`, `cols`, and `data: list[list[float]]` compiles; `multiply`, `transpose`, and `get` methods pass. |
| 12 | NumPy-style 1D vectors | **Pass** | `np.array`, `np.zeros`, `np.ones`, `np.dot` (1D and 2D), `np.sum`, and elementwise `arr * 2 + 1` compile and pass. |
| 13 | Incremental builds | **Pass** | `--force`, `--cache-dir`, `AERO_FORGE_CACHE_DIR`, and `rustc` version keys tested. |
| 14 | GPU scaffolding | **Partial** | `--gpu` and `# @accelerate gpu` detection work; CPU fallback when `nvcc` is unavailable. |
| 15 | WASM target | **Pass** | `--target wasm32-unknown-unknown` compiles scalar functions to `.wasm` and a `.js` loader; verified with Node.js. |
| 16 | Distributed builds | **Partial** | `--distribute` compiles multiple source files in parallel worker processes locally; cross-machine worker pool is future work. |

## Adding a new stress test

1. Create a directory `stress_tests/levelX_<name>/`.
2. Add a `blueprint.aero`, source `.py`, and `tests/test_*.py`.
3. Import only the module that the current source file becomes (the file stem).
4. Add a test class in `stress_tests/test_stress.py`.
5. For unsupported patterns, assert `returncode != 0` and check for the word
   `Unsupported` in the output.

## Known limitations

See the main `README.md` for the authoritative list of supported and unsupported
Python constructs.
