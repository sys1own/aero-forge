# Prompt Engineering Report

This report summarizes a live campaign that ran seven prompt templates against
four reference test cases using the DeepSeek API.  The goal was to find a
system prompt that produces **first-attempt compiles ≥ 80 %** and **< 3 average
iterations**.

> **Architecture note:** Prompt engineering is part of the *upstream intent*
> layer. LLMs are used to interpret the user's natural language request and
> emit Python source/tests. Once generated, the transpile → compile → test →
> heal pipeline is fully deterministic and never calls an LLM.

## Campaign Setup

- Provider: `deepseek` (`https://api.deepseek.com/v1`)
- Model: default (`deepseek-chat`)
- Test cases (4): `fibonacci`, `factorial`, `gcd`, `is_prime`
- Templates evaluated: `v2_structured` through `v8_iterative`
- `v1_minimal` was omitted from the live run because it lacks the transpiler
  constraints required by Aero-Forge and is expected to fail; it is kept in the
  harness as a baseline.

## Results

| Template | Success Rate | First-Attempt Rate | Avg Iterations | Avg Build Time (s) |
|----------|--------------|--------------------|----------------|--------------------|
| v2_structured | 100 % | 100 % | 1.00 | 5.35 |
| v3_algorithm | 75 % | 75 % | 2.00 | 3.17 |
| v4_performance | 100 % | 100 % | 1.00 | 2.04 |
| v5_balanced | 100 % | 100 % | 1.00 | 2.93 |
| v6_creative | 100 % | 100 % | 1.00 | 2.44 |
| v7_conservative | 100 % | 100 % | 1.00 | 2.64 |
| v8_iterative | 100 % | 100 % | 1.00 | 2.27 |

### Notes

- `v3_algorithm` failed on `is_prime` because the generated test contained a
  logic error (`assert not is_prime(2) == True`), not because the implementation
  was wrong.  This indicates that the algorithm-focused prompt can over-fit to
  terse outputs and produce fragile test cases.
- `v4_performance` produced the fastest average build time while still hitting
  100 % success, likely because it encourages flat numeric kernels that are
  easy for the scaffold engine to transpile.
- `v5_balanced` and `v8_iterative` both achieved 100 % success with low latency
  and are good general-purpose defaults.

## Selected Default

`v5_balanced` is used as the default prompt template for `aero-forge generate`,
`aero-forge build` (with a `prompt` field), and `aero-forge chat` because it
combines the structured output rules of `v2`, the algorithmic focus of `v3`, and
the performance guidance of `v4`.

## How to Reproduce

```bash
export DEEPSEEK_API_KEY=...
export AERO_FORGE_LLM_PROVIDER=deepseek
python -m aero_forge.prompt_engineering
```

The script writes `prompt_engineering_report.json` with per-case metrics.

## Prompt System Improvements Added

- Added **DeepSeek v4 / DeepSeek API** provider (`deepseek`) in
  `aero_forge/llm/clients.py`.
- Added **ten prompt templates** in `aero_forge/prompts.py` and a
  `--prompt-template` CLI option, including `v9_transpiler_friendly` and
  `v10_correctness_focused`.
- Added **smoke-test generation** as a fallback when the LLM does not produce
  tests.
- Added **deterministic router-level sanitization** that strips unsupported `raise` and
  `assert` statements before transpilation. This is a static AST transform, not an LLM call.
- Added `aero_forge/prompt_engineering.py` campaign harness and
  `tests/test_prompt_engineering.py`.

## Explanations and Reviews

Newer prompts support structured explanation output:

- `--explain` appends a request for an `## Explanation` section describing
  algorithm choice, complexity, and tradeoffs.  The response is parsed by
  `extract_explanation()` and displayed to the user.
- `--review` adds a second LLM pass with a senior-engineer system prompt that
  checks correctness, performance, security, and style, then outputs a corrected
  implementation in a single fenced code block.

## Recommendations

1. Use `v5_balanced` for general prompt-to-build workflows.
2. Use `v4_performance` when the user explicitly asks for "fast" or
   "optimized" code.
3. Use `v9_transpiler_friendly` when the transpiler rejects advanced Python idioms.
4. Use `v10_correctness_focused` when the generated code compiles but fails tests.
5. Avoid `v1_minimal` for production use; it does not supply the transpiler
   constraints that prevent invalid Rust output.
6. Consider regenerating or post-editing LLM-generated tests when using
   `v3_algorithm`, as the terse style can introduce test-side logic errors.
