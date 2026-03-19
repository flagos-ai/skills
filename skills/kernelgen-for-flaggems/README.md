# kernelgen-for-flaggems: GPU Kernel Operator Generation for FlagGems

[中文版](README_zh.md)

## Overview

`kernelgen-for-flaggems` is an AI coding skill that generates GPU kernel operators specifically for the FlagGems project via the `kernelgen-mcp` MCP service.

### Problem Statement

FlagGems has specific conventions for operator implementation: `pointwise_dynamic` wrappers, promotion methods, `flag_gems.utils` imports, categorized test files, and a unique operator registration system (`_FULL_CONFIG`). Generating code that follows all these conventions manually is tedious and error-prone.

This skill automates the entire FlagGems-specific workflow: **environment check → MCP generation → FlagGems convention adaptation → operator registration → categorized testing → benchmarking**, spanning 9 steps with FlagGems-aware code transformation.

### Usage

```bash
# Generate a FlagGems operator
/kernelgen-for-flaggems relu

# Generate with explicit function type
/kernelgen-for-flaggems layer_norm --func-type normalization
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `operator_name` | Yes | — | Operator name matching `torch.ops.aten` (e.g. `relu`, `silu`, `layer_norm`) |
| `--func-type` | No | Auto-inferred | One of: `unary_pointwise`, `binary_pointwise`, `reduction`, `normalization`, `blas`, `other` |

---

## Generation Pipeline (9 Steps)

```
┌──────────────────────────────────────────────────────────┐
│  Step 0   Pre-flight: Environment & MCP Check            │
│  Step 1   Understand the Operator Request                │
│  Step 2   Check Whether Operator Already Exists          │
│  Step 3   Research Context (flagos_wiki)                 │
│  Step 4   Call kernelgen-mcp                             │
│  Step 5   Adapt and Place Code (FlagGems conventions)    │
│  Step 5.5 Pre-test Validation                            │
│  Step 6   Run Accuracy Tests                             │
│  Step 7   Run Performance Benchmark                      │
│  Step 8   Summary Report                                 │
└──────────────────────────────────────────────────────────┘
```

### Key Features

- **FlagGems-native code** — generates `pointwise_dynamic` style kernels with correct promotion methods
- **Automatic registration** — adds to `__init__.py` and `_FULL_CONFIG` in alphabetical order
- **Categorized test placement** — appends tests to `tests/test_<category>_ops.py` following existing patterns
- **Three generation modes** — new operator, replace existing, or side-by-side custom variant (v2)
- **Error recovery** — structured retry protocol with MCP re-generation and optimization

---

## Directory Structure

```
skills/kernelgen-for-flaggems/
├── SKILL.md        # Skill definition (entry point)
├── LICENSE.txt     # Apache 2.0 license
├── README.md       # This document (English)
└── README_zh.md    # Chinese version
```

---

## Related Skills

- [`kernelgen`](../kernelgen/) — General purpose version for any Python/Triton repository
- [`kernelgen-for-vllm`](../kernelgen-for-vllm/) — Specialized for vLLM repositories
- [`kernelgen-submit-feedback`](../kernelgen-submit-feedback/) — Submit bug reports and feedback

---

## License

This project is licensed under the Apache 2.0 License. See [LICENSE.txt](LICENSE.txt) for details.
