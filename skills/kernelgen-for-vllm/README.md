# kernelgen-for-vllm: GPU Kernel Operator Generation for vLLM

[中文版](README_zh.md)

## Overview

`kernelgen-for-vllm` is an AI coding skill that generates GPU kernel operators specifically for the vLLM project via the `kernelgen-mcp` MCP service.

### Problem Statement

vLLM has specific conventions for kernel implementation: SPDX license headers, `vllm.logger.init_logger` logging, `@triton.autotune` configurations, specific file placement patterns (`vllm/kernels/`, `csrc/`, `tests/kernels/`), and custom op registration (`vllm/_custom_ops.py`). Generating code that follows all these conventions manually is complex and error-prone.

This skill automates the entire vLLM-specific workflow: **environment check → MCP generation → vLLM convention adaptation → operator registration → accuracy testing → benchmarking**, spanning 9 steps with vLLM-aware code transformation.

### Usage

```bash
# Generate a vLLM kernel operator
/kernelgen-for-vllm rms_norm

# Generate with explicit function type
/kernelgen-for-vllm silu_and_mul --func-type activation
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `operator_name` | Yes | — | Operator name in snake_case (e.g. `rms_norm`, `silu_and_mul`, `rotary_embedding`) |
| `--func-type` | No | Auto-inferred | One of: `activation`, `norm`, `attention`, `quantization`, `moe`, `sampling`, `other` |

---

## Generation Pipeline (9 Steps)

```
┌──────────────────────────────────────────────────────────┐
│  Step 0   Pre-flight: Environment & MCP Check            │
│  Step 1   Understand the Operator Request                │
│  Step 2   Check Whether Operator Already Exists          │
│  Step 3   Research Context (flagos_wiki)                 │
│  Step 4   Call kernelgen-mcp                             │
│  Step 5   Adapt and Place Code (vLLM conventions)        │
│  Step 6   Run Accuracy Tests                             │
│  Step 7   Run Performance Benchmark                      │
│  Step 8   Summary Report                                 │
└──────────────────────────────────────────────────────────┘
```

### Key Features

- **vLLM-native code** — generates Triton kernels with SPDX headers, `init_logger`, and proper `@triton.autotune`
- **Smart file placement** — places kernels in `vllm/kernels/`, tests in `tests/kernels/`, benchmarks in `benchmarks/kernels/`
- **Custom op registration** — integrates with `vllm/_custom_ops.py` when applicable
- **Three generation modes** — new operator, replace existing, or side-by-side custom variant (v2)
- **Comprehensive testing** — per-dtype tolerances (fp32: 1e-5, fp16: 1e-2, bf16: 2e-2)
- **Error recovery** — structured retry protocol with MCP re-generation and optimization

---

## Directory Structure

```
skills/kernelgen-for-vllm/
├── SKILL.md        # Skill definition (entry point)
├── LICENSE.txt     # Apache 2.0 license
├── README.md       # This document (English)
└── README_zh.md    # Chinese version
```

---

## Related Skills

- [`kernelgen`](../kernelgen/) — General purpose version for any Python/Triton repository
- [`kernelgen-for-flaggems`](../kernelgen-for-flaggems/) — Specialized for FlagGems repositories
- [`kernelgen-submit-feedback`](../kernelgen-submit-feedback/) — Submit bug reports and feedback

---

## License

This project is licensed under the Apache 2.0 License. See [LICENSE.txt](LICENSE.txt) for details.
