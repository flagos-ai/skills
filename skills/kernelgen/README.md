# kernelgen: GPU Kernel Operator Generation (General Purpose)

[中文版](README_zh.md)

## Overview

`kernelgen` is an AI coding skill that generates GPU kernel operators via the `kernelgen-mcp` MCP service and integrates them into any Python/Triton repository.

### Problem Statement

Writing high-performance GPU kernels is complex and error-prone. Developers must handle Triton pointer arithmetic, memory access patterns, autotuning configurations, and project-specific conventions. Each repository has its own file layout, coding style, and testing patterns, making it hard to generate code that fits seamlessly.

This skill automates the entire workflow: **environment check → repo discovery → MCP generation → code adaptation → accuracy testing → performance benchmarking**, spanning 10 steps with automatic convention detection and code transformation.

### Usage

```bash
# Generate a kernel operator
/kernelgen relu

# Generate with explicit function type
/kernelgen rms_norm --func-type normalization

# Generate for any operator
/kernelgen silu_and_mul
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `operator_name` | Yes | — | Operator name in snake_case (e.g. `relu`, `rms_norm`, `silu_and_mul`) |
| `--func-type` | No | Auto-inferred | One of: `unary_pointwise`, `binary_pointwise`, `reduction`, `normalization`, `attention`, `activation`, `quantization`, `moe`, `blas`, `sampling`, `other` |

---

## Generation Pipeline (10 Steps)

```
┌──────────────────────────────────────────────────────────┐
│  Step 0   Pre-flight: Environment & MCP Check            │
│  Step 1   Understand the Operator Request                │
│  Step 2   Discover Repository Structure                  │
│  Step 3   Check Whether Operator Already Exists          │
│  Step 4   Research Context (flagos_wiki)                 │
│  Step 5   Call kernelgen-mcp                             │
│  Step 6   Adapt and Place Code into Repository           │
│  Step 6.5 Pre-test Validation                            │
│  Step 7   Run Accuracy Tests                             │
│  Step 8   Run Performance Benchmark                      │
│  Step 9   Summary Report                                 │
└──────────────────────────────────────────────────────────┘
```

### Key Features

- **Dynamic repo discovery** — automatically detects project structure, operator directories, test directories, and conventions
- **Convention matching** — reads existing code to match import style, naming conventions, license headers, and autotune patterns
- **Smart code transformation** — adapts MCP-generated code to match repo patterns (wrapper style vs raw Triton)
- **Existing operator detection** — searches for naming variants before generating to avoid duplicates
- **Comprehensive testing** — runs accuracy tests with per-dtype tolerances and performance benchmarks
- **Error recovery** — structured retry protocol with MCP re-generation and optimization fallbacks

---

## Directory Structure

```
skills/kernelgen/
├── SKILL.md        # Skill definition (entry point)
├── LICENSE.txt     # Apache 2.0 license
├── README.md       # This document (English)
└── README_zh.md    # Chinese version
```

---

## File Descriptions

### `SKILL.md`

The skill entry point. Defines the complete 10-step workflow for generating GPU kernel operators in any Python/Triton repository. Includes environment checking, repo structure discovery, MCP code generation, convention-aware code placement, accuracy testing, and performance benchmarking.

---

## Usage in FlagOS Skills Repository

### Quick Install (via npx)

```bash
# Install this skill only
npx skills add flagos-ai/skills --skill kernelgen -a claude-code

# Or install all Flagos skills at once
npx skills add flagos-ai/skills -a claude-code
```

### Manual Install

```bash
# From your project root
mkdir -p .claude/skills
cp -r <path-to-this-repo>/skills/kernelgen .claude/skills/
```

---

## Related Skills

- [`kernelgen-for-flaggems`](../kernelgen-for-flaggems/) — Specialized for FlagGems repositories
- [`kernelgen-for-vllm`](../kernelgen-for-vllm/) — Specialized for vLLM repositories
- [`kernelgen-submit-feedback`](../kernelgen-submit-feedback/) — Submit bug reports and feedback

---

## License

This project is licensed under the Apache 2.0 License. See [LICENSE.txt](LICENSE.txt) for details.
