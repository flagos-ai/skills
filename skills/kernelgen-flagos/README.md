# kernelgen-flagos: Unified GPU Kernel Operator Generation

[中文版](README_zh.md)

## Overview

`kernelgen-flagos` is a unified AI coding skill that generates GPU kernel operators via the `kernelgen-mcp` MCP service. It automatically detects the target repository type and dispatches to the appropriate specialized workflow.

### Problem Statement

Writing high-performance GPU kernels is complex and error-prone. Different projects (FlagGems, vLLM, custom Triton repos) each have unique conventions for operator implementation, testing, and registration. Previously, users needed to install separate skills for each project type.

This unified skill bundles **four sub-skills** into one package, so users only need a single install:

| Sub-skill | Purpose |
|---|---|
| **kernelgen-general** | Generate GPU kernels for any Python/Triton repository |
| **kernelgen-for-flaggems** | Specialized for FlagGems (`pointwise_dynamic`, `_FULL_CONFIG`, categorized tests) |
| **kernelgen-for-vllm** | Specialized for vLLM (SPDX headers, `init_logger`, `@triton.autotune`, custom op registration) |
| **kernelgen-submit-feedback** | Submit bug reports via GitHub Issues or email |

### Usage

```bash
# Generate a kernel operator (auto-detects repo type)
/kernelgen-flagos relu

# Generate with explicit function type
/kernelgen-flagos rms_norm --func-type normalization

# Generate for any operator
/kernelgen-flagos silu_and_mul
```

The skill will automatically:
- Detect if you're in a **FlagGems** repo and use the FlagGems-specific workflow
- Detect if you're in a **vLLM** repo and use the vLLM-specific workflow
- Otherwise, use the **general-purpose** workflow with dynamic repo discovery

| Argument | Required | Default | Description |
|---|---|---|---|
| `operator_name` | Yes | — | Operator name in snake_case (e.g. `relu`, `rms_norm`, `silu_and_mul`) |
| `--func-type` | No | Auto-inferred | Function type category (varies by detected repo type) |

### Feedback

If you encounter any issues during generation, say "submit feedback" or "report a bug" and the skill will guide you through submitting a GitHub issue or email.

---

## How It Works

```
┌──────────────────────────────────────────────────────────┐
│  Phase 1   Detect repository type                        │
│            ├── FlagGems? → kernelgen-for-flaggems.md     │
│            ├── vLLM?     → kernelgen-for-vllm.md        │
│            └── Other?    → kernelgen-general.md          │
│                                                          │
│  Phase 2   Execute the selected sub-skill workflow       │
│            (environment check → MCP generation →         │
│             code adaptation → testing → benchmarking)    │
│                                                          │
│  Phase 3   Feedback handling (on demand)                 │
│            → kernelgen-submit-feedback.md                │
└──────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
skills/kernelgen/
├── SKILL.md                       # Unified entry point (routing logic)
├── kernelgen-general.md           # General-purpose sub-skill
├── kernelgen-for-flaggems.md      # FlagGems-specific sub-skill
├── kernelgen-for-vllm.md          # vLLM-specific sub-skill
├── kernelgen-submit-feedback.md   # Feedback submission sub-skill
├── LICENSE.txt                    # Apache 2.0 license
├── README.md                      # This document (English)
└── README_zh.md                   # Chinese version
```

---

## File Descriptions

### `SKILL.md`

The unified entry point. Contains routing logic that auto-detects the repository type (FlagGems, vLLM, or generic) and reads the appropriate sub-skill file to execute.

### `kernelgen-general.md`

Full 10-step workflow for generating GPU kernel operators in any Python/Triton repository. Includes dynamic repo structure discovery, convention detection, and adaptive code placement.

### `kernelgen-for-flaggems.md`

9-step workflow specialized for FlagGems repositories. Handles `pointwise_dynamic` wrappers, promotion methods, `_FULL_CONFIG` registration, categorized test files, and FlagGems-specific conventions.

### `kernelgen-for-vllm.md`

9-step workflow specialized for vLLM repositories. Handles SPDX license headers, `vllm.logger.init_logger`, `@triton.autotune`, custom op registration, and vLLM directory conventions.

### `kernelgen-submit-feedback.md`

Feedback submission workflow. Collects bug reports with auto-detected environment info and submits via GitHub Issues (`gh` CLI) or email fallback.

---

## Usage in FlagOS Skills Repository

### Quick Install (via npx)

```bash
# Install the unified kernelgen skill (includes all sub-skills)
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

## License

This project is licensed under the Apache 2.0 License. See [LICENSE.txt](LICENSE.txt) for details.
