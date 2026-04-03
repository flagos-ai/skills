# Overview

**FlagOS** is a fully open-source AI system software stack for heterogeneous AI chips,
allowing AI models to be developed once and seamlessly ported to a wide range of AI hardware with minimal effort.
This repository collects reusable **Skills** for FlagOS — injecting domain knowledge, workflow standards,
and best practices into AI coding agents.
>
> [中文版](README_zh.md)

## What are Skills?

Skills are **folder-based capability packages**: each skill uses documentation, scripts,
and resources to teach agents to reliably and reproducibly complete tasks in a specific domain.
Each skill folder contains a `SKILL.md` file with YAML frontmatter (name + description)
followed by detailed agent instructions.
Skills can also include reference docs, scripts, and assets.

This repository follows the [Agent Skills open standard](https://agentskills.io/specification).

## Quick Start

FlagOS Skills are compatible with **Claude Code**, **Cursor**, **Codex**, and any agent
supporting the [Agent Skills standard](https://agentskills.io/specification).

### npx (Recommended — works with all agents)

Use the [`skills`](https://www.npmjs.com/package/skills) CLI to install skills directly — no cloning needed:

```bash
# List available skills in this repository
npx skills add flagos-ai/skills --list

# Install a specific skill into your project
npx skills add flagos-ai/skills --skill model-migrate-flagos

# Install a specific skill globally (user-level)
npx skills add flagos-ai/skills --skill model-migrate-flagos --global

# Install all skills at once
npx skills add flagos-ai/skills --all

# Install for specific agents only
npx skills add flagos-ai/skills --agent claude-code cursor
```

Other useful commands:

```bash
npx skills list              # List installed skills
npx skills find              # Search for skills interactively
npx skills update            # Update all skills to latest versions
npx skills remove            # Interactive remove
```

> **Note:** No prior installation needed — `npx` downloads the [`skills`](https://skills.sh/) CLI automatically.

### Claude Code

1. Register the repository as a plugin marketplace (in Claude Code interactive mode):

   ```
   /plugin marketplace add flagos-ai/skills
   ```
   
   Or from the terminal:
   
   ```bash
   claude plugin marketplace add flagos-ai/skills
   ```

2. Install skills:

   ```
   /plugin install flagos-skills@flagos-skills
   ```
   
   Or from the terminal:
   
   ```bash
   claude plugin install flagos-skills@flagos-skills
   ```

After installation, mention the skill in your prompt — Claude automatically
loads the corresponding `SKILL.md` instructions.

### Cursor

This repository includes Cursor plugin manifests (`.cursor-plugin/plugin.json`
and `.cursor-plugin/marketplace.json`).

Install from the repository URL or local checkout via the Cursor plugin flow.

### Codex

Use the `$skill-installer` inside Codex:

```
$skill-installer install model-migrate-flagos from flagos-ai/skills
```

Or provide the GitHub directory URL:

```
$skill-installer install https://github.com/flagos-ai/skills/tree/main/skills/model-migrate-flagos
```

Alternatively, copy skill folders into Codex's standard `.agents/skills` location:

```bash
cp -r skills/model-migrate-flagos $REPO_ROOT/.agents/skills/
```

See the [Codex Skills guide](https://developers.openai.com/codex/skills/) for more details.

### Gemini CLI

```bash
gemini extensions install https://github.com/flagos-ai/skills.git --consent
```

This repo includes `gemini-extension.json` and `agents/AGENTS.md` for Gemini CLI integration.
See [Gemini CLI extensions docs](https://geminicli.com/docs/extensions/) for more help.

### Manual / Other Agents

For any agent that supports the [Agent Skills standard](https://agentskills.io/specification),
point it at the `skills/` directory in this repository.
Each skill is self-contained with a `SKILL.md` entry point.
The `agents/AGENTS.md` file can also be used as a fallback for agents that don't support skills natively.

## Skills Catalog

<!-- BEGIN_SKILLS_TABLE -->
| Category | Sub-category | Skill | Description |
|----------|-------------|-------|-------------|
| **Deployment & Release** | Base Image Selection | [`gpu-container-setup-flagos`](skills/gpu-container-setup-flagos/) | Automatically detect GPU vendor, find appropriate PyTorch container image, launch with correct mounts, and validate GPU functionality. Supports NVIDIA, Ascend, Metax, Iluvatar, and AMD/ROCm. Use when user says "setup container", "start pytorch container", or invokes /gpu-container-setup. |
|  | Model Migration | [`model-migrate-flagos`](skills/model-migrate-flagos/) | Migrate a model from the latest vLLM upstream repository into the vllm-plugin-FL project (pinned at vLLM v0.13.0). Use this skill whenever someone wants to add support for a new model to vllm-plugin-FL, port model code from upstream vLLM, or backport a newly released model. Trigger when the user says things like "migrate X model", "add X model support", "port X from upstream vLLM", "make X work with the FL plugin", or simply "/model-migrate-flagos model_name". The model_name argument uses snake_case (e.g. qwen3_5, kimi_k25, deepseek_v4). Do NOT use for models already supported by vLLM 0.13.0 core, or for multimodal-only components that don't need backporting. |
|  | Release Pipeline | [`flagrelease-entrance-flagos`](skills/flagrelease-entrance-flagos/) | Full FlagRelease pipeline orchestrator. Runs the complete LLM deployment, verification, and benchmarking pipeline for multi-chip GPU backends. Executes: install-stack → env-verify → model-verify → perf-test in sequence, passing state between steps and producing a final structured report. Assumes gpu-container-setup (Step 1) is already done — a running container with PyTorch + GPU access must exist. |
|  | Stack Installation | [`install-stack-flagos`](skills/install-stack-flagos/) | Install the 5-package multi-chip software stack (vLLM, FlagTree, FlagGems, FlagCX, vllm-plugin-FL) inside a GPU container. Handles network mirror detection, dependency ordering, wheel selection, and per-package validation. Use after gpu-container-setup has produced a running container with PyTorch + GPU access. |
|  | Plugin Environment Setup | [`vllm-plugin-fl-setup-flagos`](skills/vllm-plugin-fl-setup-flagos/) | Install and configure vLLM-Plugin-FL for multiple hardware backends including NVIDIA, Ascend, MetaX, Iluvatar, Moore Threads, and more. Automates the full setup workflow: detect hardware → install vLLM-Plugin-FL → install FlagGems → (optionally) install FlagCX → backend-specific configuration → inference verification. Trigger when the user says "setup vllm-plugin-fl", "install vllm-plugin-fl", "configure FL plugin", "set up FlagGems", or "set up FlagCX". |
| **Benchmarking & Eval** | Accuracy & Performance Test | [`perf-test-flagos`](skills/perf-test-flagos/) | Run accuracy benchmarks (FlagEval, when available) and performance benchmarks (vllm bench serve) against a served model. Covers 5 workload profiles: short/long prefill x short/long decode + high concurrency. Collects throughput, latency, TTFT, TPOT metrics. |
|  | Deployment A/B Verification | [`model-verify-flagos`](skills/model-verify-flagos/) | Verify the serving stack with a user-specified target model. Runs twice: first with FlagGems/FlagCX disabled (isolate model-specific errors), then with full multi-chip stack enabled. Diffs the two runs to pinpoint which layer caused any failure. |
|  | FlagPerf Case Creation | *Planned* | Generate FlagPerf-compliant directory structures, config files, run scripts, and expected metric baselines for new model/chip benchmark cases. |
|  | Post-Deploy Auto Eval | *Planned* | Automatically trigger evaluation after model deployment, track evaluation status, report errors on failure, and push notifications with results upon completion. |
| **Kernel & Operator Development** | Complex Operator Dev | *Planned* | Generate skeleton code for multi-step fused operators (fused attention, fused MoE, etc.), handling shared memory tiling strategies and multi-backend branching. |
|  | Experimental Op Promotion | *Planned* | Scan FlagGems ~130 experimental ops, check test coverage, align signatures, complete `_FULL_CONFIG` registration, and generate migration PRs to promote them to main ops. |
|  | Kernel Gen for FlagGems | [`kernelgen-flagos`](skills/kernelgen-flagos/kernelgen-generate-for-flaggems.md) | FlagGems-specific kernel generation with `@pointwise_dynamic` wrapper rewriting, `_FULL_CONFIG` registration, and operator signature alignment. |
|  | Kernel Gen for vLLM | [`kernelgen-flagos`](skills/kernelgen-flagos/kernelgen-generate-for-vllm.md) | vLLM-specific kernel generation with SPDX headers, `@triton.autotune`, custom op registration, and dispatch integration. |
|  | Kernel Generation | [`kernelgen-flagos`](skills/kernelgen-flagos/) | Unified GPU kernel operator generation and optimization skill. Automatically detects the target repository type (FlagGems, vLLM, or general Python/Triton) and dispatches to the appropriate specialized sub-skill. Includes operator generation, MCP-based iterative optimization, and feedback submission sub-skills. Use this skill when the user wants to generate or optimize a GPU kernel operator, create a Triton kernel, or says things like "generate an operator", "create a kernel for X", "optimize triton kernel", or "/kernelgen-flagos". |
|  | MCP Service Setup | [`kernelgen-flagos`](skills/kernelgen-flagos/kernelgen-mcp-setup.md) | Auto-detect and configure the `kernelgen-mcp` MCP service. Checks project-local config files for existing setup, guides the user through token acquisition if needed, and writes the configuration automatically. Runs before any generation/optimization/specialization sub-skill. |
|  | Kernel Optimization | [`kernelgen-flagos`](skills/kernelgen-flagos/kernelgen-optimize.md) | General-purpose Triton kernel optimization via MCP iterative loop. Analyzes existing kernels, identifies bottlenecks, and applies optimizations through multiple rounds until the target speedup is reached. |
|  | Kernel Optimization for FlagGems | [`kernelgen-flagos`](skills/kernelgen-flagos/kernelgen-optimize-for-flaggems.md) | FlagGems-specific kernel optimization with 3 modes: optimize built-in operators in-place, optimize external operators and integrate into experimental_ops, or optimize existing experimental operators. Includes accuracy tests and performance benchmarks. |
|  | Kernel Optimization for vLLM | [`kernelgen-flagos`](skills/kernelgen-flagos/kernelgen-optimize-for-vllm.md) | vLLM-specific kernel optimization with CustomOp registration, accuracy tests, and performance benchmark integration. Optimizes Triton operators and automatically integrates them into the vLLM project. |
|  | Kernel Platform Specialization | [`kernelgen-flagos`](skills/kernelgen-flagos/kernelgen-specialize.md) | Platform specialization for Triton operators via MCP `specialize_kernel` tool. Migrates GPU Triton operators to target platforms (e.g., Huawei Ascend NPU), handling architecture differences, Grid configuration, and memory alignment. |
|  | Kernel Specialization for FlagGems | [`kernelgen-flagos`](skills/kernelgen-flagos/kernelgen-specialize-for-flaggems.md) | Combines MCP platform specialization with FlagGems framework integration. Supports four integration modes: vendor-ops, vendor-fused, override-builtin, and experimental. Includes automated testing and performance benchmarking. |
|  | Operator Diagnosis | *Planned* | Diagnose abnormal operators in the FlagOS stack — identify precision errors, performance regressions, and backend-specific failures across chips. |
| **Multi-Chip Backend Onboarding** | Dispatch Op Extension | *Planned* | Query dispatchable ops from `base.py`, generate impl template files, add `OpImpl` registration to `register_ops.py`, and create unit test skeletons. |
|  | FlagCX Comm Backend | *Planned* | Parse 20+ device and 15+ CCL function pointer signatures from header files, generate all stub implementations with trivial function fills, plus CMake build configuration. |
|  | FlagGems Chip Backend | *Planned* | Generate the full `_vendor/` scaffold: `__init__.py` (VendorInfoBase config) + `heuristics_config_utils.py` + `tune_configs.yaml` + `ops/` directory following the FlagGems backend contribution guide. |
|  | Heterogeneous Training Config | *Planned* | Generate valid FlagScale heterogeneous training configs from hardware topology descriptions, auto-compute `hetero_process_meshes` / `hetero_pipeline_layer_split`, and validate constraints (TP×DP×PP = device count). |
|  | vLLM Vendor Backend | *Planned* | Scaffold a new vllm-plugin-FL vendor backend from the template: generate vendor directory, `Backend` subclass, `is_available` detection, `register_ops` framework, and test skeleton. |
| **Developer Tooling** | Feedback Submission | [`kernelgen-flagos`](skills/kernelgen-flagos/kernelgen-submit-feedback.md) | Auto-collect environment info, construct structured GitHub issues, and submit to flagos-ai/skills with email fallback when GitHub CLI is unavailable. |
|  | General | [`tle-developer-flagos`](skills/tle-developer-flagos/) | Self-contained orchestration skill for writing high-performance TLE kernels and shipping TLE feature changes with reproducible validation. Use when the user wants to write/optimize TLE kernels, implement TLE API/verifier/lowering features, or debug TLE correctness/performance issues. Trigger on phrases like "write a TLE kernel", "optimize TLE operator", and "debug TLE local_ptr". |
|  | Local Dev Environment | *Planned* | Set up local development and debugging environments for FlagOS modules (FlagGems / FlagTree / FlagCX / etc.) — configure dependencies, environment variables, and debug toolchains. |
|  | Skill Development | [`skill-creator-flagos`](skills/skill-creator-flagos/) | Create new skills, modify existing skills, and validate skill quality for the FlagOS skills repository. Use this skill whenever someone wants to create a skill from scratch, improve or edit an existing skill, scaffold a new skill directory, validate skill structure, or run test cases against a skill. Trigger when the user says things like "create a skill", "make a new skill for X", "scaffold a skill", "improve this skill", "validate my skill", or simply "/skill-creator-flagos". Also trigger when users mention turning a workflow into a reusable skill, or want to package a repeated process as a skill. |
<!-- END_SKILLS_TABLE -->

### Using skills in your agent

Once a skill is installed, mention it directly in your prompt:

- "Use model-migrate-flagos to migrate the Qwen3-5 model from upstream vLLM"
- "/model-migrate-flagos qwen3_5"
- "Port the DeepSeek-V4 model to vllm-plugin-FL"

Your agent automatically loads the corresponding `SKILL.md` instructions and helper scripts.

## Repository Structure

```none
├── .claude-plugin/          # Claude Code plugin manifest
│   └── marketplace.json
├── .cursor-plugin/          # Cursor plugin manifest
│   ├── marketplace.json
│   └── plugin.json
├── agents/                  # Codex / Gemini CLI fallback
│   └── AGENTS.md
├── assets/                  # Repository-level static resources
├── contributing.md          # Contribution guidelines
├── gemini-extension.json    # Gemini CLI extension manifest
├── scripts/                 # Repository-level utility scripts
│   └── validate_skills.py   # Batch validate all skills
├── skills/                  # Skill directories
│   ├── model-migrate-flagos/    # Model migration workflow
│   └── ...
├── spec/                    # Agent Skills standard & local conventions
│   ├── README.md
│   └── agent-skills-spec.md
└── template/                # Template for creating new skills
    └── SKILL.md
```

## Creating a New Skill

1. **Create directory & copy template**

   ```bash
   mkdir skills/<skill-name>
   cp template/SKILL.md skills/<skill-name>/SKILL.md
   ```

2. **Edit frontmatter** — `name` (lowercase + hyphens, must match directory name) and `description` (what it does + when to trigger)

3. **Write the body** — Overview, Prerequisites, Execution steps, Examples (2-3), Troubleshooting

4. **Add supporting files** (optional) — `references/`, `scripts/`, `assets/`, `LICENSE.txt`

5. **Validate**

   ```bash
   python scripts/validate_skills.py
   ```

See [contributing.md](contributing.md) for the full contribution guide.

## License

This project is licenced under the [Apache License version 2.0](LICENSE) license.
