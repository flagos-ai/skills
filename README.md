> **FlagOS** is a fully open-source AI system software stack for heterogeneous AI chips, allowing AI models to be developed once and seamlessly ported to a wide range of AI hardware with minimal effort. This repository collects reusable **Skills** for FlagOS — injecting domain knowledge, workflow standards, and best practices into AI coding agents.
>
> [中文版](README_zh.md)

## What are Skills?

Skills are **folder-based capability packages**: each skill uses documentation, scripts, and resources to teach agents to reliably and reproducibly complete tasks in a specific domain. Each skill folder contains a `SKILL.md` file with YAML frontmatter (name + description) followed by detailed agent instructions. Skills can also include reference docs, scripts, and assets.

This repository follows the [Agent Skills open standard](https://agentskills.io/specification).

## Quick Start

FlagOS Skills are compatible with **Claude Code**, **Cursor**, **Codex**, and any agent supporting the [Agent Skills standard](https://agentskills.io/specification).

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

After installation, mention the skill in your prompt — Claude automatically loads the corresponding `SKILL.md` instructions.

### Cursor

This repository includes Cursor plugin manifests (`.cursor-plugin/plugin.json` and `.cursor-plugin/marketplace.json`).

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

This repo includes `gemini-extension.json` and `agents/AGENTS.md` for Gemini CLI integration. See [Gemini CLI extensions docs](https://geminicli.com/docs/extensions/) for more help.

### Manual / Other Agents

For any agent that supports the [Agent Skills standard](https://agentskills.io/specification), point it at the `skills/` directory in this repository. Each skill is self-contained with a `SKILL.md` entry point. The `agents/AGENTS.md` file can also be used as a fallback for agents that don't support skills natively.

## Skills Catalog

<!-- BEGIN_SKILLS_TABLE -->
| Category | Sub-category | Skill | Description |
|----------|-------------|-------|-------------|
| **Inference & Serving** | Model Migration | [`model-migrate-flagos`](skills/model-migrate-flagos/) | Migrate a model from upstream vLLM into vllm-plugin-FL (pinned at v0.13.0). Automates the full 13-step copy-then-patch workflow with E2E verification. |
| | Serving Deployment | [PR #6 `flagrelease`](https://github.com/flagos-ai/skills/pull/6) | Deploy and configure vLLM-FL / SGLang-FL serving instances across multi-chip environments. |
| | Preflight Check | *Planned* | Verify GPU/accelerator availability, driver versions, Python env, and chip compatibility before running inference. |
| **Training & RLHF** | Training Migration | *Planned* | Adapt training scripts for FlagScale / Megatron-LM-FL across different AI chips. |
| | RLHF Pipeline | *Planned* | Set up and debug verl-FL reinforcement learning workflows. |
| **Operator & Compiler** | TLE Primitive Dev | [PR #2 `tle-developer`](https://github.com/flagos-ai/skills/pull/2) | Develop TLE (Triton Language Extensions) primitives and build operators using TLE-Lite / TLE-Struct / TLE-Raw across FlagTree backends. |
| | Operator Optimization | *Planned* | Guided iterative performance tuning for existing FlagGems / FlagAttention operators — profiling, bottleneck analysis, and optimization suggestions. |
| | Kernel Generation | [PR #10 `kernelgen`](https://github.com/flagos-ai/skills/pull/10) | General-purpose GPU kernel generation via KernelGen MCP for any Python/Triton project, covering multi-chip targets (NVIDIA, Ascend, Cambricon, Moore Threads, Iluvatar, etc.). |
| | Kernel Gen for FlagGems | [PR #10 `kernelgen-for-flaggems`](https://github.com/flagos-ai/skills/pull/10) | FlagGems-specific kernel generation with promotion rules, `pointwise_dynamic` wrappers, and `_FULL_CONFIG` registration. |
| | Kernel Gen for vLLM | [PR #10 `kernelgen-for-vllm`](https://github.com/flagos-ai/skills/pull/10) | vLLM-specific kernel generation with SPDX headers, `vllm.logger`, `@triton.autotune`, and custom op registration. |
| | KernelGen Feedback | [PR #10 `kernelgen-submit-feedback`](https://github.com/flagos-ai/skills/pull/10) | Submit bug reports and improvement suggestions for KernelGen as structured GitHub issues. |
| | Operator Diagnosis | *Planned* | Diagnose abnormal operators in the FlagOS stack — identify precision errors, performance regressions, and backend-specific failures across chips. |
| | Compiler Backend Adaptation | *Planned* | Port and debug FlagTree / Triton compiler backends for new AI chip architectures. |
| **Communication** | Collective Ops | *Planned* | Adapt and benchmark FlagCX cross-chip communication primitives (AllReduce, AllGather, Send/Recv, etc.) across 11+ backends (NCCL, IXCCL, CNCL, MCCL, etc.). |
| **Benchmarking & Eval** | Performance Benchmark | [PR #6 `perf-test`](https://github.com/flagos-ai/skills/pull/6) | Run and analyze FlagPerf benchmarks; generate multi-dimensional comparison reports (throughput, memory, scaling) across chips. |
| | E2E Accuracy Eval | [PR #6 `model-verify`](https://github.com/flagos-ai/skills/pull/6) | Token-level accuracy verification between different serving backends or chip targets. |
| **Environment & Deployment** | Stack Installation | [PR #6 `install-stack`](https://github.com/flagos-ai/skills/pull/6) | One-click FlagOS software stack installation on a target chip — auto-detect hardware, resolve dependencies, and configure the full toolchain (FlagTree + FlagGems + vLLM-FL + FlagCX). |
| | Base Image Selection | [PR #5 `gpu-container-setup`](https://github.com/flagos-ai/skills/pull/5) | Find and recommend the optimal base Docker image for domestic AI chip model deployment — matching chip type, driver version, CUDA/SDK compatibility, and framework requirements. |
| | Container Build | *Planned* | Build and publish multi-chip Docker images with correct driver/library dependencies. |
| | CI Pipeline | *Planned* | Configure and debug FlagOps CI/CD pipelines for multi-chip build matrices. |
| **Developer Tooling** | Skill Development | [`skill-creator-flagos`](skills/skill-creator-flagos/) | Create, improve, and validate skills for this repository. Scaffolding, conventions check, and test case evaluation. |
| | Chip Onboarding | *Planned* | Guide new chip vendors through the FlagOS adaptation process end-to-end. |
<!-- END_SKILLS_TABLE -->

### Using skills in your agent

Once a skill is installed, mention it directly in your prompt:

- "Use model-migrate-flagos to migrate the Qwen3-5 model from upstream vLLM"
- "/model-migrate-flagos qwen3_5"
- "Port the DeepSeek-V4 model to vllm-plugin-FL"

Your agent automatically loads the corresponding `SKILL.md` instructions and helper scripts.

## Repository Structure

```
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

[Apache License 2.0](LICENSE)
