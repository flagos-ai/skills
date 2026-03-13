# model-migrate-fl: vLLM Model Migration Skill

[中文版](README_zh.md)

## Overview

`model-migrate-fl` is an AI coding skill that migrates model code from the **latest vLLM upstream** into the **vllm-plugin-FL project** (pinned at vLLM v0.13.0).

### Problem Statement

vLLM upstream iterates rapidly with new models added frequently. However, vllm-plugin-FL is pinned at v0.13.0 — directly copying upstream code won't work due to API incompatibilities, missing config classes, broken relative imports, etc. Manual adaptation is time-consuming and error-prone.

This skill automates the entire migration workflow: **copy from upstream -> adapt for 0.13.0 -> register model -> verify correctness**, spanning 13 steps with a final token-level accuracy comparison against the upstream server.

### Usage

```bash
# Invoke in your AI coding assistant
/model-migrate-fl qwen3_5
/model-migrate-fl kimi_k25
/model-migrate-fl glm5 /path/to/upstream /path/to/plugin
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `model_name` | Yes | — | snake_case model name (e.g. `qwen3_5`, `kimi_k25`) |
| `upstream_folder` | No | `/tmp/vllm-upstream-ref` | Path to upstream vLLM source |
| `plugin_folder` | No | Current working directory | Path to vllm-plugin-FL project |

---

## Migration Pipeline (13 Steps)

```
┌──────────────────────────────────────────────────────────┐
│  Step 1   Baseline unit tests (record pre-migration state)│
│  Step 2   Clone/update upstream vLLM source               │
│  Step 3   Study existing plugin patterns                  │
│  Step 4   Add model files (Config Bridge + Copy-then-Patch)│
│  Step 5   Register model in __init__.py                   │
│  Step 6   Code review (automated script + manual checks)  │
│  Step 7   Regression unit tests (compare vs Step 1)       │
│  Step 8   Functional tests                                │
│  Step 9   Benchmark verification (dummy weights)          │
│  Step 10  Serve + Request smoke test                      │
│  Step 11  E2E accuracy verification (token-level vs GT)   │
│  Step 12-13  Final report                                 │
└──────────────────────────────────────────────────────────┘
```

### Core Principle: Copy-then-Patch

The approach is **not** "read and rewrite", but:
1. Directly `cp` the upstream file into the plugin directory
2. Apply compatibility patches (P1-P13) via targeted edits
3. Verify import passes

This maximally preserves the upstream implementation and minimizes human error.

---

## Directory Structure

```
skills/model-migrate-fl/
├── SKILL.md                          # Skill definition (entry point)
├── README.md                         # This document (English)
├── README_zh.md                      # Chinese version
├── references/                       # Reference documents
│   ├── procedure.md                  # 13-step migration procedure (detailed)
│   ├── compatibility-patches.md      # 0.13.0 compatibility patch catalog (P1-P13)
│   └── operational-rules.md          # Operational rules (communication, tasks, GPU, etc.)
├── scripts/                          # Executable scripts
│   ├── validate_migration.py         # Automated code review
│   ├── benchmark.sh                  # Benchmark verification
│   ├── serve.sh                      # Start local vLLM server
│   ├── request.sh                    # Smoke test request
│   ├── e2e_eval.py                   # E2E accuracy comparison tool
│   ├── e2e_test_prompts.json         # Test prompt set (5 text + 5 multimodal)
│   ├── e2e_config.template.json      # E2E config template
│   └── e2e_remote_serve.sh           # Remote GT server management
└── results/                          # E2E comparison output
```

---

## File Descriptions

### Skill Definition

#### `SKILL.md`
The skill entry point. Defines trigger conditions, argument format, execution overview, script index, and troubleshooting guide. The AI coding assistant uses this file to identify and invoke the skill.

### Reference Documents (`references/`)

#### `procedure.md` — Migration Procedure
Complete operating manual for all 13 steps, including:
- Specific commands and code templates
- Placeholder resolution rules (e.g. `{{model_name}}`, `{{ModelClassName}}`)
- Decision logic (when Config Bridge is needed, which patches apply)
- Final report template

#### `compatibility-patches.md` — 0.13.0 Compatibility Patches

Catalog of all known upstream -> 0.13.0 incompatibilities and their fixes. Each patch includes:
- **Before/After** code examples
- **Why** the change is needed
- **When** the patch applies

| ID | Patch Name | Description |
|---|---|---|
| P1 | Relative -> Absolute imports | `from .xxx` -> `from vllm.*` or `from vllm_fl.*` |
| P2 | Config import redirect | Point to plugin's Config Bridge |
| P3 | Remove missing APIs | `MambaStateCopyFunc` etc. don't exist in 0.13.0 |
| P4 | Replace context manager init | `_mark_tower_model` -> direct initialization |
| P5 | Import verification | Quick `python3 -c "import ..."` check |
| P6 | Parent `__init__` reimplementation | When parent uses 0.13.0-missing APIs |
| P7 | MoE nested config fix | MoE params live under `text_config` in VL configs |
| P8 | MergedColumnParallelLinear | TP sharding fix for merged projections (critical!) |
| P9 | MambaStateDtypeCalculator signature | 3-arg -> 2-arg |
| P10 | mamba_cache_mode -> enable_prefix_caching | Cache mode API difference |
| P11 | Custom Op import path dedup | Avoid `Duplicate op name` errors |
| P12 | Complete `__init__` override checklist | Don't miss attributes when skipping parent init |
| P13 | Do NOT upgrade transformers | Use Config Bridge instead of upgrading transformers version |

#### `operational-rules.md` — Operational Rules
- **Communication protocol**: Report progress at every step boundary, no silent execution
- **TaskList management**: 13 tasks map to 13 steps, supports interruption recovery
- **Bash command rules**: Single-line commands, no process substitution, avoids permission prompts
- **GPU resource management**: Force-release GPUs before benchmark/serve, never skip
- **Network retry**: Auto-retry git clone etc. up to 3 times on failure

### Scripts (`scripts/`)

#### `validate_migration.py` — Automated Code Review (Step 6)
```bash
python3 validate_migration.py <plugin_folder> <model_file> [config_file]
```
Automatically checks:
- Residual relative imports
- References to 0.13.0-missing APIs
- Config Bridge `model_type` and `PretrainedConfig` inheritance
- Registration consistency in `__init__.py`
- Code smells (bare `except:`, hardcoded paths)

Exit code: 0=pass, 1=issues found.

#### `benchmark.sh` — Benchmark Verification (Step 9)
```bash
bash benchmark.sh Qwen3.5-397B-A17B-Real
```
Uses `vllm bench throughput` with dummy weights for quick forward-pass verification without loading real weights. Parameters: input_len=128, output_len=128, num_prompts=2, TP=8.

#### `serve.sh` — Start Local Server (Step 10/11)
```bash
bash serve.sh Qwen3.5-397B-A17B-Real [PORT]
```
Starts a vLLM OpenAI-compatible API server. Default port 8122, `VLLM_FL_PREFER_ENABLED=false` (FlagGems disabled for clean comparison). TP=8, GPU utilization 0.9.

#### `request.sh` — Smoke Test (Step 10)
```bash
bash request.sh Qwen3.5-397B-A17B-Real [PORT]
```
Sends a simple chat completion request to verify the server responds correctly.

#### `e2e_eval.py` — E2E Accuracy Verification (Step 11, Core Tool)
```bash
# Full pipeline: GT + Local + Compare
python3 e2e_eval.py --model Qwen3.5-397B-A17B-Real --gt-host <GT_HOST_IP> --mode text

# With config file
python3 e2e_eval.py --config e2e_config.json --model Qwen3.5-397B-A17B-Real --mode text

# Phased execution
python3 e2e_eval.py --model ... --gt-only        # Collect GT results only
python3 e2e_eval.py --model ... --local-only      # Collect local results only
python3 e2e_eval.py --model ... --compare-only    # Compare existing results only
```

Workflow:
1. Send test prompts to GT server (upstream vLLM), collect token_ids + token strings
2. Send the same prompts to local server (plugin vLLM)
3. Compare the first N tokens (default 32) token-by-token, generate report

Output files:
- `{model}_gt.json` — GT raw results
- `{model}_local.json` — Local raw results
- `{model}_e2e_report.txt` — Comparison report

Key parameters:
| Parameter | Default | Description |
|---|---|---|
| `--gt-host` | (required) | GT server IP |
| `--gt-port` | 8122 | GT server port |
| `--local-port` | 8122 | Local server port |
| `--mode` | text | `text` / `multimodal` / `all` |
| `--token-match-count` | 32 | Number of tokens to compare |
| `--max-tokens` | 256 | Max generation length |
| `--seed` | 42 | Random seed (for reproducibility) |

#### `e2e_test_prompts.json` — Test Prompt Set
- **text** (5 prompts): English Q&A, Chinese Q&A, code generation, math reasoning, technical explanation
- **multimodal** (5 prompts): 5 synthetic small images (base64 inline), testing image description, color recognition, pattern recognition

#### `e2e_config.template.json` — E2E Config Template
```json
{
  "gt_machine": {
    "host": "<GT_HOST_IP>",
    "docker_container": "<CONTAINER_NAME>",
    "conda_env": "<CONDA_ENV>",
    "vllm_port": 8122,
    "env_vars": { "VLLM_FL_PREFER_ENABLED": "false" },
    "extra_serve_args": "--trust-remote-code --load-format fastsafetensors"
  },
  "local": { "vllm_port": 8122 },
  "eval": { "max_tokens": 256, "token_match_count": 32, "seed": 42 }
}
```
Copy to `e2e_config.json` and fill in actual values before use.

> **Prerequisites for remote GT server:**
> 1. Set up passwordless SSH to the GT machine: `ssh-copy-id -i ~/.ssh/id_ed25519 <user>@<GT_HOST_IP>`
> 2. Verify connectivity: `ssh -i ~/.ssh/id_ed25519 <user>@<GT_HOST_IP> hostname`
> 3. If the GT server runs inside a Docker container, ensure the container name and conda env are correct in the config

#### `e2e_remote_serve.sh` — Remote GT Server Management
```bash
bash e2e_remote_serve.sh start  e2e_config.json Qwen3.5-397B-A17B-Real  # Start
bash e2e_remote_serve.sh stop   e2e_config.json                          # Stop
bash e2e_remote_serve.sh status e2e_config.json                          # Status
bash e2e_remote_serve.sh logs   e2e_config.json                          # Logs
```
Manages the GT vLLM server on a remote machine via SSH -> Docker exec -> Conda run. Supports automatic readiness wait (up to 600 seconds).

---

## Migrated Model Examples

| Model | Pattern | Key Characteristics |
|---|---|---|
| `qwen3_5` | Complex MoE + hybrid attention | Full model file adaptation, Config Bridge, GDN merged projections |
| `kimi_k25` | Lightweight wrapper | Delegates to DeepseekV2, minimal changes |
| `qwen3_next` | Hybrid attention | Custom attention integration (linear + full attention) |
| `minicpmo` | Multimodal | Vision + language composition |

---

## Usage in vllm-plugin-FL

Skills are typically placed under `.claude/skills/` (or the equivalent skills directory for your editor) in the project root. To use this skill in vllm-plugin-FL:

### Quick Install (via npx)

```bash
# Install this skill only
npx skills add flagos-ai/skills --skill model-migrate-fl -a claude-code

# Or install all Flagos skills at once
npx skills add flagos-ai/skills -a claude-code
```

### Manual Install

```bash
# From vllm-plugin-FL project root
mkdir -p .claude/skills
cp -r <path-to-this-repo>/skills/model-migrate-fl .claude/skills/
```

---

## License

This project is licensed under the Apache 2.0 License. See [LICENSE.txt](LICENSE.txt) for details.
