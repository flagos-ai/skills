---
name: model-migrate-flagos
description: >
  Migrate a model from the latest vLLM upstream repository into the vllm-plugin-FL project
  (pinned at vLLM v0.13.0). Use this skill whenever someone wants to add support for a new
  model to vllm-plugin-FL, port model code from upstream vLLM, or backport a newly released
  model. Trigger when the user says things like "migrate X model", "add X model support",
  "port X from upstream vLLM", "make X work with the FL plugin", or simply
  "/model-migrate-flagos model_name". The model_name argument uses snake_case
  (e.g. qwen3_5, kimi_k25, deepseek_v4).
  Do NOT use for models already supported by vLLM 0.13.0 core, or for
  multimodal-only components that don't need backporting.
argument-hint: model_name [upstream_folder] [plugin_folder]
user-invokable: true
compatibility: "Requires vLLM 0.13.0, Python 3.8+, GPU with CUDA, vllm-plugin-FL installed via pip install -e"
metadata:
  version: "1.0.0"
  author: flagos-ai
  category: workflow-automation
  tags: [model-migration, vllm, backport, e2e-verification]
allowed-tools: "Bash(pytest:*) Bash(python3:*) Bash(git:*) Bash(vllm:*) Bash(cp:*) Bash(curl:*) Bash(bash:*) Bash(test:*) Bash(ls:*) Bash(pip:*) Bash(cd:*) Read Edit Write Glob Grep AskUserQuestion TaskCreate TaskUpdate TaskList TaskGet"
---

# FL Plugin — Model Migration Skill

## Usage

```
/model-migrate-flagos <model_name> [upstream_folder] [plugin_folder]
```

| Argument | Required | Default |
|---|---|---|
| `model_name` | Yes | — |
| `upstream_folder` | No | `/tmp/vllm-upstream-ref` |
| `plugin_folder` | No | current working directory |

## Execution

### Step 1: Parse arguments and validate paths

Extract from user input:
- `{{model_name}}` = first argument (required, snake_case)
- `{{upstream_folder}}` = second argument or `/tmp/vllm-upstream-ref`
- `{{plugin_folder}}` = third argument or current working directory

If `{{upstream_folder}}` doesn't exist, ask user whether to clone it. If `{{plugin_folder}}` doesn't exist, error out.

**→ Tell user**: Confirm parsed model name and paths.

### Step 2: Load references and resolve placeholders

Read these files (relative to this SKILL.md):
- `references/procedure.md` — step-by-step migration procedure
- `references/compatibility-patches.md` — 0.13.0 patch catalog
- `references/operational-rules.md` — communication, TaskList, bash rules, resilience

The procedure references executable scripts in `scripts/`:
- `scripts/validate_migration.py` — automated code review (Step 6)
- `scripts/benchmark.sh` — benchmark verification (Step 9)
- `scripts/serve.sh` — serve model locally (Step 10.1, also used for E2E)
- `scripts/request.sh` — test request (Step 10.2)
- `scripts/e2e_eval.py` — E2E correctness verification (Step 11)
- `scripts/e2e_test_prompts.json` — test prompts for E2E (5 text + 5 multimodal)
- `scripts/e2e_config.template.json` — E2E config template (copy to `e2e_config.json` and fill in)
- `scripts/e2e_remote_serve.sh` — manage GT server on remote machine via SSH

Then investigate upstream source + HuggingFace to resolve all placeholders:

| Placeholder | How to derive |
|---|---|
| `{{model_name}}` | Direct from argument |
| `{{model_name_lower}}` | Lowercase of model_name (usually identical, e.g. `qwen3_5`) — used in file paths |
| `{{MODEL_DISPLAY_NAME}}` | From upstream code or HF model card |
| `{{ModelClassName}}` | From upstream model class (PascalCase) |
| `{{model_type}}` | From HF config.json `model_type` field |
| `{{ConfigClassName}}` | From upstream or derive from model_type |
| `{{skill_root}}` | Absolute path to this skill's folder (the directory containing this SKILL.md) |

Naming conventions vary per model — always verify from actual source, never guess.

**→ Tell user**: Present all resolved values. Use AskUserQuestion if anything is ambiguous.

### Step 3: Execute procedure

With placeholders resolved, execute every step in `procedure.md` sequentially. Apply patches from `compatibility-patches.md` during the copy-then-patch step. Follow `operational-rules.md` throughout.

**→ Tell user**: Before starting, output a numbered plan. Report progress at each step boundary.

## Scripts Reference

| Script | Step | Description |
|---|---|---|
| `validate_migration.py` | 6 | Automated import/API/registration checks |
| `benchmark.sh` | 9 | `vllm bench throughput` with dummy weights |
| `serve.sh` | 10, 11 | Start local vLLM server (port 8122, `VLLM_FL_PREFER_ENABLED=false`) |
| `request.sh` | 10 | Quick smoke-test request |
| `e2e_eval.py` | 11 | Token-level comparison vs upstream GT server |
| `e2e_test_prompts.json` | 11 | 5 text + 5 multimodal test prompts |
| `e2e_config.template.json` | 11 | Config template (GT machine, local port, eval params) |
| `e2e_remote_serve.sh` | 11 | SSH-based GT server lifecycle (start/stop/status/logs) |

## Examples

**Example 1: Typical new model**
```
User says: "/model-migrate-flagos kimi_k25"
Actions:
  1. Parse → model_name=kimi_k25, defaults for upstream/plugin paths
  2. Clone upstream, find vllm/model_executor/models/kimi_k25.py
  3. Discover it wraps DeepseekV2 → follow kimi_k25 (wrapper) pattern
  4. Copy file, apply P1+P2 patches, create config bridge
  5. Register, validate, test, benchmark, serve+request
  6. E2E verification against upstream GT
Result: kimi_k25 fully working in plugin, all 11 steps passed
```

**Example 2: Re-run after upstream update**
```
User says: "migrate qwen3_5 again, upstream updated"
Actions:
  1. Idempotent re-run — overwrite existing files with fresh upstream copy
  2. Re-apply patches, re-validate, re-test
  3. Re-run E2E to confirm no regression
Result: qwen3_5 updated to match latest upstream, no regressions
```

## Troubleshooting

**General principle**: When any runtime error occurs, **first compare vLLM upstream code** against both the plugin adaptation and the installed 0.13.0 environment. The diff is the fastest path to root cause. See `operational-rules.md § Debugging Priority: Upstream-First` for the full protocol.

| Problem | Typical Cause | Fix |
|---|---|---|
| `ImportError` after copy-then-patch | Missing P1 fix (relative→absolute imports) | Verify all `from .xxx` converted to `from vllm.*` or `from vllm_fl.*` |
| `AttributeError: module 'vllm' has no attribute X` | API doesn't exist in 0.13.0 | Check P3 in compatibility-patches.md; stub or remove |
| Config not recognized by vLLM | model_type mismatch or config bridge missing | Verify `_CONFIG_REGISTRY[model_type]` matches HF config.json exactly |
| Registration has no effect | Class name or import path typo | Compare with existing registrations in `__init__.py` |
| Benchmark `KeyError` on config field | Config bridge missing a field | Compare upstream config class vs bridge; add missing fields with defaults |
| Benchmark/Serve fails with OOM or "insufficient memory" | GPUs occupied by other processes | Kill GPU processes: `nvidia-smi --query-compute-apps=pid --format=csv,noheader \| xargs -r kill -9` then retry. **Never skip these steps.** |
| Model outputs garbled/gibberish text | `ColumnParallelLinear` used for merged projections with different sub-dimensions (TP sharding mismatch) | Override `__init__` to use `MergedColumnParallelLinear(output_sizes=[...])`. See P8 in compatibility-patches.md |
| `AssertionError: Duplicate op name` | Child class imports custom op from different module path than parent | Use same import path as parent module (e.g. `vllm_fl.ops.fla` not `vllm_fl.models.fla_ops`). See P11 |
| `AttributeError` on `fused_recurrent_*` during CUDA graph warmup | `__init__` override with `nn.Module.__init__(self)` missed attributes used by inherited `_forward_core` | Create ALL attributes from parent's `__init__`, especially custom ops. See P12 |
| E2E: local server not reachable | `serve.sh` port doesn't match `e2e_config.json` local port | Ensure both use same port (default 8122) |
| E2E: GT server not reachable | GT machine down or docker/conda env wrong | Check `e2e_remote_serve.sh status` or SSH manually |
| E2E: early token divergence (first 5 tokens) | Weight loading bug, TP sharding error | Check `load_weights`, `stacked_params_mapping`, MergedColumnParallelLinear |
| E2E: late minor divergence (token #15+) | Numerical noise from different op implementations | Usually acceptable; document in report |
| `resolve_op` fails with `VLLM_FL_PREFER_ENABLED=false` | Op not registered in dispatch, no fallback | Add try/except fallback to `flag_gems` in op import code |
