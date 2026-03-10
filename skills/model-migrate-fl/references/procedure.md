# Migration Procedure

## Goal

Enable benchmarking **{{MODEL_DISPLAY_NAME}}** — the target command is `bash scripts/benchmark.sh {{MODEL_DISPLAY_NAME}}` (see `scripts/benchmark.sh` for exact flags).

## Constraints

* vLLM version: **0.13.0**
* Modify **only** `vllm-plugin-FL` (installed with `-e`)
* Do NOT modify `/usr/local/lib` or `/models/`
* Do NOT import code from **vLLM > 0.13.0**
* Do NOT add any **extra environment variables** during final verification
* Reuse existing plugin patterns (e.g. `qwen3_5`, `kimi_k25`, `qwen3_next`, `minicpmo`)
* **Idempotent** — if target files already exist, overwrite with latest upstream (re-adapted). Safe to re-run.

---

## Project Structure

```
vllm-plugin-FL/
  vllm_fl/
    __init__.py            # register() and register_model() entry points
    configs/               # HuggingFace config bridges
      qwen3_5_moe.py       # Example: Qwen3.5 MoE config
    models/                # Model implementations
      qwen3_5.py            # Example: Complex MoE + hybrid attention
      kimi_k25.py           # Example: Wrapper around DeepseekV2
      qwen3_next.py         # Example: Hybrid attention model
      minicpmo.py           # Example: Multimodal model
      fla_ops.py            # Shared linear attention operations
    ops/                   # Custom operators
    dispatch/              # Dispatch backends (flaggems, cuda, etc.)
  setup.py                # Entry points: vllm.platform_plugins, vllm.general_plugins
```

---

## Step 1: Baseline Unit Tests

> **→ Tell user**: `🔍 Step 1: Running baseline unit tests before making any changes...`

Run all unit tests **before modifying any plugin code**:

```bash
pytest {{plugin_folder}}/tests/unit_tests/ -v --tb=short
```

Record: total passed / failed / skipped / errors, and any **pre-existing failure** names.

- **All pass** → clean baseline; any post-migration failure is a regression
- **Some fail** → note exact test names to exclude from regression comparison

> **→ Tell user**: Report baseline (count + pre-existing failures if any).

---

## Step 2: Prepare Upstream Reference Source

> **→ Tell user**: `🔍 Step 2: Cloning/updating upstream vLLM repository...`

```bash
test -d {{upstream_folder}} && cd {{upstream_folder}} && git pull || git clone --depth 1 https://github.com/vllm-project/vllm.git {{upstream_folder}}
```

From `{{upstream_folder}}`, find the model files:
- Search `vllm/model_executor/models/` for `{{model_name_lower}}`
- Check `vllm/transformers_utils/config.py` for config registration
- Note class names, import paths, and `model_type`

> **→ Tell user**: Report files found, class names, model_type.

---

## Step 3: Study Existing Plugin Patterns

> **→ Tell user**: `🔍 Step 3: Studying existing plugin patterns...`

| Model | Pattern | Key characteristic |
|---|---|---|
| `qwen3_5` | Complex MoE + hybrid attention | Full model file adaptation |
| `kimi_k25` | Wrapper around DeepseekV2 | Lightweight delegation |
| `qwen3_next` | Hybrid attention model | Custom attention integration |
| `minicpmo` | Multimodal model | Vision + language composition |

Learn: config bridge pattern, model registration in `__init__.py`, 0.13.0 adaptation techniques.

> **→ Tell user**: Which existing model is most similar, which pattern you'll follow, and why.

---

## Step 4: Add Model Files

> **→ Tell user**: `🔍 Step 4: Creating model files for {{MODEL_DISPLAY_NAME}}...`

### 4.1 Config Bridge (if needed)

If `{{model_type}}` is NOT known to vLLM 0.13.0's transformers, create:
- File: `{{plugin_folder}}/vllm_fl/configs/{{model_name_lower}}.py`
- Subclass `transformers.PretrainedConfig`, set `model_type = "{{model_type}}"`
- Copy constructor parameters from upstream config class

> **→ Tell user**: Whether config bridge is needed and why.

### 4.2 Model File — Copy-then-Patch

**IMPORTANT: Copy-then-patch, NOT read-and-rewrite.**

**A.** Copy upstream file:
```bash
cp {{upstream_folder}}/vllm/model_executor/models/{{model_name_lower}}.py {{plugin_folder}}/vllm_fl/models/{{model_name_lower}}.py
```

**B.** Apply patches from `compatibility-patches.md` using the Edit tool.

**C.** Verify import:
```bash
python3 -c "from vllm_fl.models.{{model_name_lower}} import {{ModelClassName}}; print('OK')"
```
If it fails, fix the specific error and retry. Do NOT rewrite the whole file.

> **→ Tell user**: Single summary after all patches (files created, patches applied, import test result).

---

## Step 5: Register Model

> **→ Tell user**: `🔍 Step 5: Registering {{MODEL_DISPLAY_NAME}} in plugin entry point...`

In `{{plugin_folder}}/vllm_fl/__init__.py`, add to `register_model()`:

1. (If config bridge created) Register config:
```python
from vllm.transformers_utils.config import _CONFIG_REGISTRY
from vllm_fl.configs.{{model_name_lower}} import {{ConfigClassName}}
_CONFIG_REGISTRY["{{model_type}}"] = {{ConfigClassName}}
```

2. Register model class:
```python
ModelRegistry.register_model(
    "{{ModelClassName}}",
    "vllm_fl.models.{{model_name_lower}}:{{ModelClassName}}"
)
```

Both wrapped in try/except to avoid breaking other models.

> **→ Tell user**: Registration summary (config + model entries added).

---

## Step 6: Post-Migration Code Review

> **→ Tell user**: `🔍 Step 6: Reviewing migrated code for correctness...`

### 6.1 Automated checks

Run the validation script (relative to this skill's root):

```bash
python3 {{skill_root}}/scripts/validate_migration.py {{plugin_folder}} {{plugin_folder}}/vllm_fl/models/{{model_name_lower}}.py {{plugin_folder}}/vllm_fl/configs/{{model_name_lower}}.py
```

Omit the config file argument if no config bridge was created. The script checks:
- **Imports**: relative imports, missing `vllm_fl.*` modules
- **API compatibility**: known 0.13.0-missing APIs (`_mark_tower_model`, `MambaStateCopyFunc`, etc.)
- **Config consistency**: `model_type` defined, `PretrainedConfig` subclass, `__init__` present
- **Registration**: class names and import paths in `__init__.py` match actual code
- **Code cleanliness**: bare `except:`, hardcoded external paths

Exit code 0 = passed, 1 = issues found. **Fix all ISSUES before proceeding.** WARNINGS are informational.

### 6.2 Manual review (items the script cannot check)

- Inherited base class method signatures match 0.13.0 (compare with `/usr/local/lib/python*/dist-packages/vllm/`)
- Config bridge fields match upstream `config.json` defaults (no missing required fields)
- No commented-out upstream code left behind (remove or adapt)
- No references to vLLM > 0.13.0 features beyond the known API list
- **TP sharding correctness** (see P8): If the model has merged projections (e.g. `in_proj_qkvz`, `in_proj_ba`) where sub-outputs have different dimensions, verify they use `MergedColumnParallelLinear` (NOT `ColumnParallelLinear`). Check: does the parent's `__init__` create `ColumnParallelLinear` for these? If so, the child MUST override `__init__` to use `MergedColumnParallelLinear` with correct `output_sizes`.
- **Complete `__init__` override** (see P12): If the model overrides `__init__` with `nn.Module.__init__(self)`, verify ALL attributes used by inherited methods (`_forward_core`, etc.) are created with correct constructor args.
- **Plugin import paths** (see P11): If custom ops are imported, verify the import path matches the parent module's import path (avoid duplicate `CustomOp.register`).

> **→ Tell user**: Report script output + manual review findings. Fix any issues before proceeding.

---

## Step 7: Regression Unit Tests

> **→ Tell user**: `🔍 Step 7: Running unit tests to check for regressions...`

```bash
pytest {{plugin_folder}}/tests/unit_tests/ -v --tb=short
```

Compare with Step 1 baseline:
- **New failures** → regression from our migration → **MUST FIX** before proceeding
- **Same failures as baseline** → pre-existing, not us → continue
- **All pass** → clean

> **→ Tell user**: Results + comparison with baseline. Fix and re-run on regressions.

---

## Step 8: Functional Tests

> **→ Tell user**: `🔍 Step 8: Running functional tests...`

```bash
pytest {{plugin_folder}}/tests/functional_tests/ -v -s --tb=short
```

- **Auto-skip** (missing weights / GPU) → expected, warn and continue
- **Pass** → existing inference not broken
- **Fail** → warn and continue (may be pre-existing or env-specific), include in final report

> **→ Tell user**: Results with notes on skips/failures.

---

## Step 9: Verify — Benchmark

> **→ Tell user**: `🔍 Step 9: Running benchmark...`

**9.0 — Release GPU resources** (see operational-rules.md § GPU Resource Management):
```bash
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader
```
If GPUs are occupied (free < 50% of total), forcefully release:
```bash
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9
sleep 5
```

**9.1 — Run benchmark**:
```bash
bash {{skill_root}}/scripts/benchmark.sh {{MODEL_DISPLAY_NAME}}
```

> **→ Tell user**: Pass/fail + throughput metrics. On failure: analyze, fix, and re-run.

---

## Step 10: Verify — Serve + Request

> **→ Tell user**: `🔍 Step 10: Starting serve + request verification...`

**10.0 — Release GPU resources** (same as Step 9.0 — skip if GPUs were already freed in Step 9).

### 10.1 Serve
```bash
bash {{skill_root}}/scripts/serve.sh {{MODEL_DISPLAY_NAME}}
```

### 10.2 Request
```bash
bash {{skill_root}}/scripts/request.sh {{MODEL_DISPLAY_NAME}}
```

> **→ Tell user**: Pass/fail for both. On failure: analyze, fix, retry.

---

## Step 11: E2E Correctness Verification

> **→ Tell user**: `🔍 Step 11: Running E2E correctness verification against upstream GT...`

This step compares the local plugin's output token-by-token against an upstream (vanilla) vLLM server to verify migration correctness.

### Prerequisites

- **GT server**: An upstream vLLM instance serving the same model. Can be remote (managed via `e2e_remote_serve.sh`) or any reachable endpoint.
- **Local server**: Must already be running from Step 10.1 (`serve.sh`).

### 11.1 (Optional) Start GT server remotely

```bash
bash {{skill_root}}/scripts/e2e_remote_serve.sh start {{skill_root}}/scripts/e2e_config.json {{MODEL_DISPLAY_NAME}}
```

Skip if GT server is already running. The `e2e_remote_serve.sh` script requires a config file; use `e2e_config.template.json` as base or pass a custom one.

### 11.2 Run E2E evaluation

All parameters have defaults — no config file needed. CLI args override everything:

```bash
python3 {{skill_root}}/scripts/e2e_eval.py --model {{MODEL_DISPLAY_NAME}} --gt-host <GT_IP> --gt-port 8122 --local-port 8122 --mode text
```

Optionally load a config file (CLI args still override):
```bash
python3 {{skill_root}}/scripts/e2e_eval.py --config {{skill_root}}/scripts/e2e_config.json --model {{MODEL_DISPLAY_NAME}} --mode text
```

Key CLI args:
- `--gt-host` — GT server IP (default: 10.0.9.57)
- `--gt-port` — GT server port (default: 8122)
- `--local-port` — local server port (default: 8122, matching `serve.sh`)
- `--results-dir` — where to save results (default: `scripts/../results/`)
- `--max-tokens` — max tokens to generate (default: 256)
- `--token-match-count` — how many tokens to compare (default: 32)

Prompt modes:
- `--mode text` — text-only prompts (5 prompts, default)
- `--mode multimodal` — multimodal prompts (5 prompts, requires VL model)
- `--mode all` — all 10 prompts

Phase control:
- `--gt-only` — only collect GT results (useful when GT server is temporary)
- `--local-only` — only collect local results (GT results already saved)
- `--compare-only` — only compare previously saved results

### 11.4 Interpret results

The script compares the first N tokens (default 32) between GT and local output. Results are saved to `shared_storage.results_dir`.

- **All MATCH** → migration correctness verified
- **Minor divergence** (e.g. 1-2 tokens differ after token #15) → likely acceptable numerical noise
- **Early divergence** (within first 5 tokens) → likely a code bug, investigate

> **→ Tell user**: Report E2E results (passed/failed/errors count, any divergence details). On failure: investigate weight loading, TP sharding, or op implementation differences.

---

## Final Report

After ALL steps complete, output:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Migration Complete: {{MODEL_DISPLAY_NAME}}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Files created/modified:
  - vllm_fl/configs/xxx.py     (config bridge)
  - vllm_fl/models/xxx.py      (model impl — ~N lines)
  - vllm_fl/__init__.py         (registration added)

Compatibility fixes applied:
  1. [brief per-fix description]

Code review results:
  - [issues found and fixed, or "Clean — no issues"]

Test results:
  - Unit (baseline):    N passed, M failed, K skipped
  - Unit (regression):  N passed, M failed, K skipped — regressions: none/list
  - Functional:         N passed, M failed, K skipped — notes

Verification:
  - Benchmark:  ✅ / ❌
  - Serve:      ✅ / ❌
  - Request:    ✅ / ❌
  - E2E (text): ✅ / ❌  (N/M prompts matched, first 32 tokens)
  - E2E (mm):   ✅ / ❌  (N/M prompts matched, first 32 tokens)

Pre-existing issues: [list or "None"]
Known issues / TODOs: [list or "None"]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
