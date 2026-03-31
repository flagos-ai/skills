# Upgrade Procedure

## Goal

Upgrade vllm-plugin-FL's infrastructure code from its current vLLM base (v0.13.0) to match
`{{upstream_folder}}` (target: v0.18+). After this upgrade, the plugin's worker, model_runner,
platform, compilation, and ops layers should be compatible with the new vLLM APIs. Models now
natively supported upstream should be removed from the plugin.

## Constraints

* Modify **only** `vllm-plugin-FL` (installed with `-e --no-deps`)
* Do NOT modify the upstream vLLM source or installed vLLM packages
* Preserve ALL plugin customizations (dispatch, custom ops, platform, IO dumper, FL envs)
* **Idempotent** — safe to re-run; existing files get overwritten with fresh adaptations
* Reuse the copy-then-patch pattern from model-migrate-flagos
* Do NOT replace `current_platform.torch_device_fn.*` calls with `torch.cuda.*`. The plugin supports multiple backends (NVIDIA CUDA, Ascend NPU, MetaX MACA, etc.). `torch_device_fn` is the platform abstraction that routes to the correct device API at runtime. Hardcoding `torch.cuda.*` breaks non-CUDA platforms.

## Conda Environments

| Environment | Activate | Use for |
|---|---|---|
| `vllm_plugin_update` | `conda activate vllm_plugin_update` | Target env — vLLM v0.18.0+ installed. Run all import checks, tests, and verification here. |
| `vllm_0.13.0_plugin_tree` | `conda activate vllm_0.13.0_plugin_tree` | Old env — vLLM v0.13.0 + current plugin. Use for comparing old behavior or checking old API signatures. |

Each bash command runs in a fresh shell, so always prefix with `conda activate`:
```bash
conda activate vllm_plugin_update && <command>
```

All verification commands in this procedure should run in `vllm_plugin_update` unless stated otherwise.

---

## Plugin Infrastructure Map

| Plugin file | Upstream counterpart | Role |
|---|---|---|
| `vllm_fl/worker/worker.py` | `vllm/v1/worker/gpu_worker.py` | GPU worker (init, KV cache, exec loop) |
| `vllm_fl/worker/model_runner.py` | `vllm/v1/worker/gpu_model_runner.py` | Model runner (loading, forward, sampling) |
| `vllm_fl/platform.py` | `vllm/platforms/interface.py` + `vllm/platforms/cuda.py` | Platform abstraction |
| `vllm_fl/compilation/graph.py` | `vllm/compilation/cuda_graph.py` | CUDA graph wrapper |
| `vllm_fl/ops/fused_moe/layer.py` | `vllm/model_executor/layers/fused_moe/layer.py` | Fused MoE layer |
| `vllm_fl/ops/fused_moe/fused_moe.py` | `vllm/model_executor/layers/fused_moe/fused_moe.py` | Fused MoE kernel |
| `vllm_fl/ops/activation.py` | `vllm/model_executor/layers/activation.py` | Activation ops |
| `vllm_fl/ops/layernorm.py` | `vllm/model_executor/layers/layernorm.py` | LayerNorm ops |
| `vllm_fl/ops/rotary_embedding.py` | `vllm/model_executor/layers/rotary_embedding.py` | Rotary embedding ops |

---

## Step 1: Baseline Unit Tests

> **→ Tell user**: `🔍 Step 1: Running baseline unit tests before making any changes...`

```bash
conda activate vllm_plugin_update && pytest {{plugin_folder}}/tests/unit_tests/ -v --tb=short 2>&1 | tail -20
```

---

## Step 2: API Diff Analysis

> **→ Tell user**: `🔍 Step 2: Analyzing API differences between plugin base (v0.13.0) and upstream...`

For each plugin infrastructure file, compare it against its upstream counterpart.

### 2.1 Measure change scope

For each file pair in the Infrastructure Map:

```bash
cd {{upstream_folder}}
git log --oneline v0.13.0..HEAD -- <upstream_file_path> | wc -l
git diff v0.13.0..HEAD --stat -- <upstream_file_path>
```

### 2.2 Extract plugin customizations

For each plugin file, identify FL-specific additions that must be preserved:

```bash
diff {{plugin_folder}}/vllm_fl/worker/worker.py {{upstream_folder}}/vllm/v1/worker/gpu_worker.py | grep "^<" | head -80
```

Document each customization: which method, what it does, exact code block.

### 2.3 Categorize changes

For each upstream file, categorize into:
- **Import changes** — new/removed/renamed imports
- **Method signature changes** — parameters added/removed
- **New methods/classes** — didn't exist before
- **Removed methods/classes** — deleted
- **Structural changes** — class hierarchy, mixin, file splits
- **Config changes** — new config fields, renamed classes

Cross-reference with `api-changes.md` for known breaking changes.

> **→ Tell user**: Report summary per component (commit count, major API changes).

---

## Step 3: Upgrade platform.py

> **→ Tell user**: `🔧 Step 3: Upgrading platform.py...`

1. Read upstream: `vllm/platforms/interface.py` + `vllm/platforms/cuda.py`
2. Compare with plugin's `platform.py`:
   - New abstract methods needing implementation
   - Renamed methods or changed signatures
   - New platform capabilities or enums
3. Update `platform.py`:
   - Add new required method implementations
   - Update signatures to match new base class
   - Preserve FL-specific logic (multi-chip detection, DeviceInfo, etc.)
4. Verify:
   ```bash
   conda activate vllm_plugin_update && python3 -c "from vllm_fl.platform import PlatformFL; print('platform OK')"
   ```

---

## Step 4: Upgrade compilation/graph.py

> **→ Tell user**: `🔧 Step 4: Upgrading compilation/graph.py...`

1. Read upstream: `vllm/compilation/cuda_graph.py`
2. Compare: CUDAGraphWrapper, CUDAGraphStat, graph capture API changes
3. Update: Adapt plugin's GraphWrapper to new upstream APIs, preserve FL customizations
4. Verify:
   ```bash
   conda activate vllm_plugin_update && python3 -c "from vllm_fl.compilation.graph import GraphWrapper; print('compilation OK')"
   ```

---

## Step 5: Upgrade worker/worker.py

> **→ Tell user**: `🔧 Step 5: Upgrading worker/worker.py...`

### 5.1 Backup and extract customizations

```bash
cp {{plugin_folder}}/vllm_fl/worker/worker.py {{plugin_folder}}/vllm_fl/worker/worker.py.bak
```

Extract plugin customizations:
- FL platform initialization hooks
- Dispatch system integration (`register_oot_ops`, `managed_inference_mode`)
- IO dumper integration
- FL-specific env handling (`fl_envs`)
- `MemorySnapshot` dataclass (if FL-specific)
- `get_flag_gems_whitelist_blacklist` calls

### 5.2 Copy-then-patch

1. Copy upstream:
   ```bash
   cp {{upstream_folder}}/vllm/v1/worker/gpu_worker.py {{plugin_folder}}/vllm_fl/worker/worker.py
   ```
2. Add header: `# Adapted from vllm/v1/worker/gpu_worker.py @ v{{target_version}}`
3. Apply patches:
   - Add FL-specific imports
   - Re-apply all FL customizations from 5.1
   - Ensure `register_oot_ops` called at the right point
   - Preserve `managed_inference_mode` integration
4. Verify:
   ```bash
   conda activate vllm_plugin_update && python3 -c "from vllm_fl.worker.worker import Worker; print('worker OK')"
   ```

---

## Step 6: Upgrade worker/model_runner.py

> **→ Tell user**: `🔧 Step 6: Upgrading worker/model_runner.py — the largest component...`

### 6.1 Backup and catalog customizations

```bash
cp {{plugin_folder}}/vllm_fl/worker/model_runner.py {{plugin_folder}}/vllm_fl/worker/model_runner.py.bak
```

Catalog ALL plugin customizations:
- `managed_inference_mode` context manager usage
- IO dumper hooks
- Dispatch integration (`resolve_op`, `get_flag_gems_whitelist_blacklist`)
- Custom CUDA graph wrapper (`GraphWrapper` instead of `CUDAGraphWrapper`)
- FL envs (`fl_envs`)
- Custom op registration (`register_oot_ops`)
- FlagGems-specific attention backend selection
- Custom warmup logic
- FL-specific forward context modifications

Document each: which method, what it does, exact code block.

### 6.2 Copy-then-patch

1. Copy upstream:
   ```bash
   cp {{upstream_folder}}/vllm/v1/worker/gpu_model_runner.py {{plugin_folder}}/vllm_fl/worker/model_runner.py
   ```
2. Add header noting upstream version
3. Apply import patches:
   - Add FL-specific imports
   - Replace `CUDAGraphWrapper` with `GraphWrapper`
   - Handle moved imports (check `api-changes.md`)
4. Re-apply each plugin customization from catalog
5. Handle new upstream features:
   - New methods plugin doesn't customize → leave as-is
   - New methods interacting with customized code → adapt
   - Removed methods plugin was overriding → remove override
6. Verify:
   ```bash
   conda activate vllm_plugin_update && python3 -c "from vllm_fl.worker.model_runner import GPUModelRunner; print('model_runner OK')"
   ```

---

## Step 7: Upgrade ops layer

> **→ Tell user**: `🔧 Step 7: Checking ops layer compatibility...`

For each op file, compare plugin wrapper with upstream interface:

1. `fused_moe/layer.py` — FusedMoE class interface
2. `fused_moe/fused_moe.py` — kernel function signatures
3. `activation.py` — activation op signatures
4. `layernorm.py` — layernorm op signatures
5. `rotary_embedding.py` — rotary embedding signatures
6. `custom_ops.py` — OOT op registration

For each:
- If upstream interface changed → update plugin wrapper
- If new ops added upstream → check if dispatch needs updating
- Preserve dispatch integration (`resolve_op` pattern)

Known API migration patterns for ops:
- `from vllm._custom_ops import <fn>` — some functions removed in v0.18.1 (e.g. `silu_and_mul`). Use `torch.ops._C.<fn>` instead. Check each import against upstream `vllm/_custom_ops.py`.
- `from vllm.utils.import_utils import init_cached_hf_modules` → removed in v0.18.1. Delete the call entirely.
- Platform-specific fixes (e.g. CUDA kernel paths) should be marked with inline comments in code, not documented here. Keep this skill platform-agnostic.

```bash
conda activate vllm_plugin_update && python3 -c "from vllm_fl.ops import custom_ops; print('ops OK')"
```

---

## Step 8: Clean dead code (models, configs, patches)

> **→ Tell user**: `🔧 Step 8: Identifying and removing dead model/config/patch code...`

After upgrading, many models/configs/patches may now be natively supported upstream.
Do NOT blindly delete everything — some configs or patches may still be needed if they
haven't been upstreamed yet.

### 8.1 Identify dead model files

For each `.py` file in `{{plugin_folder}}/vllm_fl/models/` (excluding `__init__.py`):

1. Check if the model class exists in upstream `{{upstream_folder}}/vllm/model_executor/models/`
2. Check if the model class is registered in upstream's `registry.py`
3. Check if the FL model is still registered in `{{plugin_folder}}/vllm_fl/__init__.py` `register_model()`
4. If the model exists upstream AND is not registered by the plugin → it's dead code, delete it

### 8.2 Identify dead config files

For each `.py` file in `{{plugin_folder}}/vllm_fl/configs/` (excluding `__init__.py`):

1. Check if the config's `model_type` exists in upstream's `_CONFIG_REGISTRY` in
   `{{upstream_folder}}/vllm/transformers_utils/config.py`
2. Check if the config is still imported/used by any remaining plugin code
3. If the config exists upstream AND is not imported by remaining plugin code → delete it

### 8.3 Identify dead patch files

For each `.py` file in `{{plugin_folder}}/vllm_fl/patches/` (excluding `__init__.py`):

1. Check if the patches fix issues that are already resolved in the target upstream version
2. Check if the patches are still imported/called from `__init__.py` or other plugin code
3. Only delete patches that are confirmed no longer needed

### 8.4 Verify no dangling imports

```bash
conda activate vllm_plugin_update && grep -rn 'from vllm_fl\.models\.\|from vllm_fl\.configs\.\|from vllm_fl\.patches\.' \
  {{plugin_folder}}/vllm_fl/ --include='*.py' | grep -v __pycache__
```

Any remaining imports of deleted files must be removed.

> **→ Tell user**: Report which files were deleted and which were kept (with reasons).

---

## Step 9: Clean up __init__.py

> **→ Tell user**: `🔧 Step 9: Cleaning up __init__.py...`

1. In `register_model()`: remove registration code for models/configs that are now upstream
2. Keep `register_model()` if it still registers anything not yet upstream (e.g. custom configs)
3. If `register_model()` is completely empty after cleanup, delete the function entirely
4. In `register()`: remove patch imports/calls for patches that were deleted in Step 8
5. Remove `_patch_transformers_compat()` only if no remaining patches or models need it
6. Keep `register()` function — platform registration is still needed
7. Verify:
   ```bash
   conda activate vllm_plugin_update && python3 -c "import vllm_fl; print('init OK')"
   ```

---

## Step 10: Import Verification & Regression Tests

> **→ Tell user**: `🔍 Step 10: Running full verification...`

### 10.1 Import verification

```bash
conda activate vllm_plugin_update && python3 -c "
import vllm_fl
print('vllm_fl OK')
from vllm_fl.platform import PlatformFL
print('platform OK')
from vllm_fl.compilation.graph import GraphWrapper
print('compilation OK')
from vllm_fl.worker.worker import Worker
print('worker OK')
from vllm_fl.worker.model_runner import GPUModelRunner
print('model_runner OK')
from vllm_fl.ops.custom_ops import register_oot_ops
print('ops OK')
print('All imports passed!')
"
```

### 10.2 Offline inference end-to-end test

This is the **primary verification gate**. The upgrade is considered successful only when this
command completes without error:

```bash
conda activate vllm_plugin_update && python {{plugin_folder}}/examples/qwen3_5_offline_inference.py
```

This script exercises the full plugin stack (platform → worker → model_runner → ops → dispatch)
with a real model inference. If it runs to completion and produces output, the upgrade is verified.

If it fails, debug the traceback — it will point to the exact plugin component that still has
incompatibilities with the new vLLM version.

### 10.3 Regression unit tests

```bash
conda activate vllm_plugin_update && pytest {{plugin_folder}}/tests/unit_tests/ -v --tb=short 2>&1 | tail -30
```

Compare with Step 1 baseline. Fix new failures.

### 10.4 Functional tests

```bash
conda activate vllm_plugin_update && pytest {{plugin_folder}}/tests/functional_tests/ -v --tb=short 2>&1 | tail -30
```

### 10.5 Performance benchmark

Run the throughput benchmark to establish a post-upgrade performance baseline. Compare with
pre-upgrade numbers (if available) to detect performance regressions.

```bash
conda activate vllm_plugin_update && bash {{plugin_folder_relative_to_skill}}/scripts/benchmark.sh <MODEL_DISPLAY_NAME>
```

For example, with Qwen3.5-397B:

```bash
conda activate vllm_plugin_update && bash scripts/benchmark.sh Qwen3.5-397B-A17B-Real
```

The script runs `vllm bench throughput` with:
- 100 prompts, input_len=6144, output_len=1024
- TP=4, gpu_memory_utilization=0.9, dummy weights

Key metrics to check:
- **Throughput** (tokens/s): should not regress significantly vs pre-upgrade
- **TTFT** (time to first token): latency should remain comparable
- **No OOM or CUDA errors**: confirms memory management is correct after upgrade

> **→ Tell user**: Report throughput numbers and comparison with pre-upgrade baseline (if available).

---

## Step 11: Update Version References

> **→ Tell user**: `🔧 Step 11: Updating version references...`

1. Update header comments in worker.py and model_runner.py to reference new base version
2. Update `pyproject.toml` vLLM version pin if needed
3. Update `vllm_fl/version.py` if needed

---

## Final Report

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Plugin Upgrade Complete: vLLM v0.13.0 → v{{target_version}}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Infrastructure upgraded:
  - vllm_fl/platform.py
  - vllm_fl/compilation/graph.py
  - vllm_fl/worker/worker.py
  - vllm_fl/worker/model_runner.py
  - vllm_fl/ops/...
  - vllm_fl/__init__.py

Models deleted (now upstream):
  - models/, configs/, patches/ — all cleared

Plugin customizations preserved:
  - dispatch, custom ops, platform, compilation, IO dumper, FL envs

Verification:
  - Offline inference (qwen3_5_offline_inference.py): PASS / FAIL
  - Unit (baseline):    N passed, M failed, K skipped
  - Unit (regression):  N passed, M failed, K skipped
  - Functional:         N passed, M failed, K skipped

Performance benchmark:
  - Model: <MODEL_DISPLAY_NAME>
  - Throughput: X.XX tokens/s (pre-upgrade: Y.YY tokens/s)
  - Regression: +/-Z.Z%

Known issues / TODOs:
  - [list or "None"]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
