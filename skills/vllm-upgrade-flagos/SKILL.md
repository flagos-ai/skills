---
name: vllm-upgrade-flagos
description: >
  Upgrade the vllm-plugin-FL plugin's infrastructure code to match a newer upstream vLLM version.
  This skill handles upgrading core components like worker, model_runner, platform, compilation,
  ops, and dispatch — plus cleaning up model/config code that is now natively supported upstream.
  Use this skill whenever someone wants to upgrade the plugin's vLLM base version, sync the plugin
  with a newer vLLM release, update the plugin's worker or model_runner, or bump the vLLM pin.
  Trigger when the user says things like "upgrade plugin to vLLM 0.18", "sync plugin with upstream
  vLLM", "update vllm-plugin-FL base version", "bump vLLM version for the plugin", or simply
  "/vllm-upgrade-flagos".
argument-hint: "[upstream_folder] [plugin_folder]"
user-invocable: true
compatibility: "Requires vLLM source (upstream), vllm-plugin-FL source, Python 3.10+, Linux"
metadata:
  version: "1.0.0"
  author: flagos-ai
  category: workflow-automation
  tags: [vllm, upgrade, plugin, infrastructure, version-bump]
allowed-tools: "Bash(pytest:*) Bash(python3:*) Bash(git:*) Bash(diff:*) Bash(cp:*) Bash(grep:*) Bash(find:*) Bash(wc:*) Bash(head:*) Bash(tail:*) Bash(cat:*) Bash(ls:*) Bash(pip:*) Bash(cd:*) Bash(mkdir:*) Bash(bash:*) Bash(test:*) Read Edit Write Glob Grep AskUserQuestion TaskCreate TaskUpdate TaskList TaskGet"
---

# FL Plugin — vLLM Version Upgrade Skill

## Usage

```
/vllm-upgrade-flagos [upstream_folder] [plugin_folder]
```

| Argument | Required | Default |
|---|---|---|
| `upstream_folder` | No | `/workspace/vllm_update/vllm` |
| `plugin_folder` | No | `/workspace/vllm_update/vllm-plugin-FL` |

## Overview

This skill upgrades the vllm-plugin-FL plugin's **infrastructure layer** to be compatible with a
newer upstream vLLM version. The plugin is an out-of-tree (OOT) extension that overrides or extends
vLLM's worker, model_runner, platform, compilation, and ops subsystems. When vLLM's internal APIs
change between versions, the plugin code must be updated to match.

After upgrading to a newer vLLM version, models that were previously only available via the plugin
may now be natively supported upstream. In that case, the plugin's model files, config bridges, and
registration code should be cleaned up — the plugin no longer needs to carry them.

## Execution

### Step 1: Parse arguments and detect versions

Extract from user input:
- `{{upstream_folder}}` = first argument or `/workspace/vllm_update/vllm`
- `{{plugin_folder}}` = second argument or `/workspace/vllm_update/vllm-plugin-FL`

Detect current state:
```bash
# Upstream vLLM version
cd {{upstream_folder}} && git describe --tags 2>/dev/null || git log --oneline -1

# Verify target env vLLM version
conda activate vllm_plugin_update && python3 -c "import vllm; print(vllm.__version__)"

# Current plugin base version (check comments in worker/model_runner files)
head -5 {{plugin_folder}}/vllm_fl/worker/model_runner.py
head -5 {{plugin_folder}}/vllm_fl/worker/worker.py
```

Report detected versions (current plugin base → target upstream) to the user.

### Step 2: Load references and create task list

Read these files (relative to this SKILL.md):
- `references/procedure.md` — step-by-step upgrade procedure
- `references/api-changes.md` — known API changes between vLLM versions
- `references/operational-rules.md` — communication, TaskList, bash rules

Then create the full TaskList per `operational-rules.md`.

### Step 3–11: Follow procedure.md

The detailed upgrade procedure is in `references/procedure.md`. It covers:

1. Baseline tests
2. API diff analysis (upstream vs plugin)
3. Platform upgrade
4. Compilation/graph upgrade
5. Worker upgrade
6. Model runner upgrade
7. Ops layer upgrade
8. Delete all model/config/patch code (now upstream)
9. Clean up __init__.py (remove register_model, model patches)
10. Import verification, offline inference end-to-end test & regression tests
11. Final report

## Key Principles

**Copy-then-patch, not rewrite.** Copy the upstream file, then apply plugin-specific patches
(dispatch hooks, custom ops, FL platform logic). This makes future upgrades easier because you
can diff against the upstream version. Always add a header comment noting which upstream version
the file was adopted from.

**Preserve plugin customizations.** The plugin adds significant custom logic on top of vLLM:
- Dispatch system (`vllm_fl/dispatch/`) — routes ops to different backends (FlagGems, CUDA, Ascend)
- Custom ops (`vllm_fl/ops/`) — activation, layernorm, rotary, fused_moe, fla
- Platform abstraction (`vllm_fl/platform.py`) — multi-chip support
- Compilation hooks (`vllm_fl/compilation/`) — custom graph capture
- IO dumping (`vllm_fl/dispatch/io_dumper.py`, `io_common.py`) — inference debugging
- FL envs (`vllm_fl/envs.py`) — plugin-specific environment variables

When upgrading, these customizations must be preserved and adapted to the new API.

**Do NOT replace `current_platform.torch_device_fn.*` with `torch.cuda.*`.** The plugin supports
multiple backends (NVIDIA CUDA, Ascend NPU, MetaX MACA, etc.). `torch_device_fn` is the platform
abstraction that routes to the correct device API at runtime. Hardcoding `torch.cuda.*` breaks
non-CUDA platforms.

**Delete all model code.** After upgrading to v0.18+, all models are natively supported upstream.
The plugin no longer needs to carry any model-specific code. Delete entirely:
- All model files in `vllm_fl/models/`
- All config bridges in `vllm_fl/configs/` (keep `__init__.py` empty)
- All patches in `vllm_fl/patches/` (keep `__init__.py` empty)
- The entire `register_model()` function in `vllm_fl/__init__.py`
- Any model-specific patch calls in `register()` (e.g. `glm_moe_dsa` platform patches)

**Incremental verification.** After each major component upgrade, run import checks before
moving to the next component. Don't upgrade everything at once.

**Upstream-first debugging.** When something breaks, first compare the upstream code with the
plugin adaptation. The diff is the fastest path to root cause.

## Component Dependency Order

Upgrade components in this order (each depends on the previous):

```
1. platform.py          (foundation — device detection, capabilities)
2. compilation/         (graph capture, CUDA graph wrappers)
3. worker/worker.py     (process management, initialization)
4. worker/model_runner.py (model execution — the largest and most complex file)
5. ops/                 (custom operators — may need new signatures)
6. __init__.py          (remove register_model, model patches)
7. models/ + configs/ + patches/  (delete all — now upstream)
```

## Identifying Plugin Customizations

Before overwriting a plugin file with the upstream version, extract the plugin-specific additions.
These are the lines that exist in the plugin but NOT in the upstream counterpart. Common patterns:

```python
# FL-specific imports
from vllm_fl.ops.custom_ops import register_oot_ops
from vllm_fl.dispatch.io_common import managed_inference_mode
from vllm_fl.utils import get_flag_gems_whitelist_blacklist
import vllm_fl.envs as fl_envs

# FL-specific initialization
register_oot_ops()
managed_inference_mode(...)
get_flag_gems_whitelist_blacklist(...)

# FL-specific compilation hooks
from vllm_fl.compilation.graph import FLGraphWrapper  # replaces CUDAGraphWrapper

# FL-specific platform checks
from vllm_fl.platform import PlatformFL
```

Use `diff` to identify these before starting the copy-then-patch process.

## Examples

**Example 1: Full upgrade from v0.13.0 to v0.18+**
```
User: "upgrade plugin to match the latest vLLM"
Actions:
  1. Detect: plugin based on v0.13.0, upstream at v0.18.1
  2. API diff: identify changed imports, new classes, removed APIs
  3. Upgrade platform.py, compilation/graph.py, worker.py, model_runner.py
  4. Upgrade ops layer
  5. Clean dead model/config/patch code — check what's upstream before deleting
  6. Clean up __init__.py — remove obsolete registrations
  7. Verify imports, run tests
  8. Run performance benchmark: bash scripts/benchmark.sh Qwen3.5-397B-A17B-Real
Result: plugin infrastructure upgraded, dead code cleaned, performance verified
```

**Example 2: Incremental sync after upstream minor bump**
```
User: "upstream vLLM updated from 0.18.0 to 0.18.1, sync plugin"
Actions:
  1. Detect: small version bump, few API changes
  2. Diff only changed files between versions
  3. Apply targeted patches to affected plugin files
  4. Verify, test, and benchmark
Result: plugin synced with minor upstream changes
```

## Troubleshooting

| Problem | Typical Cause | Fix |
|---|---|---|
| `ImportError: cannot import name X from vllm` | API renamed or moved in new version | Check `api-changes.md`; update import path |
| `TypeError: __init__() got unexpected keyword` | Constructor signature changed | Compare upstream class signature; update call site |
| `AttributeError: module has no attribute X` | Module restructured | Check upstream module layout; update import |
| Worker crashes on init | New required config fields or init steps | Compare upstream worker `__init__` with plugin |
| CUDA graph capture fails | CUDAGraphWrapper API changed | Check compilation changes; update `graph.py` |
| `pip install -e .` overwrites vLLM | Plugin pulls vLLM as dependency | Always use `pip install --no-build-isolation --no-deps -e .` |
