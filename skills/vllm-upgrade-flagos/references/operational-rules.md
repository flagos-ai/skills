# Operational Rules

Rules that apply throughout the entire upgrade process.

## Conda Environments

Two separate conda environments are used:

| Environment | Activate | Contents |
|---|---|---|
| `vllm_plugin_update` | `conda activate vllm_plugin_update` | vLLM v0.18.0+ installed (target version) |
| `vllm_0.13.0_plugin_tree` | `conda activate vllm_0.13.0_plugin_tree` | vLLM v0.13.0 + vllm-plugin-FL installed (current version) |

**Which environment to use when:**
- **Reading upstream code / git diff**: either environment works (just reading files)
- **Running plugin tests / import checks**: use `vllm_plugin_update` (the target — plugin must work against new vLLM)
- **Comparing old behavior**: use `vllm_0.13.0_plugin_tree`

**Activating in bash**: Each bash command runs in a fresh shell, so always activate at the start:
```bash
conda activate vllm_plugin_update && python3 -c "import vllm; print(vllm.__version__)"
```

## Communication Protocol

Actively communicate at every step boundary. Silent execution is unacceptable.

**Status line patterns:**
- Starting: `🔍 Step N: <what>...`
- Finding: `📋 Found: <what>`
- Decision: `✅ Decision: <what and why>`
- Issue: `⚠️ Issue: <problem>` → `🔧 Fix: <action>`
- Complete: `✅ Step N complete: <summary>`

**What to report** (concise, at step boundaries only):
1. Version info — detected current base and target versions (once)
2. File operations — files created/modified/deleted
3. Patch summary — list of FL customizations re-applied (batch)
4. Verification results — pass/fail with key output
5. Issues — only blocking problems or decisions needing user input

Use AskUserQuestion for ambiguity or choices.

## TaskList Integration

The TaskList is both a progress indicator AND the recovery mechanism after API interruptions.

### On first invocation — create all tasks upfront

After detecting versions, create ALL tasks at once:

```
Task 1:  Baseline unit tests
Task 2:  API diff analysis (upstream v0.13.0 → v{{target}})
Task 3:  Upgrade platform.py
Task 4:  Upgrade compilation/graph.py
Task 5:  Upgrade worker/worker.py
Task 6:  Upgrade worker/model_runner.py
Task 7:  Upgrade ops layer
Task 8:  Clean up obsolete models/configs/patches
Task 9:  Clean up __init__.py registration
Task 10: Import verification & regression tests
Task 11: Final report
```

### Auto-resume protocol

**ALWAYS** start every turn with `TaskList`. Then:
- `in_progress` tasks exist → continue immediately (do NOT ask user)
- All `pending`, none `in_progress` → fresh start from first pending
- All `completed` → output final report
- User says "continue"/"继续" → resume from first non-completed task

**NEVER ask whether to continue.** After an interruption, just read tasks and keep going.

### Task state discipline

- Mark `in_progress` BEFORE starting work
- Mark `completed` ONLY after fully done and verified
- Keep `in_progress` on failure; fix the issue first
- Task descriptions = single source of truth (enough detail for cold-start)

### Work-until-done principle

Keep working until ALL tasks are completed. Do not stop after one step and wait. Make maximum progress each turn.

## Bash Command Rules

### Timeout management

| Command type | Timeout |
|---|---|
| `pytest` (unit tests) | 120s |
| `python3 -c "import ..."` | 30s |
| `git diff`, `git log` | 30s |
| `diff`, `grep`, `find` | 15s |
| `pip install` | 300s |

### Output management

- Always pipe long output through `tail -N` or `head -N`
- For pytest: `pytest ... 2>&1 | tail -30`
- For git diff: `git diff ... | head -200`

### Common failure modes (lessons learned)

1. **Stopping at blocking steps instead of working through them.** If a step requires waiting, use background tasks and continue with independent work.
2. **Accidentally overwriting vLLM.** Running `pip install -e .` for the plugin can pull vLLM as a dependency. Always use `pip install --no-build-isolation --no-deps -e .` for the plugin.
3. **If vLLM gets accidentally overwritten**, restore with `MAX_JOBS=96 pip install --no-build-isolation -v -e <vllm_source>`.
4. **Not verifying imports after each component.** Run `python3 -c "from vllm_fl.xxx import ..."` after each upgrade step.

## Debugging Protocol

### Upstream-first

When a runtime error occurs:
1. Identify the failing import/call
2. Check the upstream file at the target version — what's the correct API?
3. Compare with what the plugin is using
4. Fix the plugin to match upstream

### CUDA_LAUNCH_BLOCKING

Only use `CUDA_LAUNCH_BLOCKING=1` when:
- A CUDA kernel crashes with an unhelpful traceback
- You need to pinpoint the exact failing kernel

Do NOT use for:
- Import errors or config parsing errors (CPU-side)
- Normal development/testing

## Resilience

### Auto-resume after interruption

If TaskList shows some tasks `completed` and some not → previous session was interrupted. Do NOT start over. Find first non-completed task, read its description, continue from there. NEVER re-do completed tasks.

## Permission Assumptions

- Full **read/write/execute** for vllm-plugin-FL project directory
- Full **read** for upstream vLLM source directory
- No `sudo`/`chmod` needed
- All fixes stay **inside the plugin directory**

## vLLM Version Protection

**NEVER** modify the installed vLLM version or its source code. Specifically:

1. **Do not change upstream vLLM code** — all customisation goes into the plugin.
2. **Do not reinstall/upgrade vLLM accidentally** — always use `pip install --no-build-isolation --no-deps -e .` for the plugin.
3. **If vLLM gets accidentally overwritten**, restore with `MAX_JOBS=96 pip install --no-build-isolation -v -e <vllm_source_dir>`.
