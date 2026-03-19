# Operational Rules

Rules that apply throughout the entire migration process.

## Communication Protocol

Actively communicate at every step boundary. Silent execution is unacceptable.

**Status line patterns:**
- Starting: `🔍 Step N: <what>...`
- Finding: `📋 Found: <what>`
- Decision: `✅ Decision: <what and why>`
- Issue: `⚠️ Issue: <problem>` → `🔧 Fix: <action>`
- Complete: `✅ Step N complete: <summary>`

**What to report** (concise, at step boundaries only):
1. Model identity — resolved placeholder values (once)
2. File operations — files created/modified
3. Patch summary — list of patches applied (batch)
4. Verification results — pass/fail with key output
5. Issues — only blocking problems or decisions needing user input

Use AskUserQuestion for ambiguity or choices (e.g. HuggingFace repo, inheritance strategy).

## TaskList Integration

The TaskList is both a progress indicator AND the recovery mechanism after API interruptions.

### On first invocation — create all tasks upfront

After parsing the model name, create ALL tasks at once. Each description MUST include concrete model context for cold-start recovery:

```
Task 1:  Baseline unit tests
Task 2:  Clone/update upstream vLLM
Task 3:  Investigate model & resolve placeholders
Task 4:  Study existing plugin patterns
Task 5:  Create config bridge
Task 6:  Create model file (copy-then-patch)
Task 7:  Register model in __init__.py
Task 8:  Post-migration code review
Task 9:  Regression unit tests
Task 10: Functional tests
Task 11: Benchmark verification
Task 12: Serve + request verification
Task 13: E2E correctness verification (text + multimodal vs upstream GT)
```

Once placeholders are resolved, UPDATE task descriptions with concrete values.

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

### Common failure modes (lessons learned)

1. **Stopping at blocking steps instead of working through them.** If a step requires waiting (e.g. compilation, server startup), use background tasks and continue with independent work. Never leave a session with incomplete tasks when you could have continued.
2. **Getting stuck in monkey-patching rabbit holes.** When a compatibility issue arises (e.g. tokenizer class not found), try the simplest fix first: `.pth` file, wrapper script, or config override. Don't spend multiple rounds on increasingly complex import hook approaches.
3. **Accidentally overwriting vLLM.** Running `pip install -e .` for the plugin can pull vLLM as a dependency. Always use `--no-deps` to prevent this. If it happens, restore with `MAX_JOBS=96 pip install --no-build-isolation -v -e <vllm_source>`.
4. **Not running E2E to completion.** The migration is NOT done until E2E correctness verification passes. Benchmark/serve smoke tests are intermediate checkpoints, not the finish line.

## Bash Command Rules

Prevents permission prompts during migration:

1. **Single-line commands only** — no backslash `\` continuation. Chain with `&&` or use separate calls.
2. **No process substitution** — no `<()` or `>()`. Use pipes or temp files.
3. **No quoted flag values** — write `--load-format dummy` not `--load-format "dummy"`.
4. **Use Edit tool for file modifications** — never `sed -i`.
5. **Simple command prefixes** — start with `ls`, `cp`, `grep`, `python3`, `git`, `vllm`. Avoid `then`, `else`, `{`, `(`.
6. **Complex scripts** — write to temp `.sh`/`.py` file, then execute in separate call.

## GPU Resource Management

Before running benchmark (Step 9) or serve (Step 10), **always check GPU availability first**. If GPUs are occupied by other processes, **forcefully release them** — do NOT skip or mark as "skipped due to resource constraints".

### Protocol

1. Check GPU status:
```bash
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader
```

2. If free memory per GPU is insufficient (< 50% of total), kill the occupying processes:
```bash
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9
```

3. Wait briefly for memory to be released:
```bash
sleep 5
```

4. Verify GPUs are freed, then proceed with benchmark/serve.

**NEVER skip benchmark or serve due to GPU memory.** Always free GPUs and complete all verification steps.

## Serve / Benchmark Command Defaults

When launching `vllm serve` or `vllm bench`, **always** include these flags unless there is a specific reason not to:

1. **`--load-format fastsafetensors`** — enables parallel shard loading via GDS/io_uring, cutting weight load time from ~11 min to ~2 min for large models. This is the standard on this cluster.
2. **`VLLM_USE_DEEP_GEMM=0`** — `deep_gemm` is not installed in this environment. Without this env var, FP8 MLA models will crash at inference time with `RuntimeError: DeepGEMM backend is not available`.

Example:
```bash
VLLM_USE_DEEP_GEMM=0 vllm serve /models/<MODEL> --tensor-parallel-size 8 --trust-remote-code --port 8000 --max-model-len 4096 --load-format fastsafetensors
```

## Debugging Priority: Upstream-First

When inference errors, crashes, or unexpected behavior occur during or after migration, **always check vLLM upstream code first** before attempting local fixes.

### Protocol

1. **Compare upstream vs local**: diff the upstream model file (`{{upstream_folder}}/vllm/model_executor/models/`) against the plugin model file to identify any adaptation errors introduced during migration.
2. **Compare upstream vs 0.13.0 environment**: diff the upstream vLLM framework code (attention backends, config, utils, etc.) against the installed 0.13.0 version (`/mine/vllm/` or `/usr/local/lib/python*/dist-packages/vllm/`) to find API or behavior differences that the model code depends on.
3. **Root-cause from the diff**: most runtime errors fall into one of:
   - Migration patch introduced a bug (wrong import path, missing attribute, signature mismatch)
   - Upstream model code relies on a newer vLLM API not present in 0.13.0 (needs compatibility patch)
   - Environment difference (CUDA version, deep_gemm version, library availability)
4. **Only after upstream comparison** should you attempt speculative fixes, monkey-patches, or workarounds.

### Why

The plugin's model code is a direct copy-then-patch of upstream. When something breaks, the diff between upstream and our adaptation is the fastest path to root cause. Guessing or adding blind patches without comparing wastes time and often introduces new issues.

### CUDA Error Localization

When a serve or inference step crashes with a CUDA error (e.g. `CUDA error: an illegal memory access`, `CUDA error: misaligned address`, `RuntimeError: CUDA error`), the default stack trace is **misleading** because CUDA kernels execute asynchronously — the error is reported at a later CUDA synchronization point, not at the line that actually triggered it.

**Use `CUDA_LAUNCH_BLOCKING=1`** to force synchronous CUDA execution, which makes the error appear at the exact offending line:

```bash
CUDA_LAUNCH_BLOCKING=1 VLLM_USE_DEEP_GEMM=0 vllm serve /models/<MODEL> --tensor-parallel-size 8 --trust-remote-code --port 8122 --max-model-len 4096 --load-format fastsafetensors
```

**When to use:**
- Server starts but crashes on first inference request
- Inference returns CUDA errors with unhelpful tracebacks
- Suspected operator-level bugs (custom kernels, attention backends, quantization ops)

**When NOT to use:**
- Normal development/testing — it significantly slows down execution
- Import errors or config parsing errors — these are CPU-side, not CUDA-related

Once the exact failing line is identified, cross-reference with the upstream-first protocol above to determine whether it's a migration patch issue or an API compatibility gap.

## Resilience

### Network retry

For network-dependent commands (git clone/pull, curl, pip, wget), on failure retry up to 2 more times with 5s sleep between attempts:

```bash
git clone --depth 1 https://github.com/vllm-project/vllm.git /tmp/vllm-upstream-ref
```

If the command fails, run `sleep 5` then retry the same command. Repeat once more if still failing (3 attempts total). If all 3 fail, report the error to the user.

### Auto-resume after interruption

If TaskList shows some tasks `completed` and some not → previous session was interrupted. Do NOT start over. Find first non-completed task, read its description, continue from there. NEVER re-do completed tasks.

## Permission Assumptions

- Full **read/write/execute** for vllm-plugin-FL project directory
- Full **read** for `/usr/local/lib/` (vLLM 0.13.0 source), `/tmp/` (upstream clone), `/models/` (HF checkpoints)
- No `sudo`/`chmod` needed
- All fixes stay **inside the plugin directory**

## vLLM Version Protection

**NEVER** modify the installed vLLM version or its source code unless explicitly debugging a vLLM bug. Specifically:

1. **Do not change vLLM code** — the plugin depends on vLLM v0.13.0 as-is. All customisation goes into the plugin.
2. **Do not reinstall/upgrade vLLM** — running `pip install -e .` for the plugin may pull vLLM as a dependency and overwrite the compiled version. Always use `pip install --no-build-isolation --no-deps -e .` for the plugin to avoid this.
3. **If vLLM gets accidentally overwritten** (e.g. C extensions fail with `libcudart` errors), restore it with `MAX_JOBS=96 pip install --no-build-isolation -v -e <vllm_source_dir>` — do NOT modify vLLM source to work around the issue.
