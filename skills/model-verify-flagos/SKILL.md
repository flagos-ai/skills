---
name: model-verify-flagos
description: |
  Verify the serving stack with a user-specified target model. Runs twice: first with
  FlagGems/FlagCX disabled (isolate model-specific errors), then with full multi-chip
  stack enabled. Diffs the two runs to pinpoint which layer caused any failure.
user-invokable: true
allowed-tools: "Bash(*) Read Edit Write Glob Grep WebSearch WebFetch AskUserQuestion"
---

# Target Model Verification

Same layer-peeling approach as env-verify, but with the real target model that may
require multi-GPU tensor parallelism. Runs the model twice (without and with
multi-chip stack) and diffs results to isolate failures.

## Skill Components

```
model-verify/
├── SKILL.md                            # This file — execution flow
├── scripts/
│   └── diff_analysis.py                # Compare Run A vs Run B, classify errors (JSON)
└── references/
    └── multichip-errors.md             # Multi-chip error patterns and diff truth table
```

**Reused from env-verify:**
- `env-verify/scripts/run_offline_inference.py` — Phase A test (parameterized)
- `env-verify/scripts/test_serve_mode.py` — Phase B test (parameterized)
- `env-verify/references/error-classification.md` — Layer-based error rules

## Prerequisites

- Running container with software stack installed (from `install-stack`)
- env-verify completed (at least Phase A passed)
- User must provide model path

If invoked standalone, ask for container name, vendor, and model path.
If invoked from `/flagrelease`, these are passed as context.

## Execution Flow

### Step 1: Get Model Info from User

**Ask the user for** (use AskUserQuestion if not provided):

1. **Model path** (required) — local path inside container OR ModelScope/HuggingFace ID
2. **--tensor-parallel-size** (optional) — default to GPU count
3. **Additional vllm args** (optional)

Get default TP size:
```bash
docker exec <CONTAINER> python3 -c "
import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 1)
"
```

**If user does not provide model path → ask and wait. Do not guess.**

### Step 2: Download Model (if needed)

If model path is a remote ID (not starting with `/`):

```bash
docker exec <CONTAINER> python3 -c "
from modelscope import snapshot_download
snapshot_download('<MODEL_ID>', local_dir='/data/models/<MODEL_NAME>')
"
```

If local directory, verify `config.json` exists:
```bash
docker exec <CONTAINER> test -f <MODEL_PATH>/config.json
```

**Timeout:** 600s for large model downloads.

### Step 3: Run A — WITHOUT Multi-Chip Stack

Copy the test scripts from env-verify into the container (if not already there):

```bash
docker cp <ENV_VERIFY_DIR>/scripts/run_offline_inference.py <CONTAINER>:/tmp/
docker cp <ENV_VERIFY_DIR>/scripts/test_serve_mode.py <CONTAINER>:/tmp/
```

**Phase A (offline):**
```bash
docker exec <CONTAINER> bash -c '
export USE_FLAGGEMS=0
unset FLAGCX_PATH
timeout 300 python3 /tmp/run_offline_inference.py \
    --model <MODEL_PATH> \
    --tp <TP_SIZE> \
    --trust-remote-code
' > /tmp/run_a_offline.json
```

**Phase B (serve):**
```bash
docker exec <CONTAINER> bash -c '
export USE_FLAGGEMS=0
unset FLAGCX_PATH
timeout 360 python3 /tmp/test_serve_mode.py \
    --model <MODEL_PATH> \
    --tp <TP_SIZE> \
    --trust-remote-code \
    --health-timeout 300
' > /tmp/run_a_serve.json
```

### Step 4: Run B — WITH Full Multi-Chip Stack

**Skip logic:** If ALL of FlagGems, FlagTree, FlagCX failed install → skip Run B.
Report: "Run B skipped: no multi-chip packages installed."
Check install-stack results to decide.

**Phase A (offline):**
```bash
docker exec <CONTAINER> bash -c '
export USE_FLAGGEMS=1
export FLAGCX_PATH=/tmp/FlagCX
export VLLM_PLUGINS=fl
timeout 300 python3 /tmp/run_offline_inference.py \
    --model <MODEL_PATH> \
    --tp <TP_SIZE> \
    --trust-remote-code
' > /tmp/run_b_offline.json
```

**Phase B (serve):**
```bash
docker exec <CONTAINER> bash -c '
export USE_FLAGGEMS=1
export FLAGCX_PATH=/tmp/FlagCX
export VLLM_PLUGINS=fl
timeout 360 python3 /tmp/test_serve_mode.py \
    --model <MODEL_PATH> \
    --tp <TP_SIZE> \
    --trust-remote-code \
    --health-timeout 300
' > /tmp/run_b_serve.json
```

### Step 5: Diff Analysis

Copy and run `scripts/diff_analysis.py` to compare the two runs:

```bash
docker cp <SKILL_DIR>/scripts/diff_analysis.py <CONTAINER>:/tmp/
docker exec <CONTAINER> python3 /tmp/diff_analysis.py \
    --run-a /tmp/run_a_offline.json \
    --run-b /tmp/run_b_offline.json
```

Read `references/multichip-errors.md` to interpret the diff and classify errors.

### Step 6: Produce Report

```json
{
  "status": "PASS | PARTIAL | FAIL",
  "stage": "model-verify",
  "model": "<MODEL_PATH>",
  "tensor_parallel_size": 8,
  "run_a_without_multichip": {
    "flags": {"USE_FLAGGEMS": "0", "FLAGCX_PATH": "unset"},
    "phase_a_offline": "PASS | FAIL",
    "phase_b_serve": "PASS | FAIL",
    "output_sample": "...",
    "errors": []
  },
  "run_b_with_multichip": {
    "flags": {"USE_FLAGGEMS": "1", "FLAGCX_PATH": "/tmp/FlagCX"},
    "skipped": false,
    "phase_a_offline": "PASS | FAIL",
    "phase_b_serve": "PASS | FAIL",
    "output_sample": "...",
    "errors": []
  },
  "diff_analysis": {
    "conclusion": "BOTH_PASS | MULTICHIP_ERROR | SAME_ERROR | DIFFERENT_ERRORS",
    "detail": "...",
    "multichip_component": "FlagGems | FlagTree | FlagCX | plugin | null",
    "recommended_stack": "full | base | none"
  }
}
```

**`recommended_stack`** — tells downstream skills which stack to use:
- `full` — Run B passed (USE_FLAGGEMS=1, FLAGCX_PATH set)
- `base` — only Run A passed (USE_FLAGGEMS=0, FLAGCX_PATH unset)
- `none` — Run A also failed (model can't serve)

**Status logic:**
- `PASS` — both Run A and Run B succeed
- `PARTIAL` — Run A passes, Run B fails
- `FAIL` — Run A fails

## Error Handling

| Failure | Behavior |
|---------|----------|
| Model path not provided | Ask user, wait |
| Model path not found | Report exact path, exit with error |
| Model too large for memory | Report OOM, suggest reducing TP or dtype |
| TP > available GPUs | Report "requested TP=X but only Y available" |
| Server hangs | Kill after timeout, capture last logs |
| Run A and Run B both fail | Report both errors separately |

**Rule:** Run BOTH runs regardless of individual failures. Maximize error coverage.

## Timeout Rules

| Operation | Timeout |
|-----------|---------|
| Model download | 600s |
| Phase A (offline) | 300s |
| Phase B (serve + test) | 360s |
