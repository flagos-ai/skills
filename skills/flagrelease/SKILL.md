---
name: flagrelease
description: |
  Full FlagRelease pipeline orchestrator. Runs the complete LLM deployment, verification,
  and benchmarking pipeline for multi-chip GPU backends. Executes: install-stack →
  env-verify → model-verify → perf-test in sequence, passing state between steps
  and producing a final structured report. Assumes gpu-container-setup (Step 1) is
  already done — a running container with PyTorch + GPU access must exist.
user-invokable: true
allowed-tools: "Bash(*) Read Edit Write Glob Grep WebSearch WebFetch AskUserQuestion Agent"
---

# FlagRelease Pipeline Orchestrator

End-to-end LLM deployment + testing pipeline for multi-chip GPU backends.
Orchestrates 4 sub-skills in sequence and produces a final report.

## Skill Components

```
flagrelease/
├── SKILL.md                            # This file — orchestration flow
└── references/
    └── pipeline-state.md               # Pipeline state schema, gate logic, data flow
```

**Sub-skills (each independently invokable):**

```
../install-stack/                       # Step 2: Install 5 packages
│   ├── SKILL.md
│   ├── scripts/
│   │   ├── detect_network.py           # Probe GitHub/PyPI, return mirror config
│   │   ├── collect_env_info.py         # Python/glibc/arch/vendor/disk info
│   │   ├── select_flagtree_wheel.py    # Match vendor+python+glibc → wheel
│   │   └── validate_packages.py        # Import-test all 5 packages
│   └── references/
│       ├── vendor-mappings.md          # FlagCX make flags, adaptor names
│       └── network-mirrors.md          # Mirror config rules

../env-verify/                          # Step 3: Qwen3-0.6B smoke test
│   ├── SKILL.md
│   ├── scripts/
│   │   ├── run_offline_inference.py    # Phase A: offline inference test
│   │   └── test_serve_mode.py          # Phase B: serve + health + chat test
│   └── references/
│       └── error-classification.md     # Layer-based error classification

../model-verify/                        # Step 4: Target model ± multi-chip
│   ├── SKILL.md
│   ├── scripts/
│   │   └── diff_analysis.py            # Compare Run A vs Run B results
│   └── references/
│       └── multichip-errors.md         # Multi-chip error patterns

../perf-test/                           # Steps 5+6: Accuracy + Performance
│   ├── SKILL.md
│   ├── scripts/
│   │   ├── run_benchmark.py            # Run single benchmark profile
│   │   └── run_all_benchmarks.py       # Run all profiles + summarize
│   └── references/
│       └── benchmark-profiles.md       # Profile definitions and metrics
```

## Pipeline Overview

```
[Prerequisite: /gpu-container-setup already done by another team]
       │
       ▼
  install-stack   →  Install 5 packages (vLLM, FlagTree, FlagGems, FlagCX, plugin)
       │                scripts: detect_network, collect_env_info, select_flagtree_wheel
       │
       │  GATE: vLLM + plugin must succeed
       ▼
  env-verify      →  Smoke test with Qwen3-0.6B (FlagGems/CX OFF)
       │                scripts: run_offline_inference, test_serve_mode
       │
       │  Verify Layers 0-3
       ▼
  model-verify    →  Target model test (OFF then ON), diff analysis
       │                scripts: run_offline_inference, test_serve_mode, diff_analysis
       │
       │  Determine which stack works (full vs base)
       ▼
  perf-test       →  Accuracy (placeholder) + Performance benchmarks
       │                scripts: run_benchmark, run_all_benchmarks
       ▼
  Final Report
```

## Prerequisites

A running Docker container with:
- PyTorch installed and GPU-accessible
- Container name known (e.g. `flagrelease-worker`)

This container is produced by `/gpu-container-setup` (maintained by another team).

## Execution Flow

Read `references/pipeline-state.md` for the full state schema and gate logic.

### Step 0: Gather Initial Context

Ask user for container name (or detect running containers):

```bash
docker ps --format '{{.Names}}' | head -10
```

Verify the container is running:
```bash
docker inspect --format='{{.State.Status}}' <CONTAINER> | grep -q running
```

Initialize pipeline state (see `references/pipeline-state.md`).

### Step 1: Install Software Stack

Read and follow `../install-stack/SKILL.md`.

The install-stack skill will:
1. Copy `scripts/collect_env_info.py` into container → get vendor, Python, glibc
2. Copy `scripts/detect_network.py` into container → get mirror config
3. Install 5 packages in order, using `scripts/select_flagtree_wheel.py` for FlagTree
4. Run `scripts/validate_packages.py` inside container → get final status

**Gate check:** If `gate_passed` is false (vLLM or plugin failed) → **STOP pipeline**.
Report FAIL with install errors.

Store result in pipeline state.

### Step 2: Environment Verification

Read and follow `../env-verify/SKILL.md`.

The env-verify skill will:
1. Download Qwen3-0.6B (if not cached)
2. Copy `scripts/run_offline_inference.py` into container → Phase A
3. Copy `scripts/test_serve_mode.py` into container → Phase B
4. Classify errors using `references/error-classification.md`

**Decision:** Fatal error → STOP. Non-fatal → record and continue.

Store result in pipeline state.

### Step 3: Model Verification

Read and follow `../model-verify/SKILL.md`.

**This step is interactive** — will ask user for model path.

The model-verify skill will:
1. Get model info from user (AskUserQuestion)
2. Reuse `run_offline_inference.py` and `test_serve_mode.py` for Run A and Run B
3. Run `scripts/diff_analysis.py` to compare results
4. Determine `recommended_stack` (full/base/none)

**Decision:** If `recommended_stack` is `none` (Run A failed) → STOP.

Store result in pipeline state (including model_path, tp_size, recommended_stack).

### Step 4: Performance Test

Read and follow `../perf-test/SKILL.md`.

The perf-test skill will:
1. Start vllm serve with recommended stack
2. Copy `scripts/run_all_benchmarks.py` into container → run 5 profiles
3. Collect metrics and produce summary table

Store result in pipeline state.

### Step 5: Final Report

Compile all results from pipeline state into a final report:

```json
{
  "status": "PASS | PARTIAL | FAIL",
  "pipeline": "flagrelease",
  "container": "<name>",
  "vendor": "<vendor>",
  "model": "<path>",
  "tensor_parallel_size": 8,
  "steps": {
    "install_stack": { "status": "...", "packages": {...} },
    "env_verify":    { "status": "...", "phase_a": "...", "phase_b": "..." },
    "model_verify":  { "status": "...", "run_a": "...", "run_b": "...", "recommended_stack": "..." },
    "perf_test":     { "status": "...", "profiles_passed": "5/5", "summary_table": "..." }
  },
  "errors": [...],
  "conclusion": "Pipeline completed. ..."
}
```

Present to user with clear summary:
1. Which packages installed / failed
2. Whether base stack works
3. Whether multi-chip stack works (and which component failed if not)
4. Performance numbers (summary table)
5. All errors with layer classification

**Overall status:**
- `PASS` — all steps pass, full multi-chip stack works
- `PARTIAL` — model works with degraded stack, or some perf profiles failed
- `FAIL` — model cannot serve (gate or Run A failure)

## Design Rules

- **Every operation has a timeout** — no hangs allowed
- **Every error is caught** with precise location (step, phase, layer, cause)
- **Pipeline always completes** with success or structured error report
- **One sub-step failure does NOT skip unrelated steps** (unless gate failure)
- **Network uses mirrors** when direct access fails
- **Scripts produce JSON** — structured, parseable, comparable across runs
