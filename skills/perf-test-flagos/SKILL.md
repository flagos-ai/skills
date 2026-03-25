---
name: perf-test-flagos
description: |
  Run accuracy benchmarks (FlagEval, when available) and performance benchmarks
  (vllm bench serve) against a served model. Covers 5 workload profiles: short/long
  prefill x short/long decode + high concurrency. Collects throughput, latency,
  TTFT, TPOT metrics.
user-invokable: true
allowed-tools: "Bash(*) Read Edit Write Glob Grep WebSearch WebFetch AskUserQuestion"
---

# Accuracy + Performance Test

Start vLLM serve with the target model, run accuracy benchmarks (when FlagEval is
available) and performance benchmarks (vllm bench serve) across multiple profiles.

## Skill Components

```
perf-test/
├── SKILL.md                            # This file — execution flow
├── scripts/
│   ├── run_benchmark.py                # Run single benchmark profile (JSON output)
│   └── run_all_benchmarks.py           # Run all 5 profiles, collect + summarize (JSON)
└── references/
    └── benchmark-profiles.md           # Profile definitions, metrics, vllm bench usage
```

**Reused from env-verify:**
- `env-verify/scripts/test_serve_mode.py` — can be used to verify server is healthy
  before benchmarking (optional pre-check)

## Prerequisites

- Running container with software stack installed
- model-verify completed — know which stack to use (`full` vs `base`)
- Model path, TP size, and recommended stack from model-verify

If invoked standalone, ask for container name, model path, TP size, and stack config.
If invoked from `/flagrelease`, these are passed as context.

## Execution Flow

### Step 1: Start vLLM Server

Use the stack recommended by model-verify. Read `references/benchmark-profiles.md`
for the vllm serve command pattern.

```bash
docker exec -d <CONTAINER> bash -c '
export USE_FLAGGEMS=<0|1>
export FLAGCX_PATH=<path_or_unset>
export VLLM_PLUGINS=<fl_or_unset>
vllm serve <MODEL_PATH> \
    --tensor-parallel-size <TP_SIZE> \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 256 \
    --trust-remote-code \
    --port 8000 \
    <EXTRA_ARGS>
'
```

Wait for server ready (poll /health, timeout 300s):
```bash
docker exec <CONTAINER> bash -c '
for i in $(seq 1 150); do
    if curl -s http://localhost:8000/health 2>/dev/null | grep -qE "ok|200|\{\}"; then
        echo "SERVER_READY"; break
    fi
    sleep 2
done
'
```

If server doesn't start, report error and exit.

### Step 2: Get Model Name from Server

```bash
docker exec <CONTAINER> bash -c '
curl -s http://localhost:8000/v1/models | python3 -c "
import json, sys; print(json.load(sys.stdin)[\"data\"][0][\"id\"])"
'
```

---

## Part A: Accuracy Test (FlagEval) — PLACEHOLDER

**STATUS:** FlagEval test client not yet available.

When FlagEval becomes available, update this section with:
- [ ] Docker image URL or pip package name
- [ ] Supported benchmarks (MMLU, GSM8K, HumanEval, etc.)
- [ ] Required arguments and configuration
- [ ] Expected output format
- [ ] Pass/fail criteria (accuracy thresholds)

**Current behavior:** Report accuracy test as SKIPPED.

---

## Part B: Performance Benchmarks

### Step 3: Run All Benchmark Profiles

Copy scripts into the container and run:

```bash
docker cp <SKILL_DIR>/scripts/run_benchmark.py <CONTAINER>:/tmp/
docker cp <SKILL_DIR>/scripts/run_all_benchmarks.py <CONTAINER>:/tmp/

docker exec <CONTAINER> python3 /tmp/run_all_benchmarks.py \
    --model <MODEL_NAME> \
    --tokenizer <MODEL_PATH> \
    --port 8000 \
    --output-dir /data/results/perf
```

The script runs all 5 default profiles (see `references/benchmark-profiles.md`),
saves per-profile JSON to `/data/results/perf/`, and outputs a combined JSON report
with a summary table.

**Important:** One profile failure does NOT skip remaining profiles.

### Step 4: Stop Server

```bash
docker exec <CONTAINER> bash -c 'pkill -f "vllm serve" || true'
```

### Step 5: Produce Report

```json
{
  "status": "PASS | PARTIAL | FAIL",
  "stage": "perf-test",
  "model": "<MODEL_PATH>",
  "tensor_parallel_size": 8,
  "flags": {"USE_FLAGGEMS": "1|0", "FLAGCX_PATH": "..."},
  "accuracy": {
    "status": "SKIPPED",
    "reason": "FlagEval test client not yet available"
  },
  "performance": {
    "status": "PASS | PARTIAL | FAIL",
    "profiles_passed": "5/5",
    "profiles": [ "...per-profile results..." ],
    "summary_table": "...markdown table..."
  }
}
```

Present the summary table to the user:
```
| Profile | Input | Output | Prompts | Req/s | Tok/s | TTFT(ms) | TPOT(ms) | P99(ms) | Status |
|---------|-------|--------|---------|-------|-------|----------|----------|---------|--------|
| ...     | ...   | ...    | ...     | ...   | ...   | ...      | ...      | ...     | ...    |
```

**Status logic:**
- `PASS` — all profiles completed
- `PARTIAL` — some passed, some failed
- `FAIL` — server didn't start or all profiles failed

## Error Handling

| Failure | Behavior |
|---------|----------|
| Server fails to start | Report error; exit |
| `vllm bench serve` not found | Report vllm version issue |
| Single profile fails | Report error, continue remaining profiles |
| Single profile times out | Kill after 600s, report partial, continue |
| Server crashes mid-benchmark | Capture logs, report which profile caused crash |
| OOM during high concurrency | Report, suggest reducing num_prompts |

## Timeout Rules

| Operation | Timeout |
|-----------|---------|
| Server startup | 300s |
| Per profile benchmark | 600s |
