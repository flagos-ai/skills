# Pipeline State Schema

State object that flows between skills during orchestrated execution.

## Schema

```json
{
  "container": "<CONTAINER_NAME>",
  "vendor": "<GPU_VENDOR>",

  "install_stack": {
    "status": "PASS | PARTIAL | FAIL",
    "gate_passed": true,
    "packages": {
      "vllm": "PASS | FAIL",
      "flagtree": "PASS | FAIL | SKIPPED",
      "flaggems": "PASS | FAIL | SKIPPED",
      "flagcx": "PASS | FAIL | SKIPPED",
      "vllm_plugin_fl": "PASS | FAIL"
    },
    "flagcx_path": "/tmp/FlagCX | null",
    "network": {"github_mirror": true, "pypi_mirror": true},
    "python_version": "3.11",
    "glibc_version": "2.34",
    "errors": []
  },

  "env_verify": {
    "status": "PASS | FAIL",
    "phase_a": "PASS | FAIL",
    "phase_b": "PASS | FAIL",
    "errors": []
  },

  "model_verify": {
    "status": "PASS | PARTIAL | FAIL",
    "model_path": "<path>",
    "tensor_parallel_size": 8,
    "run_a": "PASS | FAIL",
    "run_b": "PASS | FAIL | SKIPPED",
    "recommended_stack": "full | base | none",
    "diff_conclusion": "BOTH_PASS | MULTICHIP_ERROR | ...",
    "errors": []
  },

  "perf_test": {
    "status": "PASS | PARTIAL | FAIL | SKIPPED",
    "accuracy": "SKIPPED",
    "profiles_passed": "5/5",
    "summary_table": "...",
    "errors": []
  }
}
```

## Gate Logic

```
install-stack gate_passed == false  →  STOP pipeline (status: FAIL)
install-stack gate_passed == true   →  continue

env-verify fatal error (segfault/OOM/device not found)  →  STOP
env-verify non-fatal error                              →  continue (record)

model-verify recommended_stack == "none" (Run A failed)  →  STOP
model-verify recommended_stack == "base" or "full"       →  continue

perf-test always runs if model-verify passed Run A
```

## Data Flow Between Skills

```
flagrelease (orchestrator)
    │
    ├── passes to install-stack:  container, vendor
    │   receives:                 install report (packages, gate, flagcx_path)
    │
    ├── passes to env-verify:     container, install results
    │   receives:                 env verification report
    │
    ├── passes to model-verify:   container, vendor, install results, env results
    │   receives:                 model report, recommended_stack, model_path, tp_size
    │
    └── passes to perf-test:      container, model_path, tp_size, recommended_stack
        receives:                 perf report with metrics table
```
