#!/bin/bash
# Step 10: Benchmark verification
# Usage: bash scripts/benchmark.sh <MODEL_DISPLAY_NAME>
set -euo pipefail

MODEL="${1:?Usage: benchmark.sh <MODEL_DISPLAY_NAME>}"

export USE_FLAGGEMS=0
vllm bench throughput \
    --model "/models/${MODEL}" \
    --dataset-name random \
    --input-len 6144 \
    --output-len 1024 \
    --num-prompts 100 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --load-format dummy \
    --trust-remote-code
