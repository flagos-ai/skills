#!/bin/bash

# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Start local vLLM server for verification (E2E or manual testing)
# Usage: bash scripts/serve.sh <MODEL_DISPLAY_NAME> [PORT]
set -euo pipefail

MODEL="${1:?Usage: serve.sh <MODEL_DISPLAY_NAME> [PORT]}"
PORT="${2:-8122}"

export VLLM_FL_PREFER_ENABLED=false
export VLLM_USE_DEEP_GEMM=0

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve "/models/${MODEL}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 10 \
    --load-format fastsafetensors \
    --trust-remote-code \
    --reasoning-parser deepseek_r1
