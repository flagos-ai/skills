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

# Step 9: Benchmark verification
# Usage: bash scripts/benchmark.sh <MODEL_DISPLAY_NAME>
set -euo pipefail

MODEL="${1:?Usage: benchmark.sh <MODEL_DISPLAY_NAME>}"

vllm bench throughput \
    --model "/models/${MODEL}" \
    --dataset-name random \
    --input-len 128 \
    --output-len 128 \
    --num-prompts 2 \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.9 \
    --load-format dummy \
    --max-num-seqs 10 \
    --trust-remote-code
