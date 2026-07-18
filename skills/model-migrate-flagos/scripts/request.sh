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

# Step 10.2: Send test request to verify serving
# Usage: bash scripts/request.sh <MODEL_DISPLAY_NAME> [PORT]
set -euo pipefail

MODEL="${1:?Usage: request.sh <MODEL_DISPLAY_NAME> [PORT]}"
PORT="${2:-8121}"

curl "http://localhost:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"/models/${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"介绍一下 vLLM 的核心优势\"}],\"max_tokens\":10000}"
