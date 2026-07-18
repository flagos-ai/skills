<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->

# Multi-Chip Error Classification

When Run A (base stack) passes but Run B (full stack) fails, the error is
caused specifically by the multi-chip operator/communication stack. Classify by:

## Component Identification

| Error Pattern | Component | Meaning |
|--------------|-----------|---------|
| `flag_gems.*not found` / `FlagGems` | FlagGems | Missing op implementation for this model |
| `Triton compilation error` / `triton` | FlagTree | Backend compiler issue |
| `flagcx.*failed` / `all_reduce` / `all_gather` | FlagCX | Communication backend error |
| `vllm_fl.*dispatch` / `platform_plugins` | vllm-plugin-FL | Routing to wrong operator |
| `NaN` / `inf` / `numerical mismatch` | FlagGems (precision) | Operator precision issue |
| `Hang during collective op` | FlagCX (deadlock) | Deadlock in communication |

## Diff Analysis Truth Table

| Run A (base) | Run B (full) | Conclusion | Action |
|-------------|-------------|------------|--------|
| PASS | PASS | Full stack works | Use full stack for perf-test |
| PASS | FAIL | Multi-chip error | Use base stack; report component |
| FAIL | FAIL (same error) | Base error | Not multi-chip related |
| FAIL | FAIL (different) | Two issues | Report both separately |
| FAIL | PASS | Unexpected | Investigate; use full stack |

## Environment Variables for Stack Control

| Config | Base Stack (OFF) | Full Stack (ON) |
|--------|-----------------|-----------------|
| `USE_FLAGGEMS` | `0` | `1` |
| `FLAGCX_PATH` | unset | `/tmp/FlagCX` |
| `VLLM_PLUGINS` | *(default)* | `fl` |
