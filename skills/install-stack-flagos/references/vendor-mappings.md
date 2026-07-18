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

# Vendor Mappings for FlagCX

## Make Flags (Phase 1: C++ build)

| Vendor | Make Flag |
|--------|----------|
| nvidia | `USE_NVIDIA=1` |
| ascend | `USE_ASCEND=1` |
| iluvatar | `USE_ILUVATAR_COREX=1` |
| metax | `USE_METAX=1` |
| mthreads | `USE_MUSA=1` |
| amd | `USE_AMD=1` |
| enflame | `USE_ENFLAME=1` |

## FLAGCX_ADAPTOR Values (Phase 2: torch plugin)

| Vendor | FLAGCX_ADAPTOR |
|--------|---------------|
| nvidia | `nvidia` |
| ascend | `ascend` |
| iluvatar | `iluvatar_corex` |
| metax | `metax` |
| mthreads | `musa` |
| amd | `amd` |
| enflame | `enflame` |

## Source Repositories

| Package | Repository |
|---------|-----------|
| vLLM | `pip install vllm==0.13.0` |
| FlagTree | FlagOS PyPI (pre-compiled wheel) |
| FlagGems | `<GITHUB_PREFIX>/FlagOpen/FlagGems` |
| FlagCX | `<GITHUB_PREFIX>/flagos-ai/FlagCX` |
| vllm-plugin-FL | `<GITHUB_PREFIX>/flagos-ai/vllm-plugin-FL` |

## Dependency Chain

```
FlagTree (compiler) ← FlagGems (operators) ← vllm_plugin_FL (routing)
                                             ↑
FlagCX (communication) ──────────────────────┘
                                             ↑
vLLM 0.13.0 (serving) ──────────────────────┘
```

Install order: vLLM → FlagTree → FlagGems → FlagCX → vllm-plugin-FL

## Gate Logic

- **vLLM + vllm-plugin-FL** MUST succeed → otherwise pipeline EXITS
- **FlagTree / FlagGems / FlagCX** may fail → pipeline continues with PARTIAL status
