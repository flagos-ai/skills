# vllm-plugin-fl-setup-flagos: vLLM-Plugin-FL Setup Skill

[中文版](README_zh.md)

## Overview

`vllm-plugin-fl-setup-flagos` is a **standalone** AI coding skill that installs and configures **vLLM-Plugin-FL** for multiple hardware backends including NVIDIA, Ascend, MetaX, Iluvatar, Moore Threads, and more.

> **Note:** This skill is designed for **standalone use** — it focuses exclusively on environment setup and dependency installation. It does not depend on any other skill and can be used independently to get a working vLLM-Plugin-FL environment from scratch.

### Problem Statement

vLLM-Plugin-FL extends vLLM to support model inference/serving across diverse hardware backends via FlagOS's unified operator library **FlagGems** and communication library **FlagCX**. Setting up the full stack — vLLM-Plugin-FL, FlagGems, FlagCX, and backend-specific drivers — involves many steps with hardware-dependent variations. Manual setup is tedious and error-prone, especially for less-documented backends.

This skill automates the entire setup workflow: **detect hardware → install vLLM-Plugin-FL → install FlagGems → (optionally) install FlagCX → apply backend-specific configuration → verify with inference test**.

### Usage

This skill is triggered automatically when users say things like:

- "setup vllm-plugin-fl"
- "install vllm-plugin-fl"
- "configure FL plugin"
- "set up FlagGems"
- "set up FlagCX"

---

## Prerequisites

- Linux OS (Ubuntu 20.04+ recommended)
- Python 3.10+
- vLLM **v0.13.0** — from [official release](https://github.com/vllm-project/vllm/tree/v0.13.0) or [vllm-FL fork](https://github.com/flagos-ai/vllm-FL)
- GPU with appropriate drivers (NVIDIA CUDA, Huawei Ascend, etc.)
- `pip` package manager
- Git

---

## Installation Workflow (5 Steps)

```
┌──────────────────────────────────────────────────────────┐
│  Step 1   Identify hardware backend (nvidia-smi, etc.)   │
│  Step 2   Install vLLM-Plugin-FL from source              │
│  Step 3   Install FlagGems (+ FlagTree for Ascend)        │
│  Step 4   (Optional) Install FlagCX for multi-device      │
│  Step 5   Backend-specific setup (per reference docs)     │
└──────────────────────────────────────────────────────────┘
```

### Step 1: Identify Hardware Backend

The skill detects the hardware backend by probing available CLI tools:

| Backend | Detection Command |
|---|---|
| NVIDIA GPU | `nvidia-smi` |
| Huawei NPU | `npu-smi info` |
| Moore Threads GPU | `mthreads-gmi` |
| Iluvatar GPU | `ixsmi` |

### Step 2: Install vLLM-Plugin-FL

Clone and install from source:

```bash
mkdir -p ~/flagos-workspace && cd ~/flagos-workspace
git clone https://github.com/flagos-ai/vllm-plugin-FL
cd vllm-plugin-FL
pip install -r requirements.txt
pip install --no-build-isolation .
export VLLM_PLUGINS='fl'
```

### Step 3: Install FlagGems

> **Ascend NPU users** must install FlagTree first. See [references/npu.md](references/npu.md).

```bash
pip install -U scikit-build-core==0.11 pybind11 ninja cmake
cd ~/flagos-workspace
git clone https://github.com/flagos-ai/FlagGems
cd FlagGems
pip install --no-build-isolation .
```

### Step 4: (Optional) Install FlagCX

For multi-device distributed inference. Skip for single-device setups or Ascend NPU.

```bash
cd ~/flagos-workspace
git clone https://github.com/flagos-ai/FlagCX.git
cd FlagCX
git submodule update --init --recursive
make USE_NVIDIA=1  # Adjust for your platform
export FLAGCX_PATH="$PWD"
cd plugin/torch/
FLAGCX_ADAPTOR=[xxx] pip install --no-build-isolation .
```

### Step 5: Backend-Specific Setup

| Backend | Chip Vendor | Reference |
|---|---|---|
| Ascend NPU | Huawei | [references/npu.md](references/npu.md) |
| Iluvatar GPU (BI-V150) | Iluvatar | [references/iluvatar_gpu.md](references/iluvatar_gpu.md) |
| Moore Threads GPU | Moore Threads | [references/mthreads_gpu.md](references/mthreads_gpu.md) |
| MetaX GPU | MetaX | TBD |
| Pingtouge-Zhenwu | Pingtouge | TBD |
| Tsingmicro | Tsingmicro | TBD |
| Hygon DCU | Hygon | TBD |

---

## Quick Test

After installation, the skill verifies the full stack by running offline batched inference:

```python
from vllm import LLM, SamplingParams

model_path = "<resolved_model_path>"
prompts = ["Hello, my name is"]
sampling_params = SamplingParams(max_tokens=10, temperature=0.0)

llm = LLM(model=model_path, max_num_batched_tokens=16384, max_num_seqs=2048)
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(f"Prompt: {output.prompt!r}, Generated text: {output.outputs[0].text!r}")
```

The skill will search the machine for a local model copy, or ask the user for a model path.

---

## Directory Structure

```
skills/vllm-plugin-fl-setup-flagos/
├── SKILL.md                          # Skill definition (entry point)
├── README.md                         # This document (English)
├── README_zh.md                      # Chinese version
├── LICENSE.txt                       # Apache 2.0 License
└── references/                       # Backend-specific setup guides
    ├── npu.md                        # Huawei Ascend NPU setup
    ├── iluvatar_gpu.md               # Iluvatar BI-V150 GPU setup
    └── mthreads_gpu.md               # Moore Threads GPU setup
```

---

## File Descriptions

### Skill Definition

#### `SKILL.md`
The skill entry point. Defines trigger conditions, prerequisites, the 5-step installation workflow, backend-specific references, quick test procedure, and troubleshooting guide. The AI coding assistant uses this file to identify and invoke the skill.

### Reference Documents (`references/`)

#### `npu.md` — Huawei Ascend NPU Setup
Ascend-specific configuration including FlagTree installation (required before FlagGems), CANN toolkit setup, and eager execution requirements.

#### `iluvatar_gpu.md` — Iluvatar GPU Setup
Iluvatar BI-V150 specific configuration, including driver setup and `enforce_eager=True` requirement.

#### `mthreads_gpu.md` — Moore Threads GPU Setup
Moore Threads specific configuration, including additional environment variables (`USE_FLAGGEMS=1`, `VLLM_MUSA_ENABLE_MOE_TRITON=1`) and vLLM launch parameters (`enforce_eager=True`, `block_size=64`).

---

## Troubleshooting

| Problem | Typical Cause | Fix |
|---|---|---|
| Out of memory on model load | GPU memory exhausted | Use `gpu_memory_utilization=0.8` parameter |
| FlagGems build failures | Missing build deps | Install `scikit-build-core`, `pybind11`, `ninja`, `cmake` |
| Plugin not loaded | Env var not set | Ensure `VLLM_PLUGINS='fl'` is exported |
| FlagCX communication errors | Path or build mismatch | Verify `FLAGCX_PATH` and platform build flag |
| Ascend-specific issues | FlagTree missing | See [references/npu.md](references/npu.md) |
| Cannot connect to GitHub | Network restrictions | Configure `http_proxy` / `https_proxy` |

---

## Usage in Your Project

Skills are typically placed under `.claude/skills/` (or the equivalent skills directory for your editor) in the project root.


### Install

```bash
# From your project root
mkdir -p .claude/skills
cp -r <path-to-this-repo>/skills/vllm-plugin-fl-setup-flagos .claude/skills/
```

---

## License

This project is licensed under the Apache 2.0 License. See [LICENSE.txt](LICENSE.txt) for details.
