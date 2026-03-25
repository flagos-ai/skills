---
name: vllm-plugin-fl-setup-flagos
description: >
  Install and configure vLLM-Plugin-FL for multiple hardware backends including NVIDIA, Ascend
  and etc. Use when setting up vllm-plugin-fl, configuring the environment for specific hardware
  backend, installing dependencies, checking whether dependencies are installed successfully,
  resolving runtime issues, and launching inference to verify successful model serving. Trigger
  when the user says things like "setup vllm-plugin-fl", "install vllm-plugin-fl",
  "configure FL plugin", "set up FlagGems", or "set up FlagCX".
argument-hint: "[backend]"
user-invocable: false
compatibility: "Linux (Ubuntu 20.04+), Python 3.10+, vLLM v0.13.0, GPU with appropriate drivers"
metadata:
  version: "1.0.0"
  author: flagos-ai
  category: environment-setup
  tags: [vllm, vllm-plugin-fl, flaggems, flagcx, setup, installation]
---

# vLLM-Plugin-FL Setup

## Overview

vLLM-Plugin-FL extends vLLM to support model inference/serving across diverse hardware backends (NVIDIA, Ascend, MetaX, Iluvatar, etc.) via FlagOS's unified operator library FlagGems and communication library FlagCX. This skill covers installation, hardware-specific environment configuration, and dependency setup.

## Prerequisites

- Linux OS (Ubuntu 20.04+ recommended)
- Python 3.10+
- vLLM **v0.13.0** — install from the official [v0.13.0 release](https://github.com/vllm-project/vllm/tree/v0.13.0) or the fork [vllm-FL](https://github.com/flagos-ai/vllm-FL)
- GPU with appropriate drivers (NVIDIA CUDA, Huawei Ascend, etc.)
- `pip` package manager
- Git

Verify vLLM version before proceeding:

```bash
python -c "import vllm; print(vllm.__version__)"
# Expected output: 0.13.0
```

## Installation Workflow

### Step 1: Identify Hardware Backend

```bash
# NVIDIA GPU
nvidia-smi

# Huawei NPU
npu-smi info

# Moore Threads GPU
mthreads-gmi

# Iluvatar GPU
ixsmi
```

### Step 2: Install vLLM-Plugin-FL

First create a workspace directory and try cloning the source code:

```bash
mkdir -p ~/flagos-workspace && cd ~/flagos-workspace
git clone https://github.com/flagos-ai/vllm-plugin-FL
```

If `git clone` fails due to network issues, ask the user for their network proxy settings (e.g. `http_proxy` / `https_proxy`), configure the proxy, then retry the clone.

Then install from the source directory:

```bash
cd vllm-plugin-FL
pip install -r requirements.txt
pip install --no-build-isolation .
# Required to enable vLLM-Plugin-FL when running vLLM
export VLLM_PLUGINS='fl'
```

Verify vLLM-Plugin-FL installation:

```bash
python -c "import vllm_fl; print('vllm-plugin-FL installed successfully')"
```

### Step 3: Install FlagGems

> **Ascend NPU users:** Before installing FlagGems, you **must** first install FlagTree. See [references/npu.md](references/npu.md) and complete the FlagTree installation step there before proceeding. Otherwise the FlagGems verification will fail repeatedly and keep reinstalling Triton.

```bash
# Install build dependencies
pip install -U scikit-build-core==0.11 pybind11 ninja cmake

# Clone FlagGems source code
cd ~/flagos-workspace
git clone https://github.com/flagos-ai/FlagGems
```

If `git clone` fails due to network issues, ask the user for their network proxy settings (e.g. `http_proxy` / `https_proxy`), configure the proxy, then retry the clone.

Then install from the source directory:

```bash
cd FlagGems
pip install --no-build-isolation .
```

Verify FlagGems installation:

```bash
python -c "import flag_gems; print('FlagGems installed successfully')"
```

### Step 4: (Optional) Install FlagCX

FlagCX is a unified communication library for multi-device distributed inference, supporting both homogeneous and heterogeneous setups. Skip this step if running on a single device.

> **Note:** Ascend NPU does not need FlagCX — skip this step for Ascend backends.

```bash
cd ~/flagos-workspace
git clone https://github.com/flagos-ai/FlagCX.git
```

If `git clone` fails due to network issues, ask the user for their network proxy settings (e.g. `http_proxy` / `https_proxy`), configure the proxy, then retry the clone.

Then build and install from the source directory:

```bash
cd FlagCX

git submodule update --init --recursive

# Build for your platform (e.g. USE_NVIDIA=1 for NVIDIA)
make USE_NVIDIA=1

export FLAGCX_PATH="$PWD"

# Install Python binding (replace [xxx] with your platform: nvidia, ascend, etc.)
cd plugin/torch/
FLAGCX_ADAPTOR=[xxx] pip install --no-build-isolation .
```

Verify FlagCX installation:

```bash
python -c "import flagcx; print('FlagCX installed successfully')"
```

### Step 5: Backend-Specific Setup

Some hardware backends require additional setup. See the corresponding reference document:

| Backend | Chip Vendor | Reference |
|---------|-------------|-----------|
| Ascend NPU | Huawei | [references/npu.md](references/npu.md) |
| MetaX GPU | MetaX | TBD |
| Iluvatar GPU (BI-V150) | Iluvatar | [references/iluvatar_gpu.md](references/iluvatar_gpu.md) |
| Pingtouge-Zhenwu | Pingtouge | TBD |
| Tsingmicro | Tsingmicro | TBD |
| Moore Threads GPU | Moore Threads | [references/mthreads_gpu.md](references/mthreads_gpu.md) |
| Hygon DCU | Hygon | TBD |

## Quick Test

1. Ask the user for the model name they want to test (e.g. `Qwen3-4B`, `DeepSeek-R1`).
2. Search the machine for a local copy of that model:
   ```bash
   find / -maxdepth 5 -type d -name "<user_provided_model_name>" 2>/dev/null
   ```
3. If found, use the discovered path. If not found, tell the user and ask them to provide a different model name or a full local path, then repeat the search. If after 3 attempts no valid model is found, skip the quick test and inform the user to prepare a model before retrying.
4. Ensure the FL plugin is enabled before running inference:
   ```bash
   export VLLM_PLUGINS='fl'
   ```
   For **Moore Threads GPU**, also set:
   ```bash
   export USE_FLAGGEMS=1
   export FLAGCX_PATH=/workspace/FlagCX  # MUST point to the actual FlagCX installation directory; this is only an example
   export VLLM_MUSA_ENABLE_MOE_TRITON=1
   ```
5. Once a valid model path is resolved, run offline batched inference to verify the full stack:

```python
from vllm import LLM, SamplingParams

model_path = "<resolved_model_path>"
prompts = [
    "Hello, my name is",
]
sampling_params = SamplingParams(max_tokens=10, temperature=0.0)

# For Moore Threads GPU, add: enforce_eager=True, block_size=64, attention_config={"backend": "TORCH_SDPA"}
# For Iluvatar BI-V150, add: enforce_eager=True
llm = LLM(model=model_path, max_num_batched_tokens=16384, max_num_seqs=2048)
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```


## Troubleshooting

**Out of memory on model load**: Use `gpu_memory_utilization` parameter to limit memory. Start with 0.8 and adjust:
```python
from vllm import LLM
llm = LLM(model="...", gpu_memory_utilization=0.8)
```

**FlagGems build failures**: Ensure build dependencies are installed (`scikit-build-core`, `pybind11`, `ninja`, `cmake`). Check that your compiler supports C++17.

**Plugin not loaded**: If vLLM does not use the FL plugin, verify that `VLLM_PLUGINS='fl'` is set in your environment.

**FlagCX communication errors**: Ensure `FLAGCX_PATH` is correctly set and the library was built for your platform. For NVIDIA, verify with `make USE_NVIDIA=1`.

**Ascend-specific issues**: See [references/npu.md](references/npu.md) for Ascend NPU troubleshooting, including FlagTree setup and eager execution requirements.

**Cannot connect to GitHub**: Ask the user for their network proxy settings (e.g. `http_proxy` / `https_proxy`), configure the proxy, then retry the `git clone` command.

## References

- [vLLM-Plugin-FL GitHub](https://github.com/flagos-ai/vllm-plugin-FL)
- [FlagGems GitHub](https://github.com/flagos-ai/FlagGems)
- [FlagCX GitHub](https://github.com/flagos-ai/FlagCX)
- For non-NVIDIA chips, refer to the [references](references/) directory for hardware-specific configurations and setup instructions