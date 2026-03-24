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
  version: "2.0.0"
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

Detect the hardware platform on the current machine. Try common device management CLI tools for known chip vendors (e.g. NVIDIA, Huawei Ascend, Moore Threads, Iluvatar, etc.) to determine which backend is available. If uncertain, ask the user what hardware they are using.

Record the detected backend — it will be needed when the dependency READMEs ask you to choose platform-specific options.

### Step 2: Clone vLLM-Plugin-FL and Read Its README

```bash
mkdir -p ~/flagos-workspace && cd ~/flagos-workspace
git clone https://github.com/flagos-ai/vllm-plugin-FL
```

If `git clone` fails due to network issues, ask the user for their network proxy settings (e.g. `http_proxy` / `https_proxy`), configure the proxy, then retry.

**After cloning, read the repository's `README.md` carefully.** The README is the single source of truth for:
- How to install vLLM-Plugin-FL itself (e.g. `pip install` commands, required environment variables).
- The list of dependencies (e.g. FlagGems, FlagCX, FlagTree, etc.) and their GitHub repository URLs.
- Any ordering constraints between dependencies (e.g. "install X before Y").
- Backend-specific notes or environment variable requirements.

Follow the README's instructions to install vLLM-Plugin-FL first, then proceed to install each dependency it lists.

### Step 3: Install Each Dependency by Following Its Own README

For **each dependency** listed in the vLLM-Plugin-FL README:

1. **Clone** the dependency repository using the URL from the README.
2. **Read the dependency's own `README.md`** to find installation instructions, build flags, and platform-specific options for the detected backend.
3. **Follow those instructions** to build and install the dependency, selecting the correct backend/platform options identified in Step 1.
4. **Verify** the installation succeeds (the dependency README usually provides a verification command or import check — use it).

> **Important:** Respect any ordering constraints stated in the vLLM-Plugin-FL README (e.g. if it says "install A before B", do so). If a dependency's README references further sub-dependencies, follow the same clone-read-install pattern recursively.

### Step 4: Backend-Specific Setup

Some hardware backends require additional configuration beyond what the upstream READMEs cover. Scan the [references/](references/) directory in this skill for any document that matches the detected backend. If a matching reference file exists, read it and apply the additional steps it describes (extra environment variables, build flags, execution constraints, etc.).

## Quick Test

1. Ask the user for the model name they want to test (e.g. `Qwen3-4B`, `DeepSeek-R1`).
2. Search the machine for a local copy of that model:
   ```bash
   find / -maxdepth 5 -type d -name "<user_provided_model_name>" 2>/dev/null
   ```
3. If found, use the discovered path. If not found, ask the user to provide a different model name or a full local path. If after 3 attempts no valid model is found, skip the quick test and inform the user to prepare a model before retrying.
4. Set any environment variables required by the vLLM-Plugin-FL README and the backend-specific reference document (e.g. `VLLM_PLUGINS`, `FLAGCX_PATH`, etc.).
5. Run offline batched inference to verify the full stack. Adapt `LLM` constructor parameters based on the backend-specific reference document (e.g. `enforce_eager`, `block_size`, `attention_config`):

```python
from vllm import LLM, SamplingParams

model_path = "<resolved_model_path>"
prompts = ["Hello, my name is"]
sampling_params = SamplingParams(max_tokens=10, temperature=0.0)

# Add backend-specific LLM parameters as documented in the reference files
llm = LLM(model=model_path, max_num_batched_tokens=16384, max_num_seqs=2048)
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(f"Prompt: {output.prompt!r}, Generated text: {output.outputs[0].text!r}")
```

## Troubleshooting

**Out of memory on model load**: Use `gpu_memory_utilization` parameter to limit memory. Start with 0.8 and adjust:
```python
from vllm import LLM
llm = LLM(model="...", gpu_memory_utilization=0.8)
```

**Dependency build failures**: Re-read the dependency's README for build prerequisites. Common missing items include C++17-compatible compilers and Python build tools (`scikit-build-core`, `pybind11`, `ninja`, `cmake`).

**Plugin not loaded**: If vLLM does not use the FL plugin, verify that the environment variable specified in the vLLM-Plugin-FL README is set (e.g. `VLLM_PLUGINS='fl'`).

**Communication library errors**: Re-check the communication library's README for correct build flags and `PATH` variable setup for your platform.

**Backend-specific issues**: See the [references/](references/) directory for hardware-specific troubleshooting.

**Cannot connect to GitHub**: Ask the user for their network proxy settings (e.g. `http_proxy` / `https_proxy`), configure the proxy, then retry the `git clone` command.

## References

- [vLLM-Plugin-FL GitHub](https://github.com/flagos-ai/vllm-plugin-FL)
- For non-NVIDIA chips, refer to the [references/](references/) directory for hardware-specific configurations and setup instructions