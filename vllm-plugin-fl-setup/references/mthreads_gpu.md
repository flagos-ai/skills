# Additional Steps for Moore Threads GPU

If using Moore Threads GPU, the following extra steps and overrides apply.

## Docker Environment (Recommended)

Moore Threads provides pre-built Docker images with the full software stack (vLLM, vLLM-Plugin-FL, FlagGems, FlagCX) already installed. Using the Docker image is the recommended approach.

### Select Image by CPU Type

The correct image depends on the host CPU. Detect the CPU vendor first:

```bash
grep -m1 'vendor_id' /proc/cpuinfo
```

| CPU Vendor | `vendor_id` | Image |
|------------|-------------|-------|
| Intel | `GenuineIntel` | `registry.mthreads.com/presale/devtech/vllm_plugin:20260311` |
| Hygon (海光) | `HygonGenuine` | `registry.mthreads.com/presale/devtech/vllm_plugin:20260313hg` |

### Start Container (if not already inside one)

In most cases you are already working inside a container. If you need to create a new one, use the image selected above:

```bash
docker run -itd --privileged --net host \
  --name=vllm_plugin_test -w /workspace \
  -v /data/:/data/ \
  --env MTHREADS_VISIBLE_DEVICES=all \
  --shm-size=300g \
  <image> /bin/bash

docker exec -it vllm_plugin_test bash
```


If the Docker image already has everything installed, you can skip directly to the [Environment Variables](#environment-variables) section below.

## Installation Overrides

If you need to reinstall the software stack (e.g. for development), the following overrides apply to the main workflow steps. **FlagGems follows the standard installation** — no Moore Threads–specific changes needed.

### vLLM-Plugin-FL (overrides Step 2)

Moore Threads uses a separate fork and branch:

```bash
cd ~/flagos-workspace
git clone https://github.com/jiamingwang-mt/vllm-plugin-FL.git
cd vllm-plugin-FL
git checkout musa/dev
pip install --no-build-isolation .
```

### FlagCX (overrides Step 4)

Moore Threads uses a separate fork and builds with MUSA support:

```bash
cd ~/flagos-workspace
git clone https://github.com/jiamingwang-mt/FlagCX.git
cd FlagCX
git checkout dev
git submodule update --init --recursive
make USE_MUSA=1

cd plugin/torch/
python setup.py develop --adaptor musa
```

Set the FlagCX path:

```bash
export FLAGCX_PATH=/path/to/FlagCX
```

## Environment Variables

These must be set before launching vLLM:

```bash
export VLLM_PLUGINS='fl'
export USE_FLAGGEMS=1
export FLAGCX_PATH=/workspace/FlagCX  # MUST point to the actual FlagCX installation directory; this is only an example
export VLLM_MUSA_ENABLE_MOE_TRITON=1
```

## Inference Notes

Moore Threads requires additional parameters when constructing the `LLM` object. These are automatically applied in the Quick Test (main workflow) when the backend is detected as Moore Threads:

- `enforce_eager=True`
- `block_size=64`
- `attention_config={"backend": "TORCH_SDPA"}`

> **Important:** Do not run inference from the `/workspace` directory — `cd /` or use another path first.

## Supported Models

The following models have been tested on Moore Threads GPU:

**Text generation:**
- Qwen3.5-35B-A3B
- Qwen3-0.6B
- Qwen3-8B
- Qwen3-Next-80B-A3B-Instruct

**Multimodal:**
- Qwen3.5-35B-A3B
- Qwen3-VL-8B-Instruct

> **Note:** First-time inference for some models may take a long time due to Triton kernel compilation.
