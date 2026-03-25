# Additional Steps for Moore Threads GPU

If using Moore Threads GPU, the following extra configuration applies on top of the main workflow. All source code repositories use the same `flagos-ai` GitHub organization as other backends — no separate forks are needed.

## Installation Differences from NVIDIA

The main workflow (Steps 2–4) applies as-is with one difference:

### FlagCX Build Flag (Step 4)

When building FlagCX, use `USE_MUSA=1` instead of `USE_NVIDIA=1`, and set the adaptor to `musa`:

```bash
# In the FlagCX directory (from Step 4 of the main workflow)
make USE_MUSA=1

cd plugin/torch/
FLAGCX_ADAPTOR=musa pip install --no-build-isolation .
```

## Environment Variables

These must be set **in addition to** the standard `VLLM_PLUGINS='fl'` before launching vLLM:

```bash
export USE_FLAGGEMS=1
export FLAGCX_PATH=/path/to/FlagCX  # MUST point to the actual FlagCX installation directory
export VLLM_MUSA_ENABLE_MOE_TRITON=1
```

## Inference Notes

Moore Threads requires additional parameters when constructing the `LLM` object:

- `enforce_eager=True`
- `block_size=64`
- `attention_config={"backend": "TORCH_SDPA"}`

> **Note:** First-time inference for some models may take a long time due to Triton kernel compilation.
