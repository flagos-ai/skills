# Additional Steps for Moore Threads GPU

If using Moore Threads GPU, the following extra configuration applies on top of the main workflow.

## Environment Variables

These must be set **in addition to** the standard vLLM-Plugin-FL environment variables before launching vLLM:

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
