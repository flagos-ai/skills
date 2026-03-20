# Additional Steps for Iluvatar GPU (BI-V150)

If using Iluvatar BI-V150 GPU, the following extra configuration applies on top of the main workflow. All source code repositories use the same `flagos-ai` GitHub organization as other backends — no separate forks are needed.

## Installation Differences from NVIDIA

The main workflow (Steps 2–4) applies as-is with one difference:

### FlagCX Build Flag (Step 4)

When building FlagCX, use `USE_ILUVATAR_COREX=1` instead of `USE_NVIDIA=1`, and set the adaptor to `iluvatar_corex`:

```bash
# In the FlagCX directory (from Step 4 of the main workflow)
make USE_ILUVATAR_COREX=1

cd plugin/torch/
FLAGCX_ADAPTOR=iluvatar_corex pip install --no-build-isolation .
```

## Environment Variables

These must be set **in addition to** the standard `VLLM_PLUGINS='fl'` before launching vLLM:

```bash
export VLLM_ENGINE_ITERATION_TIMEOUT_S=36000
export VLLM_RPC_TIMEOUT=36000000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600
```

For enabling FlagOS operator overrides (e.g. attention backend, MoE), also set:

```bash
export VLLM_FL_FLAGOS_WHITELIST="attention_backend,unquantized_fused_moe_method"
export VLLM_FL_OOT_WHITELIST="unquantized_fused_moe_method"
```

## Inference Notes

Iluvatar BI-V150 requires `enforce_eager=True` when launching inference:

```python
from vllm import LLM
llm = LLM(model="<model_path>", enforce_eager=True)
```

Or via the CLI:

```bash
vllm serve <model_path> --enforce-eager --served-model-name <name> --port 8010
```

### Cross-Node Distributed Inference

For multi-node inference, set the network interface environment variables:

```bash
export NCCL_SOCKET_IFNAME=<interface>   # e.g. ens1f0
export GLOO_SOCKET_IFNAME=<interface>   # e.g. ens1f0
```
