# Additional Steps for Iluvatar GPU (BI-V150)

If using Iluvatar BI-V150 GPU, the following extra configuration applies on top of the main workflow.

## Environment Variables

For enabling FlagOS operator overrides (e.g. attention backend, MoE), set these **in addition to** the standard vLLM-Plugin-FL environment variables before launching vLLM:

```bash
export VLLM_FL_FLAGOS_WHITELIST="attention_backend,unquantized_fused_moe_method"
export VLLM_FL_OOT_WHITELIST="unquantized_fused_moe_method"
```

## Inference Notes

Iluvatar BI-V150 requires `enforce_eager=True` when launching inference.

### Cross-Node Distributed Inference

For multi-node inference, set the network interface environment variables:

```bash
export NCCL_SOCKET_IFNAME=<interface>   # e.g. ens1f0
export GLOO_SOCKET_IFNAME=<interface>   # e.g. ens1f0
```
