# Additional Steps for Ascend NPU

If using Huawei Ascend NPU, the following extra configuration applies on top of the main workflow.

## Dependency Ordering Constraint

FlagTree must be installed **before** FlagGems. If FlagTree is not installed first, the FlagGems verification will fail repeatedly. Refer to the vLLM-Plugin-FL README for the FlagTree repository URL and installation instructions.

## Environment Variables

```bash
export TRITON_ALL_BLOCKS_PARALLEL=1
```

## Inference Notes

Ascend requires eager execution. Add `enforce_eager=True` to the `LLM` constructor or pass `--enforce-eager` on the command line.
