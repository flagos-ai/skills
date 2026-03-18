# Additional Steps for Ascend NPU

If using Huawei Ascend NPU, the following extra steps are required. **The FlagTree installation below must be completed before installing or verifying FlagGems (Step 3 in the main workflow).** If FlagTree is not installed first, the FlagGems verification will fail repeatedly and keep reinstalling Triton.

## Install FlagTree (before FlagGems)

> **Important:** This step must be done **before** proceeding to Step 3 (Install FlagGems) in the main workflow.

```bash
RES="--index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple --trusted-host=https://resource.flagos.net"
pip install flagtree==0.4.0+ascend3.2 $RES
```

Verify FlagTree installation:

```bash
python -c "import flagtree; print('FlagTree installed successfully')"
```

## Set Environment Variables

```bash
export TRITON_ALL_BLOCKS_PARALLEL=1
```

## Enable Eager Execution

Ascend requires eager execution. Add `enforce_eager=True` to the `LLM` constructor or pass `--enforce-eager` on the command line.

```python
from vllm import LLM
llm = LLM(model="Qwen/Qwen3-4B", enforce_eager=True)
```
