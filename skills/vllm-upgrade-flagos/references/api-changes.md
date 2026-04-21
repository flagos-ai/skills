# Known API Changes: vLLM v0.13.0 → v0.18+

This catalog documents breaking API changes between vLLM versions that affect the plugin's
infrastructure code. When upgrading, check each change against the plugin files.

---

## AC1: Attention Layer Import Path Change

**Versions**: v0.14+

```python
# v0.13.0
from vllm.attention.layer import Attention, MLAAttention

# v0.18+
from vllm.model_executor.layers.attention import Attention, MLAAttention
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
```

**Affected**: `model_runner.py`

---

## AC2: CUDAGraphWrapper Moved

**Versions**: v0.15+

```python
# v0.13.0
from vllm.compilation.cuda_graph import CUDAGraphStat  # CUDAGraphWrapper in same file

# v0.18+
from vllm.compilation.cuda_graph import CUDAGraphStat, CUDAGraphWrapper
```

**Affected**: `model_runner.py`, `compilation/graph.py`

---

## AC3: set_forward_context Replaces get_forward_context

**Versions**: v0.15+

```python
# v0.13.0
from vllm.forward_context import BatchDescriptor, get_forward_context

# v0.18+
from vllm.forward_context import BatchDescriptor, set_forward_context
```

**Affected**: `model_runner.py`

---

## AC4: CompilationMode Import Path

**Versions**: v0.14+

```python
# v0.13.0
from vllm.config.compilation import CompilationMode

# v0.18+ (may be merged into vllm.config)
from vllm.config import CompilationMode
```

**Affected**: `worker.py`, `model_runner.py`

---

## AC5: set_current_vllm_config Added

**Versions**: v0.15+

```python
# v0.18+
from vllm.config import set_current_vllm_config
```

New function used in model runner initialization. May need to be called in plugin's model runner.

**Affected**: `model_runner.py`

---

## AC6: RoutedExpertsCapturer Added

**Versions**: v0.16+

```python
# v0.18+
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import RoutedExpertsCapturer
```

New class for MoE CUDA graph capture. Plugin's fused_moe layer may need to integrate.

**Affected**: `model_runner.py`, `ops/fused_moe/layer.py`

---

## AC7: Model Offloader API

**Versions**: v0.16+

```python
# v0.18+
from vllm.model_executor.offloader import create_offloader, get_offloader, set_offloader
```

New model offloading system. Plugin worker/model_runner may need to support it.

**Affected**: `model_runner.py`

---

## AC8: Worker Base Class Changes

**Versions**: v0.14+

```python
# v0.13.0
from vllm.v1.worker.worker_base import WorkerBase

# v0.18+ — WorkerBase may have new abstract methods
```

Check for new abstract methods that the plugin's Worker class must implement.

**Affected**: `worker.py`

---

## AC9: KVCacheConfig / KVCacheSpec Changes

**Versions**: v0.15+

The KV cache interface may have new fields or changed method signatures. Check:
- `vllm/v1/kv_cache_interface.py`
- Worker's `determine_available_memory()` and `initialize_cache()` methods

**Affected**: `worker.py`

---

## AC10: SchedulerOutput Changes

**Versions**: v0.14+

```python
# v0.13.0
from vllm.v1.core.sched.output import SchedulerOutput

# v0.18+ — may have new fields (GrammarOutput, etc.)
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
```

**Affected**: `worker.py`, `model_runner.py`

---

## AC11: ModelRunnerOutput Changes

**Versions**: v0.15+

```python
# v0.13.0
from vllm.v1.outputs import ModelRunnerOutput

# v0.18+ — new output types
from vllm.v1.outputs import AsyncModelRunnerOutput, DraftTokenIds, ModelRunnerOutput
```

**Affected**: `model_runner.py`

---

## AC12: Parallel State API Changes

**Versions**: v0.14+

```python
# v0.13.0
from vllm.distributed.parallel_state import get_pp_group, get_tp_group, graph_capture

# v0.18+ — new functions
from vllm.distributed.parallel_state import (
    get_dcp_group, get_pp_group, get_tp_group,
    graph_capture, is_global_first_rank,
    prepare_communication_buffer_for_model,
)
```

**Affected**: `worker.py`, `model_runner.py`

---

## AC13: validate_cudagraph_capturing_enabled Renamed

**Versions**: v0.16+

```python
# v0.13.0
from vllm.compilation.monitor import validate_cudagraph_capturing_enabled

# v0.18+
from vllm.compilation.monitor import set_cudagraph_capturing_enabled
```

**Affected**: `compilation/graph.py`, `model_runner.py`

---

## AC14: AttentionBackend Import Path

**Versions**: v0.14+

```python
# v0.13.0
from vllm.attention.backends.abstract import AttentionBackend, AttentionMetadata, AttentionType, MultipleOf

# v0.18+ — check if still same path or moved
```

**Affected**: `model_runner.py`

---

## AC15: ReconfigureDistributedRequest

**Versions**: v0.16+

```python
# v0.18+
from vllm.v1.engine import ReconfigureDistributedRequest, ReconfigureRankType
```

New distributed reconfiguration API. Plugin worker may need to handle it.

**Affected**: `worker.py`

---

## AC16: EplbState (Expert Load Balancing)

**Versions**: v0.17+

```python
# v0.18+
from vllm.distributed.eplb.eplb_state import EplbState
```

New expert-level load balancing. Plugin model_runner may need to integrate.

**Affected**: `model_runner.py`

---

## AC17: MultiModal API Changes

**Versions**: v0.15+

```python
# v0.18+
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.encoder_budget import MultiModalBudget
from vllm.multimodal.inputs import BatchedTensorInputs, MultiModalKwargsItem, PlaceholderRange
from vllm.multimodal.utils import group_and_batch_mm_kwargs
```

Significant multimodal API evolution. Plugin model_runner's multimodal handling needs updating.

**Affected**: `model_runner.py`

---

## AC18: LoRA API Changes

**Versions**: v0.15+

```python
# v0.18+
from vllm.lora.layers import LoRAMapping, LoRAMappingType
```

**Affected**: `model_runner.py`

---

## AC19: GPU Worker Submodule Split

**Versions**: v0.17+

In v0.18+, the GPU worker code is split into `vllm/v1/worker/gpu/` subdirectory with many
submodules (model_runner.py, input_batch.py, buffer_utils.py, etc.). The top-level
`gpu_model_runner.py` and `gpu_worker.py` may still exist as compatibility shims or may be
the actual files. Check which is the canonical location.

**Affected**: `worker.py`, `model_runner.py`

---

## Adding New Changes

When a new API incompatibility is discovered during upgrade, append it as AC20, AC21, etc.
Include:
- **What** changed (before/after import or signature)
- **Which versions** introduced the change
- **Affected** plugin files
