# vLLM 0.13.0 Compatibility Patches

Apply these patches after copying upstream model files into the plugin. Not all patches apply to every model — apply only those relevant to the specific model being migrated.

## P1: Relative Imports → Absolute Imports

Upstream files use relative imports that break when moved to the plugin. Convert them:

- **Intra-plugin references** (files that also live in `vllm_fl/models/`) use `vllm_fl.models.*`
- **All other relative imports** use the vLLM 0.13.0 absolute path `vllm.*`

```python
# Before (upstream relative imports):
from .interfaces import ...
from .qwen3_next import ...
from .qwen3_vl import ...
from .qwen2_moe import ...
from .utils import ...

# After (absolute imports):
from vllm.model_executor.models.interfaces import ...
from vllm_fl.models.qwen3_next import ...           # plugin file → vllm_fl
from vllm.model_executor.models.qwen3_vl import ...  # vLLM core → vllm
from vllm.model_executor.models.qwen2_moe import ... # vLLM core → vllm
from vllm.model_executor.models.utils import ...     # vLLM core → vllm
```

**Rule**: If a module also lives in `vllm_fl/models/` (because the plugin may have fixes for it), import from `vllm_fl.models.*`. Otherwise, import from `vllm.model_executor.models.*`.

## P2: Config Imports

Upstream config imports point to paths that don't exist in 0.13.0. Redirect to the plugin's config bridges:

```python
# Before:
from vllm.transformers_utils.configs.qwen3_5 import ...
from vllm.transformers_utils.configs.qwen3_5_moe import ...

# After:
from vllm_fl.configs.qwen3_5 import ...
from vllm_fl.configs.qwen3_5_moe import ...
```

## P3: Remove Missing APIs

These APIs don't exist in 0.13.0 and must be removed:

- `MambaStateCopyFunc` and `MambaStateCopyFuncCalculator` — remove from imports
- `get_mamba_state_copy_func` classmethod — remove the entire method (it references the missing types)

## P4: Replace Context Manager Init Pattern

`_mark_tower_model` and `_mark_language_model` context managers are not available in 0.13.0. Replace with direct initialization:

```python
# Before (upstream context manager pattern):
with self._mark_tower_model(vllm_config, {"image", "video"}):
    self.visual = Qwen3_VisionTransformer(config.vision_config, ...)
with self._mark_language_model(vllm_config):
    self.language_model = XxxForCausalLM(...)

# After (0.13.0 compatible direct init):
if not multimodal_config.get_limit_per_prompt("image") and not multimodal_config.get_limit_per_prompt("video"):
    self.visual = None
else:
    self.visual = Qwen3_VisionTransformer(
        config.vision_config, ...,
        multimodal_config=multimodal_config, ...
    )
self.language_model = XxxForCausalLM(...)
```

Key: add `multimodal_config=multimodal_config` to the `Qwen3_VisionTransformer()` constructor.

## P5: Import Verification

After applying all patches, verify the model imports correctly:

```bash
python3 -c "from vllm_fl.models.{{model_name_lower}} import {{ModelClassName}}; print('OK')"
```

If this fails, fix the specific import error and retry.

## When Patches Aren't Enough

If the model uses APIs not covered above:

1. Test the specific failing import: `python3 -c "from xxx import yyy"`
2. Check if an equivalent exists in 0.13.0 (inspect `/usr/local/lib/python*/dist-packages/vllm/`)
3. If truly missing, stub it out or remove the dependent code
4. Only read the 0.13.0 source file when comparing a specific method signature

## P6: Parent Class __init__ Reimplementation

When a child class inherits from a parent that uses 0.13.0-missing APIs in its `__init__` (e.g. `_mark_tower_model`, `_mark_language_model`), you cannot simply call `super().__init__()`. Instead, reimplement the `__init__` body, replacing the missing APIs with their 0.13.0 equivalents (see P4).

```python
# Before (upstream — child delegates to parent __init__):
class Qwen3_5MoeForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    # inherits __init__ from parent which uses _mark_tower_model

# After (0.13.0 — reimplement __init__ with direct initialization):
class Qwen3_5MoeForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    def __init__(self, *, vllm_config, prefix=""):
        # Copy parent __init__ body, replacing _mark_tower_model/
        # _mark_language_model with direct init (P4 pattern)
        super(Qwen3_5ForConditionalGeneration, self).__init__(...)  # skip parent, call grandparent
        # ... direct initialization of visual + language_model ...
```

**Why**: The parent's `__init__` calls context managers that don't exist in 0.13.0. Calling `super().__init__()` would crash.

**When**: Any VL (vision-language) model class whose parent class uses `_mark_tower_model` / `_mark_language_model` in `__init__`.

## P7: MoE Nested Config — hf_config → hf_text_config Patching

When a model's HuggingFace config is a VL (vision-language) config with a nested `text_config` containing MoE parameters (like `num_experts`, `expert_top_k`, `moe_intermediate_size`), and the model inherits from a parent MoE class that reads these params from `hf_config` directly, you must wrap the parent class to redirect `hf_config` to `hf_text_config`.

```python
# Before (upstream — parent reads MoE params from hf_config):
class Qwen3NextSparseMoeBlock(nn.Module):
    def __init__(self, ...):
        config = vllm_config.model_config.hf_config  # expects MoE params here
        self.num_experts = config.num_experts  # fails: num_experts is in text_config

# After (0.13.0 — create a wrapper subclass):
class Qwen3_5SparseMoeBlock(Qwen3NextSparseMoeBlock):
    def __init__(self, *, vllm_config, prefix=""):
        patched_vllm_config = _patch_vllm_config_for_moe(vllm_config)
        super().__init__(vllm_config=patched_vllm_config, prefix=prefix)

def _patch_vllm_config_for_moe(vllm_config):
    """Replace hf_config with hf_text_config so MoE params are visible."""
    import copy
    patched = copy.copy(vllm_config)
    patched_model = copy.copy(patched.model_config)
    patched_model.hf_config = patched_model.hf_text_config
    patched.model_config = patched_model
    return patched
```

**Why**: In VL models, MoE parameters live under `text_config`, not at the top level. Parent MoE classes read `hf_config` directly and fail with `AttributeError`.

**When**: Any MoE model whose top-level config is a VL config (has `text_config`), inheriting from a parent class that reads MoE params from `hf_config`.

## P8: MergedColumnParallelLinear for Merged Projections with Different Sub-dimensions

When a model has a merged projection combining sub-outputs with **different dimensions** (e.g. `in_proj_qkvz` = q + k + v + z where `key_dim ≠ value_dim`), you **MUST** use `MergedColumnParallelLinear` instead of `ColumnParallelLinear`. This is critical for correct Tensor Parallel (TP) sharding.

**Why `ColumnParallelLinear` fails**: It does contiguous TP slicing — rank i gets rows `[i*N/tp : (i+1)*N/tp]`. But when sub-outputs have different sizes (e.g. q=2048, k=2048, v=8192, z=8192), contiguous slicing gives each GPU a mix of wrong sub-projections. `MergedColumnParallelLinear` slices each sub-output independently, which is what the forward pass expects.

**How to detect**: Look at the HF checkpoint weight names. If you see split weights like `in_proj_qkv` + `in_proj_z` (or `in_proj_b` + `in_proj_a`), AND the parent class creates a single `ColumnParallelLinear` for the merged name, check whether the sub-dimensions differ. If they do → must override with `MergedColumnParallelLinear`.

**Three-part fix**:

### Part 1: Override `__init__` with `MergedColumnParallelLinear`

```python
class MyGatedDeltaNet(ParentGatedDeltaNet):
    def __init__(self, config, ...):
        nn.Module.__init__(self)  # Skip parent's __init__ entirely
        # ... set all scalar attributes (tp_size, hidden_size, etc.) ...

        # Use MergedColumnParallelLinear with explicit output_sizes
        self.in_proj_qkvz = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[key_dim, key_dim, value_dim, value_dim],  # q, k, v, z
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.in_proj_qkvz",
        )
        self.in_proj_ba = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[num_v_heads, num_v_heads],  # b, a
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.in_proj_ba",
        )
```

### Part 2: Add stacked_params_mapping with tuple shard_ids

```python
stacked_params_mapping = [
    # ... existing entries ...
    # GDN: tuple shard_id = load one HF weight into multiple shards
    ("in_proj_qkvz", "in_proj_qkv", (0, 1, 2)),  # qkv → shards 0,1,2
    ("in_proj_qkvz", "in_proj_z", 3),              # z → shard 3
    ("in_proj_ba", "in_proj_b", 0),
    ("in_proj_ba", "in_proj_a", 1),
]
```

### Part 3: Handle tuple shard_id in load_weights loop

```python
param = params_dict[name]
weight_loader = param.weight_loader
if isinstance(shard_id, tuple):
    # Split by MergedColumnParallelLinear.output_sizes and load each shard
    layer = weight_loader.__self__
    split_sizes = [layer.output_sizes[s] for s in shard_id]
    output_dim = getattr(param, "output_dim", 0)
    parts = loaded_weight.split(split_sizes, dim=output_dim)
    for s, part in zip(shard_id, parts):
        weight_loader(param, part, s)
else:
    weight_loader(param, loaded_weight, shard_id)
```

**When**: Any model with merged linear projections where sub-outputs have different dimensions (e.g. GDN layers in Qwen3.5, hybrid attention models with mixed q/k/v/z sizes).

## P9: MambaStateDtypeCalculator — 2-Argument Signature

In 0.13.0, `MambaStateDtypeCalculator.gated_delta_net_state_dtype()` takes **2 arguments** (`self`, `dt_dtype`), not 3. Remove the `mamba_ssm_cache_dtype` third argument:

```python
# Before (upstream — 3-arg call):
mamba_state_dtype = MambaStateDtypeCalculator.gated_delta_net_state_dtype(
    dt_dtype=self.dt_proj.weight.dtype,
    mamba_ssm_cache_dtype=vllm_config.model_config.dtype,
)

# After (0.13.0 — 2-arg call):
mamba_state_dtype = MambaStateDtypeCalculator.gated_delta_net_state_dtype(
    dt_dtype=self.dt_proj.weight.dtype,
)
```

**Why**: The `mamba_ssm_cache_dtype` parameter was added after 0.13.0.

**When**: Any model with hybrid attention that uses `MambaStateDtypeCalculator.gated_delta_net_state_dtype()`.

## P10: mamba_cache_mode → enable_prefix_caching

`vllm_config.cache_config.mamba_cache_mode` does not exist in 0.13.0. Replace usage with the available `enable_prefix_caching`:

```python
# Before (upstream):
if vllm_config.cache_config.mamba_cache_mode == "full":
    ...

# After (0.13.0):
if vllm_config.cache_config.enable_prefix_caching:
    ...
```

**Why**: `mamba_cache_mode` was introduced after 0.13.0 to control Mamba/hybrid model caching behavior. In 0.13.0, `enable_prefix_caching` is the closest equivalent.

**When**: Any hybrid attention model (mixing full attention + linear attention/Mamba layers) that checks `mamba_cache_mode`.

## P11: Plugin Custom Op Import Path Deduplication

The plugin has multiple modules that register the same custom op name (e.g. `chunk_gated_delta_rule`). If a parent module (like `qwen3_next.py`) already imports and registers a custom op via one path (e.g. `vllm_fl.ops.fla`), child classes **MUST** import from the **same path**, not from an alternative module (e.g. `vllm_fl.models.fla_ops`).

```python
# WRONG — causes "Duplicate op name: chunk_gated_delta_rule"
# (qwen3_next.py already imported from vllm_fl.ops.fla)
from vllm_fl.models.fla_ops import ChunkGatedDeltaRuleOp

# CORRECT — same path as parent module, module already in sys.modules
from vllm_fl.ops.fla import ChunkGatedDeltaRuleOp
```

**How to detect**: Check the parent module's imports (e.g. `grep "ChunkGatedDeltaRuleOp" vllm_fl/models/qwen3_next.py`). Use the same import path.

**Why**: Python's `@CustomOp.register("name")` decorator runs at module load time. If two different modules register the same op name, the second import triggers `AssertionError: Duplicate op name`. Importing from the same path is a no-op (module already cached in `sys.modules`).

**When**: Any model that inherits from a parent class in the plugin (e.g. `Qwen3NextGatedDeltaNet`) and needs to create custom op instances in its own `__init__`.

## P12: Complete `__init__` Override — Inherited Method Attribute Checklist

When overriding `__init__` with `nn.Module.__init__(self)` (skipping the parent entirely), you must create **ALL** attributes used by inherited methods — not just the ones used in the overridden `forward()`.

**Checklist**:

1. **Read the parent's `__init__`** — list every `self.xxx = ...` assignment
2. **Search inherited methods for `self.xxx` access** — especially `_forward_core`, `rearrange_mixed_qkv`, and any method called by custom ops (e.g. `gdn_attention_core` → `self._forward_core`)
3. **Create ALL required attributes** with matching constructor arguments

Common missed attributes (Qwen3.5 / GDN models):

```python
# These are used by the inherited _forward_core method,
# NOT by the overridden forward() — easy to miss!
self.chunk_gated_delta_rule = ChunkGatedDeltaRuleOp(
    output_final_state=True,
    use_qk_l2norm_in_kernel=True,
)
self.fused_recurrent_gated_delta_rule_multi_query = FusedRecurrentGatedDeltaRuleOp(
    inplace_final_state=True,
    use_qk_l2norm_in_kernel=True,
)
self.fused_recurrent_gated_delta_rule_remain_query = FusedRecurrentGatedDeltaRuleOp(
    inplace_final_state=True,
    use_qk_l2norm_in_kernel=True,
)
```

**Why**: Custom ops like `gdn_attention_core` dispatch to `self._forward_core()` via `compilation_config.static_forward_context`. If `_forward_core` is inherited (not overridden), it accesses attributes from the parent that were never created → `AttributeError` at runtime (typically during CUDA graph warmup, not at import time).

**When**: Any model class that uses `nn.Module.__init__(self)` to skip the parent's `__init__`, while inheriting runtime methods from the parent.

---

## P13: Do NOT Upgrade transformers to Support New Models

The transformers version is determined by the pinned vLLM version (e.g. vLLM 0.13.0 → transformers 4.57.6). When migrating a new model whose HuggingFace config declares a newer `transformers_version` (e.g. `"5.0.2.dev0"`), do NOT upgrade the transformers package. Instead, create a config bridge in `vllm_fl/configs/`.

**Why**: Silently upgrading transformers changes code paths — some ops may fall through to transformers' own implementations instead of the plugin's OOT operators or FlagGems, causing correctness issues. The transformers version should only change when the underlying vLLM version is upgraded.

**When**: Every migration under the current vLLM pin. Use the config bridge approach (P2) to handle unrecognized `model_type` values.

---

## Adding New Patches

When a new incompatibility is discovered, append it as P14, P15, etc. Include:

- **What** to change (before/after code example)
- **Why** it's needed (what's missing or different in 0.13.0)
- **When** this patch applies (which model architectures or patterns trigger it)
