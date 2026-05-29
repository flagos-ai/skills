# 代码修改常见模式

## 1. dtype 修复

### Benchmark: 硬编码 → 常量

```python
# Before
dtypes=[torch.float16, torch.float32, torch.bfloat16]

# After
from . import consts
dtypes=consts.FLOAT_DTYPES
```

### Test: 硬编码 → 常量

```python
# Before
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])

# After
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
```

### 例外：dtype 不支持时

某些算子（linalg、special 等）不支持 Half/BFloat16，此时可以硬编码但**必须加注释**：

```python
# torch.linalg.cholesky does not support float16/bfloat16 on CUDA
dtypes=[torch.float32, torch.float64]
```

---

## 2. Benchmark 封装类替换

### 自定义代码 → UnaryPointwiseBenchmark

```python
# Before (错误 — 自定义 pytest parametrize)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_perf_foo(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    ...

# After (正确)
@pytest.mark.foo
def test_foo():
    bench = base.UnaryPointwiseBenchmark(
        op_name="foo",
        torch_op=torch.foo,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
```

### 需要自定义输入 → GenericBenchmark + input_fn

```python
@pytest.mark.foo
def test_foo():
    def input_fn(shape, dtype, device):
        return (torch.randn(shape, dtype=dtype, device=device),
                torch.randn(shape, dtype=dtype, device=device))

    bench = base.GenericBenchmark(
        op_name="foo",
        torch_op=torch.foo,
        input_fn=input_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
```

### 需要自定义 shape → 继承 + set_more_shapes

```python
class FooBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        # Custom shapes for matrix operations
        return [(64, 64), (128, 128), (256, 256), (512, 512)]

@pytest.mark.foo
def test_foo():
    bench = FooBenchmark(
        op_name="foo",
        torch_op=torch.foo,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
```

### BLAS 算子 → BlasBenchmark

```python
@pytest.mark.foo
def test_foo():
    bench = base.BlasBenchmark(
        op_name="foo",
        torch_op=torch.foo,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
```

---

## 3. Import 修复

### 测试文件相对导入

```python
# Before (错误)
import accuracy_utils as utils
from flag_gems.testing import accuracy_utils as utils

# After (正确)
from . import accuracy_utils as utils
```

### 删除无用 import

```python
# flake8 F401 — 检查是否真的使用了
# 删除不需要的 import（常见：benchmark 中不需要 consts 但 import 了）
```

### import 顺序（isort）

```python
# 正确顺序：stdlib → third-party → local
import pytest          # third-party
import torch           # third-party

import flag_gems       # local (absolute)
from . import base     # local (relative)
from . import consts   # local (relative)
```

---

## 4. libdevice → tl_extra_shim

```python
# Before
from triton.language.extra.cuda import libdevice
result = libdevice.erfc(x)

# After
from flag_gems.runtime import tl_extra_shim
result = tl_extra_shim.erfc(x)
```

---

## 5. print → logger

```python
# Before
print(f"Processing {op_name}")

# After (kernel 文件)
import logging
logger = logging.getLogger(__name__)
logger.debug("Processing %s", op_name)

# 或者直接删除（大多数情况）
```

---

## 6. gems_assert_close 修复

### 删除 rtol 参数

```python
# Before (错误 — rtol 不支持)
utils.gems_assert_close(res, ref, dtype, rtol=1e-3)

# After (正确 — 只用 atol)
utils.gems_assert_close(res, ref, dtype, atol=1e-3)
```

### NaN 比较

```python
# 涉及 NaN 的算子
utils.gems_assert_close(res, ref, dtype, equal_nan=True)
```

### 位精确算子 → gems_assert_equal

```python
# 位精确算子（如 bitwise_and, eq, logical_or）不需要容差比较
# 使用 gems_assert_equal 做严格逐元素相等断言
utils.gems_assert_equal(res, ref)
```

---

## 7. operators.yaml 修复

### description 补充

```yaml
# Before
- id: foo
  description: ""

# After — 从 PyTorch docs 复制一句话
- id: foo
  description: |
    Computes the element-wise error function of input.
```

### 字母序插入

operators.yaml 中的条目必须按 `id` 字母序排列。如果插入位置不对，移动到正确位置。

---

## 8. 测试函数修复

### to_reference 遗漏

```python
# Before (错误 — ref_inp 没有 to_reference)
ref_out = torch.foo(inp)

# After (正确)
ref_inp = utils.to_reference(inp)
ref_out = torch.foo(ref_inp)
```

### 多输入 tensor 都要转

```python
# 所有参与参考计算的 tensor 都需要 to_reference
ref_a = utils.to_reference(a)
ref_b = utils.to_reference(b)
ref_out = torch.foo(ref_a, ref_b)
```

---

## 9. `_FULL_CONFIG` 注册修复

在 `src/flag_gems/__init__.py` 的 `_FULL_CONFIG` 列表中注册算子。

### 基本注册

```python
# 普通算子（不带 aten:: 前缀）
("foo", foo)

# 下划线前缀算子 — aten name 保留下划线
("_foo", _foo)
```

### Overload 变体

```python
# Tensor overload
("foo.Tensor", foo)

# out 变体
("foo.out", foo_out)
```

### Inplace 变体

```python
# inplace — 尾部下划线保留
("foo_", foo_)
```

### 常见错误

```python
# 错误：多加了 aten:: 前缀（实际不需要）
("aten::foo", foo)  # 应为 ("foo", foo)

# 错误：下划线前缀算子漏掉下划线
("foo", _foo)  # 应为 ("_foo", _foo)

# 错误：overload 用下划线而非点号
("foo_Tensor", foo)  # 应为 ("foo.Tensor", foo)
```

---

## 10. is_cuda → flag_gems.device

```python
# Before (错误 — 不应使用 is_cuda)
if input.is_cuda:
    ...

# After (正确 — 使用 flag_gems.device 进行设备判断)
import flag_gems
if input.device == flag_gems.device:
    ...

# 或者：如果设备检查完全不必要，直接删除整个 if 分支
```

---

## 11. autotune config 提取到配置文件

```python
# Before (错误 — inline autotune config)
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=2),
    ],
    key=["M", "N"],
)
@triton.jit
def foo_kernel(...):
    ...

# After (正确 — config 从配置文件读取)
from flag_gems.runtime import config

# configs 定义在独立的 config 文件中（如 flag_gems/runtime/configs/foo.yaml），
# kernel 通过 config 模块引用，不在 kernel 文件中内联硬编码。
@triton.jit
def foo_kernel(...):
    ...
```

---

## 12. logger 格式规范

```python
# Before (错误 — 格式不对)
logger.debug("running foo kernel")
logger.debug(f"GEMS {op_name}")

# After (正确 — 使用标准格式)
logger.debug("GEMS FOO")
# 格式要求：logger.debug("GEMS <OP_NAME_UPPER>")
# OP_NAME_UPPER 为算子名大写形式
```

---

## 13. Benchmark 公平性修复

```python
# Before (不公平 — torch wrapper 包含额外开销)
def torch_op(input):
    output = input.requires_grad_(True)
    result = torch.foo(output)
    result.backward(torch.ones_like(result))
    return output.grad

# After (公平 — 直接调用 aten backward op)
def torch_op(grad_output, input):
    return torch.ops.aten.foo_backward(grad_output, input)
```

对于 backward 类 benchmark，torch wrapper 必须与 gems wrapper 测量相同的操作范围。使用 `torch.ops.aten.<op>_backward(...)` 直接调用底层 aten op，而非通过 autograd 间接测量。

---

## 14. 缺失的测试/Benchmark 文件

### 新增测试文件模板

```python
# tests/test_<op>_ops.py
import pytest
import torch

from . import accuracy_utils as utils


@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
class TestFoo:
    def test_foo(self, shape, dtype):
        inp = torch.randn(shape, dtype=dtype, device="cuda")
        ref_inp = utils.to_reference(inp)

        res = torch.foo(inp)
        ref = torch.foo(ref_inp)
        utils.gems_assert_close(res, ref, dtype)
```

### 新增 Benchmark 文件模板

```python
# benchmarks/test_<op>_perf.py
import pytest
import torch

from . import base
from . import consts


@pytest.mark.foo
def test_foo():
    bench = base.GenericBenchmark(
        op_name="foo",
        torch_op=torch.foo,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
```

---

## 15. ops/ → fused/ 目录迁移

当 reviewer 指出算子应该放在 `fused/` 目录时：

```bash
# 1. 移动文件
git mv src/flag_gems/ops/foo.py src/flag_gems/fused/foo.py

# 2. 更新 ops/__init__.py — 删除旧 import
# Before
from .foo import foo

# 3. 更新 fused/__init__.py — 添加新 import
from .foo import foo

# 4. 更新 __init__.py 注册（如有必要）
# fused 算子通常通过 fused/__init__.py 自动注册
```

**判断标准：** 融合了多个基本算子的实现（如 fused_attention、fused_norm）应放在 `fused/`，单个标准 aten op 的实现放在 `ops/`。

---

## 16. Benchmark shapes 去重

```python
# Before (错误 — 重复了 core_shapes.yaml 中的 shapes)
class FooBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return [
            (256,), (512,), (1024,),       # 这些在 core_shapes.yaml 已有
            (4096, 2048),                    # 只有这个是自定义的
        ]

# After (正确 — 只保留不在 core_shapes.yaml 中的自定义 shapes)
class FooBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        # Additional shapes specific to foo operation
        return [(4096, 2048)]
```

`set_more_shapes` 只用于添加 `core_shapes.yaml` 中没有的、算子特有的 shapes。基础 shapes 已由框架自动加载。
