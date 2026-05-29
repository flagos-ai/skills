# FlagGems PR Review 常见意见模式与标准回复

## 1. 命名类

### mark 与 yaml id 不一致

**Reviewer 意见示例：**
> "pytest mark should match the yaml id"
> "mark 需要和 operators.yaml 的 id 一致"

**修复：**
```python
# Before
@pytest.mark.reflection_pad3d  # yaml id 是 reflection_pad3d_out

# After
@pytest.mark.reflection_pad3d_out
```

**回复模板：**
```
Fixed. Updated pytest mark and op_name to match yaml id `<id>`.
```

### 前导下划线处理错误

**Reviewer 意见示例：**
> "id 中不需要前导下划线"
> "mark 里去掉下划线"

**规则：**
- yaml `id`：去掉前导下划线（`_foo` → `foo`）
- pytest mark：去掉前导下划线
- 文件名/函数名/import：保留下划线
- `_FULL_CONFIG` 的 aten name：保留下划线

**回复模板：**
```
Fixed. Removed leading underscore from yaml id and pytest mark while keeping it in filenames and imports per naming convention.
```

---

## 2. Benchmark 类

### 使用了错误的 benchmark 封装类

**Reviewer 意见示例：**
> "请使用 base 封装类"
> "不要自定义 benchmark 框架"

**修复：** 根据算子类型选择正确的封装类（参考 `code-fix-patterns.md` §2）

**回复模板：**
```
Fixed. Replaced custom benchmark with `base.<ClassName>` as required.
```

### dtype 硬编码

**Reviewer 意见示例：**
> "dtype 请使用常量"
> "不要硬编码 dtype"

**修复：**
```python
# Before
dtypes=[torch.float32, torch.float16]

# After (benchmark)
dtypes=consts.FLOAT_DTYPES

# After (test)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
```

**回复模板：**
```
Fixed. Replaced hardcoded dtypes with `consts.FLOAT_DTYPES` (benchmark) / `utils.FLOAT_DTYPES` (test).
```

---

## 3. 代码质量类

### 删除 print

**Reviewer 意见示例：**
> "请删除 print"
> "不要 print，用 logger"

**修复：** 删除 `print()` 或改为 `logger.debug()`

**回复模板：**
```
Fixed. Removed print statements.
```

### import 问题

**Reviewer 意见示例：**
> "import 顺序不对"
> "不需要这个 import"
> "用相对导入"

**修复：** 运行 `isort`，删除无用 import，测试文件用 `from . import accuracy_utils as utils`

**回复模板：**
```
Fixed. Removed unused import / corrected import order / switched to relative import.
```

---

## 4. 功能/逻辑类

### libdevice 兼容性

**Reviewer 意见示例：**
> "需要跨后端兼容"
> "libdevice 只支持 CUDA"

**修复：** 将 `tl.extra.cuda.libdevice.xxx` 替换为 `tl_extra_shim` 的跨后端封装

**回复模板：**
```
Fixed. Replaced `tl.extra.cuda.libdevice.<func>` with the cross-backend `tl_extra_shim` wrapper for portability.
```

### description 需要补充

**Reviewer 意见示例：**
> "yaml 里 description 需要补充"
> "描述太简单了"

**修复：** 从 PyTorch 官方文档复制一句话描述

**回复模板：**
```
Fixed. Updated description in operators.yaml with the official PyTorch doc summary.
```

---

## 5. Git/提交类

### Co-Authored-By 问题

**Reviewer 意见示例：**
> "删掉 Co-Authored-By"
> "CLA 会失败"

**修复：** 新 commit 不加 Co-Authored-By（已 push 的无法修改，但后续 commit 不再包含）

**回复模板：**
```
Fixed in subsequent commit. Co-Authored-By removed — CLA should pass now.
```

---

## 6. Kernel 规范类

### 不要使用 is_cuda

**Reviewer 意见示例：**
> "don't use is_cuda, use flag_gems.device"
> "不要用 is_cuda 判断设备"

**修复：** 将 `.is_cuda` 替换为 `flag_gems.device` 比较，或直接删除不必要的设备检查

**回复模板：**
```
Fixed. Replaced `is_cuda` check with `flag_gems.device` comparison.
```

### autotune config 不要内联

**Reviewer 意见示例：**
> "move autotune config to config file"
> "autotune 配置放到配置文件里"

**修复：** 将 `@triton.autotune` 内联配置提取到配置文件，使用 `@torch_device_fn.register_triton_autotune_config` 注册

**回复模板：**
```
Fixed. Extracted inline autotune configs to config file and registered via `register_triton_autotune_config`.
```

### logger 格式不对

**Reviewer 意见示例：**
> "logger format wrong"
> "logger 格式不对，用 GEMS <OP> 格式"

**修复：** 使用 `logger.debug("GEMS <OP_NAME_UPPER>")` 标准格式

**回复模板：**
```
Fixed. Updated logger format to `logger.debug("GEMS <OP_NAME_UPPER>")`.
```

---

## 7. Benchmark 公平性

### benchmark 不公平

**Reviewer 意见示例：**
> "benchmark is unfair"
> "torch 和 gems 测量范围不一致"

**修复：** 确保 torch wrapper 与 gems wrapper 测量相同操作范围。对 backward 类算子使用 `torch.ops.aten.<op>_backward(...)` 直接调用

**回复模板：**
```
Fixed. Aligned torch wrapper to measure the same operation scope as gems by calling `torch.ops.aten.<op>_backward` directly.
```

---

## 8. 缺失文件类

### 缺少测试 / Benchmark

**Reviewer 意见示例：**
> "missing test file"
> "需要添加 benchmark"
> "没有对应的测试"

**修复：** 按照标准模板创建缺失的测试或 benchmark 文件（参考 `code-fix-patterns.md` §14）

**回复模板：**
```
Fixed. Added test/benchmark file `<filename>` following the standard template.
```

---

## 9. 目录结构类

### 算子应放在 fused/ 目录

**Reviewer 意见示例：**
> "this should be in fused/"
> "融合算子放到 fused 目录"

**修复：** 将文件从 `ops/` 移动到 `fused/`，并更新相应的 `__init__.py` import

**回复模板：**
```
Fixed. Moved `<op>.py` from `ops/` to `fused/` and updated imports.
```

### Benchmark shapes 与 core_shapes.yaml 重复

**Reviewer 意见示例：**
> "shapes duplicate core_shapes.yaml"
> "这些 shapes 在 core_shapes 里已经有了"

**修复：** 删除与 `core_shapes.yaml` 重复的 shapes，`set_more_shapes` 中只保留算子特有的自定义 shapes

**回复模板：**
```
Fixed. Removed shapes already covered by `core_shapes.yaml`, kept only op-specific custom shapes.
```

---

## 10. 通用回复原则

1. **用英文回复**（上游是国际项目）
2. **简洁具体** — 说明改了什么，不写无意义的感谢
3. **引用 commit** — `Fixed in <short_sha>.` 让 reviewer 快速定位
4. **批量回复** — 同类问题可以合并回复："Fixed all naming issues in <sha>."
5. **不争论** — 如果不认同，礼貌说明技术原因即可
