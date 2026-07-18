<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->

# FlagGems PR Review 常见问题

来自历史 PR Review 的经验总结，提交前务必逐条排查。

## 1. 算子命名问题

### 命名转换规则
- aten 算子需要进行名字转换（例如 `a.out` → `a_out`，`a.self_out` → `a_self_out`）
- **前导下划线去除（mark 和 yaml id 中）**：`_foo` → `foo`
  - 文件名、函数名、import、`_FULL_CONFIG` 的 aten name 保留下划线
  - 只有 pytest mark 和 yaml `id` 去掉前导下划线
  - 例如：`_cholesky_solve_helper` → yaml `id: cholesky_solve_helper`，mark `@pytest.mark.cholesky_solve_helper`
- 尾部下划线保留：`b_` → `b_`（inplace 变体）
- fused 算子以算子本身名称为 ID，不用参考实现的名称
- **禁止随意 mark**

### 命名冲突
- 提交前必须检查是否与现有算子冲突
- 运行：`grep "id: <op>" conf/operators.yaml`

## 2. 测试文件问题

### 禁止 print
- 测试中 `print()` 会干扰正常数据采集
- 跳过测例用 `@pytest.mark.skip(reason=…)`

### 运行时间
- 注意测试运行时间，避免不必要的开销
- 审视是否真的需要很重的 shapes 来执行精度测试

### mark 不匹配
- 测例的 mark 必须与算子 ID 严格匹配
- 错误示例：算子名 `special_erfcx` 但 mark 是 `@pytest.mark.erfcx`

## 3. Benchmark 问题

### 禁止自定义框架
- **必须使用 pytest + base 封装类**
- 不允许用自定义框架替代 pytest

### 封装类选择
| 算子类型 | 使用的类 |
|---------|---------|
| 一元 pointwise | `base.UnaryPointwiseBenchmark` |
| 二元 pointwise | `base.BinaryPointwiseBenchmark` |
| Reduction | `base.UnaryReductionBenchmark` |
| Linalg/自定义 | 继承 `base.GenericBenchmark`，传入自定义 `input_fn` 参数 |

> **自定义扩展点**：
> - `input_fn(shape, dtype, device)` — 生成输入 tensor（最常用，传给 GenericBenchmark 构造函数）
> - `get_input_iter(dtype)` — 如需完全控制输入生成逻辑，继承后覆盖此方法
> - `set_more_shapes()` — 如需自定义额外 shape 列表，覆盖此方法

### dtypes 不要硬编码
- **必须使用 `consts.FLOAT_DTYPES`**，不要写 `[torch.float32]`
- 参考已合入的 PR（如 #3278 atan2）

## 4. 代码质量问题

### 无意义代码
- 不提交无意义的代码片段
- 不用无意义的冗余函数名
- 不做无意义的封装

### Import 规范
- 禁止奇怪的 import（如 `from flag_gems.fused import fp8_einsum, fp8_einsum_ref`）
- import 放在文件顶部，不在函数中间 import

### 测试文件必须用相对导入
- 测试文件中 import utils 必须用 `from . import accuracy_utils as utils`
- **禁止** `import accuracy_utils as utils` 或 `from flag_gems.testing import accuracy_utils as utils`

### 日志规范
- 算子名使用 `logger.debug` 输出，不用 `print`

### 重复函数
- 检查是否有重复的函数定义（同名函数会互相覆盖）
- 检查 kernel 函数名与导出函数名是否合理

## 5. 文件与格式规范

- 遵循项目命名约定，不乱起文件名
- 维持已有列表的排列顺序（字母序），不随意打乱
- 文件末尾必须有换行
- 行长度不超过 120 字符

## 6. 上游结构差异（重要）

当前上游已大幅重构，**不能直接 cherry-pick**：

| 项目 | 旧格式 | 上游当前格式 |
|------|-------|------------|
| 测试文件 | 共享文件追加 | 每算子独立文件 `tests/test_<op>.py` |
| Benchmark | 共享文件追加 | 每算子独立文件 `benchmark/test_<op>.py` |
| Benchmark API | 直接 pytest parametrize | `base.UnaryPointwiseBenchmark()` 等封装类 |
| operators.yaml | `name` 字段 | `id` 字段 |

## 7. pre-commit 常见问题

| Hook | 常见问题 | 修复方法 |
|------|---------|---------|
| `end-of-file-fixer` | 文件末尾缺换行 | 自动修复，重新 stage |
| `flake8` | `F401` 未使用的 import | 删掉多余 import（如 kernel 中不需要的 `import torch`） |
| `flake8` | `F401` 未使用的 `consts` | benchmark 中不 import `consts` 除非真的用了 |
| `isort` | import 顺序不对 | 自动修复，重新 stage |
| `black` | 格式不对 | 自动修复，重新 stage |
| `trailing-whitespace` | 行尾有空白 | 自动修复，重新 stage |

## 8. 多重载算子 mark/op_name 未与 yaml 对齐

当一个 PR 提交多个算子重载（如 `reflection_pad3d` + `reflection_pad3d_out`，或 `eq` + `eq_scalar`），且 `operators.yaml` 中为每个重载注册了独立 `id` 时，benchmark 中**每个重载的测试函数必须使用与其 yaml id 一致的 mark 和 op_name**。

**错误示例**（yaml 拆了独立条目但 benchmark 未对齐）：
```python
# yaml 有 id: reflection_pad3d 和 id: reflection_pad3d_out
# 但 benchmark 中 _out 变体写的是：
@pytest.mark.reflection_pad3d          # ✗ 应为 reflection_pad3d_out
def test_reflection_pad3d_out():
    bench = ReflectionPad3dBenchmark(
        op_name="reflection_pad3d",    # ✗ 应为 reflection_pad3d_out
    )
```

**正确做法**：
```python
@pytest.mark.reflection_pad3d_out      # ✓ 与 yaml id 一致
def test_reflection_pad3d_out():
    bench = ReflectionPad3dBenchmark(
        op_name="reflection_pad3d_out", # ✓ 与 yaml id 一致
    )
```

**规则**：yaml `id` = pytest mark = benchmark `op_name`，三者必须完全一致。适用于所有重载形式（`_out`、`_scalar`、`_tensor`、`_mode` 等）。

## 9. libdevice 兼容性

- 部分算子使用了 `tl.extra.cuda.libdevice`（如 special_erfcx）
- 上游要求跨后端兼容
- 如果 reviewer 提出此问题，需改用 `tl_extra_shim`

## 10. git 操作注意

- **绝不用 `git add -A` 或 `git add .`**（仓库有 687 个 worktree 和大目录）
- 必须逐文件 stage
- 分支命名统一用 `pr/<operator>`
- 每个分支基于 `upstream/master` 创建

## 11. 禁止使用 `.is_cuda` 进行设备判断 (KERNEL_NO_IS_CUDA)

- **禁止** `tensor.is_cuda`、`device.type == "cuda"` 等硬编码 CUDA 设备判断
- 必须使用 `flag_gems.device` 代替，以保证多后端兼容性
- 错误示例：`if x.is_cuda:` 或 `if device.type == "cuda":`
- 正确做法：使用 `flag_gems.device` 提供的接口进行设备判断

## 12. Autotune 配置不得内联硬编码 (KERNEL_AUTOTUNE_CONFIG)

- Autotune configs（`@triton.autotune` 的 `configs` 列表）必须放在 config 文件中统一管理
- **禁止**在 kernel 文件中内联硬编码 autotune 配置
- 正确做法：将配置提取到对应的 config 文件中，kernel 通过引用使用

## 13. Logger 格式规范 (KERNEL_LOGGER_FORMAT)

- `logger.debug` 消息必须使用固定格式：`"GEMS <OP_NAME_UPPER>"`
- `<OP_NAME_UPPER>` 为算子名的大写形式，使用下划线分隔
- 错误示例：`logger.debug("running fractional_max_pool2d")`
- 正确示例：`logger.debug("GEMS FRACTIONAL_MAX_POOL2D")`

## 14. Benchmark 公平性 (BENCH_FAIRNESS)

- Benchmark 中 torch 参考实现和 gems 实现**必须测量相同范围的计算**
- 不允许 torch 端额外包含预处理/后处理而 gems 端不包含（或反之）
- 确保两者的 `input_fn` 和调用方式测量的是完全等价的操作

## 15. 测试和 Benchmark 文件必须存在 (TEST_EXISTS / BENCH_EXISTS)

- 每个在 `operators.yaml` 中注册的算子**必须有对应的测试文件和 benchmark 文件**
- 测试文件：`tests/test_<op>.py`
- Benchmark 文件：`benchmark/test_<op>.py`
- 缺少任一文件的 PR 将被拒绝

## 16. Fused 算子目录归属 (FUSED_DIR)

- Fused 算子（如 `AddRMSNorm`、`FusedRoPE`、`SkipLayerNorm` 等）必须放在 `src/flag_gems/fused/` 目录下
- **禁止**将 fused 算子放在 `src/flag_gems/ops/` 目录中
- 判断标准：算子名称含 "Fused"、"Add"（复合前缀）、多算子融合的，属于 fused 类

## 17. Benchmark shapes 不得重复 (BENCH_SHAPES_NO_DUP)

- 自定义的 benchmark shapes 不得与 `core_shapes.yaml` 中已有的 shapes 重复
- 提交前检查：`grep` 你的 shapes 是否已在 `core_shapes.yaml` 中定义
- 如果 `core_shapes.yaml` 已覆盖所需 shapes，直接使用默认即可，无需自定义
