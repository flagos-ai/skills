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

# FlagGems 算子 PR 提交检查清单

每个算子 PR 提交前逐项检查：

## 文件完整性

- [ ] `src/flag_gems/ops/<op>.py` — kernel 实现文件
- [ ] `src/flag_gems/ops/__init__.py` — 添加 import 行 + `__all__` 条目（按字母序）
- [ ] `src/flag_gems/__init__.py` — `_FULL_CONFIG` 添加条目（按字母序）
- [ ] `tests/test_<op>.py` — 独立测试文件
- [ ] `benchmark/test_<op>.py` — 独立 benchmark 文件
- [ ] `conf/operators.yaml` — 添加算子条目（按 `id` 字母序）

## Kernel 代码规范

- [ ] 使用 `logging` 模块（`logger = logging.getLogger(__name__)`），不使用 `print`
- [ ] 函数名 snake_case，类名 PascalCase，常量 UPPER_CASE
- [ ] 无重复函数定义
- [ ] 无多余 `import torch`（纯 triton kernel 不需要 torch）
- [ ] 避免魔法数字，使用命名常量
- [ ] 无无意义的封装或冗余函数

## 注册规范

### ops/__init__.py
- [ ] `from flag_gems.ops.<op> import <func>` — 按模块名字母序插入
- [ ] `__all__` 列表中添加导出名 — 按字母序，下划线前缀的排在前面

### __init__.py (_FULL_CONFIG)
- [ ] `("<aten_op_name>", <func>)` — 按 aten op 名字母序插入
- [ ] inplace 变体用下划线后缀：`("<op>_", <func>_)`
- [ ] 带 overload 的用点号：`("<op>.Tensor", <func>)`

### operators.yaml
- [ ] `id` 字段与 pytest mark 严格一致
- [ ] `description` 从 PyTorch 文档提取
- [ ] `for` 列出对应 aten op 名
- [ ] `labels` 必须包含 `aten` + `KernelGen`，可加 `pointwise`/`reduction` 等分类
- [ ] `kind` 类别：`Math`/`Reduction`/`NeuralNetwork`/`LinearAlg`/`Tensor`
- [ ] `stages` 新增算子统一用 `alpha: '5.1'`

## 测试文件规范

- [ ] 独立文件 `tests/test_<op>.py`
- [ ] 使用相对导入：`from . import accuracy_utils as utils`
- [ ] pytest mark 与算子名严格对齐：`@pytest.mark.<op_name>`
- [ ] 多重载对齐：yaml 中每个独立 `id` 对应的测试函数，mark 必须与该 id 完全一致
- [ ] CPU-FP64 作为 Golden Reference（`utils.to_reference`）
- [ ] 浮点运算用 `gems_assert_close`，位精确操作用 `gems_assert_equal`
- [ ] 容差使用统一标准（fp32: 1.3e-6, fp16: 1e-3, bf16: 0.016）
- [ ] 使用 `utils.POINTWISE_SHAPES`/`utils.FLOAT_DTYPES` 等标准参数
- [ ] **禁止 print()**

## Benchmark 文件规范

- [ ] 独立文件 `benchmark/test_<op>.py`
- [ ] 使用 `from . import base, consts`（相对导入）
- [ ] pytest mark 与算子名一致
- [ ] 多重载对齐：yaml 中每个独立 `id` 对应的 benchmark 函数，mark 和 op_name 必须与该 id 完全一致（适用于 `_out`、`_scalar`、`_tensor` 等所有重载）
- [ ] 使用 benchmark 封装类：
  - 一元 pointwise → `base.UnaryPointwiseBenchmark`
  - 二元 pointwise → `base.BinaryPointwiseBenchmark`
  - Reduction → `base.UnaryReductionBenchmark`
  - 其他 → 参考上游同类型算子
- [ ] **不得自己编写新的 Benchmark 框架**

## 提交规范

- [ ] pre-commit 全部通过（black、isort、flake8、end-of-file-fixer、trailing-whitespace）
- [ ] `git add` 只添加指定文件（**禁止 `git add -A` 或 `git add .`**）
- [ ] commit message 格式：`[KernelGen][Nvidia] Add <op> operator with Triton kernel`
- [ ] 分支命名：`pr/<operator>`
- [ ] PR body 包含：PR Category / Type of Change / Description / Changes / Performance

## 提交前最终确认

- [ ] 算子不存在于上游（`git show upstream/master:src/flag_gems/ops/<op>.py` 应报错）
- [ ] 算子名与现有算子不冲突
- [ ] 运行 `scripts/check_operator.py <op>` 全部通过
