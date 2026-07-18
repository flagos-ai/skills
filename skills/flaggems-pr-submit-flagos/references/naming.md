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

# 下划线前缀算子命名规则

示例算子：`_cholesky_solve_helper`

## 去掉前导下划线的位置

| 位置 | 示例 |
|------|------|
| pytest mark | `@pytest.mark.cholesky_solve_helper` |
| operators.yaml `id` | `id: cholesky_solve_helper` |
| benchmark `op_name` | `op_name="cholesky_solve_helper"` |
| 测试文件名 | `tests/test_cholesky_solve_helper.py` |
| benchmark 文件名 | `benchmark/test_cholesky_solve_helper.py` |
| 测试函数名 | `def test_cholesky_solve_helper(...)` |

## 保留前导下划线的位置

| 位置 | 示例 |
|------|------|
| kernel 文件名 | `src/flag_gems/ops/_cholesky_solve_helper.py` |
| 函数名 | `_cholesky_solve_helper` |
| Import / `__all__` | `from flag_gems.ops._cholesky_solve_helper import _cholesky_solve_helper` |
| `_FULL_CONFIG` aten name | `("_cholesky_solve_helper", _cholesky_solve_helper)` |
| operators.yaml `for` | `- _cholesky_solve_helper` |
| 代码中实际调用 | `torch._cholesky_solve_helper(...)` |

## 尾部下划线（inplace）

尾部 `_` 始终保留：`bernoulli_` → mark `bernoulli_`、id `bernoulli_`、op_name `bernoulli_`

## 多重载算子命名

当 yaml 为算子的每个重载注册了独立 `id` 时，三者必须完全对齐：

| yaml `id` | pytest mark | benchmark `op_name` |
|------|------|------|
| `reflection_pad3d` | `@pytest.mark.reflection_pad3d` | `op_name="reflection_pad3d"` |
| `reflection_pad3d_out` | `@pytest.mark.reflection_pad3d_out` | `op_name="reflection_pad3d_out"` |
| `eq` | `@pytest.mark.eq` | `op_name="eq"` |
| `eq_scalar` | `@pytest.mark.eq_scalar` | `op_name="eq_scalar"` |

适用于所有重载后缀：`_out`、`_scalar`、`_tensor`、`_mode`、`_backward` 等。是否拆分为独立条目以 yaml 实际配置为准——如果 yaml 中该重载没有独立 `id`，则 benchmark 可复用主算子的 mark。
