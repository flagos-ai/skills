# 命名相关 Review 意见修复指南

## 命名规则速查

| 位置 | `_foo` 算子 | `foo` 算子 | `foo_out` 变体 |
|------|-------------|-----------|---------------|
| 文件名 | `_foo.py` | `foo.py` | `foo.py`（同文件） |
| kernel 函数名 | `_foo` | `foo` | `foo_out` |
| yaml `id` | `foo`（去下划线） | `foo` | `foo_out` |
| pytest mark | `@pytest.mark.foo` | `@pytest.mark.foo` | `@pytest.mark.foo_out` |
| benchmark `op_name` | `"foo"` | `"foo"` | `"foo_out"` |
| `__init__.py` import | `from flag_gems.ops._foo import _foo` | `from flag_gems.ops.foo import foo` | — |
| `__all__` 条目 | `"_foo"` | `"foo"` | — |
| `_FULL_CONFIG` aten | `"_foo"` | `"foo"` | `"foo.out"` |

## 常见修复场景

### 场景 1: yaml id 有前导下划线

```yaml
# Before (错误)
- id: _cholesky_solve_helper

# After (正确)
- id: cholesky_solve_helper
```

对应修改 pytest mark 和 benchmark op_name。

### 场景 2: mark 与 id 不一致（overload 变体）

多重载算子（如 `eq` + `eq_scalar`）每个变体的 mark 和 op_name 必须与其各自的 yaml id 一致。

```python
# 错误：_out 变体用了主算子的 mark
@pytest.mark.foo       # 应为 foo_out
def test_foo_out():
    bench = FooBenchmark(op_name="foo")  # 应为 "foo_out"

# 正确
@pytest.mark.foo_out
def test_foo_out():
    bench = FooBenchmark(op_name="foo_out")
```

### 场景 3: 文件名 vs mark 命名冲突

测试/benchmark 文件名去掉前导下划线（与 mark/id 一致），但 kernel 文件名保留：
- kernel 文件：`src/flag_gems/ops/_foo.py` ← 保留前导下划线
- 测试文件：`tests/test_foo.py` ← 去掉前导下划线（不是 `test__foo.py`）
- benchmark 文件：`benchmark/test_foo.py` ← 去掉前导下划线
- mark：`@pytest.mark.foo` ← 去掉前导下划线

### 场景 4: inplace 变体命名

Inplace 算子尾部下划线保留：
- `add_` 的 yaml id = `add_`
- mark = `@pytest.mark.add_`
- op_name = `"add_"`

## 批量修复命令

当需要全局替换命名时：
```bash
# 查看所有需要修改的位置
grep -rn "pytest.mark.<old_name>" tests/ benchmark/
grep -rn "op_name=\"<old_name>\"" benchmark/
grep -rn "id: <old_name>" conf/operators.yaml
```
