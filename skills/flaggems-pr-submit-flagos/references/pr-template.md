# PR Description Template

## gh 命令

```bash
export GH_TOKEN="$GH_TOKEN"  # 需提前设置，禁止硬编码 token

gh pr create \
  --repo ${FLAGGEMS_UPSTREAM:-flagos-ai/FlagGems} \
  --head <FORK_OWNER>:pr/<op> \  # FORK_OWNER 由 submit_operator.py 从 git remote 自动推断
  --base master \
  --title "[KernelGen][Nvidia] Add <op> operator with Triton kernel" \
  --body "$(cat <<'EOF'
## Summary
Adds a Triton kernel for `<op>`. <one-line description>.

## Testing
- Parametrized tests over `<parameter dims>` and `<dtypes>`
- Validated against reference on device via `to_reference(inp, True)`
- Tested on: Nvidia, Tianshu, Muxi, Ascend, Hygon

## Performance
Test command: `pytest benchmark/test_<op>.py --level core` (NVIDIA H20)

| Configuration | Torch Latency (ms) | Gems Latency (ms) | Speedup | TFLOPS |
|---|---|---|---|---|
| <shape1> | <torch_ms> | <gems_ms> | <speedup> | <tflops> |
| **Arithmetic Mean** | — | — | **<am_speedup>** | — |

## Multi-backend Testing
| Backend | Accuracy Test | Benchmark | Speedup (mean) | Notes |
|---|---|---|---|---|
| Nvidia (H20) | PASS | PASS (<N> cases, --level core) | <am_speedup> | Primary |
| Tianshu | <acc> | <bench> (<N> cases) | <mean_speedup> | <notes> |
| Muxi | <acc> | <bench> (<N> cases) | <mean_speedup> | <notes> |
| Ascend | <acc> | <bench> (<N> cases) | <mean_speedup> | <notes> |
| Hygon | <acc> | <bench> (<N> cases) | <mean_speedup> | <notes> |

## Files Changed
- `src/flag_gems/ops/<op>.py`: Triton kernel implementation
- `tests/test_<op>.py`: Accuracy test
- `benchmark/test_<op>.py`: Performance benchmark
- `src/flag_gems/ops/__init__.py`: Register import and `__all__`
- `src/flag_gems/__init__.py`: Register to `_FULL_CONFIG`
- `conf/operators.yaml`: Add operator entry (kind: <kind>, stage: alpha 5.1)
EOF
)"
```

## JSON 字段映射

`gen_pr_description.py` 输出的 JSON 字段直接映射到模板：

| 模板位置 | JSON 字段 |
|---------|-----------|
| Performance 表格行 | `nvidia_benchmark.rows[]` (shape, torch_ms, gems_ms, speedup, tflops) |
| Arithmetic Mean | `nvidia_benchmark.arithmetic_mean_speedup` |
| Performance case 数 | `nvidia_benchmark.case_count` |
| Multi-backend PASS/FAIL | `domestic_gpu.<backend>.accuracy_passed` / `benchmark_passed` |
| Multi-backend Speedup | `domestic_gpu.<backend>.bench_mean_speedup` |
| Multi-backend case 数 | `domestic_gpu.<backend>.bench_case_count` |
| Multi-backend Notes | `domestic_gpu.<backend>.test_error` / `bench_error`（截短至一句话） |

## 填写规则

- **全部用英文**
- Performance 数据必须来自 CI 日志或 `gen_pr_description.py` 脚本
- 国产卡 Speedup (mean) 从 summary JSON 的 `benchmark.data` 计算
- 通过时 Notes 填 `—`，benchmark 未运行填 `N/A`，Speedup 填 `—`
- Notes 中失败原因截短至一句话（如 "CompilationError" 而非完整 traceback）
