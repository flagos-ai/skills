---
name: flaggems-pr-submit-flagos
description: >
  This skill should be used when submitting FlagGems operator PRs, reviewing operator code before
  submission, preparing operator code for PR, or when the user mentions "提PR", "提交算子",
  "submit operator", "PR提交", "代码审核", "pre-commit". It automates code review, validates
  completeness and compliance, runs pre-commit and worktree tests, and directly submits PR
  to upstream with full description including speedup data.
---

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


# FlagGems 算子 PR 提交 Skill

提交流程：规范名查询 → 建分支 → 提取 worktree 代码(6文件) → 脚本验证 → pre-commit → push → 创建 PR → 回填链接。

## Rules（违反会导致 PR 被拒）

> 25+ 项检查已由 `check_operator.py` 自动执行（详见下方检查表），以下仅列出**模型需主动注意**的规则。

### 流程规则
1. **先跑脚本再 commit** — `check_operator.py --strict` 必须 0 errors
2. **使用规范命名** — 提交前用 `operator_registry.py lookup` 查询
3. **回填 PR 链接** — PR 创建后必须 `operator_registry.py backfill`
4. **PR 描述由脚本生成** — `gen_pr_description.py` 输出 JSON，映射到模板（英文）

### 代码规则
5. **代码必须与 worktree 原版一致** — 不允许重写测试逻辑，仅允许 import 调整和格式化
6. **不删 worktree 现有注释**
7. **下划线命名** — 前导 `_` 的算子，mark/yaml id/文件名去掉下划线，其余保留（详见 `references/naming.md`）
8. **dtype 默认用常量** — test 用 `utils.FLOAT_DTYPES`，benchmark 用 `consts.FLOAT_DTYPES`；CUDA 不支持时可硬编码但必须加注释
9. **非 pointwise benchmark** — 简单场景用 `GenericBenchmark(input_fn=...)`；需要自定义 shape 时继承并覆盖 `set_shapes`
10. **hardcode size 需加注释** — kernel BLOCK、test shapes、benchmark shapes 都需注释说明原因
11. **不支持的 dtype 在 wrapper 加 assert**
12. **overloaded ops yaml 拆成独立条目** — 参考 `eq` / `eq_scalar` 模式
13. **禁止 .is_cuda** — 设备判断使用 `flag_gems.device`，不用 `.is_cuda` 或 `device.type == "cuda"`（多后端）
14. **Autotune 配置放 config 文件** — 不在 kernel 中内联硬编码 autotune configs
15. **Logger 格式** — `logger.debug("GEMS <OP_NAME_UPPER>")`，不用其他格式
16. **Fused 算子放 fused/ 目录** — `src/flag_gems/fused/`，不放 `src/flag_gems/ops/`

### 提交规则
17. **不修改上游已有测试** — 只新增，不改已有函数
18. **先提交通用版，再提交特化版**
19. **概率算子用统计验证** — mean ≈ p，不能只查 0/1
20. **nan 比较用 gems_assert_close(equal_nan=True)**

## Environment

| 环境变量 | 必需 | 说明 | 默认值 |
|---------|------|------|-------|
| `FLAGGEMS_REPO` | 是 | FlagGems 仓库本地路径 | — |
| `GH_TOKEN` | 是 | GitHub Personal Access Token | — |
| `FLAGGEMS_FORK` | 否 | Fork 仓库（自动从 git remote 推断） | `XDYuanzhuLee/FlagGems` |
| `FLAGGEMS_UPSTREAM` | 否 | 上游仓库 | `flagos-ai/FlagGems` |
| `FLAGGEMS_NORM_XLSX` | 否 | 规范名 Excel 路径 | `/workspace/规范名.xlsx` |
| `FLAGGEMS_PR_XLSX` | 否 | 待提交算子 Excel 路径 | `/workspace/第一批pr算子.xlsx` |
| `FLAGGEMS_DOMESTIC_DIR` | 否 | 国产 GPU 测试数据目录 | `/workspace/国产GPU算子测试情况` |

每次新会话需确保 `FLAGGEMS_REPO` 和 `GH_TOKEN` 已设置（禁止硬编码）。

## Workflow（模型只需调用 3 个命令）

### Phase 0: Name Lookup
```bash
cd $FLAGGEMS_REPO
python <SCRIPTS_DIR>/operator_registry.py lookup <op>
```

### Phase 1: Preparation
```bash
cd $FLAGGEMS_REPO
git checkout -b pr/<op> upstream/master
```
确认算子不存在于上游。Never cherry-pick or rebase。

### Phase 2: Extract Code（一步完成，禁止手动编写）
```bash
python <SCRIPTS_DIR>/extract_from_worktree.py <op> --repo-dir $FLAGGEMS_REPO
```
脚本自动从 worktree 提取 6 个文件：kernel、test、benchmark、ops/__init__.py、__init__.py、operators.yaml。
**所有注册按字母序插入，所有代码从 worktree 原样提取。禁止手动编写 test/benchmark 代码。**

脚本完成后检查 operators.yaml 的 description 是否需要补充（脚本可能使用默认占位描述）。

### Phase 3-7: Validate, Test, Submit（一步完成，禁止手动跳过）
```bash
CUDA_VISIBLE_DEVICES=<N> python <SCRIPTS_DIR>/submit_operator.py <op> --repo-dir $FLAGGEMS_REPO
```
脚本串行执行 9 步（含 1.5 多重载检查）：check_operator → 多重载一致性检查 → pre-commit → **本地测试** → **本地 benchmark** → PR描述生成 → commit → push → 创建 PR → 回填链接。
**任何一步失败立即中断退出。不允许手动执行单独步骤来绕过。**

可选参数：
- `--dry-run` — 只验证不提交（调试用）
- `--skip-test` — 跳过本地测试（仅当环境不支持时）
- `--skip-benchmark` — 跳过 benchmark，仅查国产卡数据

## References

- `references/workflow.md` — Phase 2 六文件详细模板、代码 review 要点
- `references/pr-template.md` — PR 描述模板、JSON 字段映射
- `references/naming.md` — 下划线算子命名规则对照表
- `references/pr-checklist.md` — 提交前逐项检查清单
- `references/common-issues.md` — 历史 review 问题汇总
- `scripts/check_operator.py` — 自动化验证脚本（25+ 检查项）
- `scripts/check_overload_consistency.py` — 多重载算子 yaml/mark/op_name 一致性检查
- `scripts/extract_from_worktree.py` — 从 worktree 提取 6 个 PR 文件
- `scripts/gen_pr_description.py` — PR 描述数据生成
- `scripts/submit_operator.py` — 一站式提交（9 步串行）
- `scripts/operator_registry.py` — 规范名查询 + PR 链接回填
- `scripts/pr_gate_check.sh` — git push 前门禁（PreToolUse hook）

## check_operator.py 自动检查项一览

| 检查 | 级别 |
|------|------|
| Kernel 文件存在 + KernelGen 首行 | error |
| 无 print()、无重复函数 | error |
| ops/__init__.py 注册 + 字母序 | error |
| _FULL_CONFIG 注册（映射到 wrapper） | error |
| Fallback 递归检测（kernel 中禁止 torch.<op>()） | error |
| operators.yaml 完整性 + 唯一性 | error |
| 测试 pytest mark + import 方式 + gems_assert | error |
| gems_assert_close 无 rtol | error |
| dtype 硬编码检查（需常量或加注释） | error |
| 私有 API torch._xxx | error |
| Benchmark pytest mark + op_name + dtype | error |
| Git commit 无 Co-Authored-By | error |
| Inplace mark 守卫 | error |
| yaml/config 一致性交叉验证 | error |
| Anti-hack Layer 1 (AST) + Layer 2 (dual execution) | error |
| 单算子 PR（git diff 验证） | error |
| @use_tl_extra 未使用桩函数 | error |
| 硬编码超参无注释 | error |
| 上游冲突检查 | error |
| 测试函数命名规范 | warning |
| logging 模块、torch import | warning |
| 代码质量（行长、EOF、行尾空白） | warning |
| 别名封装、NaN 处理、Worktree 一致性、Wrapper dtype assert | warning |
| Benchmark case 数量 | info |
| `.is_cuda` / `device.type == "cuda"` 硬编码设备判断 | error |
| Autotune configs 内联硬编码（应在 config 文件） | error |
| Logger 格式不符合 `"GEMS <OP_NAME_UPPER>"` | error |
| Benchmark torch/gems 测量范围不一致 | error |
| 测试文件或 Benchmark 文件缺失 | error |
| Fused 算子放在 ops/ 而非 fused/ 目录 | error |
| 自定义 shapes 与 core_shapes.yaml 重复 | warning |

## 模型必须人工检查的规则（无法自动化）

以下规则依赖领域判断，模型每次提交前必须逐项确认：

- [ ] **dtype 预检测** — 该算子 PyTorch 参考实现是否支持 Half/BFloat16？linalg/special/cdist 等通常不支持
- [ ] **benchmark 类选择** — 非 pointwise 算子是否使用了正确的封装类？需要自定义 shape 时是否覆盖了 `set_shapes`？
- [ ] **无效分支清理** — pointwise_dynamic 已处理类型分发时，是否有多余的 isinstance 判断？
- [ ] **不修改上游已有测试** — 新增算子时是否无意中改动了同文件中其他算子的测试？
- [ ] **概率算子** — 如果是 dropout/bernoulli/rand 等，测试是否用统计方法验证（mean/variance）而非精确比较？
- [ ] **先通用后特化** — 特化版本是否依赖尚未 merge 的通用版？
- [ ] **设备判断禁止 is_cuda** — kernel/wrapper 中是否使用了 `.is_cuda` 或 `device.type == "cuda"`？必须用 `flag_gems.device`
- [ ] **Autotune 配置位置** — autotune configs 是否放在 config 文件中？禁止在 kernel 内联硬编码
- [ ] **Logger 格式** — `logger.debug` 是否使用 `"GEMS <OP_NAME_UPPER>"` 格式？
- [ ] **Benchmark 公平性** — torch 和 gems 是否测量了相同范围的计算？
- [ ] **Fused 算子目录** — fused 算子（AddRMSNorm、FusedRoPE 等）是否放在 `src/flag_gems/fused/` 而非 `ops/`？
- [ ] **Benchmark shapes 去重** — 自定义 shapes 是否与 `core_shapes.yaml` 已有条目重复？

## 强制执行策略（模型必须遵守）

- **check_operator warning = error** — `submit_operator.py` 使用 `--strict` 模式，所有 warning 升级为 error。模型不得手动绕过脚本。
- **AI 生成代码不可信** — anti-hack Layer 2 (dual-execution) 会验证 kernel 是否真正使用 Triton 计算。如检测到 hack，拦截 PR 不提交该算子。
- **异常自动记录** — `submit_operator.py` 的 `fatal()` 自动追加事件到 `pr状态记录.md`，无需手动记录。
