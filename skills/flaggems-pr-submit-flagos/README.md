# FlagGems PR Submit Skill

Claude Code Skill，用于自动化 FlagGems 算子的 PR 提交流程。

覆盖从 worktree 代码提取、19 项自动化检查、到一站式提交（测试/benchmark/commit/push/PR 创建/回填）的完整流程。

## 目录结构

```
├── SKILL.md                        # Skill 定义（触发词、规则、工作流）
├── references/
│   ├── common-issues.md            # Code review 常见问题与修复指引
│   ├── naming.md                   # 下划线前缀算子命名规范
│   ├── pr-checklist.md             # PR 提交前检查清单
│   ├── pr-template.md              # PR 描述模板与 gh 命令
│   └── workflow.md                 # 完整工作流程文档
└── scripts/
    ├── check_operator.py           # 19 项自动化检查（AST 分析、命名、注册一致性等）
    ├── check_overload_consistency.py # 多重载 yaml/mark/op_name 一致性检查
    ├── extract_from_worktree.py    # 从 worktree 提取 6 个 PR 文件
    ├── gen_pr_description.py       # 生成 PR 描述数据（benchmark + 国产卡）
    ├── operator_registry.py        # 规范名查询 + PR 链接回填
    ├── pr_gate_check.sh            # git push 前门禁（PreToolUse hook）
    └── submit_operator.py          # 9 步一站式提交
```

## 环境要求

### 环境变量

| 变量 | 必需 | 说明 | 默认值 |
|------|------|------|-------|
| `FLAGGEMS_REPO` | 是 | FlagGems 仓库本地路径 | — |
| `GH_TOKEN` | 是 | GitHub Personal Access Token | — |
| `FLAGGEMS_FORK` | 否 | Fork 仓库（自动从 git remote 推断） | `XDYuanzhuLee/FlagGems` |
| `FLAGGEMS_UPSTREAM` | 否 | 上游仓库 | `flagos-ai/FlagGems` |
| `FLAGGEMS_NORM_XLSX` | 否 | 规范名 Excel 路径 | `/workspace/规范名.xlsx` |
| `FLAGGEMS_PR_XLSX` | 否 | 待提交算子 Excel 路径 | `/workspace/第一批pr算子.xlsx` |
| `FLAGGEMS_DOMESTIC_DIR` | 否 | 国产 GPU 测试数据目录 | `/workspace/国产GPU算子测试情况` |

### Quick Start

```bash
export FLAGGEMS_REPO=/path/to/your/FlagGems
export GH_TOKEN=ghp_xxx
# 可选：export FLAGGEMS_FORK=yourname/FlagGems（不设则自动从 git remote 推断）
```

### 依赖

- **Python 包**：`pyyaml`, `openpyxl`, `pandas`
- **工具链**：`gh` (GitHub CLI), `pre-commit`, `pytest`
- **数据文件**：`$FLAGGEMS_NORM_XLSX`, `$FLAGGEMS_PR_XLSX`（用于规范名查询和回填，可选）

## 工作流程

```
Phase 0: 规范名查询
    └─ operator_registry.py lookup <op>

Phase 1: 代码提取（从 worktree 生成 6 个 PR 文件）
    └─ python extract_from_worktree.py <op> --repo-dir $FLAGGEMS_REPO

Phase 2: 自动化检查（19 项）
    └─ python check_operator.py <op> --repo-dir $FLAGGEMS_REPO --strict

Phase 3: 一站式提交（测试 → benchmark → commit → push → PR → 回填）
    └─ python submit_operator.py <op> --repo-dir $FLAGGEMS_REPO --gpu 0
```

## 脚本说明

### check_operator.py

19 项自动化检查，包括 kernel 文件合规、注册一致性、命名规范、dtype 硬编码检测、anti-hack AST 扫描等。

```bash
python scripts/check_operator.py special_erfcx --repo-dir /path/to/FlagGems
python scripts/check_operator.py special_erfcx --strict    # warning 也视为失败
python scripts/check_operator.py special_erfcx --list-files # 仅列出涉及文件
```

### extract_from_worktree.py

从 `.worktrees/gen-<op>` 提取算子代码，自动生成 kernel、test、benchmark、__init__.py 注册、operators.yaml 条目共 6 个文件。

```bash
python scripts/extract_from_worktree.py _cholesky_solve_helper --dry-run  # 预览
python scripts/extract_from_worktree.py _cholesky_solve_helper            # 执行
```

### submit_operator.py

串行执行 9 个步骤：check → pre-commit → test → benchmark → gen PR data → commit → push → create PR → backfill。

```bash
python scripts/submit_operator.py special_erfcx --gpu 0
python scripts/submit_operator.py special_erfcx --dry-run        # 只验证不提交
python scripts/submit_operator.py special_erfcx --skip-benchmark # 跳过 benchmark
```

### gen_pr_description.py

采集 Nvidia benchmark 数据和国产卡（天数/沐曦/华为/海光）测试结果，输出 JSON。

```bash
python scripts/gen_pr_description.py special_erfcx --repo /path/to/FlagGems
python scripts/gen_pr_description.py special_erfcx --nvidia-stdin < bench.log
python scripts/gen_pr_description.py special_erfcx --skip-run  # 仅查国产卡数据
```

### check_overload_consistency.py

检查多重载算子的 yaml id、benchmark mark、op_name 三方一致性。通过 yaml id 精确匹配和 for 字段关联的并集来发现相关重载。

```bash
python scripts/check_overload_consistency.py reflection_pad3d --repo-dir /path/to/FlagGems
python scripts/check_overload_consistency.py eq
python scripts/check_overload_consistency.py max
```

### operator_registry.py

查询算子规范名、加速比，回填 PR 链接到 Excel。

```bash
python scripts/operator_registry.py lookup _cholesky_solve_helper
python scripts/operator_registry.py backfill _cholesky_solve_helper https://github.com/.../pull/123
python scripts/operator_registry.py pending --limit 20
```

## 参考文档

| 文件 | 用途 |
|------|------|
| `common-issues.md` | 从真实 review 反馈提炼的常见问题，附具体修复指引 |
| `naming.md` | 下划线前缀算子（如 `_cholesky_solve_helper`）的命名规则速查表 |
| `pr-checklist.md` | 提交前检查清单，覆盖 kernel/test/benchmark/注册/git 规范 |
| `pr-template.md` | PR 描述 Markdown 模板和 `gh pr create` 命令模板 |
| `workflow.md` | 完整工作流程，含每个阶段的详细步骤和代码模板 |
