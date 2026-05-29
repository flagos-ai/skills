---
name: flaggems-pr-review-flagos
description: >
  This skill should be used when reviewing FlagGems operator PRs, performing code review,
  self-reviewing before submission, or when the user mentions "review PR", "审PR", "代码review",
  "code review", "review #123", "self-review", "自审", "检查PR", "审查算子", "review operator".
  It fetches PR diffs, applies FlagGems domain-specific review rules (structural checks, naming,
  registration, tests, benchmarks), and posts inline review comments directly on GitHub.
---

# FlagGems PR Review Skill

作为 reviewer 角色审查 FlagGems 算子 PR。支持审查别人的 PR，也支持提交前自审。

## Rules

1. **先拉取完整 diff** — 必须获取 PR 所有变更文件和完整内容后再开始审查
2. **逐文件检查** — 按 kernel → 注册 → yaml → test → benchmark 顺序检查
3. **标注严重级别** — error（必须修）、warning（应该修）、suggestion（建议）
4. **不重复标记** — 同一位置的同一问题只标记一次
5. **给出修复建议** — 每条 finding 附带具体的修复方式
6. **原子提交** — 所有 inline comments 合并为一个 review 提交（一次通知）
7. **只审格式规范，不审算子正确性** — Skill 聚焦于格式、命名、注册、文件结构等规范性问题。算子实现的正确性（Triton API 用法、数学逻辑、backward 实现等）不在 scope 内，由 CI 测试覆盖

## Environment

| 环境变量 | 必需 | 说明 | 默认值 |
|---------|------|------|-------|
| `GH_TOKEN` | 是 | GitHub Personal Access Token | — |
| `FLAGGEMS_REPO` | 否 | FlagGems 仓库本地路径（未设置时使用当前目录） | — |

**每次新会话需确保环境已配置（禁止硬编码路径）。**

## Workflow

### Phase -1: 环境验证（每次会话必须执行）

**在开始审查前，必须按顺序验证以下配置。任何一项失败都要询问用户，不要自己做决策（如自动克隆仓库）。**

#### 1. 验证 GH_TOKEN

```bash
gh auth status
```

- **成功**：继续下一步
- **失败**：提示用户 `请先运行 gh auth login 或设置 GH_TOKEN 环境变量`，停止执行

#### 2. 验证仓库位置

```bash
# 优先使用 FLAGGEMS_REPO
if [ -n "$FLAGGEMS_REPO" ]; then
  cd "$FLAGGEMS_REPO" && git rev-parse --show-toplevel
else
  # 否则检查当前目录
  git rev-parse --show-toplevel 2>/dev/null || echo "not in git repo"
fi
```

- **成功**：记录仓库路径，继续下一步
- **失败**：询问用户提供仓库路径，提供以下选项：
  - 输入 FlagGems 仓库的绝对路径
  - 如果 `/tmp/FlagGems` 存在且是有效仓库，询问是否使用
  - **不要自动克隆仓库**

#### 3. 验证 remote 配置

```bash
cd <repo_path>
git remote -v | grep -iE "FlagGems|flaggems"
```

- **成功**：确认这是 FlagGems 仓库，继续
- **失败**：警告用户 `当前仓库不是 FlagGems，请确认路径是否正确`，询问是否继续

#### 4. 记录工作目录

```bash
export REVIEW_WORK_DIR=<validated_repo_path>
cd "$REVIEW_WORK_DIR"
```

**只有所有验证通过后，才进入 Phase 0。**

---

### Phase 0: 确定目标 + 选择模式

```bash
# 审查别人的 PR（传入 PR number）
gh pr view <PR_NUMBER> --json title,headRefName,files

# 自审模式（从当前分支 diff）
git log --oneline upstream/master..HEAD
```

**必须询问用户选择发布模式：**

| 模式 | 说明 |
|------|------|
| **直接发送** | Review 完成后自动发布到 GitHub（适合已确认规则无误时） |
| **预览确认**（默认） | 先展示 findings 给用户，用户确认后再发布到 GitHub |
| **仅本地** | 只输出到 terminal，不发布到 GitHub（适合自审） |

如果用户未明确指定，默认使用**预览确认**模式。

### Phase 1: 获取数据

```bash
# PR 模式
python scripts/fetch_pr_diff.py <PR_NUMBER>

# 自审模式
python scripts/fetch_pr_diff.py --local --base upstream/master
```

输出 JSON 包含：operator name、changed files（path + patch + full content）、commits

### Phase 2: 格式检查

```bash
# PR 模式
python scripts/review_operator.py <PR_NUMBER> --json > findings.json

# 自审模式
python scripts/review_operator.py --local --json > findings.json
```

运行 31 条结构化规则（详见 `references/review-rules-structural.md`），输出 findings JSON。

**这是 skill 的核心功能，只检查格式规范，不检查算子正确性。**

模型不需要做任何额外的人工判断或代码分析，直接使用脚本输出的 findings 即可。

### Phase 3: 发布 Review（必须执行）

此步骤为 workflow 的必要终点，不可跳过。

**预览确认模式（默认）：**
1. 先向用户展示完整的 findings 报告（表格形式）
2. 询问用户："确认发送到 GitHub？" 或 "需要修改？"
3. 用户确认后执行发布：
```bash
python scripts/post_review.py <PR_NUMBER> --findings findings.json
```

**直接发送模式：**
```bash
python scripts/post_review.py <PR_NUMBER> --findings findings.json
```

**仅本地模式：**
```bash
python scripts/post_review.py <PR_NUMBER> --findings findings.json --dry-run
```
展示结果后结束，不发布到 GitHub。

**发布行为：**
- 脚本将 findings 映射为 GitHub review：
  - 有 file + line 的 → inline comment（定位到 diff 具体位置）
  - 无具体 line 的 → review body
  - 有 error → 自动 REQUEST_CHANGES；无 error → COMMENT
- 所有 comments 合并为一个 review 原子提交（PR 作者只收一个通知）

## 检查规则速览

### 结构化检查（31 条）

所有检查由 `review_operator.py` 自动执行，模型只需使用脚本输出的结果。

| 类别 | Rule ID | 检查内容 |
|------|---------|---------|
| Kernel | KERNEL_HEADER | KernelGen 首行 |
| Kernel | KERNEL_NO_PRINT | 无 print() |
| Kernel | KERNEL_NO_DUP_FUNC | 无重复函数 |
| Kernel | KERNEL_NO_FALLBACK | 无 torch.op() 递归 |
| Kernel | KERNEL_NO_PRIVATE_API | 无 torch._xxx |
| Kernel | KERNEL_NO_IS_CUDA | 禁止 is_cuda，用 flag_gems.device |
| Kernel | KERNEL_AUTOTUNE_CONFIG | autotune config 不硬编码 |
| Kernel | KERNEL_LOGGER_FORMAT | logger 须用 "GEMS OP_NAME" 格式 |
| 注册 | REG_OPS_INIT | ops/__init__.py 注册 + 字母序 |
| 注册 | REG_FULL_CONFIG | _FULL_CONFIG 正确注册 |
| 注册 | REG_YAML_COMPLETE | yaml 字段完整 |
| 注册 | REG_YAML_UNIQUE | yaml id 不重复 |
| 注册 | REG_CONSISTENCY | yaml ↔ _FULL_CONFIG 一致 |
| 测试 | TEST_MARK | mark 与 yaml id 一致 |
| 测试 | TEST_IMPORT | 相对导入 |
| 测试 | TEST_ASSERT | 使用 gems_assert_close/equal |
| 测试 | TEST_NO_RTOL | 无 rtol 参数 |
| 测试 | TEST_NO_PRINT | 无 print() |
| 测试 | TEST_TO_REFERENCE | ref 计算须用 to_reference() |
| Bench | BENCH_MARK | mark 与 yaml id 一致 |
| Bench | BENCH_OP_NAME | op_name 与 yaml id 一致 |
| Bench | BENCH_CLASS | 使用标准封装类 |
| Bench | BENCH_DTYPE | dtype 用常量或加注释 |
| Bench | BENCH_FAIRNESS | torch/gems 测量范围一致 |
| 跨文件 | DTYPE_HARDCODE | dtype 硬编码无注释 |
| 跨文件 | SINGLE_OP | 单算子 PR |
| 跨文件 | GIT_COAUTHOR | 无 Co-Authored-By |
| 跨文件 | HYPERPARAMS | 魔法数字须有注释 |
| 跨文件 | CODE_QUALITY | 行长/尾空白/EOF |
| 完整性 | TEST_EXISTS | 注册算子必须有 test 文件 |
| 完整性 | BENCH_EXISTS | 注册算子必须有 benchmark 文件 |

## References

- `references/review-rules-structural.md` — 31 条结构化规则详情 + 正反示例
- `references/naming-conventions.md` — 完整命名规则速查表
- `references/registration-patterns.md` — 三处注册的正确/错误 pattern
- `references/benchmark-patterns.md` — benchmark class 选择 + dtype 规则
- `references/test-patterns.md` — 测试文件规范
