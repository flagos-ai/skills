---
name: flaggems-review-reply-flagos
description: >
  This skill should be used when handling FlagGems PR review comments, replying to reviewer feedback,
  fixing code based on review, or when the user mentions "回复review", "处理review意见", "review回复",
  "fix review comments", "address review", "回复评论", "处理反馈". It reads PR review comments,
  analyzes issues using FlagGems domain knowledge, applies code fixes, pushes updates, and replies
  to reviewers on GitHub.
---

# FlagGems PR Review Reply Skill

处理流程：读取 PR review comments → 分类分析 → 修改代码 → 验证 → push → 回复 reviewer。

## Rules

### 流程规则
1. **先读后改** — 必须先完整读取所有 review comments，理解全貌后再开始修改
2. **逐条处理** — 每条 review comment 单独分析、修改、回复，不跳过任何一条
3. **修改后验证** — 代码修改后必须运行 `pre-commit` 和相关测试确认无误
4. **回复必须具体** — 回复中说明做了什么修改、在哪里修改、为什么这样改
5. **不认同要说明** — 如果 reviewer 意见有误或不适用，需在回复中礼貌说明理由

### 代码修改规则
6. **最小化修改** — 只改 reviewer 指出的问题，不做额外重构
7. **保持一致性** — 修改后的代码必须与项目规范一致（参考 `references/`）
8. **不破坏已有功能** — 修改不能引入新的测试失败
9. **遵循上游规范** — 所有修改必须符合 FlagGems 上游的代码规范

### Git 规则
10. **不 force push** — 使用追加 commit 的方式修复，不 rebase/amend 已 push 的 commit
11. **commit message 说明 review fix** — 格式: `fix: address review - <简要说明>`
12. **不用 git add -A** — 只 stage 本次修改的文件，避免误提交无关变更

## Environment

| Item | Value |
|------|-------|
| Token | 环境变量 `$GH_TOKEN` |

工作目录为当前 FlagGems 仓库根目录。upstream repo 和 fork 信息从 git remote 自动获取。

## Workflow

### Phase 0: 获取 Review Comments

```bash
# 查看当前分支对应的 PR
gh pr list --json number,title,headRefName

# 获取 PR 的所有 review comments
gh api repos/{owner}/{repo}/pulls/<PR_NUMBER>/comments --paginate \
  --jq '.[] | {id: .id, path: .path, line: .line, body: .body, user: .user.login, created_at: .created_at}'

# 获取 PR review（含 review-level comments）
gh api repos/{owner}/{repo}/pulls/<PR_NUMBER>/reviews --paginate \
  --jq '.[] | {id: .id, state: .state, body: .body, user: .user.login}'

# 获取 PR 基本信息
gh pr view <PR_NUMBER> --json title,body,headRefName,files
```

> `{owner}/{repo}` 从 git remote 获取，如 `gh repo view --json nameWithOwner -q .nameWithOwner`。

### Phase 1: 分类分析

将每条 review comment 分为以下类别：

| 类别 | 说明 | 处理方式 |
|------|------|----------|
| `code-fix` | 代码需要修改 | 修改代码 + 回复 |
| `style` | 代码风格/格式问题 | 修改代码 + 回复 |
| `naming` | 命名不规范 | 修改代码 + 回复（参考 `references/naming-fixes.md`） |
| `question` | reviewer 有疑问 | 仅回复解释 |
| `suggestion` | 建议性意见 | 评估后决定采纳或解释 |
| `nit` | 小问题 | 修改代码 + 回复 |
| `invalid` | 意见不适用 | 回复说明理由 |

### Phase 2: 修改代码

```bash
# 切到 PR 分支
git checkout pr/<op>

# 根据 review 意见修改相关文件
# 修改后运行项目检查脚本（如有）
pre-commit run --files <modified_files>
```

### Phase 3: 验证

```bash
# 可选：运行测试
CUDA_VISIBLE_DEVICES=<N> python -m pytest tests/test_<op>.py -x -v
```

### Phase 4: Commit & Push

```bash
# 逐文件 stage
git add <file1> <file2> ...

# commit（不加 Co-Authored-By）
# 注意：不使用 Claude Code 默认的 commit 流程（会自动追加 Co-Authored-By），必须手动执行 git commit。
git commit -m "fix: address review - <简要说明>"

# push 到 fork
git push origin pr/<op>
```

### Phase 5: 回复 Review Comments

```bash
# 回复单条 review comment
gh api repos/{owner}/{repo}/pulls/<PR_NUMBER>/comments/<COMMENT_ID>/replies \
  -f body="<reply_message>"

# 发表 review-level comment（GitHub API 不支持回复已有 review，此命令创建新的 review comment）
gh api repos/{owner}/{repo}/pulls/<PR_NUMBER>/reviews \
  -f body="<reply_message>" -f event="COMMENT"
```

### 回复模板

**代码已修改：**
```
Fixed in <commit_sha>. <具体说明做了什么修改>.
```

**解释说明：**
```
This is intentional because <原因>. <补充解释>.
```

**采纳建议：**
```
Good catch! Fixed in <commit_sha>. Changed <old> to <new>.
```

**不采纳但说明：**
```
Thanks for the suggestion. In this case, <原因> so I've kept the current approach. <补充>.
```

## References

- `references/common-review-patterns.md` — FlagGems PR 常见 review 意见模式与标准回复
- `references/naming-fixes.md` — 命名相关 review 意见的修复指南
- `references/code-fix-patterns.md` — 代码修改常见模式（dtype、benchmark class、import 等）

## 常见 Review 意见速查

| Review 意见 | 修复方式 | 参考 |
|------------|---------|------|
| "mark 与 yaml id 不一致" | 统一 mark、op_name、yaml id | `naming-fixes.md` 场景 2 |
| "dtype 不要硬编码" | 改用 `consts.FLOAT_DTYPES` / `utils.FLOAT_DTYPES` | `code-fix-patterns.md` §1 |
| "用封装类" | 改用 `base.UnaryPointwiseBenchmark` 等 | `code-fix-patterns.md` §2 |
| "删掉 print" | 删除或改 `logger.debug` | `code-fix-patterns.md` §5 |
| "import 顺序" | 运行 `isort` / `pre-commit` | `code-fix-patterns.md` §3 |
| "libdevice 兼容性" | 改用 `tl_extra_shim` | `code-fix-patterns.md` §4 |
| "description 补充" | 更新 yaml description | `code-fix-patterns.md` §7 |
| "不要 Co-Authored-By" | 后续 commit 不再包含 | `common-review-patterns.md` §5 |
| "don't use is_cuda" | 改用 `flag_gems.device` 比较或删除设备检查 | `code-fix-patterns.md` §10 |
| "autotune config 放到配置文件" | 提取到 config 文件，用 `register_triton_autotune_config` | `code-fix-patterns.md` §11 |
| "logger 格式不对" | 使用 `logger.debug("GEMS <OP_NAME_UPPER>")` | `code-fix-patterns.md` §12 |
| "benchmark 不公平" | torch wrapper 用 `torch.ops.aten.<op>_backward` 对齐范围 | `code-fix-patterns.md` §13 |
| "缺少测试/benchmark" | 按标准模板创建文件 | `code-fix-patterns.md` §14 |
| "应该放在 fused/ 目录" | 移动文件到 `fused/` 并更新 import | `code-fix-patterns.md` §15 |
| "shapes 与 core_shapes 重复" | 删除重复 shapes，只保留自定义 | `code-fix-patterns.md` §16 |
