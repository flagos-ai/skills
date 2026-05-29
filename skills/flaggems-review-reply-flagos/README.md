# FlagGems Review Reply Skill

Claude Code Skill，用于自动化处理 FlagGems PR 的 review 意见回复。

覆盖从读取 review comments、分类分析、代码修改、验证到回复 reviewer 的完整流程。

## 目录结构

```
├── SKILL.md                              # Skill 定义（触发词、规则、工作流）
├── references/
│   ├── common-review-patterns.md         # 常见 review 意见模式与标准回复
│   ├── naming-fixes.md                   # 命名相关 review 意见修复指南
│   └── code-fix-patterns.md              # 代码修改常见模式
└── scripts/
    ├── fetch_reviews.py                  # 获取 PR review comments
    ├── reply_comment.py                  # 回复 review comment
    └── batch_reply.py                    # 批量处理 review 意见
```

## 环境要求

| 变量 | 必需 | 说明 |
|------|------|------|
| `GH_TOKEN` | 是 | GitHub Personal Access Token |

### 依赖

- **工具链**：`gh` (GitHub CLI), `pre-commit`

## 工作流程

```
Phase 0: 获取 Review Comments
    └─ gh api / fetch_reviews.py

Phase 1: 分类分析
    └─ code-fix / style / naming / question / suggestion / nit / invalid

Phase 2: 修改代码 + 验证
    └─ 修改 → pre-commit → 可选: pytest

Phase 3: Commit & Push
    └─ git commit -m "fix: address review - ..." → git push

Phase 4: 回复 Reviewer
    └─ gh api / reply_comment.py
```
