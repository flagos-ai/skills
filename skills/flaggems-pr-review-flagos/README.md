# FlagGems PR Review Skill

Claude Code Skill，作为 reviewer 角色审查 FlagGems 算子 PR。

支持审查别人的 PR（直接在 GitHub 上发 review comments），也支持提交前自审（输出到 terminal）。

## 目录结构

```
├── SKILL.md
├── references/
│   ├── review-rules-structural.md     # 31 条结构化检查规则
│   ├── naming-conventions.md          # 命名规则速查
│   ├── registration-patterns.md       # 三处注册正确/错误示例
│   ├── benchmark-patterns.md          # benchmark class + dtype 规则
│   └── test-patterns.md              # 测试文件规范
└── scripts/
    ├── fetch_pr_diff.py              # 获取 PR diff 和文件内容
    ├── review_operator.py            # 核心检查脚本，输出 findings JSON
    └── post_review.py                # 将 findings 发布为 GitHub review
```

## 环境要求

| 变量 | 必需 | 说明 |
|------|------|------|
| `GH_TOKEN` | 是 | GitHub Personal Access Token |

### 依赖

- **工具链**：`gh` (GitHub CLI), Python 3.10+
- **Python 包**：`pyyaml`

## 使用方式

### 审查 GitHub PR

```bash
python scripts/fetch_pr_diff.py 3456
python scripts/review_operator.py 3456 --json > findings.json
python scripts/post_review.py 3456 --findings findings.json
```

### 提交前自审

```bash
python scripts/review_operator.py --local --base upstream/master
```
