# kernelgen-submit-feedback：技能反馈提交

## 概述

`kernelgen-submit-feedback` 是一个 AI 编程技能，帮助用户为 FlagOS skills 仓库中的技能提交 bug 报告、缺陷报告和改进建议。

### 解决的问题

当用户遇到 bug 或有改进建议时，需要收集环境信息、正确格式化报告并通过正确的渠道提交。这个过程很繁琐，经常导致 bug 报告不完整，缺少关键的环境详情。

本技能自动化了整个反馈工作流程：**收集信息 → 自动采集环境 → 验证 GitHub CLI → 构建 issue → 提交**，当 GitHub CLI 不可用时自动回退到邮件方式。

### 使用方式

```bash
# 交互式提交反馈
/kernelgen-submit-feedback

# 为特定技能提交反馈
/kernelgen-submit-feedback kernelgen
```

| 参数 | 必填 | 默认值 | 说明 |
|---|---|---|---|
| `skill-name` | 否 | 从上下文推断 | 反馈所针对的技能名称 |

---

## 提交工作流

```
┌──────────────────────────────────────────────────────────┐
│  步骤 1   确定提交方式（GitHub 或邮件）                    │
│  步骤 2   从用户收集信息                                   │
│  步骤 3   自动采集环境信息                                 │
│  步骤 4   验证 GitHub CLI 可用性                           │
│  步骤 5   确定 issue 类型和标签                            │
│  步骤 6   构建 issue                                      │
│  步骤 7   与用户确认                                       │
│  步骤 8   提交 issue                                      │
│  步骤 9   确认创建                                        │
└──────────────────────────────────────────────────────────┘
```

### 核心特性

- **自动环境检测** — 自动收集 OS、Python、PyTorch、CUDA 和 Triton 版本信息
- **GitHub CLI 集成** — 通过 `gh issue create` 直接提交 issue
- **邮件回退** — 当 GitHub CLI 不可用时生成邮件草稿和 mailto 链接
- **智能上下文复用** — 从对话历史中提取信息，避免重复询问
- **标签管理** — 提交前验证标签是否存在，避免错误

---

## 目录结构

```
skills/kernelgen-submit-feedback/
├── SKILL.md        # 技能定义（入口文件）
├── LICENSE.txt     # Apache 2.0 许可证
├── README.md       # 英文文档
└── README_zh.md    # 本文档（中文版）
```

---

## 相关技能

- [`kernelgen`](../kernelgen/) — 通用 GPU 算子生成
- [`kernelgen-for-flaggems`](../kernelgen-for-flaggems/) — FlagGems 专用算子生成
- [`kernelgen-for-vllm`](../kernelgen-for-vllm/) — vLLM 专用算子生成

---

## 许可证

本项目基于 Apache 2.0 许可证。详见 [LICENSE.txt](LICENSE.txt)。
