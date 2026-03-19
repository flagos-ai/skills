# kernelgen-for-flaggems：FlagGems GPU 算子生成

## 概述

`kernelgen-for-flaggems` 是一个 AI 编程技能，通过 `kernelgen-mcp` MCP 服务专门为 FlagGems 项目生成 GPU 算子。

### 解决的问题

FlagGems 项目有特定的算子实现规范：`pointwise_dynamic` 包装器、类型提升方法、`flag_gems.utils` 导入、分类的测试文件以及独特的算子注册系统（`_FULL_CONFIG`）。手动编写符合所有这些规范的代码既繁琐又容易出错。

本技能自动化了 FlagGems 特定的工作流程：**环境检查 → MCP 代码生成 → FlagGems 规范适配 → 算子注册 → 分类测试 → 性能基准测试**，涵盖 9 个步骤，支持 FlagGems 感知的代码转换。

### 使用方式

```bash
# 生成 FlagGems 算子
/kernelgen-for-flaggems relu

# 指定函数类型
/kernelgen-for-flaggems layer_norm --func-type normalization
```

| 参数 | 必填 | 默认值 | 说明 |
|---|---|---|---|
| `operator_name` | 是 | — | 匹配 `torch.ops.aten` 的算子名称（如 `relu`、`silu`、`layer_norm`） |
| `--func-type` | 否 | 自动推断 | 可选: `unary_pointwise`、`binary_pointwise`、`reduction`、`normalization`、`blas`、`other` |

---

## 生成流水线（9 步）

```
┌──────────────────────────────────────────────────────────┐
│  步骤 0   预检：环境与 MCP 检查                           │
│  步骤 1   理解算子需求                                    │
│  步骤 2   检查算子是否已存在                               │
│  步骤 3   研究上下文（flagos_wiki）                        │
│  步骤 4   调用 kernelgen-mcp                              │
│  步骤 5   适配代码（FlagGems 规范）                        │
│  步骤 5.5 测试前验证                                      │
│  步骤 6   运行准确性测试                                   │
│  步骤 7   运行性能基准测试                                 │
│  步骤 8   生成报告                                        │
└──────────────────────────────────────────────────────────┘
```

### 核心特性

- **FlagGems 原生代码** — 生成 `pointwise_dynamic` 风格的算子，包含正确的类型提升方法
- **自动注册** — 按字母顺序添加到 `__init__.py` 和 `_FULL_CONFIG`
- **分类测试放置** — 遵循现有模式将测试追加到 `tests/test_<category>_ops.py`
- **三种生成模式** — 新建算子、替换已有、或并列自定义变体（v2）
- **错误恢复** — 结构化重试协议，支持 MCP 重新生成和优化

---

## 目录结构

```
skills/kernelgen-for-flaggems/
├── SKILL.md        # 技能定义（入口文件）
├── LICENSE.txt     # Apache 2.0 许可证
├── README.md       # 英文文档
└── README_zh.md    # 本文档（中文版）
```

---

## 相关技能

- [`kernelgen`](../kernelgen/) — 通用版本，适用于任意 Python/Triton 仓库
- [`kernelgen-for-vllm`](../kernelgen-for-vllm/) — vLLM 仓库专用版
- [`kernelgen-submit-feedback`](../kernelgen-submit-feedback/) — 提交 bug 报告和反馈

---

## 许可证

本项目基于 Apache 2.0 许可证。详见 [LICENSE.txt](LICENSE.txt)。
