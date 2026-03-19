# kernelgen：GPU 算子生成（通用版）

## 概述

`kernelgen` 是一个 AI 编程技能，通过 `kernelgen-mcp` MCP 服务生成 GPU 算子，并将其集成到任意 Python/Triton 仓库中。

### 解决的问题

编写高性能 GPU 算子复杂且容易出错。开发者需要处理 Triton 指针运算、内存访问模式、自动调优配置以及项目特定的代码规范。每个仓库都有各自的文件布局、编码风格和测试模式，生成符合项目规范的代码非常困难。

本技能自动化了整个工作流程：**环境检查 → 仓库结构发现 → MCP 代码生成 → 代码适配 → 准确性测试 → 性能基准测试**，涵盖 10 个步骤，支持自动检测项目规范并进行代码转换。

### 使用方式

```bash
# 生成算子
/kernelgen relu

# 指定函数类型
/kernelgen rms_norm --func-type normalization

# 生成任意算子
/kernelgen silu_and_mul
```

| 参数 | 必填 | 默认值 | 说明 |
|---|---|---|---|
| `operator_name` | 是 | — | snake_case 格式的算子名称（如 `relu`、`rms_norm`、`silu_and_mul`） |
| `--func-type` | 否 | 自动推断 | 可选: `unary_pointwise`、`binary_pointwise`、`reduction`、`normalization`、`attention`、`activation`、`quantization`、`moe`、`blas`、`sampling`、`other` |

---

## 生成流水线（10 步）

```
┌──────────────────────────────────────────────────────────┐
│  步骤 0   预检：环境与 MCP 检查                           │
│  步骤 1   理解算子需求                                    │
│  步骤 2   发现仓库结构                                    │
│  步骤 3   检查算子是否已存在                               │
│  步骤 4   研究上下文（flagos_wiki）                        │
│  步骤 5   调用 kernelgen-mcp                              │
│  步骤 6   适配代码并放置到仓库                             │
│  步骤 6.5 测试前验证                                      │
│  步骤 7   运行准确性测试                                   │
│  步骤 8   运行性能基准测试                                 │
│  步骤 9   生成报告                                        │
└──────────────────────────────────────────────────────────┘
```

### 核心特性

- **动态仓库发现** — 自动检测项目结构、算子目录、测试目录和代码规范
- **规范匹配** — 读取现有代码以匹配导入风格、命名规范、许可证头和自动调优模式
- **智能代码转换** — 将 MCP 生成的代码适配为仓库特定的模式（包装器风格 vs 原始 Triton）
- **已有算子检测** — 生成前搜索命名变体以避免重复
- **全面测试** — 按数据类型精度运行准确性测试和性能基准测试
- **错误恢复** — 结构化重试协议，支持 MCP 重新生成和优化回退

---

## 目录结构

```
skills/kernelgen/
├── SKILL.md        # 技能定义（入口文件）
├── LICENSE.txt     # Apache 2.0 许可证
├── README.md       # 英文文档
└── README_zh.md    # 本文档（中文版）
```

---

## 相关技能

- [`kernelgen-for-flaggems`](../kernelgen-for-flaggems/) — FlagGems 仓库专用版
- [`kernelgen-for-vllm`](../kernelgen-for-vllm/) — vLLM 仓库专用版
- [`kernelgen-submit-feedback`](../kernelgen-submit-feedback/) — 提交 bug 报告和反馈

---

## 许可证

本项目基于 Apache 2.0 许可证。详见 [LICENSE.txt](LICENSE.txt)。
