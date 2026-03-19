# kernelgen-for-vllm：vLLM GPU 算子生成

## 概述

`kernelgen-for-vllm` 是一个 AI 编程技能，通过 `kernelgen-mcp` MCP 服务专门为 vLLM 项目生成 GPU 算子。

### 解决的问题

vLLM 项目有特定的算子实现规范：SPDX 许可证头、`vllm.logger.init_logger` 日志记录、`@triton.autotune` 配置、特定的文件放置模式（`vllm/kernels/`、`csrc/`、`tests/kernels/`）以及自定义算子注册（`vllm/_custom_ops.py`）。手动编写符合所有这些规范的代码既复杂又容易出错。

本技能自动化了 vLLM 特定的工作流程：**环境检查 → MCP 代码生成 → vLLM 规范适配 → 算子注册 → 准确性测试 → 性能基准测试**，涵盖 9 个步骤，支持 vLLM 感知的代码转换。

### 使用方式

```bash
# 生成 vLLM 算子
/kernelgen-for-vllm rms_norm

# 指定函数类型
/kernelgen-for-vllm silu_and_mul --func-type activation
```

| 参数 | 必填 | 默认值 | 说明 |
|---|---|---|---|
| `operator_name` | 是 | — | snake_case 格式的算子名称（如 `rms_norm`、`silu_and_mul`、`rotary_embedding`） |
| `--func-type` | 否 | 自动推断 | 可选: `activation`、`norm`、`attention`、`quantization`、`moe`、`sampling`、`other` |

---

## 生成流水线（9 步）

```
┌──────────────────────────────────────────────────────────┐
│  步骤 0   预检：环境与 MCP 检查                           │
│  步骤 1   理解算子需求                                    │
│  步骤 2   检查算子是否已存在                               │
│  步骤 3   研究上下文（flagos_wiki）                        │
│  步骤 4   调用 kernelgen-mcp                              │
│  步骤 5   适配代码（vLLM 规范）                            │
│  步骤 6   运行准确性测试                                   │
│  步骤 7   运行性能基准测试                                 │
│  步骤 8   生成报告                                        │
└──────────────────────────────────────────────────────────┘
```

### 核心特性

- **vLLM 原生代码** — 生成带有 SPDX 头、`init_logger` 和正确 `@triton.autotune` 的 Triton 算子
- **智能文件放置** — 算子放到 `vllm/kernels/`，测试放到 `tests/kernels/`，基准测试放到 `benchmarks/kernels/`
- **自定义算子注册** — 在适用时集成 `vllm/_custom_ops.py`
- **三种生成模式** — 新建算子、替换已有、或并列自定义变体（v2）
- **全面测试** — 按数据类型精度设置容差（fp32: 1e-5, fp16: 1e-2, bf16: 2e-2）
- **错误恢复** — 结构化重试协议，支持 MCP 重新生成和优化

---

## 目录结构

```
skills/kernelgen-for-vllm/
├── SKILL.md        # 技能定义（入口文件）
├── LICENSE.txt     # Apache 2.0 许可证
├── README.md       # 英文文档
└── README_zh.md    # 本文档（中文版）
```

---

## 相关技能

- [`kernelgen`](../kernelgen/) — 通用版本，适用于任意 Python/Triton 仓库
- [`kernelgen-for-flaggems`](../kernelgen-for-flaggems/) — FlagGems 仓库专用版
- [`kernelgen-submit-feedback`](../kernelgen-submit-feedback/) — 提交 bug 报告和反馈

---

## 许可证

本项目基于 Apache 2.0 许可证。详见 [LICENSE.txt](LICENSE.txt)。
