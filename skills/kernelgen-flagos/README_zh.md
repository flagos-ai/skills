# kernelgen-flagos：统一 GPU 算子生成技能（Skills）

[English](README.md)

## 产品概述

KernelGen `kernelgen-flagos` 技能（Skills）是 FlagOS 推出的统一 AI 编程技能。通过调用 `kernelgen-mcp` KernelGen 算子开发 MCP 工具集服务，该技能兼容国内外多种芯片平台，可生成高精度 GPU 算子。它能够自动检测目标代码仓库的类型，并分发至对应的专属工作流执行。

## 快速入门

本节介绍如何快速启动一个任务，例如为特定平台创建、优化、特化 Kernel，或提交反馈。您可以使用提示词或命令行来执行这些任务。本节仅介绍如何使用提示词。如需了解命令行或手动配置的使用方式，请参见 [KernelGen Skills 用户指南](https://docs.flagos.io/projects/kernelgen/zh-cn/latest/skills_user_guide/skills-user-guide.html)。

除获取 KernelGen Token 外，请在您的 AI 智能体客户端中执行以下所有步骤，例如 OpenClaw、已激活 GitHub Copilot 的 VSCode，或 Claude Code。

### 前提条件

* Claude Code 2.1 及以上版本

* OpenClaw 2026.3.2 及以上版本

* 已激活 GitHub Copilot 的 VSCode

### 操作步骤

1. 从以下地址获取 **KernelGen Token**：[https://kernelgen.flagos.io/mcp](https://kernelgen.flagos.io/mcp)。

2. 在您的 AI 智能体客户端中，发送提示词以连接 KernelGen 算子开发 MCP 工具集，例如：

   ```{code-block} shell
   连接 MCP，其 URL 为 https://kernelgen.flagos.io/sse，token 为 <你的 KernelGen Token>。
   ```

   或者

   ```{code-block} shell
   请配置 kernelgen MCP，URL 为 https://kernelgen.flagos.io/sse，token 为 <你的 KernelGen Token>。
   ```

   **注意**：您可能需要重启 AI 智能体以使设置生效。请参阅相关 AI 智能体的文档进行确认。

3. 验证 KernelGen 算子开发 MCP 工具集连接，发送提示词：

   ```{code-block} shell
   请验证 kernelgen MCP 连接是否成功。
   ```

4. 发送提示词以安装 `kernelgen-flagos` 统一技能（包含所有子技能），例如：

   * **Copilot**：

   ```{code-block} shell
   从 <https://github.com/flagos-ai/skills/tree/main/skills/kernelgen-flagos> 在Copilot中安装 kernelgen-flagos skills。
   ```

   或者

   * **其他 AI 智能体客户端**：
  
   ```{code-block} shell
   从 <https://github.com/flagos-ai/skills/tree/main/skills/kernelgen-flagos> 安装 kernelgen-flagos skills。
   ```

5. 验证技能是否安装成功，发送提示词：

   ```{code-block} shell
   请验证 kernelgen-flagos skills 是否正常工作。
   ```

6. 发送提示词，通过调用对应子技能来运行任务。

### 发送提示词以调用子技能运行任务

KernelGen 会自动检测项目是 FlagGems 仓库、vLLM 仓库还是其他类型，并执行对应的工作流。

* **使用案例 1**：为 vLLM 项目创建 Kernel：

  1. 预先安装 [vLLM](https://docs.vllm.ai/en/latest/getting_started/installation/)。

     **注意**：请通过 git clone 从源码安装。

  2. 发送提示词以生成 Kernel，例如：

   ```{code-block} shell
   使用 kernelgen-flagos 生成 ReLU 算子，使用沐曦（MetaX），集成至 vLLM。
   ```

   或者

   ```{code-block} python
   /kernelgen-flagos 生成 ReLU 算子，使用沐曦（MetaX），集成至 vLLM。
   ```

* **使用案例 2**：发送提示词以在 NVIDIA 上优化 Kernel，例如：
  
  ```{code-block} python
  使用 kernelgen-flagos 优化 scaled_dot_product_attention_math 算子，迭代 5 轮，该算子位于 <算子路径>。
  ```
  
  或者

  ```{code-block} python
  /kernelgen-flagos 优化 scaled_dot_product_attention_math 算子，迭代 5 轮，该算子位于 <算子路径>。
  ```
  

* **使用案例 3**：发送提示词，为 FlagGems 项目将 CUDA 实现的算子特化至华为昇腾（Huawei Ascend）：

  1. 预先安装 [FlagGems](https://docs.flagos.io/projects/FlagGems/en/latest/getting_started/install.html#)。

     **注意**：请通过 git clone 从源码安装。

  2. 发送提示词以迁移算子，例如：

     ```{code-block} python
     使用 kernelgen-flagos 将 CUDA 实现的算子 <算子路径 A> 迁移至昇腾芯片，算子文件存储在 FlagGems 仓库中，目录为 <算子路径 B>，确保精度验证通过。
     ```

     或者

     ```{code-block} python
     /kernelgen-flagos 将 CUDA 实现的算子 <算子路径 A> 迁移至昇腾芯片，算子文件存储在 FlagGems 仓库中，目录为 <算子路径 B>，确保精度验证通过。
     ```

### 输出文件

输出文件如下：

* \<operator name>\_triton.py — Triton Kernel 实现

* \<operator name>\_torch.py — PyTorch 基线实现

* test\_\<operator name>.py — 完整测试用例

* benchmark\_\<operator name>.py — 性能基准测试

**注意**：如果生成了其他文件，请发送提示词验证技能和 KernelGen 算子开发 MCP 工具集是否仍正常工作。

后续步骤：

* **拉取请求**：如果用户在 GitHub 仓库中进行开发，技能将询问是否需要提交拉取请求（PR），并可自动提交 PR 并提供链接。

* **聊天集成**：如果用户通过聊天应用（如微信或飞书）配置技能，技能将询问是否需要将修改后的文件发送至聊天应用。

* **反馈**：如果在使用过程中遇到问题，可使用本技能通过电子邮件或创建 GitHub Issue 提交反馈。发送提示词以提交反馈，例如："使用 kernelgen-flagos 技能提交反馈" 或 "使用 kernelgen-flagos 技能报告问题"。

## 为什么选择技能（Skills）？

KernelGen 技能不单纯依赖 LLM 从零生成代码，而是遵循一套结构化的四步工作流：

1. **收集 Kernel 信息**：从用户的自然语言描述中提取算子参数（输入/输出类型、形状、计算逻辑）。

2. **搜索相似代码片段**：从现有代码库中检索相似实现以供参考（用户可选择是否采用）。

3. **生成 Kernel 代码 + CUDA 实现**：同步生成 Triton Kernel 及其对应的 CUDA 实现，后者作为 PyTorch 基线。

4. **测试**：以 CUDA 实现作为 GroundTruth 对 Kernel 进行测试，输出正确性测试结果与加速比。

生成完成后，从生成到贡献的全流程完全自动化：若在 FlagGems 仓库中开发，专属子技能将自动处理格式转换与目录对齐，直接将四项核心交付物（Kernel、基线、正确性测试、加速比测试）注入 `src/flag_gems/experimental_ops/` 目录。开发者只需一键即可生成合规的 PR，无需手动调用转换脚本。

### 核心特性

编写高性能 GPU 算子既复杂又容易出错。不同项目（FlagGems、vLLM、自定义 Triton 仓库）各有独特的算子实现规范、测试模式和注册系统。过去，用户需要分别安装多个技能。

本统一技能将十个子技能打包为一，只需安装一次：

### 文件说明

### `SKILL.md`

统一入口文件。包含路由逻辑，可自动检测仓库类型（FlagGems、vLLM 或通用类型），并读取相应的子技能文件执行。

### `kernelgen-mcp-setup.md`

MCP 服务配置检查与自动设置。在任何子技能执行前，SKILL.md 会先调度至此，验证 kernelgen-mcp MCP 服务是否已配置。若未配置，则引导用户获取 Token 并自动写入配置。

### `kernelgen-generate.md`

适用于任意 Python/Triton 仓库的 GPU Kernel 算子生成完整工作流（10 步）。包括动态仓库结构发现、规范检测和自适应代码放置。

### `kernelgen-generate-for-flaggems.md`

专为 FlagGems 仓库定制的 9 步工作流。处理 `pointwise_dynamic` 包装器、提升方法、`_FULL_CONFIG` 注册、分类测试文件及 FlagGems 专属规范。

### `kernelgen-generate-for-vllm.md`

专为 vLLM 仓库定制的 9 步工作流。处理 SPDX 许可证头、`vllm.logger.init_logger`、`@triton.autotune`、自定义算子注册及 vLLM 目录规范。

### `kernelgen-submit-feedback.md`

反馈提交工作流。收集问题报告并自动检测环境信息，通过 GitHub Issues（`gh` CLI）或邮件备选方式提交。

### `kernelgen-optimize.md`

通过 MCP 迭代循环进行通用 Triton Kernel 优化。分析现有 Kernel，识别性能瓶颈，经多轮迭代应用优化直至达到目标加速比。

### `kernelgen-optimize-for-flaggems.md`

FlagGems 专属 Kernel 优化，支持 3 种模式：原地优化内置算子、优化外部算子并集成至 experimental\_ops，或优化现有实验性算子。包含正确性测试和性能基准测试。

### `kernelgen-optimize-for-vllm.md`

vLLM 专属 Kernel 优化，集成 CustomOp 注册、正确性测试和性能基准测试。优化 Triton 算子并自动集成至 vLLM 项目。

### `kernelgen-specialize.md`

通过 MCP `specialize_kernel` 工具对 Triton 算子进行平台专化。将 GPU Triton 算子迁移至目标平台（如华为昇腾（Huawei Ascend） NPU），处理架构差异、Grid 配置、UB 溢出和内存对齐问题。

### `kernelgen-specialize-for-flaggems.md`

结合 MCP 平台专化与 FlagGems 框架集成。支持四种集成模式：vendor-ops（默认）、vendor-fused、override-builtin 和 experimental。包含自动化测试和性能基准测试。

***

## 许可证

本项目基于 Apache 2.0 许可证授权。详情请参见 [LICENSE.txt](https://github.com/flagos-ai/skills/blob/main/skills/kernelgen-flagos/LICENSE.txt)。
