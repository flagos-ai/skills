> **FlagOS** 是面向异构 AI 芯片的全开源 AI 系统软件栈，让 AI 模型只需开发一次即可轻松移植到各类 AI 硬件。本仓库收集的是 FlagOS 中可复用的 **Skills**，为 AI 编程智能体注入领域知识、流程规范和最佳实践。
>
> [English](README.md)

## 什么是 Skills？

Skills 是一组**文件夹化的能力包**：每个 Skill 通过说明文档、脚本和资源，教会智能体在某一类任务上稳定、可复现地完成工作。每个 Skill 文件夹包含一个 `SKILL.md` 文件（YAML frontmatter + Markdown 正文），作为智能体的详细执行指令。还可包含参考文档、脚本和资源文件。

本仓库遵循 [Agent Skills 开放标准](https://agentskills.io/specification)。

## 快速开始

FlagOS Skills 兼容 **Claude Code**、**Cursor**、**Codex** 以及任何支持 [Agent Skills 标准](https://agentskills.io/specification) 的智能体。

### npx（推荐 — 适用于所有智能体）

使用 [`skills`](https://www.npmjs.com/package/skills) CLI 直接安装，无需克隆仓库：

```bash
# 查看本仓库中可用的 skills
npx skills add flagos-ai/skills --list

# 安装指定 skill 到当前项目
npx skills add flagos-ai/skills --skill model-migrate-flagos

# 全局安装（用户级别）
npx skills add flagos-ai/skills --skill model-migrate-flagos --global

# 一次安装所有 skills
npx skills add flagos-ai/skills --all

# 仅安装到指定智能体
npx skills add flagos-ai/skills --agent claude-code cursor
```

其他常用命令：

```bash
npx skills list              # 查看已安装的 skills
npx skills find              # 交互式搜索 skills
npx skills update            # 更新所有 skills 到最新版本
npx skills remove            # 交互式移除
```

> **提示：** 无需预先安装 — `npx` 会自动下载 [`skills`](https://skills.sh/) CLI。

### Claude Code

1. 注册本仓库为插件市场（在 Claude Code 交互模式中）：

```
/plugin marketplace add flagos-ai/skills
```

或从终端执行：

```bash
claude plugin marketplace add flagos-ai/skills
```

2. 安装 skills：

```
/plugin install flagos-skills@flagos-skills
```

或从终端执行：

```bash
claude plugin install flagos-skills@flagos-skills
```

安装后，在提示词中提及 skill 名称即可 — Claude 会自动加载对应的 `SKILL.md` 指令。

### Cursor

本仓库包含 Cursor 插件清单（`.cursor-plugin/plugin.json` 和 `.cursor-plugin/marketplace.json`）。

通过 Cursor 插件流程从仓库 URL 或本地路径安装。

### Codex

在 Codex 中使用 `$skill-installer`：

```
$skill-installer install model-migrate-flagos from flagos-ai/skills
```

或提供 GitHub 目录 URL：

```
$skill-installer install https://github.com/flagos-ai/skills/tree/main/skills/model-migrate-flagos
```

也可以直接复制 skill 文件夹到 Codex 的标准 `.agents/skills` 目录：

```bash
cp -r skills/model-migrate-flagos $REPO_ROOT/.agents/skills/
```

详见 [Codex Skills 指南](https://developers.openai.com/codex/skills/)。

### Gemini CLI

```bash
gemini extensions install https://github.com/flagos-ai/skills.git --consent
```

本仓库包含 `gemini-extension.json` 和 `agents/AGENTS.md` 用于 Gemini CLI 集成。详见 [Gemini CLI 扩展文档](https://geminicli.com/docs/extensions/)。

### 手动安装 / 其他智能体

对于任何支持 [Agent Skills 标准](https://agentskills.io/specification) 的智能体，将其指向本仓库的 `skills/` 目录即可。每个 skill 都是独立的，以 `SKILL.md` 为入口。`agents/AGENTS.md` 文件可作为不原生支持 skills 的智能体的 fallback。

## Skills 总览

<!-- BEGIN_SKILLS_TABLE -->
| 大分类 | 小分类 | Skill | 说明 |
|--------|--------|-------|------|
| **推理与服务** | 模型迁移 | [`model-migrate-flagos`](skills/model-migrate-flagos/) | 将上游 vLLM 模型迁移到 vllm-plugin-FL（锁定 v0.13.0）。自动化 13 步 copy-then-patch 流程，含 E2E 精度验证。 |
| | 服务部署 | [PR #6 `flagrelease`](https://github.com/flagos-ai/skills/pull/6) | 在多芯片环境中部署和配置 vLLM-FL / SGLang-FL 推理服务。 |
| | 环境预检 | *待开发* | 推理前检查 GPU/加速卡可用性、驱动版本、Python 环境及芯片兼容性。 |
| **训练与 RLHF** | 训练迁移 | *待开发* | 将训练脚本适配到 FlagScale / Megatron-LM-FL，支持多芯片。 |
| | RLHF 流水线 | *待开发* | 搭建和调试 verl-FL 强化学习工作流。 |
| **算子与编译器** | TLE 原语开发 | [PR #2 `tle-developer`](https://github.com/flagos-ai/skills/pull/2) | 开发 TLE（Triton Language Extensions）原语，利用 TLE-Lite / TLE-Struct / TLE-Raw 三级扩展在 FlagTree 各后端构建算子。 |
| | 算子性能优化 | *待开发* | 基于已有 FlagGems / FlagAttention 算子版本，提供性能剖析、瓶颈分析和优化建议，引导持续迭代调优。 |
| | 算子生成 | [PR #10 `kernelgen`](https://github.com/flagos-ai/skills/pull/10) | 通过 KernelGen MCP 通用算子生成，支持多芯片目标（NVIDIA、昇腾、寒武纪、摩尔线程、天数智芯等）。 |
| | 算子生成（FlagGems） | [PR #10 `kernelgen-for-flaggems`](https://github.com/flagos-ai/skills/pull/10) | FlagGems 专用算子生成，含 promotion 规则、`pointwise_dynamic` 封装和 `_FULL_CONFIG` 注册。 |
| | 算子生成（vLLM） | [PR #10 `kernelgen-for-vllm`](https://github.com/flagos-ai/skills/pull/10) | vLLM 专用算子生成，含 SPDX 头、`vllm.logger`、`@triton.autotune` 和自定义算子注册。 |
| | KernelGen 反馈 | [PR #10 `kernelgen-submit-feedback`](https://github.com/flagos-ai/skills/pull/10) | 以结构化 GitHub Issue 提交 KernelGen 的 Bug 报告和改进建议。 |
| | 异常算子诊断 | *待开发* | 为 FlagOS 技术栈诊断异常算子——定位精度错误、性能回退和后端特定故障，跨芯片排查。 |
| | 编译器后端适配 | *待开发* | 为新 AI 芯片架构移植和调试 FlagTree / Triton 编译器后端。 |
| **通信** | 集合通信 | *待开发* | 适配和基准测试 FlagCX 跨芯片通信原语（AllReduce、AllGather、Send/Recv 等），覆盖 11+ 后端（NCCL、IXCCL、CNCL、MCCL 等）。 |
| **评测与基准** | 性能评测 | [PR #6 `perf-test`](https://github.com/flagos-ai/skills/pull/6) | 运行和分析 FlagPerf 基准测试，生成多维度跨芯片对比报告（吞吐、显存、扩展性）。 |
| | E2E 精度验证 | [PR #6 `model-verify`](https://github.com/flagos-ai/skills/pull/6) | 不同推理后端或芯片目标间的 token 级精度对比验证。 |
| **环境与部署** | 软件栈安装 | [PR #6 `install-stack`](https://github.com/flagos-ai/skills/pull/6) | 在目标芯片上一键安装 FlagOS 软件栈——自动检测硬件、解析依赖、配置完整工具链（FlagTree + FlagGems + vLLM-FL + FlagCX）。 |
| | 基础镜像选型 | [PR #5 `gpu-container-setup`](https://github.com/flagos-ai/skills/pull/5) | 为国产 AI 芯片模型部署推荐最优基础 Docker 镜像——匹配芯片型号、驱动版本、CUDA/SDK 兼容性和框架需求。 |
| | 容器构建 | *待开发* | 构建含正确驱动和库依赖的多芯片 Docker 镜像。 |
| | CI 流水线 | *待开发* | 配置和调试 FlagOps 多芯片构建矩阵的 CI/CD 流水线。 |
| **开发者工具** | Skill 开发 | [`skill-creator-flagos`](skills/skill-creator-flagos/) | 创建、改进和验证本仓库中的 skill。支持脚手架、规范检查和测试用例评估。 |
| | 芯片对接 | *待开发* | 引导新芯片厂商完成 FlagOS 全流程适配。 |
<!-- END_SKILLS_TABLE -->

### 在智能体中使用 Skills

安装 skill 后，在提示词中直接提及即可：

- "使用 model-migrate-flagos 将 Qwen3-5 模型从上游 vLLM 迁移过来"
- "/model-migrate-flagos qwen3_5"
- "把 DeepSeek-V4 模型移植到 vllm-plugin-FL"

智能体会自动加载对应的 `SKILL.md` 指令和辅助脚本。

## 仓库结构

```
├── .claude-plugin/          # Claude Code 插件清单
│   └── marketplace.json
├── .cursor-plugin/          # Cursor 插件清单
│   ├── marketplace.json
│   └── plugin.json
├── agents/                  # Codex / Gemini CLI fallback
│   └── AGENTS.md
├── assets/                  # 仓库级静态资源
├── contributing.md          # 贡献指南
├── gemini-extension.json    # Gemini CLI 扩展清单
├── scripts/                 # 仓库级工具脚本
│   └── validate_skills.py   # 批量验证所有 skills
├── skills/                  # Skill 目录
│   ├── model-migrate-flagos/    # 模型迁移工作流
│   └── ...
├── spec/                    # Agent Skills 规范与本地约定
│   ├── README.md
│   └── agent-skills-spec.md
└── template/                # 新 skill 模版
    └── SKILL.md
```

## 创建新 Skill

1. **创建目录并复制模版**
   ```bash
   mkdir skills/<skill-name>
   cp template/SKILL.md skills/<skill-name>/SKILL.md
   ```

2. **编辑 frontmatter** — `name`（小写+短横线，必须与目录名一致）和 `description`（做什么+何时触发）

3. **编写正文** — 概述、前提条件、执行步骤、示例（2-3个）、排障指南

4. **添加辅助文件**（可选） — `references/`、`scripts/`、`assets/`、`LICENSE.txt`

5. **验证**
   ```bash
   python scripts/validate_skills.py
   ```

详见 [contributing.md](contributing.md) 贡献指南。

## 许可证

[Apache License 2.0](LICENSE)
