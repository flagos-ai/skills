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
| **部署与发布** | 基础镜像选型 | [`gpu-container-setup-flagos`](skills/gpu-container-setup-flagos/) | 自动检测 GPU 厂商，查找合适的 PyTorch 容器镜像，启动正确的挂载配置，并验证 GPU 功能。支持 NVIDIA、昇腾、Metax、天数智芯和 AMD/ROCm。当用户说"setup container"、"start pytorch container"或调用 /gpu-container-setup 时使用。 |
|  | 模型迁移 | [`model-migrate-flagos`](skills/model-migrate-flagos/) | 将上游最新 vLLM 仓库的模型迁移到 vllm-plugin-FL 项目（锁定 vLLM v0.13.0）。当用户想要为 vllm-plugin-FL 添加新模型支持、从上游 vLLM 移植模型代码或 backport 新发布的模型时使用。触发词如 "migrate X model"、"add X model support"、"port X from upstream vLLM"、"make X work with the FL plugin" 或直接 "/model-migrate-flagos model_name"。model_name 使用 snake_case（如 qwen3_5、kimi_k25、deepseek_v4）。不适用于 vLLM 0.13.0 核心已支持的模型，或不需要 backport 的纯多模态组件。 |
|  | 发布流水线 | [`flagrelease-entrance-flagos`](skills/flagrelease-entrance-flagos/) | 完整的 FlagRelease 流水线编排器。针对多芯片 GPU 后端运行完整的 LLM 部署、验证和基准测试流水线。按序执行：install-stack → env-verify → model-verify → perf-test，在步骤间传递状态并生成最终结构化报告。假设 gpu-container-setup（步骤1）已完成 — 必须有一个运行中的 PyTorch + GPU 访问的容器。 |
|  | 软件栈安装 | [`install-stack-flagos`](skills/install-stack-flagos/) | 在 GPU 容器内安装 5 包多芯片软件栈（vLLM、FlagTree、FlagGems、FlagCX、vllm-plugin-FL）。处理网络镜像检测、依赖排序、wheel 选择和逐包验证。在 gpu-container-setup 产生运行中的 PyTorch + GPU 访问容器后使用。 |
| **基准测试与评测** | 精度与性能测试 | [`perf-test-flagos`](skills/perf-test-flagos/) | 对已服务的模型运行精度基准测试（FlagEval，可用时）和性能基准测试（vllm bench serve）。覆盖 5 种工作负载配置：短/长 prefill × 短/长 decode + 高并发。收集吞吐量、延迟、TTFT、TPOT 指标。 |
|  | 部署 A/B 验证 | [`model-verify-flagos`](skills/model-verify-flagos/) | 使用用户指定的目标模型验证服务栈。运行两次：首次禁用 FlagGems/FlagCX（隔离模型特定错误），然后启用完整多芯片栈。对比两次运行以定位故障来自哪一层。 |
|  | FlagPerf 用例创建 | *规划中* | 为新模型/芯片基准测试用例生成 FlagPerf 兼容的目录结构、配置文件、运行脚本和预期指标基线。 |
|  | 部署后自动评测 | *规划中* | 模型部署后自动触发评测，跟踪评测状态，失败时报错，完成后推送结果通知。 |
| **算子与内核开发** | 复杂算子开发 | *规划中* | 为多步骤融合算子（fused attention、fused MoE 等）生成骨架代码，处理共享内存 tiling 策略和多后端分支。 |
|  | 实验性算子推广 | *规划中* | 扫描 FlagGems 约130个实验性算子，检查测试覆盖率，对齐签名，完成 `_FULL_CONFIG` 注册，生成迁移 PR 将其推广为主算子。 |
|  | FlagGems 内核生成 | [`kernelgen-flagos`](skills/kernelgen-flagos/kernelgen-for-flaggems.md) | FlagGems 专用内核生成，含 `@pointwise_dynamic` 封装重写、`_FULL_CONFIG` 注册和算子签名对齐。 |
|  | vLLM 内核生成 | [`kernelgen-flagos`](skills/kernelgen-flagos/kernelgen-for-vllm.md) | vLLM 专用内核生成，含 SPDX 头、`@triton.autotune`、自定义算子注册和 dispatch 集成。 |
|  | 内核生成 | [`kernelgen-flagos`](skills/kernelgen-flagos/) | 统一的 GPU 内核算子生成 skill。自动检测目标仓库类型（FlagGems、vLLM 或通用 Python/Triton）并派发到相应的专用子 skill。还包含用于 Bug 报告的反馈提交子 skill。当用户想要生成 GPU 内核算子、创建 Triton 内核，或说 "generate an operator"、"create a kernel for X"、"/kernelgen-flagos" 时使用。此单一 skill 替代了安装 kernelgen-general、kernelgen-for-flaggems、kernelgen-for-vllm 和 kernelgen-submit-feedback 的需求。 |
|  | 算子诊断 | *规划中* | 诊断 FlagOS 技术栈中的异常算子——识别精度错误、性能回退和跨芯片的后端特定故障。 |
| **多芯片后端对接** | Dispatch 算子扩展 | *规划中* | 从 `base.py` 查询可 dispatch 的算子，生成 impl 模板文件，向 `register_ops.py` 添加 `OpImpl` 注册，并创建单元测试骨架。 |
|  | FlagCX 通信后端 | *规划中* | 从头文件解析 20+ 设备和 15+ CCL 函数指针签名，生成所有 stub 实现，外加 CMake 构建配置。 |
|  | FlagGems 芯片后端 | *规划中* | 按照 FlagGems 后端贡献指南生成完整的 `_vendor/` 脚手架：`__init__.py`（VendorInfoBase 配置）+ `heuristics_config_utils.py` + `tune_configs.yaml` + `ops/` 目录。 |
|  | 异构训练配置 | *规划中* | 从硬件拓扑描述生成有效的 FlagScale 异构训练配置，自动计算 `hetero_process_meshes` / `hetero_pipeline_layer_split`，并验证约束（TP×DP×PP = 设备数）。 |
|  | vLLM 厂商后端 | *规划中* | 从模板脚手架新建 vllm-plugin-FL 厂商后端：生成厂商目录、`Backend` 子类、`is_available` 检测、`register_ops` 框架和测试骨架。 |
| **开发者工具** | 反馈提交 | [`kernelgen-flagos`](skills/kernelgen-flagos/kernelgen-submit-feedback.md) | 自动收集环境信息，构建结构化 GitHub Issue，提交到 flagos-ai/skills，GitHub CLI 不可用时回退到邮件方式。 |
|  | 通用 | [`tle-developer-flagos`](skills/tle-developer-flagos/) | 用于编写高性能 TLE 内核和交付 TLE 功能变更的自包含编排 skill，含可复现验证。当用户想要编写/优化 TLE 内核、实现 TLE API/verifier/lowering 功能或调试 TLE 正确性/性能问题时使用。触发词如 "write a TLE kernel"、"optimize TLE operator"、"debug TLE local_ptr"。 |
|  | 本地开发环境 | *规划中* | 为 FlagOS 模块（FlagGems / FlagTree / FlagCX 等）搭建本地开发和调试环境——配置依赖、环境变量和调试工具链。 |
|  | Skill 开发 | [`skill-creator-flagos`](skills/skill-creator-flagos/) | 为 FlagOS skills 仓库创建新 skill、修改现有 skill 和验证 skill 质量。当用户想要从头创建 skill、改进或编辑现有 skill、脚手架新 skill 目录、验证 skill 结构或运行测试用例时使用。触发词如 "create a skill"、"make a new skill for X"、"scaffold a skill"、"improve this skill"、"validate my skill" 或直接 "/skill-creator-flagos"。也适用于用户提到将工作流转化为可复用 skill 或想要打包重复流程为 skill 的场景。 |
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
