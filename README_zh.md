> **FlagOS** 是面向异构 AI 芯片的全开源 AI 系统软件栈，让 AI 模型只需开发一次即可轻松移植到各类 AI 硬件。本仓库收集的是 FlagOS 中可复用的 **Skills**，为 AI 编程智能体注入领域知识、流程规范和最佳实践。
>
> [English](README.md)

## 什么是 Skills？

Skills 是一组**文件夹化的能力包**：每个 Skill 通过说明文档、脚本和资源，教会智能体在某一类任务上稳定、可复现地完成工作。每个 Skill 文件夹包含一个 `SKILL.md` 文件（YAML frontmatter + Markdown 正文），作为智能体的详细执行指令。还可包含参考文档、脚本和资源文件。

本仓库遵循 [Agent Skills 开放标准](https://agentskills.io/specification)，与 [anthropics/skills](https://github.com/anthropics/skills)、[huggingface/skills](https://github.com/huggingface/skills)、[openai/skills](https://github.com/openai/skills) 对齐。

## 快速开始

FlagOS Skills 兼容 **Claude Code**、**Cursor**、**Codex** 以及任何支持 [Agent Skills 标准](https://agentskills.io/specification) 的智能体。

### npx（推荐 — 适用于所有智能体）

使用 [`skills`](https://www.npmjs.com/package/skills) CLI 直接安装，无需克隆仓库：

```bash
# 查看本仓库中可用的 skills
npx skills add flagos-ai/skills --list

# 安装指定 skill 到当前项目
npx skills add flagos-ai/skills --skill model-migrate-fl

# 全局安装（用户级别）
npx skills add flagos-ai/skills --skill model-migrate-fl --global

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
$skill-installer install model-migrate-fl from flagos-ai/skills
```

或提供 GitHub 目录 URL：

```
$skill-installer install https://github.com/flagos-ai/skills/tree/main/skills/model-migrate-fl
```

也可以直接复制 skill 文件夹到 Codex 的标准 `.agents/skills` 目录：

```bash
cp -r skills/model-migrate-fl $REPO_ROOT/.agents/skills/
```

详见 [Codex Skills 指南](https://developers.openai.com/codex/skills/)。

### Gemini CLI

```bash
gemini extensions install https://github.com/flagos-ai/skills.git --consent
```

本仓库包含 `gemini-extension.json` 和 `agents/AGENTS.md` 用于 Gemini CLI 集成。详见 [Gemini CLI 扩展文档](https://geminicli.com/docs/extensions/)。

### 手动安装 / 其他智能体

对于任何支持 [Agent Skills 标准](https://agentskills.io/specification) 的智能体，将其指向本仓库的 `skills/` 目录即可。每个 skill 都是独立的，以 `SKILL.md` 为入口。`agents/AGENTS.md` 文件可作为不原生支持 skills 的智能体的 fallback。

## 已有 Skills

<!-- BEGIN_SKILLS_TABLE -->
| 名称 | 分类 | 说明 | 文档 |
|------|------|------|------|
| [`model-migrate-fl`](skills/model-migrate-fl/) | workflow-automation | 将上游 vLLM 模型迁移到 vllm-plugin-FL 项目。用于添加新模型支持、移植模型代码或回溯新发布的模型。 | [SKILL.md](skills/model-migrate-fl/SKILL.md) |
<!-- END_SKILLS_TABLE -->

### 在智能体中使用 Skills

安装 skill 后，在提示词中直接提及即可：

- "使用 model-migrate-fl 将 Qwen3-5 模型从上游 vLLM 迁移过来"
- "/model-migrate-fl qwen3_5"
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
│   ├── model-migrate-fl/    # 模型迁移工作流
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

## Skill 分类

| 分类 | 说明 | 示例 |
|------|------|------|
| **workflow-automation** | 多步骤流程：模型迁移、厂商对接、E2E 验证 | `model-migrate-fl` |
| **deployment** | 环境检查、容器构建、多芯片 CI | — |
| **enterprise-standards** | 品牌规范、文档模版、代码规范 | — |
| **tool-integration** | CI/CD、监控系统、内部平台、第三方 SaaS | — |

## 与主流 Skills 仓库对齐

| 特性 | 本仓库 | Anthropic | HuggingFace | OpenAI |
|------|--------|-----------|-------------|--------|
| 标准 | agentskills.io | agentskills.io | agentskills.io | agentskills.io |
| IDE 支持 | Claude Code, Cursor, Codex, Gemini CLI | Claude Code, Claude.ai, API | Claude Code, Codex, Gemini, Cursor | Codex |
| 插件清单 | `.claude-plugin/` + `.cursor-plugin/` | `.claude-plugin/` | `.claude-plugin/` + `.cursor-plugin/` | — |
| 组织方式 | 平铺 | 平铺 | 平铺 | `.system/` + `.curated/` + `.experimental/` |
| 验证工具 | `scripts/validate_skills.py` | — | `scripts/publish.sh` | — |
| 贡献指南 | `contributing.md` | — | — | `contributing.md` |

## 许可证

[Apache License 2.0](LICENSE)
