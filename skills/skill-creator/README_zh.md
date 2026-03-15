# skill-creator：技能开发工具链

## 概述

`skill-creator` 是一个元技能（meta-skill），用于在 FlagOS skills 仓库中创建、改进和验证其他技能。

### 解决的问题

编写高质量的 skill 涉及大量细节：YAML frontmatter 格式、触发优化的 description、渐进式披露结构、文件组织规范、双语文档等。手动检查这些要求既繁琐又容易遗漏，新贡献者往往要到 review 阶段才发现问题。

本技能自动化了整个技能开发生命周期：**从模板脚手架 → 按规范编写 → 验证约定 → 用测试用例迭代**，提供 4 种工作模式，每个阶段都有配套工具。

### 使用方式

```bash
# 交互模式 — 询问你想做什么
/skill-creator

# 创建新 skill 脚手架
/skill-creator preflight-check --init

# 创建时指定资源目录
/skill-creator preflight-check --init --resources scripts,references

# 验证单个 skill
/skill-creator model-migrate-fl --validate

# 验证所有 skill
/skill-creator --validate

# 改进现有 skill
/skill-creator model-migrate-fl

# 运行测试用例
/skill-creator model-migrate-fl --eval
```

| 参数 | 必填 | 默认值 | 说明 |
|---|---|---|---|
| `skill-name` | 否 | — | hyphen-case 格式的技能名（如 `preflight-check`） |
| `--init` | 否 | — | 创建新 skill 目录脚手架 |
| `--validate` | 否 | — | 验证 skill 结构和规范 |
| `--eval` | 否 | — | 对 skill 运行测试用例 |
| `--resources` | 否 | — | 逗号分隔的资源目录：`scripts,references,assets` |

不带任何 flag 时进入交互模式。

---

## 四种工作模式

### 模式一：创建 — 从零构建新 Skill

```
意图捕获 → 访谈 → 脚手架 → 编写 SKILL.md → 添加资源 → 验证 → 写 README
```

**执行过程：**

1. **意图捕获** — Agent 访谈你：这个 skill 做什么、什么时候触发、需要什么工具、产出什么。如果当前对话已经包含你想封装的工作流，Agent 会先从上下文提取信息。

2. **脚手架** — 运行 `init_skill.py` 生成目录结构：
   ```
   my-skill/
   ├── SKILL.md        # 带 [TODO] 占位符的模板
   ├── LICENSE.txt      # Apache 2.0
   ├── scripts/         # （按需）
   ├── references/      # （按需）
   └── assets/          # （按需）
   ```

3. **编写 SKILL.md** — 按照 `references/writing-guide.md` 中的规范填充 frontmatter 和 body：
   - Description 优化触发（具体短语、适度"激进"）
   - Body 包含所有必要章节（Overview、Steps、Examples、Troubleshooting）
   - 渐进式披露（详细内容放 `references/`，不让 SKILL.md 膨胀）

4. **添加资源** — 按需创建脚本、参考文档和资源文件，每个文件都在 SKILL.md 中记录用法。

5. **验证 & README** — 运行验证器修复问题，编写中英文 README。

**示例会话：**

```
你：   /skill-creator preflight-check --init
Agent: preflight-check 应该做什么？什么时候触发？
你：   在运行 vLLM 前检查 GPU 可用性、驱动版本和 Python 环境
Agent: 明白了。我会创建 skill 并包含 scripts/ 目录来放检查脚本……
       [创建目录、编写 SKILL.md、添加 scripts/check_gpu.sh、验证]
       Skill 'preflight-check' 已创建，最终结构如下：……
```

### 模式二：改进 — 迭代优化现有 Skill

**执行过程：**

1. **分析** — 读取现有 SKILL.md 和所有支撑文件，识别问题：
   - Description 中触发条件模糊或缺失
   - 缺少示例或边界情况
   - 步骤没有进度汇报
   - SKILL.md 过长需要拆分
   - Troubleshooting 条目缺失

2. **设计测试用例** — 创建 2-3 个真实的测试 prompt，保存到 `evals/evals.json`。

3. **迭代** — 应用改进 → 重新验证 → 对照测试用例审查 → 收集反馈 → 重复。

**示例会话：**

```
你：   /skill-creator model-migrate-fl
Agent: 分析了 model-migrate-fl，发现 3 个可改进项：
       1. Description 可以更有利于触发
       2. 缺少多模态模型迁移的示例
       3. Troubleshooting 表没有覆盖网络超时
       先处理哪个？
```

### 模式三：验证 — 检查规范合规性

**执行过程：**

运行仓库级脚本 `scripts/validate_skills.py`，检查项：

| 检查项 | 级别 | 说明 |
|---|---|---|
| SKILL.md 存在 | Error | 必须有入口文件 |
| YAML frontmatter 合法 | Error | 必须以 `---` 开头和结尾 |
| `name` 字段存在 | Error | 必填字段 |
| `description` 字段存在 | Error | 必填字段 |
| name 与目录名一致 | Error | `name` 字段必须等于文件夹名 |
| name 格式合规 | Error | 仅小写字母 + 数字 + 连字符，最长 64 字符 |
| description 长度 | Error | 不超过 1024 字符 |
| body 内容充实 | Error | 至少 100 字符 |
| 引用文件存在 | Error | 所有 markdown 链接指向的文件必须实际存在 |
| 有 Examples 章节 | Warning | 建议所有 skill 都包含 |
| 有 Troubleshooting 章节 | Warning | 建议所有 skill 都包含 |
| 脚本有执行权限 | Warning | `.py` 和 `.sh` 文件应有 `+x` 权限 |
| LICENSE.txt 存在 | Warning | 建议每个 skill 包含 |
| README.md 存在 | Warning | 建议包含文档 |

**输出示例：**

```
============================================================
  model-migrate-fl: PASS
============================================================
    ✓ All checks passed

============================================================
  skill-creator: PASS
============================================================
  Warnings (1):
    ⚠ No README.md found (recommended)

────────────────────────────────────────────────────────────
  Summary: 2 skill(s), 0 error(s), 1 warning(s)
────────────────────────────────────────────────────────────
```

### 模式四：评估 — 运行测试用例

**执行过程：**

如果 skill 目录中存在 `evals/evals.json`，Agent 会运行每个测试 prompt 并检查断言，生成通过/失败报告。

---

## 目录结构

```
skills/skill-creator/
├── SKILL.md                          # 技能定义（入口文件）
├── LICENSE.txt                       # Apache 2.0 许可证
├── README.md                         # 英文文档
├── README_zh.md                      # 本文档（中文版）
├── references/                       # 参考文档
│   ├── writing-guide.md              # 详细的技能编写规范和最佳实践
│   └── schemas.md                    # 评估和验证的 JSON schema
└── scripts/                          # 可执行脚本
    └── init_skill.py                 # 创建新 skill 目录脚手架
```

---

## 各文件说明

### 技能定义

#### `SKILL.md`

技能的入口文件。定义了触发条件、参数格式、四种工作模式（创建、改进、验证、评估）、占位符解析、使用示例和常见问题排查。AI 编程助手根据此文件识别和调用技能。

### 参考文档（`references/`）

#### `writing-guide.md` — 技能编写规范

全面的编写指南，涵盖：

- **Skill 结构解剖** — 必填字段、FlagOS 扩展字段、目录约定
- **渐进式披露** — 三级加载系统（metadata → body → references）、何时以及如何拆分内容
- **Description 编写** — 如何写出触发优化的描述（具体短语 + 反向触发条件）
- **Body 结构** — 必要章节模板（Overview、Prerequisites、Steps、Examples、Troubleshooting）
- **写作风格** — 祈使句、解释 why 而非堆 MUST、简洁优先
- **Scripts 和 References 用法** — 何时使用、文档化要求
- **FlagOS 特有约定** — 命名规范、分类体系、双语支持、许可证

#### `schemas.md` — JSON Schema 定义

定义评估和验证使用的 JSON 格式：

| Schema | 位置 | 用途 |
|---|---|---|
| `evals.json` | `evals/evals.json` | 测试用例定义（prompt + 断言） |
| `eval_result.json` | `evals/results/eval_result.json` | 评估结果（每个断言的通过/失败状态） |
| `validation_result.json` | `evals/results/validation_result.json` | 结构化验证输出 |

### 脚本（`scripts/`）

#### `init_skill.py` — 创建新 Skill 脚手架

```bash
# 基本用法
python3 init_skill.py my-skill --path skills/

# 带资源目录
python3 init_skill.py my-skill --path skills/ --resources scripts,references,assets

# 名称自动标准化
python3 init_skill.py "My Cool Skill" --path skills/
# → 创建 skills/my-cool-skill/
```

**生成的文件：**

| 文件 | 必定生成 | 说明 |
|---|---|---|
| `SKILL.md` | 是 | 带所有 frontmatter 字段和 `[TODO]` body 占位符的模板 |
| `LICENSE.txt` | 是 | Apache 2.0 许可证 |
| `scripts/` | 按需 | 空目录 + `.gitkeep` |
| `references/` | 按需 | 空目录 + `.gitkeep` |
| `assets/` | 按需 | 空目录 + `.gitkeep` |

**执行的校验：**
- 名称 2-64 字符，仅允许小写字母 + 数字 + 连字符
- 父目录必须存在
- 目标目录不能已存在
- 资源类型必须是 `scripts`、`references`、`assets` 之一

### 仓库级验证脚本（`scripts/validate_skills.py`）

验证由仓库级脚本 `scripts/validate_skills.py` 统一处理（不在 skill-creator 内部重复，避免逻辑分散）。支持单个和批量验证：

```bash
# 验证所有 skill（默认）
python3 scripts/validate_skills.py

# 验证单个 skill
python3 scripts/validate_skills.py skills/model-migrate-fl

# 验证目录下所有 skill
python3 scripts/validate_skills.py skills/ --all
```

执行 14 项检查，分 error 和 warning 两个级别。退出码：0 = 通过，1 = 有错误。

---

## 在 FlagOS Skills 仓库中使用

### 快速安装（通过 npx）

```bash
# 仅安装本 skill
npx skills add flagos-ai/skills --skill skill-creator -a claude-code

# 或一次性安装所有 Flagos skills
npx skills add flagos-ai/skills -a claude-code
```

### 手动安装

```bash
# 在你的项目根目录执行
mkdir -p .claude/skills
cp -r <本仓库路径>/skills/skill-creator .claude/skills/
```

### 独立使用脚本

脚本可以脱离 skill 调用，独立使用：

```bash
# 在任意位置创建新 skill 脚手架
python3 skills/skill-creator/scripts/init_skill.py my-skill --path ./my-project/skills/

# 验证 skills（仓库级脚本）
python3 scripts/validate_skills.py
python3 scripts/validate_skills.py skills/my-skill
```

---

## 许可证

This project is licensed under the Apache 2.0 License. See [LICENSE.txt](LICENSE.txt) for details.
