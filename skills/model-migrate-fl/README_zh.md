# model-migrate-fl：vLLM 模型迁移技能

## 概述

`model-migrate-fl` 是一个 AI 编程技能（Skill），用于将 **vLLM 上游（upstream）最新版本** 中的模型代码迁移到 **vllm-plugin-FL 项目**（锁定在 vLLM v0.13.0）。

### 解决的问题

vLLM 上游版本迭代快，新模型不断加入。但 vllm-plugin-FL 锁定在 v0.13.0，直接复制上游代码无法运行——API 不兼容、配置类缺失、相对导入失效等。手动适配费时且容易出错。

本技能自动化了整个迁移流程：**从上游复制代码 → 适配 0.13.0 → 注册模型 → 验证正确性**，全程 13 个步骤，最终与上游服务做 token 级精度对比。

### 使用方式

```bash
# 在 AI 编程助手中调用
/model-migrate-fl qwen3_5
/model-migrate-fl kimi_k25
/model-migrate-fl glm5 /path/to/upstream /path/to/plugin
```

| 参数 | 必填 | 默认值 | 说明 |
|---|---|---|---|
| `model_name` | 是 | — | snake_case 模型名（如 `qwen3_5`、`kimi_k25`） |
| `upstream_folder` | 否 | `/tmp/vllm-upstream-ref` | 上游 vLLM 源码路径 |
| `plugin_folder` | 否 | 当前工作目录 | vllm-plugin-FL 项目路径 |

---

## 迁移流程（13 步）

```
┌─────────────────────────────────────────────────────┐
│  Step 1   基线单元测试（记录迁移前的状态）            │
│  Step 2   克隆/更新上游 vLLM 源码                    │
│  Step 3   研究已有插件模式（选择最相似的参考模型）    │
│  Step 4   添加模型文件（Config Bridge + Copy-then-Patch）│
│  Step 5   在 __init__.py 中注册模型                  │
│  Step 6   代码审查（自动化脚本 + 人工核查）          │
│  Step 7   回归单元测试（对比 Step 1 基线）           │
│  Step 8   功能测试                                   │
│  Step 9   Benchmark 验证（dummy weights）            │
│  Step 10  Serve + Request 冒烟测试                   │
│  Step 11  E2E 精度验证（与上游 GT 做 token 级对比）  │
│  Step 12-13  最终报告                                │
└─────────────────────────────────────────────────────┘
```

### 核心理念：Copy-then-Patch

**不是读代码再重写**，而是：
1. 直接 `cp` 上游文件到插件目录
2. 按兼容性补丁目录（P1-P13）逐项 Edit 修补
3. 验证 import 通过

这样最大限度保留上游实现，减少人为错误。

---

## 目录结构

```
skills/model-migrate-fl/
├── SKILL.md                          # 技能定义（入口文件）
├── README.md                         # 本文档
├── references/                       # 参考文档
│   ├── procedure.md                  # 13 步迁移流程详述
│   ├── compatibility-patches.md      # 0.13.0 兼容性补丁目录（P1-P13）
│   └── operational-rules.md          # 运行规则（通信、任务管理、GPU 管理等）
├── scripts/                          # 可执行脚本
│   ├── validate_migration.py         # 自动化代码审查
│   ├── benchmark.sh                  # Benchmark 验证
│   ├── serve.sh                      # 启动本地 vLLM 服务
│   ├── request.sh                    # 冒烟测试请求
│   ├── e2e_eval.py                   # E2E 精度对比工具
│   ├── e2e_test_prompts.json         # 测试 prompt 集（5 text + 5 multimodal）
│   ├── e2e_config.template.json      # E2E 配置模板
│   └── e2e_remote_serve.sh           # 远程 GT 服务器管理
└── results/                          # E2E 对比结果输出
```

---

## 各文件说明

### 技能定义

#### `SKILL.md`
技能的入口文件。定义了触发条件、参数格式、执行步骤概览、脚本索引和常见问题排查。AI 编程助手根据此文件识别和调用技能。

### 参考文档（`references/`）

#### `procedure.md` — 迁移流程
13 步迁移的完整操作手册，包含每一步的：
- 具体命令和代码模板
- 占位符解析规则（如 `{{model_name}}`、`{{ModelClassName}}`）
- 判断逻辑（何时需要 Config Bridge、哪些补丁适用）
- 最终报告模板

#### `compatibility-patches.md` — 0.13.0 兼容性补丁

收录了所有已知的 vLLM upstream → 0.13.0 不兼容项及其修复方案，每个补丁包含：
- **Before/After** 代码示例
- **Why** 需要改
- **When** 适用条件

| 编号 | 补丁名称 | 说明 |
|---|---|---|
| P1 | 相对导入 → 绝对导入 | `from .xxx` → `from vllm.*` 或 `from vllm_fl.*` |
| P2 | Config 导入重定向 | 指向插件的 Config Bridge |
| P3 | 移除缺失 API | `MambaStateCopyFunc` 等不存在于 0.13.0 |
| P4 | 替换 Context Manager 初始化 | `_mark_tower_model` → 直接初始化 |
| P5 | 导入验证 | 快速验证 `python3 -c "import ..."` |
| P6 | 父类 `__init__` 重实现 | 父类用了 0.13.0 缺失 API 时，子类需重写 |
| P7 | MoE 嵌套配置修正 | VL config 下 MoE 参数在 `text_config` 中 |
| P8 | MergedColumnParallelLinear | 合并投影的 TP 分片修正（关键！） |
| P9 | MambaStateDtypeCalculator 签名 | 3 参数 → 2 参数 |
| P10 | mamba_cache_mode → enable_prefix_caching | 缓存模式 API 差异 |
| P11 | 自定义 Op 导入路径去重 | 避免 `Duplicate op name` 错误 |
| P12 | 完整 `__init__` 重写检查清单 | 跳过父类 init 时不遗漏属性 |
| P13 | 不要升级 transformers | 使用 Config Bridge 而非升级 transformers 版本 |

#### `operational-rules.md` — 运行规则
- **通信协议**：每步边界报告进度，不允许静默执行
- **TaskList 管理**：13 个 Task 对应 13 步，支持中断恢复
- **Bash 命令规范**：单行命令、不用进程替换，避免权限弹窗
- **GPU 资源管理**：benchmark/serve 前强制释放 GPU，永不跳过
- **网络重试**：git clone 等命令失败自动重试 3 次

### 脚本（`scripts/`）

#### `validate_migration.py` — 自动化代码审查（Step 6）
```bash
python3 validate_migration.py <plugin_folder> <model_file> [config_file]
```
自动检查：
- 残留的相对导入
- 0.13.0 缺失 API 的引用
- Config Bridge 的 `model_type` 和 `PretrainedConfig` 继承
- `__init__.py` 中的注册一致性
- 代码异味（裸 `except:`、硬编码路径）

退出码：0=通过，1=有问题需修复。

#### `benchmark.sh` — Benchmark 验证（Step 9）
```bash
bash benchmark.sh Qwen3.5-397B-A17B-Real
```
使用 `vllm bench throughput` + dummy weights 快速验证模型能否正常 forward，不需要加载真实权重。测试参数：input_len=128, output_len=128, num_prompts=2, TP=8。

#### `serve.sh` — 启动本地服务（Step 10/11）
```bash
bash serve.sh Qwen3.5-397B-A17B-Real [PORT]
```
启动 vLLM OpenAI 兼容 API 服务，默认端口 8122，`VLLM_FL_PREFER_ENABLED=false`（关闭 FlagGems，确保对比纯净性）。TP=8, GPU 利用率 0.9。

#### `request.sh` — 冒烟测试（Step 10）
```bash
bash request.sh Qwen3.5-397B-A17B-Real [PORT]
```
发送一个简单的 chat completion 请求，验证服务能正常响应。

#### `e2e_eval.py` — E2E 精度验证（Step 11，核心工具）
```bash
# 完整流程：GT + Local + 对比
python3 e2e_eval.py --model Qwen3.5-397B-A17B-Real --gt-host <GT_HOST_IP> --mode text

# 用配置文件
python3 e2e_eval.py --config e2e_config.json --model Qwen3.5-397B-A17B-Real --mode text

# 分阶段执行
python3 e2e_eval.py --model ... --gt-only        # 只采集 GT 结果
python3 e2e_eval.py --model ... --local-only      # 只采集 Local 结果
python3 e2e_eval.py --model ... --compare-only    # 只做对比
```

工作流程：
1. 向 GT 服务器（upstream vLLM）发送测试 prompt，收集 token_ids + token 文本
2. 向 Local 服务器（plugin vLLM）发送相同 prompt
3. 逐 token 对比前 N 个 token（默认 32），生成报告

输出三个文件：
- `{model}_gt.json` — GT 原始结果
- `{model}_local.json` — Local 原始结果
- `{model}_e2e_report.txt` — 对比报告

关键参数：
| 参数 | 默认值 | 说明 |
|---|---|---|
| `--gt-host` | （必填） | GT 服务器 IP |
| `--gt-port` | 8122 | GT 服务器端口 |
| `--local-port` | 8122 | 本地服务端口 |
| `--mode` | text | `text` / `multimodal` / `all` |
| `--token-match-count` | 32 | 对比 token 数量 |
| `--max-tokens` | 256 | 最大生成长度 |
| `--seed` | 42 | 随机种子（确保可复现） |

#### `e2e_test_prompts.json` — 测试 Prompt 集
- **text**（5 条）：英文问答、中文问答、代码生成、数学推理、技术解释
- **multimodal**（5 条）：5 张合成小图（base64 内联），测试图片描述、颜色识别、模式识别

#### `e2e_config.template.json` — E2E 配置模板
```json
{
  "gt_machine": {
    "host": "<GT_HOST_IP>",
    "docker_container": "<CONTAINER_NAME>",
    "conda_env": "<CONDA_ENV>",
    "vllm_port": 8122,
    "env_vars": { "VLLM_FL_PREFER_ENABLED": "false" },
    "extra_serve_args": "--trust-remote-code --load-format fastsafetensors"
  },
  "local": { "vllm_port": 8122 },
  "eval": { "max_tokens": 256, "token_match_count": 32, "seed": 42 }
}
```
使用时复制为 `e2e_config.json` 并填入实际值。

> **远程 GT 服务器前置条件：**
> 1. 配置免密 SSH 登录 GT 机器：`ssh-copy-id -i ~/.ssh/id_ed25519 <user>@<GT_HOST_IP>`
> 2. 验证连通性：`ssh -i ~/.ssh/id_ed25519 <user>@<GT_HOST_IP> hostname`
> 3. 如果 GT 服务运行在 Docker 容器内，确保 config 中的容器名和 conda 环境名正确

#### `e2e_remote_serve.sh` — 远程 GT 服务器管理
```bash
bash e2e_remote_serve.sh start  e2e_config.json Qwen3.5-397B-A17B-Real  # 启动
bash e2e_remote_serve.sh stop   e2e_config.json                          # 停止
bash e2e_remote_serve.sh status e2e_config.json                          # 状态
bash e2e_remote_serve.sh logs   e2e_config.json                          # 日志
```
通过 SSH → Docker exec → Conda run 的方式在远程机器上管理 GT vLLM 服务。支持自动等待服务 ready（最多 600 秒）。

---

## 已迁移模型示例

| 模型 | 模式 | 特点 |
|---|---|---|
| `qwen3_5` | 复杂 MoE + 混合注意力 | 完整模型文件适配、Config Bridge、GDN 层合并投影 |
| `kimi_k25` | 轻量包装 | 委托给 DeepseekV2，少量修改 |
| `qwen3_next` | 混合注意力 | 自定义注意力集成（linear + full attention） |
| `minicpmo` | 多模态 | 视觉 + 语言组合 |

---

## 在 vllm-plugin-FL 中使用

Skill 通常放在项目根目录的 `.claude/skills/`（或你的编辑器对应的 skills 目录）下。在 vllm-plugin-FL 中使用本 skill：

### 快速安装（通过 npx）

```bash
# 仅安装本 skill
npx skills add flagos-ai/skills --skill model-migrate-fl -a claude-code

# 或一次性安装所有 Flagos skills
npx skills add flagos-ai/skills -a claude-code
```

### 手动安装

```bash
# 在 vllm-plugin-FL 项目根目录执行
mkdir -p .claude/skills
cp -r <本仓库路径>/skills/model-migrate-fl .claude/skills/
```

---

## 许可证

This project is licensed under the Apache 2.0 License. See [LICENSE.txt](LICENSE.txt) for details.
