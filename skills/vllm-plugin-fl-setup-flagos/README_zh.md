# vllm-plugin-fl-setup-flagos：vLLM-Plugin-FL 环境搭建技能

## 概述

`vllm-plugin-fl-setup-flagos` 是一个**独立使用**的 AI 编程技能（Skill），用于在多种硬件后端（NVIDIA、昇腾、MetaX、天数智芯、摩尔线程等）上安装和配置 **vLLM-Plugin-FL**。

> **说明：** 本技能专注于**环境搭建和依赖安装**，可独立使用，不依赖任何其他技能。仅需本技能即可从零搭建一个可运行的 vLLM-Plugin-FL 环境。

### 解决的问题

vLLM-Plugin-FL 通过 FlagOS 的统一算子库 **FlagGems** 和通信库 **FlagCX**，扩展 vLLM 以支持多种硬件后端的模型推理/服务。搭建完整技术栈——vLLM-Plugin-FL、FlagGems、FlagCX 以及各后端特定驱动——步骤繁多，且不同硬件有不同的配置要求。手动搭建费时且容易出错，尤其是文档较少的后端。

本技能自动化了整个搭建流程：**检测硬件 → 安装 vLLM-Plugin-FL → 安装 FlagGems → （可选）安装 FlagCX → 应用后端特定配置 → 推理验证**。

### 使用方式

当用户说以下内容时自动触发：

- "setup vllm-plugin-fl"
- "install vllm-plugin-fl"
- "configure FL plugin"
- "set up FlagGems"
- "set up FlagCX"

---

## 前提条件

- Linux 操作系统（推荐 Ubuntu 20.04+）
- Python 3.10+
- vLLM **v0.13.0** — 从[官方发布](https://github.com/vllm-project/vllm/tree/v0.13.0)或 [vllm-FL 分支](https://github.com/flagos-ai/vllm-FL)安装
- 带有适当驱动的 GPU（NVIDIA CUDA、华为昇腾等）
- `pip` 包管理器
- Git

---

## 安装流程（5 步）

```
┌─────────────────────────────────────────────────────────┐
│  Step 1   识别硬件后端（nvidia-smi 等）                  │
│  Step 2   从源码安装 vLLM-Plugin-FL                      │
│  Step 3   安装 FlagGems（昇腾需先装 FlagTree）           │
│  Step 4   （可选）安装 FlagCX 用于多设备分布式推理        │
│  Step 5   后端特定配置（参见对应参考文档）                │
└─────────────────────────────────────────────────────────┘
```

### Step 1：识别硬件后端

技能通过探测可用的 CLI 工具来检测硬件后端：

| 后端 | 检测命令 |
|---|---|
| NVIDIA GPU | `nvidia-smi` |
| 华为 NPU | `npu-smi info` |
| 摩尔线程 GPU | `mthreads-gmi` |
| 天数智芯 GPU | `ixsmi` |

### Step 2：安装 vLLM-Plugin-FL

克隆并从源码安装：

```bash
mkdir -p ~/flagos-workspace && cd ~/flagos-workspace
git clone https://github.com/flagos-ai/vllm-plugin-FL
cd vllm-plugin-FL
pip install -r requirements.txt
pip install --no-build-isolation .
export VLLM_PLUGINS='fl'
```

### Step 3：安装 FlagGems

> **昇腾 NPU 用户**必须先安装 FlagTree。详见 [references/npu.md](references/npu.md)。

```bash
pip install -U scikit-build-core==0.11 pybind11 ninja cmake
cd ~/flagos-workspace
git clone https://github.com/flagos-ai/FlagGems
cd FlagGems
pip install --no-build-isolation .
```

### Step 4：（可选）安装 FlagCX

用于多设备分布式推理。单设备或昇腾 NPU 可跳过此步。

```bash
cd ~/flagos-workspace
git clone https://github.com/flagos-ai/FlagCX.git
cd FlagCX
git submodule update --init --recursive
make USE_NVIDIA=1  # 根据你的平台调整
export FLAGCX_PATH="$PWD"
cd plugin/torch/
FLAGCX_ADAPTOR=[xxx] pip install --no-build-isolation .
```

### Step 5：后端特定配置

| 后端 | 芯片厂商 | 参考文档 |
|---|---|---|
| 昇腾 NPU | 华为 | [references/npu.md](references/npu.md) |
| 天数智芯 GPU (BI-V150) | 天数智芯 | [references/iluvatar_gpu.md](references/iluvatar_gpu.md) |
| 摩尔线程 GPU | 摩尔线程 | [references/mthreads_gpu.md](references/mthreads_gpu.md) |
| MetaX GPU | 沐曦 | 待补充 |
| 平头哥-真武 | 平头哥 | 待补充 |
| 清微智能 | 清微智能 | 待补充 |
| 海光 DCU | 海光 | 待补充 |

---

## 快速测试

安装完成后，技能通过离线批量推理验证完整技术栈：

```python
from vllm import LLM, SamplingParams

model_path = "<resolved_model_path>"
prompts = ["Hello, my name is"]
sampling_params = SamplingParams(max_tokens=10, temperature=0.0)

llm = LLM(model=model_path, max_num_batched_tokens=16384, max_num_seqs=2048)
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(f"Prompt: {output.prompt!r}, Generated text: {output.outputs[0].text!r}")
```

技能会在机器上搜索本地模型副本，或询问用户提供模型路径。

---

## 目录结构

```
skills/vllm-plugin-fl-setup-flagos/
├── SKILL.md                          # 技能定义（入口文件）
├── README.md                         # 英文文档
├── README_zh.md                      # 本文档（中文版）
├── LICENSE.txt                       # Apache 2.0 许可证
└── references/                       # 后端特定配置指南
    ├── npu.md                        # 华为昇腾 NPU 配置
    ├── iluvatar_gpu.md               # 天数智芯 BI-V150 GPU 配置
    └── mthreads_gpu.md               # 摩尔线程 GPU 配置
```

---

## 各文件说明

### 技能定义

#### `SKILL.md`
技能的入口文件。定义了触发条件、前提条件、5 步安装流程、后端特定参考文档、快速测试流程和常见问题排查。AI 编程助手根据此文件识别和调用技能。

### 参考文档（`references/`）

#### `npu.md` — 华为昇腾 NPU 配置
昇腾特定配置，包括 FlagTree 安装（FlagGems 前置依赖）、CANN 工具套件配置和 eager 执行要求。

#### `iluvatar_gpu.md` — 天数智芯 GPU 配置
天数智芯 BI-V150 特定配置，包括驱动安装和 `enforce_eager=True` 要求。

#### `mthreads_gpu.md` — 摩尔线程 GPU 配置
摩尔线程特定配置，包括额外环境变量（`USE_FLAGGEMS=1`、`VLLM_MUSA_ENABLE_MOE_TRITON=1`）和 vLLM 启动参数（`enforce_eager=True`、`block_size=64`）。

---

## 常见问题排查

| 问题 | 常见原因 | 解决方法 |
|---|---|---|
| 加载模型时显存不足 | GPU 显存耗尽 | 使用 `gpu_memory_utilization=0.8` 参数 |
| FlagGems 构建失败 | 缺少构建依赖 | 安装 `scikit-build-core`、`pybind11`、`ninja`、`cmake` |
| 插件未加载 | 环境变量未设置 | 确保已 export `VLLM_PLUGINS='fl'` |
| FlagCX 通信错误 | 路径或构建不匹配 | 检查 `FLAGCX_PATH` 和平台构建参数 |
| 昇腾相关问题 | FlagTree 未安装 | 参见 [references/npu.md](references/npu.md) |
| 无法连接 GitHub | 网络限制 | 配置 `http_proxy` / `https_proxy` |

---

## 在项目中使用

Skill 通常放在项目根目录的 `.claude/skills/`（或你的编辑器对应的 skills 目录）下。


### 安装

```bash
# 在项目根目录执行
mkdir -p .claude/skills
cp -r <本仓库路径>/skills/vllm-plugin-fl-setup-flagos .claude/skills/
```

---

## 许可证

This project is licensed under the Apache 2.0 License. See [LICENSE.txt](LICENSE.txt) for details.
