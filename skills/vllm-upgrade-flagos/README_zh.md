# vllm-upgrade-flagos：vLLM 插件版本升级工具

## 概述

`vllm-upgrade-flagos` 用于将 vllm-plugin-FL 插件的基础设施代码升级到匹配更新版本的上游 vLLM。

### 解决的问题

vllm-plugin-FL 是一个 OOT（out-of-tree）插件，覆盖或扩展了 vLLM 的 worker、model_runner、platform、compilation 和 ops 子系统。当上游 vLLM 版本升级时，内部 API 会发生变化（类重命名、函数签名变更、模块重组等），插件代码必须同步更新。

手动升级涉及大量工作：对比 API 差异、逐文件适配、处理废弃代码、清理已上游化的模型。本技能将整个流程自动化为 11 个步骤，确保升级过程可重复、不遗漏。

### 使用方式

```bash
# 默认路径
/vllm-upgrade-flagos

# 指定路径
/vllm-upgrade-flagos /path/to/upstream/vllm /path/to/vllm-plugin-FL
```

| 参数 | 必填 | 默认值 | 说明 |
|---|---|---|---|
| `upstream_folder` | 否 | `/workspace/vllm_update/vllm` | 上游 vLLM 源码目录 |
| `plugin_folder` | 否 | `/workspace/vllm_update/vllm-plugin-FL` | 插件源码目录 |

---

## 升级流程

### 步骤 1：基线单元测试

在修改任何代码之前，运行插件现有的单元测试，记录基线结果。

### 步骤 2：API 差异分析

对比插件基础设施文件与上游对应文件之间的 API 变化：
- 度量变更范围（commit 数、diff 统计）
- 提取插件自定义代码（dispatch、IO dumper、FL envs 等）
- 识别需要适配的 API 变更

### 步骤 3–7：逐模块升级

按以下顺序升级插件基础设施：

| 顺序 | 插件文件 | 上游对应 | 说明 |
|---|---|---|---|
| 3 | `vllm_fl/platform.py` | `vllm/platforms/interface.py` + `cuda.py` | 平台抽象层 |
| 4 | `vllm_fl/compilation/graph.py` | `vllm/compilation/cuda_graph.py` | CUDA 图封装 |
| 5 | `vllm_fl/worker/worker.py` | `vllm/v1/worker/gpu_worker.py` | GPU Worker |
| 6 | `vllm_fl/worker/model_runner.py` | `vllm/v1/worker/gpu_model_runner.py` | 模型运行器 |
| 7 | `vllm_fl/ops/` | `vllm/model_executor/layers/` | 算子层（MoE、激活、LayerNorm 等） |

每个模块的升级策略：
1. 从上游复制最新版本
2. 重新注入插件自定义代码
3. 修复导入路径和 API 签名
4. 验证导入通过

### 步骤 8：清理废弃代码

**不是盲目删除所有文件**，而是逐一检查：

- **模型文件**（`models/`）：检查上游是否已有对应模型类和注册。已上游化且插件不再注册的 → 删除
- **配置文件**（`configs/`）：检查上游 `_CONFIG_REGISTRY` 是否已包含对应 `model_type`。已上游化且无其他引用的 → 删除
- **补丁文件**（`patches/`）：检查补丁修复的问题是否已在上游解决。仍需要的 → 保留

删除后验证无悬空导入：

```bash
grep -rn 'from vllm_fl\.models\.\|from vllm_fl\.configs\.\|from vllm_fl\.patches\.' \
  vllm_fl/ --include='*.py' | grep -v __pycache__
```

### 步骤 9：清理 `__init__.py`

- 移除已上游化模型/配置的注册代码
- 如果 `register_model()` 仍有未上游化的注册项 → 保留函数
- 如果 `register_model()` 完全为空 → 删除函数
- 移除已删除补丁的导入和调用

### 步骤 10：验证与回归测试

五级验证：

1. **导入验证** — 确认所有插件模块可正常导入
2. **端到端推理** — 运行 `qwen3_5_offline_inference.py`，验证完整推理链路
3. **回归单元测试** — 与步骤 1 基线对比
4. **功能测试** — 运行功能测试套件
5. **性能基准测试** — 运行吞吐量基准，检测性能回退

性能测试使用 `scripts/benchmark.sh`：

```bash
conda activate vllm_plugin_update && bash scripts/benchmark.sh Qwen3.5-397B-A17B-Real
```

测试参数：100 条 prompt，input_len=6144，output_len=1024，TP=4，dummy 权重。
关注指标：吞吐量（tokens/s）、首 token 延迟（TTFT）、无 OOM/CUDA 错误。

### 步骤 11：更新版本引用

更新文件头注释、`pyproject.toml` 版本约束等。

---

## 常见 API 变更模式

本技能在升级过程中会自动处理以下常见变更：

| 变更类型 | 示例 | 处理方式 |
|---|---|---|
| 环境变量重命名/移除 | `VLLM_FUSED_MOE_CHUNK_SIZE` → 移除 | 替换为硬编码默认值或新变量名 |
| 字符串→枚举 | `activation="silu"` → `MoEActivation.SILU` | 添加枚举值归一化 |
| 函数签名变更 | `forward_native` 新增参数 | 更新方法签名匹配上游 |
| 基类方法缺失调用 | `import_kernels()` 未调用 `super()` | 添加 `super()` 调用 |
| OOT 注册门控 | FlagGems 门控阻止注册 | 移除不必要的门控 |
| 返回值语义变更 | runner 自行处理 shared experts | 调整返回值（不返回 tuple） |
| `_custom_ops` 函数移除 | `from vllm._custom_ops import silu_and_mul` → 移除 | 改用 `torch.ops._C.<fn>`，逐个检查上游 `_custom_ops.py` |
| 工具函数移除 | `from vllm.utils.import_utils import init_cached_hf_modules` → 移除 | 删除相关调用 |

> 注：平台特定的修复（如 CUDA kernel 路径）应在代码中用行内注释标注，不在此文档中列出。本技能保持平台无关。

---

## 插件基础设施架构

```
vllm-plugin-FL/
├── vllm_fl/
│   ├── __init__.py          # 插件入口：register() + register_model()
│   ├── platform.py          # 平台抽象（设备检测、内核加载、配置）
│   ├── compilation/
│   │   └── graph.py         # CUDA 图封装
│   ├── worker/
│   │   ├── worker.py        # GPU Worker（初始化、KV 缓存、执行循环）
│   │   └── model_runner.py  # 模型运行器（加载、前向、采样）
│   ├── ops/
│   │   ├── custom_ops.py    # OOT 算子注册
│   │   ├── activation.py    # 激活函数
│   │   ├── layernorm.py     # LayerNorm
│   │   ├── rotary_embedding.py  # 旋转位置编码
│   │   └── fused_moe/       # MoE 算子
│   │       ├── layer.py     # FusedMoE 层 + UnquantizedFusedMoEMethod
│   │       └── fused_moe.py # MoE 内核实现
│   ├── dispatch/             # 算子分发框架（保留不动）
│   ├── configs/              # 未上游化的模型配置
│   ├── patches/              # 兼容性补丁
│   └── models/               # 未上游化的模型（升级后通常为空）
```

---

## Conda 环境

| 环境 | 用途 |
|---|---|
| `vllm_plugin_update` | 目标环境 — 安装了新版 vLLM，所有验证和测试在此运行 |
| `vllm_0.13.0_plugin_tree` | 旧环境 — vLLM v0.13.0 + 当前插件，用于对比旧 API |

---

## 故障排查

| 问题 | 常见原因 | 解决方式 |
|---|---|---|
| `ImportError: cannot import name X from vllm` | API 在新版本中重命名或移动 | 查看 `api-changes.md`，更新导入路径 |
| `TypeError: __init__() got unexpected keyword` | 构造函数签名变更 | 对比上游类签名，更新调用 |
| `AttributeError: module has no attribute X` | 模块重组 | 检查上游模块布局，更新导入 |
| Worker 初始化崩溃 | 新增必填配置字段或初始化步骤 | 对比上游 worker `__init__` |
| CUDA 图捕获失败 | CUDAGraphWrapper API 变更 | 检查 compilation 变更，更新 `graph.py` |
| `assert self.kernel is not None` | OOT 类未注册，回退到基类 `forward_cuda` | 检查 `custom_ops.py` 注册逻辑 |
| `pip install -e .` 覆盖 vLLM | 插件拉取 vLLM 作为依赖 | 始终使用 `pip install --no-build-isolation --no-deps -e .` |

---

## 安装

### 通过 skills CLI

```bash
npx skills add flagos-ai/skills --skill vllm-upgrade-flagos -a claude-code
```

### 手动安装

```bash
mkdir -p .claude/skills
ln -s <本仓库路径>/skills/vllm-upgrade-flagos .claude/skills/
```

---

## 许可证

This project is licensed under the Apache 2.0 License.
