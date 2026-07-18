<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->

# tle-developer-flagos: TLE 内核与特性开发技能

## 概览

`tle-developer-flagos` 是一个用于 TLE 工作端到端执行的自包含技能，覆盖内核优化、编译器特性实现，以及可复现验证驱动的正确性/性能问题排查。

该技能通过固定流程保证开发过程可追踪、可复验：

`intake -> implementation -> validation -> artifacts -> merge decision`

## 何时使用

当你需要以下工作时使用本技能：

- 编写或优化 TLE 内核。
- 实现 TLE API / verifier / lowering / pipeline 相关特性。
- 排查 TLE 正确性、性能或回归问题。

典型触发短语：

- `write a TLE kernel`
- `optimize TLE operator`
- `debug TLE local_ptr`

## 使用方式

```bash
/tle-developer-flagos
```

推荐输入格式：

```text
Goal:
Non-goal:
Acceptance:
Impact scope (optional):
```

## 工作约定

- 将 `references/tle-sources.md` 作为技术事实来源。
- 将 `references/workflow-templates.md` 作为流程模板来源。
- 不依赖本技能目录之外的文档。
- 不假设固定 Python 环境名或固定构建脚本名。

## 目录结构

```text
skills/tle-developer-flagos/
├── SKILL.md
├── README.md
├── README_cn.md
├── LICENSE.txt
├── agents/
│   └── openai.yaml
└── references/
    ├── tle-sources.md
    └── workflow-templates.md
```

## 文件说明

- `SKILL.md`：技能入口，定义触发条件、约束、输出要求与完成检查项。
- `agents/openai.yaml`：面向助手的接口元数据和默认提示词。
- `references/tle-sources.md`：环境、开发路径、调试与优化参考。
- `references/workflow-templates.md`：intake/验证矩阵/修复总结/经验记录/合并决策模板。

## 许可证

本技能采用 Apache License 2.0，详见 `LICENSE.txt`。
