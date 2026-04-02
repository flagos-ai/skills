# kernelgen-flagos: Unified GPU Kernel Generation Skills

[中文版](README_zh.md)

## Overview

The KernelGen `kernelgen-flagos` skill is a unified AI programming skill launched by the Beijing Academy of Artificial Intelligence (BAAI). By leveraging the `kernelgen-mcp` KernelGen Operator Development MCP Toolkit service, it is compatible with various domestic and international chip platforms to generate high-precision GPU kernel operators. It automatically detects the type of the target code repository and dispatches it to the corresponding dedicated workflow.

## Quick Start

This section introduces how to quickly start a task, for example, creating, optimizing, specialize operators for a specific platform, or submitting feedback. You can use either prompt or the command line to execute these tasks. In this section, we only teach you how to use prompts. For how to use command lines or manual configurations, see [KernelGen Skills User Guide](https://docs.flagos.io/projects/kernelgen/en/latest/skills_user_guide/skills-user-guide.html#).

Perform all of the following steps except for obtaining the Access Token in your agent client, for example, OpenClaw, VS Code with activated Github Copilot, or Claude Code.

### Prerequisites

* Claude Code version 2.1 and later

* OpenClaw version 2026.3.2 and later

* VSCode with Github Copilot activated

### Steps

1. Obtain the **Access Token** from: [https://kernelgen.flagos.io/mcp](https://kernelgen.flagos.io/mcp).

2. In your agent client, send a prompt to connect to the KernelGen Operator Development MCP Toolkit, for example:

   * "Connect to MCP, its URL is `http://kernelgen.flagos.io/sse` and token is `<your Access Token>`."

   * "Please configure the kernelgen MCP with the URL `http://kernelgen.flagos.io/sse` and the token is `<your Access Token>`. "

   **Note**: You may need to restart your agent to let the settings take effect. Check this in the documentation of the relevant AI agent.

3. Verify KernelGen Operator Development MCP Toolkit connection, prompt “Please verify the kernelgen mcp connection is successful.”

4. Send a prompt to setup the `kernelgen-flagos` unified skill, including all sub-skills, for example:

   * **Copilot**: "Setup kernelgen-flagos skill **in Copilot** from <https://github.com/flagos-ai/skills/tree/main/skills/kernelgen-flagos>."

   * **Other agent clients**: "Setup kernelgen-flagos skills from <https://github.com/flagos-ai/skills/tree/main/skills/kernelgen-flagos>."

5. Verify skills are successfully installed: "Please verify if the kernelgen-flagos skills are working correctly."

6. Send a prompt to run a task by invoking the corresponding sub-skill.&#x20;

### Send a prompt to run a task by invoking the sub-skill

KernelGen automatically detects whether the project is a FlagGems repository, a vLLM repository, or another type, and executes the corresponding workflow.

* **Use Case 1**: Create an operator for the vLLM project：

  1. Preinstall [vLLM](https://docs.vllm.ai/en/latest/getting_started/installation/).

     **Note**: Please install from source via git clone.

  2. Send a prompt to generate an operator, for example: "Use kernelgen-flagos to generate the ReLU operator. Use MetaX. Integrated into vLLM."

* **Use Case 2**: send a prompt to optimize an operator on NVIDIA, for example:  “Use kernelgen-flagos to optimize the scaled\_dot\_product\_attention\_math operator. Optimize 5 iterations. The scaled\_dot\_product\_attention\_math operator is located at \<operator path>.”

* **Use Case 3**: Send a prompt to specialize the CUDA-implemented operator for Huawei Ascend for the FlagGems project：

  1. Preinstall [FlagGems](https://docs.flagos.io/projects/FlagGems/en/latest/getting_started/install.html#).

     **Note**: Please install from source via git clone.

  2. Send a prompt to optimize an operator, for example:

     * "Use kernelgen-flagos to migrate the CUDA-implemented operator \<operator path A> to the Ascend chip, with the operator file stored in the FlagGems repository, and the directory is \<operator path B>, ensuring that the accuracy verification passes."

     * "Use `kernelgen-flagos` to migrate the CUDA-implemented operators located in \<operator path> to the Ascend chip, and integrate them into the FlagGems repository, ensuring that precision verification passes."

### Output files

The output files are as follows:

* \<operator name>\_triton.py - Triton kernel implementation

* \<operator name>\_torch.py - PyTorch baseline implementation

* test\_\<operator name>.py - Complete test cases

* benchmark\_\<operator name>.py - Performance benchmarking

**Note**: If you have other files generated, send a prompt to verify the skills and KernelGen Operator Development MCP Toolkit are still working.

Further steps:

* Pull Requests: If the user is developing within a GitHub repository, the skill will ask if a Pull Request needs to be submitted. It can then automatically submit the PR and provide the link.

* Chat Integration: If the user configures the skill via chat applications (such as WeChat or Feishu), the skill will ask if the modified files should be sent to the chat application.

* Feedback: If you encounter bugs during usage, you can use this skill to submit feedback via email or by opening a GitHub issue. send a prompt to submit feedback, for example: "Use the kernelgen-flagos skill to submit feedback" or "Use the kernelgen-flagos skill to report a bug".

## Why Choose Skill?

KernelGen Skill doesn't rely solely on LLMs to generate code from scratch. Instead, it follows a structured four-step workflow:

1. **Collect Kernel Information**: Extracts operator parameters (input/output types, shapes, computation logic) from the user's natural language description.

2. **Search for Similar Code Snippets**: Retrieves similar implementations from the existing codebase for reference (users can choose whether to adopt them).

3. **Generate Kernel Code + CUDA Implementation**: Simultaneously generates the Triton Kernel and its corresponding CUDA implementation, with the latter serving as the PyTorch benchmark.

4. **Test**: Uses the CUDA implementation as the ground truth to test the Kernel, outputting Correctness and Speedup Ratio.

Once generation is complete, the path from generation to contribution is fully automated: if developing within the FlagGems repository, a specialized sub-skill automatically handles format conversion and directory alignment. It directly injects the four core deliverables (kernel, baseline, accuracy test, speedup test) into the `src/flag_gems/experimental_ops/` directory. Developers can generate a compliant PR with one click, without needing to manually invoke conversion scripts.

### Core Features

Writing high-performance GPU operators is complex and error-prone. Different projects (FlagGems, vLLM, custom Triton repositories) each have unique operator implementation specifications, testing patterns, and registration systems. Previously, users needed to install multiple skills separately.

This unified skill packages nine sub-skills into one, requiring installation only once:

### File Descriptions

### `SKILL.md`

The unified entry point. Contains routing logic that auto-detects the repository type (FlagGems, vLLM, or generic) and reads the appropriate sub-skill file to execute.

### `kernelgen-generate.md`

Full 10-step workflow for generating GPU kernel operators in any Python/Triton repository. Includes dynamic repo structure discovery, convention detection, and adaptive code placement.

### `kernelgen-generate-for-flaggems.md`

9-step workflow specialized for FlagGems repositories. Handles `pointwise_dynamic` wrappers, promotion methods, `_FULL_CONFIG` registration, categorized test files, and FlagGems-specific conventions.

### `kernelgen-generate-for-vllm.md`

9-step workflow specialized for vLLM repositories. Handles SPDX license headers, `vllm.logger.init_logger`, `@triton.autotune`, custom op registration, and vLLM directory conventions.

### `kernelgen-submit-feedback.md`

Feedback submission workflow. Collects bug reports with auto-detected environment info and submits via GitHub Issues (`gh` CLI) or email fallback.

### `kernelgen-optimize.md`

General-purpose Triton kernel optimization via MCP iterative loop. Analyzes existing kernels, identifies bottlenecks, and applies optimizations through multiple rounds until the target speedup is reached.

### `kernelgen-optimize-for-flaggems.md`

FlagGems-specific kernel optimization with 3 modes: optimize built-in operators in-place, optimize external operators and integrate into experimental\_ops, or optimize existing experimental operators. Includes accuracy tests and performance benchmarks.

### `kernelgen-optimize-for-vllm.md`

vLLM-specific kernel optimization with CustomOp registration, accuracy tests, and performance benchmark integration. Optimizes Triton operators and automatically integrates them into the vLLM project.

### `kernelgen-specialize.md`

Platform specialization for Triton operators via MCP `specialize_kernel` tool. Migrates GPU Triton operators to target platforms (e.g., Huawei Ascend NPU), handling architecture differences, Grid configuration, UB overflow, and memory alignment.

### `kernelgen-specialize-for-flaggems.md`

Combines MCP platform specialization with FlagGems framework integration. Supports four integration modes: vendor-ops (default), vendor-fused, override-builtin, and experimental. Includes automated testing and performance benchmarking.

***

## License

This project is licensed under the Apache 2.0 License. See [LICENSE.txt](https://github.com/flagos-ai/skills/blob/main/skills/kernelgen-flagos/LICENSE.txt) for details.

