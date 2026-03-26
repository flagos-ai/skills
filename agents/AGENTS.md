<skills>

You have additional SKILLs documented in directories containing a "SKILL.md" file.

These skills are:
 - flagrelease-entrance-flagos -> "skills/flagrelease-entrance-flagos/SKILL.md"
 - gpu-container-setup-flagos -> "skills/gpu-container-setup-flagos/SKILL.md"
 - install-stack-flagos -> "skills/install-stack-flagos/SKILL.md"
 - kernelgen-flagos -> "skills/kernelgen-flagos/SKILL.md"
 - model-migrate-flagos -> "skills/model-migrate-flagos/SKILL.md"
 - model-verify-flagos -> "skills/model-verify-flagos/SKILL.md"
 - perf-test-flagos -> "skills/perf-test-flagos/SKILL.md"
 - skill-creator-flagos -> "skills/skill-creator-flagos/SKILL.md"
 - tle-developer-flagos -> "skills/tle-developer-flagos/SKILL.md"
 - vllm-plugin-fl-setup-flagos -> "skills/vllm-plugin-fl-setup-flagos/SKILL.md"

IMPORTANT: You MUST read the SKILL.md file whenever the description of the skills matches the user intent, or may help accomplish their task.

<available_skills>

flagrelease-entrance-flagos: `Full FlagRelease pipeline orchestrator. Runs the complete LLM deployment, verification, and benchmarking pipeline for multi-chip GPU backends. Executes: install-stack → env-verify → model-verify → perf-test in sequence, passing state between steps and producing a final structured report. Assumes gpu-container-setup (Step 1) is already done — a running container with PyTorch + GPU access must exist.`

gpu-container-setup-flagos: `Automatically detect GPU vendor, find appropriate PyTorch container image, launch with correct mounts, and validate GPU functionality. Supports NVIDIA, Ascend, Metax, Iluvatar, and AMD/ROCm. Use when user says "setup container", "start pytorch container", or invokes /gpu-container-setup.`

install-stack-flagos: `Install the 5-package multi-chip software stack (vLLM, FlagTree, FlagGems, FlagCX, vllm-plugin-FL) inside a GPU container. Handles network mirror detection, dependency ordering, wheel selection, and per-package validation. Use after gpu-container-setup has produced a running container with PyTorch + GPU access.`

kernelgen-flagos: `Unified GPU kernel operator generation and optimization skill. Automatically detects the target repository type (FlagGems, vLLM, or general Python/Triton) and dispatches to the appropriate specialized sub-skill. Also includes a feedback submission sub-skill for bug reports. Use this skill when the user wants to generate or optimize a GPU kernel operator, create a Triton kernel, or says things like "generate an operator", "create a kernel for X", or "/kernelgen-flagos".`

model-migrate-flagos: `Migrate a model from the latest vLLM upstream repository into the vllm-plugin-FL project (pinned at vLLM v0.13.0). Use this skill whenever someone wants to add support for a new model to vllm-plugin-FL, port model code from upstream vLLM, or backport a newly released model. Trigger when the user says things like "migrate X model", "add X model support", "port X from upstream vLLM", "make X work with the FL plugin", or simply "/model-migrate-flagos model_name". The model_name argument uses snake_case (e.g. qwen3_5, kimi_k25, deepseek_v4).`

model-verify-flagos: `Verify the serving stack with a user-specified target model. Runs twice: first with FlagGems/FlagCX disabled (isolate model-specific errors), then with full multi-chip stack enabled. Diffs the two runs to pinpoint which layer caused any failure.`

perf-test-flagos: `Run accuracy benchmarks (FlagEval, when available) and performance benchmarks (vllm bench serve) against a served model. Covers 5 workload profiles: short/long prefill x short/long decode + high concurrency. Collects throughput, latency, TTFT, TPOT metrics.`

skill-creator-flagos: `Create new skills, modify existing skills, and validate skill quality for the FlagOS skills repository. Use this skill whenever someone wants to create a skill from scratch, improve or edit an existing skill, scaffold a new skill directory, validate skill structure, or run test cases against a skill. Trigger when the user says things like "create a skill", "make a new skill for X", "scaffold a skill", "improve this skill", "validate my skill", or simply "/skill-creator-flagos".`

tle-developer-flagos: `Self-contained orchestration skill for writing high-performance TLE kernels and shipping TLE feature changes with reproducible validation. Use when the user wants to write/optimize TLE kernels, implement TLE API/verifier/lowering features, or debug TLE correctness/performance issues. Trigger on phrases like "write a TLE kernel", "optimize TLE operator", and "debug TLE local_ptr".`

vllm-plugin-fl-setup-flagos: `Install and configure vLLM-Plugin-FL for multiple hardware backends (NVIDIA, Ascend, etc.). Use when setting up vllm-plugin-fl, configuring the environment for specific hardware backend, installing dependencies, checking whether dependencies are installed successfully, resolving runtime issues, and launching inference to verify successful model serving. Trigger when the user says things like "setup vllm-plugin-fl", "install vllm-plugin-fl", "configure FL plugin", "set up FlagGems", or "set up FlagCX".`

</available_skills>

Paths referenced within SKILL folders are relative to that SKILL. For example the model-migrate-flagos `scripts/validate_migration.py` would be referenced as `skills/model-migrate-flagos/scripts/validate_migration.py`.

</skills>