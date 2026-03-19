---
name: kernelgen-flagos
description: >
  Unified GPU kernel operator generation skill. Automatically detects the target repository type
  (FlagGems, vLLM, or general Python/Triton) and dispatches to the appropriate specialized
  sub-skill. Also includes a feedback submission sub-skill for bug reports. Use this skill when
  the user wants to generate a GPU kernel operator, create a Triton kernel, or says things like
  "generate an operator", "create a kernel for X", or "/kernelgen-flagos". This single skill replaces
  the need to install kernelgen-general, kernelgen-for-flaggems, kernelgen-for-vllm, and
  kernelgen-submit-feedback separately.
argument-hint: "<operator_name> [--func-type <type>]"
user-invokable: true
compatibility: "Python 3.8+, PyTorch with CUDA, Triton"
metadata:
  version: "1.0.0"
  author: flagos-ai
  category: gpu-kernel-generation
  tags: [kernelgen, triton, gpu, mcp, operator-generation, flaggems, vllm, feedback]
allowed-tools:
  - Bash
  - Bash(gh:*)
  - Bash(python:*)
  - Bash(python3:*)
  - Bash(command:*)
  - Read
  - Write
  - Edit
  - Glob
  - Grep
  - AskUserQuestion
---

# kernelgen-flagos — Unified GPU Operator Generation Skill

This is a **unified entry point** that bundles four sub-skills into one:

| Sub-skill file | Purpose |
|---|---|
| `kernelgen-general.md` | Generate GPU kernels for **any** Python/Triton repository |
| `kernelgen-for-flaggems.md` | Specialized generation for **FlagGems** repositories |
| `kernelgen-for-vllm.md` | Specialized generation for **vLLM** repositories |
| `kernelgen-submit-feedback.md` | Submit bug reports and feedback via GitHub or email |

All sub-skill files are located in the **same directory** as this `SKILL.md` file.

---

## Routing Protocol — Follow This BEFORE Doing Anything Else

### Phase 1: Detect Repository Type

Use the Glob tool to check for project identity files in the current working directory:

```
Glob: pyproject.toml
Glob: setup.py
Glob: setup.cfg
```

Then use the Read tool to read whichever file exists. Determine the **project name** from
the file contents (e.g., `name = "flag_gems"` in pyproject.toml, or `name='vllm'` in setup.py).

Also use the Glob tool to check for characteristic directory structures:

**FlagGems indicators** (match ANY):
- `src/flag_gems/` directory exists
- Project name is `flag_gems` or `flag-gems` or `FlagGems`
- `import flag_gems` appears in test files

**vLLM indicators** (match ANY):
- `vllm/` directory exists at the repo root (with `vllm/__init__.py`)
- Project name is `vllm`
- `csrc/` directory exists alongside `vllm/`

### Phase 2: Dispatch to Sub-skill

Based on the detection result, use the **Read tool** to read the appropriate sub-skill file
from this skill's directory, then **follow the instructions in that file exactly**.

**To locate the sub-skill files**: They are in the same directory as this SKILL.md. Use the
Glob tool to find the path:

```
Glob: **/skills/kernelgen-flagos/kernelgen-general.md
```

Then use the Read tool to read the matched path.

#### Decision Table

| Detection Result | Action |
|---|---|
| FlagGems repository detected | Read `kernelgen-for-flaggems.md` and follow it |
| vLLM repository detected | Read `kernelgen-for-vllm.md` and follow it |
| Neither detected (or unknown) | Read `kernelgen-general.md` and follow it |
| User reports a bug or requests feedback submission | Read `kernelgen-submit-feedback.md` and follow it |

**Important rules:**
1. **Always detect first, dispatch second.** Never skip detection.
2. **Read the entire sub-skill file** before starting execution — do not partially read it.
3. **Follow the sub-skill instructions exactly** as if they were the main SKILL.md. All steps,
   rules, and protocols in the sub-skill apply fully.
4. **Do not mix sub-skills.** Once you dispatch to a sub-skill, follow it to completion.
5. If the user explicitly requests a specific sub-skill (e.g., "use the FlagGems version"),
   honor that request regardless of auto-detection results.
6. **CRITICAL — MCP is mandatory**: ALL operator code generation MUST go through the
   `mcp__kernelgen-mcp__generate_operator` MCP tool. NEVER generate Triton kernels, PyTorch
   wrappers, or operator implementations yourself. If MCP is not configured, not reachable,
   or fails after all retries, STOP and report the issue — do NOT fall back to writing code
   manually.

### Phase 3: Feedback Handling

At **any point** during the workflow, if the user reports a bug, says something is broken,
or asks to submit feedback about the skill:

1. Use the Read tool to read `kernelgen-submit-feedback.md` from this skill's directory.
2. Follow the feedback submission workflow described in that file.
3. After feedback is submitted, ask the user if they want to continue with the operator
   generation workflow or stop.

---

## Quick Reference for Users

```bash
# Generate a kernel operator (auto-detects repo type)
/kernelgen-flagos relu

# Generate with explicit function type
/kernelgen-flagos rms_norm --func-type normalization

# The skill will automatically:
# - Detect if you're in a FlagGems repo → use FlagGems-specific workflow
# - Detect if you're in a vLLM repo → use vLLM-specific workflow
# - Otherwise → use the general-purpose workflow
```

If you encounter any issues during generation, just say "submit feedback" or "report a bug"
and the skill will guide you through the feedback submission process.
