# kernelgen-submit-feedback: Skill Feedback Submission

[中文版](README_zh.md)

## Overview

`kernelgen-submit-feedback` is an AI coding skill that helps users submit bug reports, defect reports, and improvement suggestions for skills in the FlagOS skills repository.

### Problem Statement

When users encounter bugs or have suggestions for skills, they need to gather environment information, format the report properly, and submit it through the right channel. This process is tedious and often results in incomplete bug reports missing critical environment details.

This skill automates the entire feedback workflow: **gather information → auto-collect environment → verify GitHub CLI → construct issue → submit**, with an automatic email fallback when GitHub CLI is unavailable.

### Usage

```bash
# Submit feedback interactively
/kernelgen-submit-feedback

# Submit feedback for a specific skill
/kernelgen-submit-feedback kernelgen
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `skill-name` | No | Inferred from context | Name of the skill the feedback is about |

---

## Submission Workflow

```
┌──────────────────────────────────────────────────────────┐
│  Step 1   Determine submission method (GitHub or Email)  │
│  Step 2   Gather information from user                   │
│  Step 3   Auto-collect environment information           │
│  Step 4   Verify GitHub CLI availability                 │
│  Step 5   Determine issue type and labels                │
│  Step 6   Construct the issue                            │
│  Step 7   Confirm with user                              │
│  Step 8   Submit the issue                               │
│  Step 9   Confirm creation                               │
└──────────────────────────────────────────────────────────┘
```

### Key Features

- **Auto environment detection** — automatically collects OS, Python, PyTorch, CUDA, and Triton versions
- **GitHub CLI integration** — submits issues directly via `gh issue create`
- **Email fallback** — generates email draft with mailto link when GitHub CLI is unavailable
- **Smart context reuse** — extracts information from conversation history instead of re-asking
- **Label management** — verifies label existence before submission to avoid errors

---

## Directory Structure

```
skills/kernelgen-submit-feedback/
├── SKILL.md        # Skill definition (entry point)
├── LICENSE.txt     # Apache 2.0 license
├── README.md       # This document (English)
└── README_zh.md    # Chinese version
```

---

## Related Skills

- [`kernelgen`](../kernelgen/) — General purpose GPU kernel generation
- [`kernelgen-for-flaggems`](../kernelgen-for-flaggems/) — FlagGems-specific kernel generation
- [`kernelgen-for-vllm`](../kernelgen-for-vllm/) — vLLM-specific kernel generation

---

## License

This project is licensed under the Apache 2.0 License. See [LICENSE.txt](LICENSE.txt) for details.
