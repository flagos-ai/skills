# skill-creator: Skill Development Toolkit

[中文版](README_zh.md)

## Overview

`skill-creator` is a meta-skill for creating, improving, and validating skills in the FlagOS skills repository.

### Problem Statement

Writing a high-quality skill involves many details: YAML frontmatter conventions, trigger-optimized descriptions, progressive disclosure structure, proper file organization, bilingual documentation, and more. Manually checking all these requirements is tedious and error-prone, and new contributors often miss conventions that only become apparent after review.

This skill automates the entire skill development lifecycle: **scaffold from template -> write with guided patterns -> validate conventions -> iterate with test cases**, spanning 4 operational modes with built-in tooling for each stage.

### Usage

```bash
# Interactive — asks what you want to do
/skill-creator

# Scaffold a new skill
/skill-creator preflight-check --init

# Scaffold with specific resource directories
/skill-creator preflight-check --init --resources scripts,references

# Validate a single skill
/skill-creator model-migrate-fl --validate

# Validate all skills
/skill-creator --validate

# Improve an existing skill
/skill-creator model-migrate-fl

# Run test cases against a skill
/skill-creator model-migrate-fl --eval
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `skill-name` | No | — | Skill name in hyphen-case (e.g. `preflight-check`) |
| `--init` | No | — | Scaffold a new skill directory |
| `--validate` | No | — | Validate skill structure and conventions |
| `--eval` | No | — | Run test cases against the skill |
| `--resources` | No | — | Comma-separated resource dirs: `scripts,references,assets` |

When no flag is provided, the skill enters interactive mode and asks what you'd like to do.

---

## Four Operational Modes

### Mode 1: Create — Build a New Skill from Scratch

```
Intent capture → Interview → Scaffold → Write SKILL.md → Add resources → Validate → Write README
```

**What happens:**

1. **Intent capture** — The agent interviews you about what the skill should do, when it should trigger, what tools it needs, and what outputs it produces. If the current conversation already contains a workflow you want to capture, the agent extracts answers from context first.

2. **Scaffold** — Runs `init_skill.py` to generate the directory structure:
   ```
   my-skill/
   ├── SKILL.md        # Template with [TODO] placeholders
   ├── LICENSE.txt      # Apache 2.0
   ├── scripts/         # (if requested)
   ├── references/      # (if requested)
   └── assets/          # (if requested)
   ```

3. **Write SKILL.md** — Fills in the frontmatter and body following the patterns in `references/writing-guide.md`. The agent ensures:
   - Description is trigger-optimized (specific phrases, slightly "pushy")
   - Body includes all required sections (Overview, Steps, Examples, Troubleshooting)
   - Progressive disclosure is applied (details in `references/`, not bloating SKILL.md)

4. **Add resources** — Creates scripts, reference docs, and assets as needed, with each file documented in SKILL.md.

5. **Validate & README** — Runs the validator, fixes issues, and writes bilingual READMEs.

**Example session:**

```
You:   /skill-creator preflight-check --init
Agent: What should preflight-check do? When should it trigger?
You:   It checks GPU availability, driver versions, and Python env before running vLLM
Agent: Got it. I'll scaffold the skill and include a scripts/ directory for the check scripts...
       [creates directory, writes SKILL.md, adds scripts/check_gpu.sh, validates]
       Skill 'preflight-check' created. Here's the final structure: ...
```

### Mode 2: Improve — Iterate on an Existing Skill

**What happens:**

1. **Analyze** — Reads the existing SKILL.md and all supporting files. Identifies issues:
   - Vague or missing trigger conditions in description
   - Missing examples or edge cases
   - Steps without progress reporting
   - Overly long SKILL.md that should be split
   - Missing troubleshooting entries

2. **Draft test prompts** — Creates 2-3 realistic test cases and saves them to `evals/evals.json` for verification.

3. **Iterate** — Applies improvements, re-validates, reviews against test prompts, collects your feedback, and repeats.

**Example session:**

```
You:   /skill-creator model-migrate-fl
Agent: I've analyzed model-migrate-fl. Found 3 potential improvements:
       1. Description could be more trigger-friendly
       2. Missing example for multimodal model migration
       3. Troubleshooting table doesn't cover network timeout
       Which should I address first?
```

### Mode 3: Validate — Check Conventions Compliance

**What happens:**

Runs the repository-level `scripts/validate_skills.py` which checks:

| Check | Severity | Description |
|---|---|---|
| SKILL.md exists | Error | Entry point file must be present |
| Valid YAML frontmatter | Error | Must start and end with `---` |
| `name` field present | Error | Required field |
| `description` field present | Error | Required field |
| Name matches directory | Error | `name` field must equal folder name |
| Name format | Error | Lowercase + hyphens only, max 64 chars |
| Description length | Error | Must not exceed 1024 chars |
| Body content | Error | Must be at least 100 chars |
| Referenced files exist | Error | All linked files must be on disk |
| Examples section | Warning | Recommended for all skills |
| Troubleshooting section | Warning | Recommended for all skills |
| Script permissions | Warning | `.py` and `.sh` files should be executable |
| LICENSE.txt present | Warning | Recommended per-skill license |
| README.md present | Warning | Recommended documentation |

**Example output:**

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

### Mode 4: Eval — Run Test Cases

**What happens:**

If `evals/evals.json` exists in the skill directory, the agent runs each test prompt and checks assertions against the expected output. Generates a report showing pass/fail status per assertion.

---

## Directory Structure

```
skills/skill-creator/
├── SKILL.md                          # Skill definition (entry point)
├── LICENSE.txt                       # Apache 2.0 license
├── README.md                         # This document (English)
├── README_zh.md                      # Chinese version
├── references/                       # Reference documents
│   ├── writing-guide.md              # Detailed skill writing patterns & best practices
│   └── schemas.md                    # JSON schemas for evals and validation
└── scripts/                          # Executable scripts
    └── init_skill.py                 # Scaffold a new skill directory
```

---

## File Descriptions

### Skill Definition

#### `SKILL.md`

The skill entry point. Defines trigger conditions, argument format, four operational modes (Create, Improve, Validate, Eval), placeholder resolution, examples, and troubleshooting guide. The AI coding assistant uses this file to identify and invoke the skill.

### Reference Documents (`references/`)

#### `writing-guide.md` — Skill Writing Patterns & Best Practices

Comprehensive guide covering:

- **Anatomy of a Skill** — required fields, FlagOS extension fields, directory conventions
- **Progressive Disclosure** — three-level loading system (metadata → body → references), when and how to split content
- **Writing the Description** — how to craft trigger-optimized descriptions with specific phrases and negative triggers
- **Structuring the Body** — required sections template (Overview, Prerequisites, Steps, Examples, Troubleshooting)
- **Writing Style** — imperative form, explain why not what, conciseness checklist
- **Scripts and References** — when to use each, documentation requirements
- **FlagOS-Specific Conventions** — naming, categories, bilingual support, licensing

#### `schemas.md` — JSON Schemas

Defines the JSON formats used for evaluation and validation:

| Schema | Location | Purpose |
|---|---|---|
| `evals.json` | `evals/evals.json` | Test case definitions (prompts + assertions) |
| `eval_result.json` | `evals/results/eval_result.json` | Evaluation results with pass/fail per assertion |
| `validation_result.json` | `evals/results/validation_result.json` | Structured validation output |

### Scripts (`scripts/`)

#### `init_skill.py` — Scaffold a New Skill

```bash
# Basic usage
python3 init_skill.py my-skill --path skills/

# With resource directories
python3 init_skill.py my-skill --path skills/ --resources scripts,references,assets

# Name auto-normalization
python3 init_skill.py "My Cool Skill" --path skills/
# → creates skills/my-cool-skill/
```

**What it generates:**

| File | Always | Description |
|---|---|---|
| `SKILL.md` | Yes | Template with all frontmatter fields and `[TODO]` body placeholders |
| `LICENSE.txt` | Yes | Apache 2.0 license |
| `scripts/` | If requested | Empty directory with `.gitkeep` |
| `references/` | If requested | Empty directory with `.gitkeep` |
| `assets/` | If requested | Empty directory with `.gitkeep` |

**Validations performed:**
- Name must be 2-64 chars, lowercase + hyphens only
- Parent directory must exist
- Target directory must not already exist
- Resource types must be one of: `scripts`, `references`, `assets`

### Repository-Level Validation (`scripts/validate_skills.py`)

Validation is handled by the repository-level script `scripts/validate_skills.py` (not bundled inside skill-creator to avoid duplication). It supports both single-skill and batch validation:

```bash
# Validate all skills (default)
python3 scripts/validate_skills.py

# Validate a single skill
python3 scripts/validate_skills.py skills/model-migrate-fl

# Validate all skills in a specific directory
python3 scripts/validate_skills.py skills/ --all
```

Performs 14 checks across two severity levels (error / warning). See the Mode 3 table above for the full list. Exit code: 0 = pass, 1 = errors found.

---

## Usage in FlagOS Skills Repository

### Quick Install (via npx)

```bash
# Install this skill only
npx skills add flagos-ai/skills --skill skill-creator -a claude-code

# Or install all Flagos skills at once
npx skills add flagos-ai/skills -a claude-code
```

### Manual Install

```bash
# From your project root
mkdir -p .claude/skills
cp -r <path-to-this-repo>/skills/skill-creator .claude/skills/
```

### Standalone Script Usage

The scripts can be used independently without invoking the skill:

```bash
# Scaffold a new skill anywhere
python3 skills/skill-creator/scripts/init_skill.py my-skill --path ./my-project/skills/

# Validate skills (repo-level script)
python3 scripts/validate_skills.py
python3 scripts/validate_skills.py skills/my-skill
```

---

## License

This project is licensed under the Apache 2.0 License. See [LICENSE.txt](LICENSE.txt) for details.
