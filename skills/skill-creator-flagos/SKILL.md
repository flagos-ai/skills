---
name: skill-creator-flagos
description: >
  Create new skills, modify existing skills, and validate skill quality for the FlagOS skills
  repository. Use this skill whenever someone wants to create a skill from scratch, improve or
  edit an existing skill, scaffold a new skill directory, validate skill structure, or run test
  cases against a skill. Trigger when the user says things like "create a skill", "make a new
  skill for X", "scaffold a skill", "improve this skill", "validate my skill", or simply
  "/skill-creator-flagos". Also trigger when users mention turning a workflow into a reusable skill,
  or want to package a repeated process as a skill.
argument-hint: "[skill-name] [--init | --validate | --eval]"
user-invokable: true
compatibility: "Python 3.8+, works with any AI coding assistant that supports the Agent Skills standard"
metadata:
  version: "1.0.0"
  author: flagos-ai
  category: workflow-automation
  tags: [skill-creation, scaffolding, validation, meta-skill, developer-tooling]
allowed-tools: "Bash(python3:*) Bash(python:*) Bash(chmod:*) Bash(mkdir:*) Bash(cp:*) Bash(ls:*) Bash(cat:*) Read Edit Write Glob Grep AskUserQuestion TaskCreate TaskUpdate TaskList TaskGet Agent"
---

# Skill Creator

A meta-skill for creating, improving, and validating skills in the FlagOS skills repository.

## Overview

This skill guides you through the full lifecycle of skill development:
1. **Create** — scaffold a new skill from template, interview the user, write SKILL.md
2. **Improve** — analyze an existing skill, identify weaknesses, iterate with test cases
3. **Validate** — check structure, frontmatter, references, and conventions compliance

## Usage

```
/skill-creator-flagos                       # Interactive — asks what you want to do
/skill-creator-flagosmy-new-skill --init    # Scaffold a new skill
/skill-creator-flagosmy-skill --validate    # Validate an existing skill
/skill-creator-flagosmy-skill --eval        # Run test prompts against a skill
```

## Execution

### Step 0: Parse arguments and determine mode

Extract from user input:
- `{{skill_name}}` — optional skill name (hyphen-case)
- `{{mode}}` — one of: `create`, `improve`, `validate`, `eval`, or `interactive` (default)

If no mode is specified, ask the user:

> What would you like to do?
> 1. Create a new skill from scratch
> 2. Improve an existing skill
> 3. Validate a skill's structure and conventions
> 4. Run test cases against a skill

**-> Tell user**: Confirm the mode and skill name.

---

## Mode 1: Create a New Skill

### Step 1: Capture intent

Start by understanding what the user wants to build. If the current conversation already contains a workflow the user wants to capture (e.g., "turn this into a skill"), extract answers from the conversation history first.

Key questions to clarify:
1. What should this skill enable the agent to do?
2. When should this skill trigger? (user phrases, contexts, file types)
3. What's the expected output or end state?
4. Does the skill need scripts, reference docs, or asset files?
5. What tools does the skill need access to?

Adapt your communication style to the user — don't assume coding jargon familiarity. Pay attention to context cues.

**-> Tell user**: Summarize the captured intent and confirm before proceeding.

### Step 2: Initialize the skill directory

Run the init script to scaffold the skill:

```bash
python3 {{skill_root}}/scripts/init_skill.py {{skill_name}} --path {{skills_dir}} [--resources scripts,references,assets]
```

Where:
- `{{skill_root}}` = absolute path to this skill-creator-flagos's directory
- `{{skills_dir}}` = path to the `skills/` directory (usually the parent of `{{skill_root}}`)

The script creates the directory with SKILL.md template, LICENSE.txt, and optional subdirectories.

**-> Tell user**: Show the created directory structure.

### Step 3: Write the SKILL.md

Based on the user interview, fill in these components:

#### Frontmatter (YAML)

| Field | Required | Guidelines |
|-------|----------|------------|
| `name` | Yes | Lowercase + hyphens, must match directory name, max 64 chars |
| `description` | Yes | 1-2 sentences: what it does AND when to trigger. Be specific and slightly "pushy" — err on the side of triggering too often rather than too rarely |
| `argument-hint` | Recommended | Show expected arguments |
| `user-invokable` | Recommended | `true` if invokable via `/skill-name` |
| `compatibility` | Optional | Environment requirements |
| `metadata` | Recommended | version, author, category, tags |
| `allowed-tools` | Recommended | Space-separated tool list with patterns |

#### Body (Markdown)

Follow the structure documented in `references/writing-guide.md`. At minimum include:
- **Overview** — what problem this solves, when to activate
- **Prerequisites** — environment requirements
- **Execution steps** — numbered steps with `**-> Tell user**` progress markers
- **Examples** — at least 2-3 realistic usage examples
- **Troubleshooting** — common problems and fixes

Read `references/writing-guide.md` for detailed patterns on progressive disclosure, output formats, domain organization, and writing style.

### Step 4: Add supporting resources

Based on the skill's needs, create:
- **`scripts/`** — executable code for deterministic/repetitive tasks
- **`references/`** — detailed docs loaded into context as needed
- **`assets/`** — files used in output (templates, icons, etc.)

Rules:
- Every script/reference must be documented in SKILL.md with usage instructions
- Scripts should have execute permissions (`chmod +x`)
- Keep SKILL.md under 500 lines; move detailed content to `references/`
- For reference files >300 lines, include a table of contents

### Step 5: Validate

Run validation to check conventions compliance:

```bash
python3 {{repo_root}}/scripts/validate_skills.py {{skills_dir}}/{{skill_name}}
```

Fix any reported issues. See Troubleshooting section below.

**-> Tell user**: Report validation results. On failure, diagnose and fix.

### Step 6: Write README files

Create `README.md` (English) and optionally `README_zh.md` (Chinese) following the pattern in existing skills. The README should cover:
- Overview / problem statement
- Usage instructions
- Directory structure
- File descriptions
- Examples
- Installation instructions

**-> Tell user**: Skill creation complete. Present the final directory tree and summary.

---

## Mode 2: Improve an Existing Skill

### Step 1: Analyze the current skill

Read the existing SKILL.md and all supporting files. Identify:
- Unclear or missing trigger conditions in the description
- Missing examples or edge cases
- Steps that lack progress reporting (`**-> Tell user**`)
- Overly long SKILL.md that should be split into references
- Missing troubleshooting entries
- Scripts without documentation

**-> Tell user**: Present findings and proposed improvements.

### Step 2: Draft test prompts

Create 2-3 realistic test prompts — things a real user would say. Share them with the user for confirmation.

Save to `evals/evals.json`:

```json
{
  "skill_name": "{{skill_name}}",
  "evals": [
    {
      "id": 1,
      "prompt": "User's task prompt",
      "expected_output": "Description of expected result",
      "assertions": ["The output includes X", "Step Y was executed"]
    }
  ]
}
```

See `references/schemas.md` for the full schema.

### Step 3: Iterate

For each round:
1. Apply improvements to SKILL.md and supporting files
2. Re-run validation
3. Review against test prompts (mentally or by spawning test runs)
4. Collect user feedback
5. Repeat until satisfied

**-> Tell user**: Report changes made in each iteration.

---

## Mode 3: Validate

Run the validation script:

```bash
python3 {{repo_root}}/scripts/validate_skills.py {{skills_dir}}/{{skill_name}}
```

The script checks:
- SKILL.md exists and has valid YAML frontmatter
- Required fields (`name`, `description`) are present
- `name` matches directory name, follows naming conventions
- `description` length within limits
- Body has sufficient content (>100 chars)
- Referenced files in `scripts/` and `references/` actually exist
- Scripts have execute permissions
- No hardcoded paths or credentials

**-> Tell user**: Report all findings with severity (error/warning).

---

## Mode 4: Run Evals

If `evals/evals.json` exists, run the test prompts against the skill:

```bash
python3 {{skill_root}}/scripts/run_eval.py {{skills_dir}}/{{skill_name}}
```

This generates a report showing which assertions passed/failed for each test case.

**-> Tell user**: Present results and suggest improvements.

---

## Placeholders

| Placeholder | How to derive |
|---|---|
| `{{skill_name}}` | From user input, normalized to hyphen-case |
| `{{skill_root}}` | Absolute path to this skill-creator-flagos's directory |
| `{{repo_root}}` | Absolute path to the repository root (parent of `skills/`) |
| `{{skills_dir}}` | Path to the `skills/` directory containing all skills |

## Examples

**Example 1: Create a new skill from scratch**
```
User says: "/skill-creator-flagospreflight-check --init"
Actions:
  1. Parse → skill_name=preflight-check, mode=create
  2. Interview user about what preflight-check should do
  3. Run init_skill.py to scaffold directory
  4. Write SKILL.md with deployment verification workflow
  5. Add scripts/check_gpu.sh, scripts/check_env.py
  6. Validate and fix any issues
  7. Write README.md
Result: Complete preflight-check skill ready for use
```

**Example 2: Improve an existing skill**
```
User says: "improve model-migrate-flagos, the description could be better"
Actions:
  1. Read existing SKILL.md and all references
  2. Analyze description — find it's too long and not trigger-friendly
  3. Propose a more concise, trigger-optimized description
  4. Create test prompts to verify triggering
  5. Apply changes, validate
Result: Improved skill with better triggering accuracy
```

**Example 3: Validate all skills**
```
User says: "/skill-creator-flagos--validate"
Actions:
  1. No skill_name provided → validate all skills in skills/
  2. Run validate_skill.py on each skill directory
  3. Report consolidated results
Result: Validation report for all skills
```

**Example 4: Turn a conversation workflow into a skill**
```
User says: "turn what we just did into a skill"
Actions:
  1. Analyze conversation history for the workflow performed
  2. Extract: tools used, sequence of steps, corrections made
  3. Interview user to fill gaps
  4. Scaffold and write the skill
Result: New skill capturing the conversation workflow
```

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `name does not match directory name` | Frontmatter `name` differs from folder name | Ensure they are identical |
| `description exceeds 1024 chars` | Description too long | Move details to the body; keep description to 1-2 sentences |
| `SKILL.md body is too short` | Insufficient instructions | Add overview, steps, examples, troubleshooting |
| `Missing required field` | `name` or `description` absent in frontmatter | Add the missing field |
| `name must be lowercase+hyphens` | Invalid characters in name | Use only `a-z`, `0-9`, `-` |
| Skill doesn't trigger | Description too narrow or vague | Make description more specific and slightly aggressive about triggering |
| Referenced file not found | Script or reference listed in SKILL.md but not on disk | Create the file or remove the reference |
