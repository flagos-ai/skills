# Skill Writing Guide

Detailed patterns and best practices for writing effective skills in the FlagOS skills repository.

## Table of Contents

- [Anatomy of a Skill](#anatomy-of-a-skill)
- [Progressive Disclosure](#progressive-disclosure)
- [Writing the Description](#writing-the-description)
- [Structuring the Body](#structuring-the-body)
- [Writing Style](#writing-style)
- [Scripts and References](#scripts-and-references)
- [FlagOS-Specific Conventions](#flagos-specific-conventions)

---

## Anatomy of a Skill

```
skill-name/
├── SKILL.md (required)
│   ├── YAML frontmatter (name, description required)
│   └── Markdown instructions
├── LICENSE.txt (recommended, Apache 2.0)
├── README.md (recommended, English)
├── README_zh.md (optional, Chinese)
└── Bundled Resources (optional)
    ├── scripts/    - Executable code for deterministic/repetitive tasks
    ├── references/ - Docs loaded into context as needed
    └── assets/     - Files used in output (templates, icons, fonts)
```

### Required Fields

| Field | Constraints | Description |
|-------|-------------|-------------|
| `name` | ≤64 chars, lowercase + hyphens, matches directory | Unique identifier |
| `description` | ≤1024 chars | What the skill does AND when to trigger |

### FlagOS Extension Fields

| Field | Description | Example |
|-------|-------------|---------|
| `argument-hint` | CLI argument hint | `"model_name [options]"` |
| `user-invokable` | Can be invoked via `/skill-name` | `true` |
| `allowed-tools` | Fine-grained tool access control | `"Bash(python3:*) Read Edit ..."` |
| `compatibility` | Environment requirements | `"Python 3.8+, CUDA required"` |
| `metadata.version` | Skill version (semver) | `"1.0.0"` |
| `metadata.author` | Author | `"flagos-ai"` |
| `metadata.category` | Category | `"workflow-automation"` |
| `metadata.tags` | Tag list | `[model-migration, vllm]` |

---

## Progressive Disclosure

Skills use a three-level loading system to manage context efficiently:

1. **Metadata** (name + description) — Always in context (~100 words)
2. **SKILL.md body** — Loaded when skill triggers (<500 lines ideal)
3. **Bundled resources** — Loaded as needed (unlimited; scripts can execute without loading)

### Key Principles

- **Keep SKILL.md under 500 lines.** If approaching this limit, split content into `references/` files with clear pointers about when to read them.
- **Reference files must be documented in SKILL.md** with guidance on when to read them.
- **For reference files >300 lines**, include a table of contents at the top.
- **Avoid deeply nested references** — keep one level deep from SKILL.md.

### Patterns

**Pattern 1: Domain-specific organization**

When a skill supports multiple domains/frameworks, organize by variant:

```
cloud-deploy/
├── SKILL.md (workflow + selection)
└── references/
    ├── aws.md
    ├── gcp.md
    └── azure.md
```

The agent reads only the relevant reference file.

**Pattern 2: Workflow with reference catalog**

```
model-migrate-fl/
├── SKILL.md (overview + step index)
└── references/
    ├── procedure.md (detailed 13-step procedure)
    ├── compatibility-patches.md (patch catalog)
    └── operational-rules.md (operational constraints)
```

SKILL.md provides the overview; detailed content lives in references.

**Pattern 3: Conditional details**

```markdown
## Basic usage
Simple inline instructions here.

**For advanced configuration**: See [advanced-config.md](references/advanced-config.md)
**For troubleshooting**: See [troubleshooting.md](references/troubleshooting.md)
```

---

## Writing the Description

The `description` field is the **primary triggering mechanism**. It determines when the agent activates the skill.

### Rules

1. **Include both WHAT and WHEN** — what the skill does and what contexts trigger it.
2. **Be specific about triggers** — list user phrases, file types, or task patterns.
3. **Be slightly "pushy"** — err on triggering too often rather than too rarely. The agent tends to under-trigger.
4. **Include negative triggers** — explicitly say what NOT to use this skill for.
5. **Keep under 1024 chars** — but use the full allowance if needed.

### Good Example

```yaml
description: >
  Migrate a model from the latest vLLM upstream repository into the vllm-plugin-FL project
  (pinned at vLLM v0.13.0). Use this skill whenever someone wants to add support for a new
  model to vllm-plugin-FL, port model code from upstream vLLM, or backport a newly released
  model. Trigger when the user says things like "migrate X model", "add X model support",
  "port X from upstream vLLM". Do NOT use for models already supported by vLLM 0.13.0 core.
```

### Bad Example

```yaml
description: Helps with model migration.
```

Too vague — the agent won't know when to trigger.

---

## Structuring the Body

### Required Sections

Every SKILL.md body should include:

1. **Overview** — what problem this solves, when to activate, expected inputs/outputs
2. **Prerequisites** (if any) — environment requirements, tools, access
3. **Execution steps** — numbered steps with progress reporting
4. **Examples** — at least 2-3 realistic usage examples
5. **Troubleshooting** — common problems and fixes

### Execution Steps Pattern

Use numbered steps with `**-> Tell user**` markers for progress reporting:

```markdown
### Step 1: Parse arguments

Extract from user input:
- `{{model_name}}` = first argument (required)
- `{{options}}` = remaining arguments

**-> Tell user**: Confirm parsed values.

### Step 2: Execute workflow

Do the actual work here.

```bash
command --example
```

**-> Tell user**: Report progress.

### Step 3: Verify

```bash
verification_command
```

**-> Tell user**: Report results. On failure, diagnose and fix.
```

### Examples Pattern

```markdown
## Examples

**Example 1: Typical usage**
```
User says: "/skill-name argument"
Actions:
  1. Parse input
  2. Execute workflow
  3. Verify result
Result: Description of expected outcome
```

**Example 2: Edge case**
```
User says: "alternative trigger phrase"
Actions:
  1. Handle the edge case
  2. Adapt workflow accordingly
Result: Description of expected outcome
```
```

### Troubleshooting Pattern

Use a table for common issues:

```markdown
## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| Common error 1 | Typical cause | How to fix |
| Common error 2 | Typical cause | How to fix |
```

---

## Writing Style

### General Principles

- **Use imperative form** for instructions ("Run the script", not "You should run the script").
- **Explain WHY, not just WHAT** — help the agent understand the reasoning so it can adapt.
- **Be concise** — the context window is a shared resource. Only add information the agent doesn't already have.
- **Prefer examples over explanations** — a concrete example communicates faster than an abstract description.
- **Don't over-constrain** — use MUSTs sparingly. Give the agent enough freedom to adapt to unexpected situations.

### What NOT to Include

- Information the agent already knows (common programming patterns, standard library usage)
- Redundant explanations of well-known tools
- Overly detailed error messages for obvious failures
- Commentary about the skill creation process itself

### Conciseness Checklist

For each paragraph, ask:
- Does the agent really need this explanation?
- Does this paragraph justify its token cost?
- Could this be replaced by a short example?
- Is this duplicated elsewhere in the skill or references?

---

## Scripts and References

### When to Use Scripts

Use `scripts/` when:
- The same code would be rewritten repeatedly
- Deterministic reliability is needed (validation, data processing)
- Complex operations need to be reproducible

Scripts can be executed without loading into context, saving tokens.

### When to Use References

Use `references/` when:
- Detailed documentation needs to be available but not always in context
- Multiple variants/domains need separate detailed guides
- Content exceeds what fits comfortably in SKILL.md

### Documentation Requirements

Every script and reference file must be documented in SKILL.md:

```markdown
## Scripts Reference

| Script | Step | Description |
|---|---|---|
| `validate_migration.py` | 6 | Automated import/API/registration checks |
| `benchmark.sh` | 9 | `vllm bench throughput` with dummy weights |
```

---

## FlagOS-Specific Conventions

### Naming

- Directory names: lowercase + hyphens (`model-migrate-fl`, `preflight-check`)
- Script filenames: lowercase + underscores (`validate_migration.py`)
- Reference docs: lowercase + hyphens (`compatibility-patches.md`)
- FlagOS skills use functional prefixes describing the action

### Categories

| Category | Description | Example |
|---|---|---|
| `workflow-automation` | Multi-step workflows | model-migrate-fl |
| `deployment-verification` | Deployment & environment validation | preflight-check |
| `build-tooling` | Build & release tools | build-vendor-image |
| `code-standard` | Coding standards & review | — |
| `operations` | Operational tasks | — |
| `developer-tooling` | Developer productivity tools | skill-creator |

### Bilingual Support

FlagOS skills should provide:
- `README.md` — English documentation
- `README_zh.md` — Chinese documentation (recommended)

Both READMEs follow the same structure but are written independently (not machine-translated).

### License

- Default: Apache License 2.0
- Place `LICENSE.txt` in each skill directory
- Third-party dependencies must be noted in SKILL.md with source and license information
