---
name: template-skill
description: >
  Replace with a description of the skill and when Claude should use it.
  This should be one or two complete sentences explaining what the skill does
  and what triggers it.
user-invokable: true
compatibility: "Describe environment requirements here"
metadata:
  version: "1.0.0"
  author: flagos-ai
  category: replace-category
  tags: [replace, with, relevant, tags]
allowed-tools: "Bash(python3:*) Read Edit Write Glob Grep AskUserQuestion TaskCreate TaskUpdate TaskList TaskGet"
---

# Skill Name

## Overview

Brief explanation of:
- What problem this skill solves
- When it should be activated
- Expected inputs and outputs

## Prerequisites

- List environment requirements
- Required tools, packages, or access

## Execution

### Step 1: First action

> **-> Tell user**: Status update

Explain what to do, with executable examples:

```bash
command --example
```

### Step 2: Second action

> **-> Tell user**: Status update

Continue the workflow.

### Step 3: Verification

Verify the result:

```bash
verification_command
```

**-> Tell user**: Report results. On failure, diagnose and fix.

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

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| Common error 1 | Typical cause | How to fix |
| Common error 2 | Typical cause | How to fix |
