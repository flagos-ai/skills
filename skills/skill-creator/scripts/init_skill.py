#!/usr/bin/env python3
"""Initialize a new skill directory from template.

Usage:
    init_skill.py <skill-name> --path <skills-dir> [--resources scripts,references,assets]

Examples:
    init_skill.py preflight-check --path skills/
    init_skill.py my-skill --path skills/ --resources scripts,references
    init_skill.py my-skill --path /absolute/path/to/skills --resources scripts,references,assets
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

MAX_NAME_LEN = 64
NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9\-]*[a-z0-9]$")
ALLOWED_RESOURCES = {"scripts", "references", "assets"}

LICENSE_TEXT = """\

                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   Copyright 2026 BAAI (Beijing Academy of Artificial Intelligence)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""


def normalize_name(name: str) -> str:
    """Normalize a skill name to lowercase hyphen-case."""
    normalized = name.strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "-", normalized)
    normalized = normalized.strip("-")
    return normalized


def make_title(name: str) -> str:
    """Convert hyphen-case name to Title Case."""
    return " ".join(word.capitalize() for word in name.split("-"))


def generate_skill_md(name: str, title: str) -> str:
    return f"""---
name: {name}
description: >
  [TODO: Replace with a description of the skill and when the agent should use it.
  This should be one or two complete sentences explaining what the skill does
  and what triggers it. Be specific about trigger phrases and contexts.]
argument-hint: "[TODO: argument hints]"
user-invokable: true
compatibility: "[TODO: Describe environment requirements]"
metadata:
  version: "1.0.0"
  author: flagos-ai
  category: [TODO: replace-category]
  tags: [TODO, replace, with, relevant, tags]
allowed-tools: "Bash(python3:*) Read Edit Write Glob Grep AskUserQuestion TaskCreate TaskUpdate TaskList TaskGet"
---

# {title}

## Overview

[TODO: Brief explanation of:]
- What problem this skill solves
- When it should be activated
- Expected inputs and outputs

## Prerequisites

- [TODO: List environment requirements]
- [TODO: Required tools, packages, or access]

## Execution

### Step 1: [TODO: First action]

[TODO: Explain what to do, with executable examples]

```bash
command --example
```

**-> Tell user**: [TODO: Status update]

### Step 2: [TODO: Second action]

[TODO: Continue the workflow]

**-> Tell user**: [TODO: Status update]

### Step 3: [TODO: Verification]

[TODO: Verify the result]

```bash
verification_command
```

**-> Tell user**: Report results. On failure, diagnose and fix.

## Examples

**Example 1: Typical usage**
```
User says: "/{name} argument"
Actions:
  1. Parse input
  2. Execute workflow
  3. Verify result
Result: [TODO: Description of expected outcome]
```

**Example 2: Edge case**
```
User says: "[TODO: alternative trigger phrase]"
Actions:
  1. Handle the edge case
  2. Adapt workflow accordingly
Result: [TODO: Description of expected outcome]
```

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| [TODO: Common error 1] | [TODO: Typical cause] | [TODO: How to fix] |
| [TODO: Common error 2] | [TODO: Typical cause] | [TODO: How to fix] |
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialize a new skill directory")
    parser.add_argument("name", help="Skill name (lowercase + hyphens)")
    parser.add_argument("--path", required=True, help="Parent directory for the skill")
    parser.add_argument(
        "--resources",
        default="",
        help="Comma-separated list of resource dirs to create: scripts,references,assets",
    )
    args = parser.parse_args()

    name = normalize_name(args.name)

    # Validate name
    if len(name) < 2:
        print(f"Error: name '{name}' is too short (min 2 chars)")
        return 1
    if len(name) > MAX_NAME_LEN:
        print(f"Error: name '{name}' exceeds {MAX_NAME_LEN} chars")
        return 1
    if not NAME_PATTERN.match(name):
        print(f"Error: name '{name}' must be lowercase letters, digits, and hyphens only")
        return 1

    # Parse resources
    resources = set()
    if args.resources:
        for r in args.resources.split(","):
            r = r.strip()
            if r and r not in ALLOWED_RESOURCES:
                print(f"Error: unknown resource type '{r}'. Allowed: {', '.join(sorted(ALLOWED_RESOURCES))}")
                return 1
            if r:
                resources.add(r)

    # Create directory
    parent = Path(args.path)
    if not parent.exists():
        print(f"Error: parent directory does not exist: {parent}")
        return 1

    skill_dir = parent / name
    if skill_dir.exists():
        print(f"Error: directory already exists: {skill_dir}")
        return 1

    title = make_title(name)

    # Create skill directory and files
    skill_dir.mkdir(parents=True)
    print(f"Created: {skill_dir}/")

    # SKILL.md
    (skill_dir / "SKILL.md").write_text(generate_skill_md(name, title), encoding="utf-8")
    print(f"Created: {skill_dir}/SKILL.md")

    # LICENSE.txt
    (skill_dir / "LICENSE.txt").write_text(LICENSE_TEXT, encoding="utf-8")
    print(f"Created: {skill_dir}/LICENSE.txt")

    # Resource directories
    for resource in sorted(resources):
        res_dir = skill_dir / resource
        res_dir.mkdir()
        (res_dir / ".gitkeep").write_text("", encoding="utf-8")
        print(f"Created: {skill_dir}/{resource}/")

    print(f"\nSkill '{name}' initialized successfully!")
    print(f"Next steps:")
    print(f"  1. Edit {skill_dir}/SKILL.md — fill in the [TODO] sections")
    print(f"  2. Add supporting files to scripts/, references/, assets/ as needed")
    print(f"  3. Run: python3 validate_skill.py {skill_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
