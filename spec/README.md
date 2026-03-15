## Agent Skills Specification & Repository Conventions

This directory contains information related to the Agent Skills specification and local conventions adopted by the `flagos-ai/skills` repository.

### Standard

This repository follows the [Agent Skills open standard](https://agentskills.io/specification). Core design patterns:

- Each skill is a self-contained directory
- Each directory must contain a `SKILL.md` file (YAML frontmatter + Markdown body)
- Optional subdirectories: `references/` (documentation), `scripts/` (executables), `assets/` (resources)
- The repository root provides a template (`template/`) and specification notes (`spec/`)

### SKILL.md Standard Fields

Per [agentskills.io/specification](https://agentskills.io/specification):

| Field | Required | Constraints | Description |
|-------|----------|-------------|-------------|
| `name` | Yes | ≤64 chars, lowercase + hyphens, must match directory name | Unique identifier |
| `description` | Yes | ≤1024 chars | What the skill does and when to trigger it |
| `license` | No | License name or file reference | Defaults to repository LICENSE |
| `compatibility` | No | ≤500 chars | Environment requirements |
| `metadata` | No | Arbitrary key-value pairs | Extension metadata |
| `allowed-tools` | No | Space-separated tool list | Experimental |

### FlagOS Extension Fields

This repository uses the following additional fields beyond the standard (see `model-migrate-fl` skill for reference):

| Field | Description | Example |
|-------|-------------|---------|
| `argument-hint` | CLI argument hint | `"model_name [upstream_folder]"` |
| `user-invokable` | Whether the skill can be invoked directly by users | `true` |
| `allowed-tools` | Fine-grained control over tools the skill can use | `"Bash(python3:*) Read Edit ..."` |
| `metadata.version` | Skill version | `"1.0.0"` |
| `metadata.author` | Author | `"flagos-ai"` |
| `metadata.category` | Category | `"workflow-automation"` |
| `metadata.tags` | Tag list | `[model-migration, vllm]` |

### Repository Conventions

#### 1. Directory Structure

```
skills/                          # Repository root
├── .claude-plugin/              # Claude Code plugin manifest
│   └── marketplace.json
├── .cursor-plugin/              # Cursor plugin manifest
│   ├── marketplace.json
│   └── plugin.json
├── .gitignore
├── agents/                      # Codex / Gemini CLI fallback
│   └── AGENTS.md                # Skills index for agents without native support
├── gemini-extension.json        # Gemini CLI extension manifest
├── LICENSE                      # Apache-2.0
├── README.md                    # Repository documentation
├── assets/                      # Repository-level static resources
├── contributing.md              # Contribution guidelines
├── scripts/                     # Repository-level utility scripts
│   └── validate_skills.py       # Batch validate all skills
├── skills/                      # Skill directories
│   ├── <skill-name>/            # One directory per skill
│   │   ├── SKILL.md             # [Required] Entry point
│   │   ├── LICENSE.txt          # [Recommended] Per-skill license
│   │   ├── references/          # [Optional] Reference documentation
│   │   ├── scripts/             # [Optional] Executable scripts
│   │   └── assets/              # [Optional] Icons, templates, etc.
│   └── ...
├── spec/                        # Specification notes
│   ├── README.md                # This file
│   └── agent-skills-spec.md     # Pointer to agentskills.io
└── template/                    # Template for new skills
    └── SKILL.md
```

#### 2. Naming Conventions

- Directory names and `name` fields use lowercase + hyphens: `my-skill-name`
- Avoid vague names (`misc`, `tmp`, `v1`)
- FlagOS-related skills should use functional prefixes: `model-migrate-fl`, `preflight-check`

#### 3. Skill Internal Structure

| File / Directory | Required | Description |
|------------------|----------|-------------|
| `SKILL.md` | Required | Entry point with frontmatter + agent instructions |
| `LICENSE.txt` | Recommended | Per-skill license (defaults to repository Apache-2.0) |
| `references/` | Optional | Detailed reference docs, referenced in SKILL.md |
| `scripts/` | Optional | Executable scripts, with usage documented in SKILL.md |
| `assets/` | Optional | Static resources (icons, templates, sample data) |
| `examples/` | Optional | Usage examples |

#### 4. Quality Standards

- `description` must be a complete sentence answering "what it does" + "when to trigger"
- Markdown body should include at minimum: overview, execution steps, 2-3 examples, troubleshooting guide
- All referenced scripts/resources must be documented in the body with usage instructions
