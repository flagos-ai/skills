# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is the **FlagOS Skills** repository — a collection of reusable **Agent Skills** for FlagOS, a fully open-source AI system software stack for heterogeneous AI chips. Skills are folder-based capability packages that teach agents to reliably perform domain-specific tasks (model migration, deployment verification, complex workflows).

This repository follows the [Agent Skills open standard](https://agentskills.io/specification), aligned with [anthropics/skills](https://github.com/anthropics/skills), [huggingface/skills](https://github.com/huggingface/skills), and [openai/skills](https://github.com/openai/skills).

There is no build system or package manager. This is a pure content repository of Markdown-based skill definitions with supporting scripts and resources.

## Repository Structure

```
├── .claude-plugin/          # Claude Code plugin manifest
│   └── marketplace.json     # Plugin bundles for Claude Code marketplace
├── .cursor-plugin/          # Cursor plugin manifest
│   ├── marketplace.json     # Cursor marketplace listing
│   └── plugin.json          # Plugin metadata
├── agents/                  # Codex / Gemini CLI fallback
│   └── AGENTS.md            # Skills index for agents without native support
├── assets/                  # Repository-level static resources
├── contributing.md          # Contribution guidelines
├── gemini-extension.json    # Gemini CLI extension manifest
├── scripts/                 # Repository-level utility scripts
│   └── validate_skills.py   # Batch validate all skills
├── skills/                  # Actual skill directories
│   ├── model-migrate-flagos/    # Example: model migration workflow
│   └── ...
├── spec/                    # Agent Skills standard & local conventions
│   ├── README.md            # Conventions and cross-reference to standards
│   └── agent-skills-spec.md # Redirect to agentskills.io/specification
└── template/                # Template for creating new skills
    └── SKILL.md
```

## Skill Structure

Each skill is a **directory** under `skills/` containing at minimum a `SKILL.md` file with:

1. **YAML frontmatter** (required fields):
   - `name`: unique identifier, lowercase with hyphens (e.g., `model-migrate-flagos`)
   - `description`: one or two sentences explaining what the skill does and when to trigger it
2. **Optional frontmatter**: `license`, `compatibility`, `metadata`, `allowed-tools`, `user-invokable`, `argument-hint`
3. **Markdown body**: instructions for the agent — applicable scenarios, constraints, steps, and examples (at least 2–3)
4. **Optional subdirectories**: `references/`, `scripts/`, `assets/`, `examples/`

## Creating a New Skill

```bash
mkdir skills/<skill-name>
cp template/SKILL.md skills/<skill-name>/SKILL.md
# Edit the SKILL.md: update frontmatter and write agent instructions
# Validate: python scripts/validate_skills.py
```

See `contributing.md` for full guidelines.

## Naming Conventions

- Directory names and `name` fields: lowercase + hyphens (`my-skill-name`)
- Max 64 characters for `name`
- Avoid vague names like `misc`, `tmp`, `v1`
- FlagOS skills use `-flagos` suffix: `model-migrate-flagos`, `skill-creator-flagos`

## Key Conventions

- Additional scripts/resources can be placed alongside `SKILL.md` but must be documented in the markdown body
- When skills conflict, the more specific/business-contextual skill takes priority
- See `spec/README.md` for detailed conventions and the Agent Skills specification
- Plugin bundles are defined in `.claude-plugin/marketplace.json` and `.cursor-plugin/`
- Run `python scripts/validate_skills.py` before submitting PRs
