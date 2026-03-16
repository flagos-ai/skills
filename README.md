> **FlagOS** is a fully open-source AI system software stack for heterogeneous AI chips, allowing AI models to be developed once and seamlessly ported to a wide range of AI hardware with minimal effort. This repository collects reusable **Skills** for FlagOS — injecting domain knowledge, workflow standards, and best practices into AI coding agents.
>
> [中文版](README_zh.md)

## What are Skills?

Skills are **folder-based capability packages**: each skill uses documentation, scripts, and resources to teach agents to reliably and reproducibly complete tasks in a specific domain. Each skill folder contains a `SKILL.md` file with YAML frontmatter (name + description) followed by detailed agent instructions. Skills can also include reference docs, scripts, and assets.

This repository follows the [Agent Skills open standard](https://agentskills.io/specification).

## Quick Start

FlagOS Skills are compatible with **Claude Code**, **Cursor**, **Codex**, and any agent supporting the [Agent Skills standard](https://agentskills.io/specification).

### npx (Recommended — works with all agents)

Use the [`skills`](https://www.npmjs.com/package/skills) CLI to install skills directly — no cloning needed:

```bash
# List available skills in this repository
npx skills add flagos-ai/skills --list

# Install a specific skill into your project
npx skills add flagos-ai/skills --skill model-migrate-flagos

# Install a specific skill globally (user-level)
npx skills add flagos-ai/skills --skill model-migrate-flagos --global

# Install all skills at once
npx skills add flagos-ai/skills --all

# Install for specific agents only
npx skills add flagos-ai/skills --agent claude-code cursor
```

Other useful commands:

```bash
npx skills list              # List installed skills
npx skills find              # Search for skills interactively
npx skills update            # Update all skills to latest versions
npx skills remove            # Interactive remove
```

> **Note:** No prior installation needed — `npx` downloads the [`skills`](https://skills.sh/) CLI automatically.

### Claude Code

1. Register the repository as a plugin marketplace (in Claude Code interactive mode):

```
/plugin marketplace add flagos-ai/skills
```

Or from the terminal:

```bash
claude plugin marketplace add flagos-ai/skills
```

2. Install skills:

```
/plugin install flagos-skills@flagos-skills
```

Or from the terminal:

```bash
claude plugin install flagos-skills@flagos-skills
```

After installation, mention the skill in your prompt — Claude automatically loads the corresponding `SKILL.md` instructions.

### Cursor

This repository includes Cursor plugin manifests (`.cursor-plugin/plugin.json` and `.cursor-plugin/marketplace.json`).

Install from the repository URL or local checkout via the Cursor plugin flow.

### Codex

Use the `$skill-installer` inside Codex:

```
$skill-installer install model-migrate-flagos from flagos-ai/skills
```

Or provide the GitHub directory URL:

```
$skill-installer install https://github.com/flagos-ai/skills/tree/main/skills/model-migrate-flagos
```

Alternatively, copy skill folders into Codex's standard `.agents/skills` location:

```bash
cp -r skills/model-migrate-flagos $REPO_ROOT/.agents/skills/
```

See the [Codex Skills guide](https://developers.openai.com/codex/skills/) for more details.

### Gemini CLI

```bash
gemini extensions install https://github.com/flagos-ai/skills.git --consent
```

This repo includes `gemini-extension.json` and `agents/AGENTS.md` for Gemini CLI integration. See [Gemini CLI extensions docs](https://geminicli.com/docs/extensions/) for more help.

### Manual / Other Agents

For any agent that supports the [Agent Skills standard](https://agentskills.io/specification), point it at the `skills/` directory in this repository. Each skill is self-contained with a `SKILL.md` entry point. The `agents/AGENTS.md` file can also be used as a fallback for agents that don't support skills natively.

## Available Skills

<!-- BEGIN_SKILLS_TABLE -->
| Name | Category | Description | Docs |
|------|----------|-------------|------|
| [`model-migrate-flagos`](skills/model-migrate-flagos/) | workflow-automation | Migrate a model from upstream vLLM into the vllm-plugin-FL project. Use when adding new model support, porting model code, or backporting newly released models. | [SKILL.md](skills/model-migrate-flagos/SKILL.md) |
<!-- END_SKILLS_TABLE -->

### Using skills in your agent

Once a skill is installed, mention it directly in your prompt:

- "Use model-migrate-flagos to migrate the Qwen3-5 model from upstream vLLM"
- "/model-migrate-flagos qwen3_5"
- "Port the DeepSeek-V4 model to vllm-plugin-FL"

Your agent automatically loads the corresponding `SKILL.md` instructions and helper scripts.

## Repository Structure

```
├── .claude-plugin/          # Claude Code plugin manifest
│   └── marketplace.json
├── .cursor-plugin/          # Cursor plugin manifest
│   ├── marketplace.json
│   └── plugin.json
├── agents/                  # Codex / Gemini CLI fallback
│   └── AGENTS.md
├── assets/                  # Repository-level static resources
├── contributing.md          # Contribution guidelines
├── gemini-extension.json    # Gemini CLI extension manifest
├── scripts/                 # Repository-level utility scripts
│   └── validate_skills.py   # Batch validate all skills
├── skills/                  # Skill directories
│   ├── model-migrate-flagos/    # Model migration workflow
│   └── ...
├── spec/                    # Agent Skills standard & local conventions
│   ├── README.md
│   └── agent-skills-spec.md
└── template/                # Template for creating new skills
    └── SKILL.md
```

## Creating a New Skill

1. **Create directory & copy template**
   ```bash
   mkdir skills/<skill-name>
   cp template/SKILL.md skills/<skill-name>/SKILL.md
   ```

2. **Edit frontmatter** — `name` (lowercase + hyphens, must match directory name) and `description` (what it does + when to trigger)

3. **Write the body** — Overview, Prerequisites, Execution steps, Examples (2-3), Troubleshooting

4. **Add supporting files** (optional) — `references/`, `scripts/`, `assets/`, `LICENSE.txt`

5. **Validate**
   ```bash
   python scripts/validate_skills.py
   ```

See [contributing.md](contributing.md) for the full contribution guide.

## Skill Categories

| Category | Description | Examples |
|----------|-------------|----------|
| **workflow-automation** | Multi-step processes: model migration, vendor onboarding, E2E validation | `model-migrate-flagos` |
| **deployment** | Environment checks, container builds, multi-chip CI | — |
| **enterprise-standards** | Brand guidelines, doc templates, code standards | — |
| **tool-integration** | CI/CD, monitoring, internal platforms, third-party SaaS | — |

## License

[Apache License 2.0](LICENSE)
