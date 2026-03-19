# Contributing to FlagOS Skills

Thanks for your interest in contributing to FlagOS Skills! This guide will help you get started.

## Getting Started

### Creating a New Skill

```bash
# 1. Create directory
mkdir skills/<skill-name>

# 2. Copy template
cp template/SKILL.md skills/<skill-name>/SKILL.md

# 3. Edit SKILL.md
#    - Update frontmatter (name, description, etc.)
#    - Write agent instructions
#    - Add examples and troubleshooting guide

# 4. (Optional) Add supporting files
mkdir skills/<skill-name>/references  # Reference docs
mkdir skills/<skill-name>/scripts     # Scripts
mkdir skills/<skill-name>/assets      # Resources

# 5. Validate
python scripts/validate_skills.py
```

### Updating an Existing Skill

1. Find the corresponding `skills/<skill-name>/SKILL.md`
2. Make your changes
3. If adding new scripts/resources, document their usage in SKILL.md
4. Run validation

## Skill Quality Checklist

Before submitting a PR, please verify:

- [ ] `SKILL.md` contains complete YAML frontmatter (at least `name` + `description`)
- [ ] `name` field matches the directory name, all lowercase with hyphens
- [ ] `description` is a complete sentence explaining "what it does" + "when to trigger"
- [ ] Body includes: Overview, Prerequisites, Execution steps, Examples (2-3), Troubleshooting
- [ ] All referenced scripts/resource files actually exist
- [ ] Scripts have execute permissions (`chmod +x`)
- [ ] No hardcoded internal paths, secrets, or credentials

## Naming Conventions

| Scope | Rule | Example |
|-------|------|---------|
| Directory name | Lowercase + hyphens | `model-migrate-flagos` |
| `name` field | Must match directory name | `model-migrate-flagos` |
| Script filenames | Lowercase + underscores | `validate_migration.py` |
| Reference docs | Lowercase + hyphens | `compatibility-patches.md` |

Avoid:
- Vague names: `misc`, `utils`, `tmp`, `v1`, `test`
- Overly long names: keep to 3-4 words

## Skill Categories

See the [Skills Catalog](README.md#skills-catalog) in README.md for the full category hierarchy. Major categories:

| Category | Description | Example |
|----------|-------------|---------|
| Inference & Serving | Model migration, serving deployment, preflight checks | `model-migrate-flagos` |
| Training & RLHF | Training adaptation, RLHF pipelines | — |
| Operator & Compiler | TLE primitive dev, operator optimization, kernel generation, operator diagnosis, compiler backend adaptation | — |
| Communication | Cross-chip collective ops (FlagCX) | — |
| Benchmarking & Eval | FlagPerf benchmarks, E2E accuracy verification | — |
| Environment & Deployment | Stack installation, base image selection, container builds, CI pipelines | — |
| Developer Tooling | Skill development, chip vendor onboarding | `skill-creator-flagos` |

## PR Template

```markdown
### Change Type
<!-- New Skill | Update Skill | Fix | Infrastructure -->

### Skill Name
<!-- e.g., model-migrate-flagos -->

### Description
<!-- Brief description of changes -->

### Testing
<!-- How to verify this skill works correctly -->
- [ ] Tested the skill in Claude Code
- [ ] Ran `python scripts/validate_skills.py`
```

## License

- Repository default license: Apache License 2.0
- If a skill has special licensing requirements, place a `LICENSE.txt` in its directory
- Third-party dependencies must be noted in SKILL.md with source and license information
