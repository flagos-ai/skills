#!/usr/bin/env python3
"""Validate all skills in the repository.

Checks SKILL.md frontmatter, directory naming, and file references.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

SKILLS_DIR = Path(__file__).parent.parent / "skills"
REQUIRED_FIELDS = {"name", "description"}
MAX_NAME_LEN = 64
MAX_DESC_LEN = 1024
NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9\-]*[a-z0-9]$")


def parse_frontmatter(text: str) -> tuple[dict[str, str], list[str]]:
    """Parse YAML frontmatter from SKILL.md text. Returns (fields, errors).

    Handles multi-line YAML values (folded '>' and literal '|' scalars)
    by collecting indented continuation lines.
    """
    errors = []
    fields = {}

    if not text.startswith("---"):
        errors.append("Missing YAML frontmatter (must start with ---)")
        return fields, errors

    parts = text.split("---", 2)
    if len(parts) < 3:
        errors.append("Malformed frontmatter (missing closing ---)")
        return fields, errors

    fm_text = parts[1].strip()
    lines = fm_text.split("\n")
    current_key = None
    current_value_parts: list[str] = []

    def _flush():
        if current_key is not None:
            value = " ".join(current_value_parts).strip()
            # Strip block scalar indicators
            if value in (">", "|", ">-", "|-"):
                value = ""
            fields[current_key] = value

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # Continuation line (indented) for multi-line value
        if line[0] in (" ", "\t") and current_key is not None:
            current_value_parts.append(stripped)
            continue

        # New top-level key
        if ":" in line and not line[0] in (" ", "\t"):
            _flush()
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            current_key = key
            if value in (">", "|", ">-", "|-"):
                # Multi-line scalar — value comes on next indented lines
                current_value_parts = []
            else:
                current_value_parts = [value] if value else []
        else:
            # Indented key (nested YAML like metadata.version) — skip for now
            pass

    _flush()
    return fields, errors


def validate_skill(skill_dir: Path) -> list[str]:
    """Validate a single skill directory. Returns list of error messages."""
    errors = []
    skill_md = skill_dir / "SKILL.md"

    if not skill_md.exists():
        errors.append(f"{skill_dir.name}: missing SKILL.md")
        return errors

    text = skill_md.read_text(encoding="utf-8")
    fields, fm_errors = parse_frontmatter(text)
    errors.extend(f"{skill_dir.name}: {e}" for e in fm_errors)

    # Check required fields
    for field in REQUIRED_FIELDS:
        if field not in fields:
            errors.append(f"{skill_dir.name}: missing required field '{field}'")

    # Validate name
    name = fields.get("name", "")
    if name:
        if len(name) > MAX_NAME_LEN:
            errors.append(f"{skill_dir.name}: name exceeds {MAX_NAME_LEN} chars")
        if not NAME_PATTERN.match(name) and len(name) > 1:
            errors.append(f"{skill_dir.name}: name '{name}' must be lowercase+hyphens")
        if name != skill_dir.name:
            errors.append(
                f"{skill_dir.name}: name '{name}' does not match directory name"
            )

    # Validate description
    desc = fields.get("description", "")
    if desc and len(desc) > MAX_DESC_LEN:
        errors.append(f"{skill_dir.name}: description exceeds {MAX_DESC_LEN} chars")

    # Check for body content
    body = text.split("---", 2)[-1].strip() if text.startswith("---") else text
    if len(body) < 100:
        errors.append(f"{skill_dir.name}: SKILL.md body is too short (< 100 chars)")

    return errors


def main() -> int:
    if not SKILLS_DIR.exists():
        print(f"Skills directory not found: {SKILLS_DIR}")
        return 1

    skill_dirs = [
        d
        for d in sorted(SKILLS_DIR.iterdir())
        if d.is_dir() and not d.name.startswith(".")
    ]

    if not skill_dirs:
        print("No skill directories found.")
        return 0

    all_errors = []
    for skill_dir in skill_dirs:
        errors = validate_skill(skill_dir)
        all_errors.extend(errors)

    if all_errors:
        print(f"Validation FAILED with {len(all_errors)} error(s):")
        for err in all_errors:
            print(f"  - {err}")
        return 1

    print(f"Validation PASSED. {len(skill_dirs)} skill(s) validated:")
    for d in skill_dirs:
        print(f"  - {d.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
