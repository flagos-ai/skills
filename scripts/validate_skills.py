#!/usr/bin/env python3
"""Validate skills in the repository.

Usage:
    validate_skills.py                             # Validate all skills
    validate_skills.py <skill-dir>                 # Validate a single skill
    validate_skills.py <skills-parent-dir> --all   # Validate all skills in directory

Examples:
    python3 scripts/validate_skills.py
    python3 scripts/validate_skills.py skills/model-migrate-fl
    python3 scripts/validate_skills.py skills/ --all
"""
from __future__ import annotations

import argparse
import os
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
    errors: list[str] = []
    fields: dict[str, str] = {}

    if not text.startswith("---"):
        errors.append("Missing YAML frontmatter (must start with ---)")
        return fields, errors

    parts = text.split("---", 2)
    if len(parts) < 3:
        errors.append("Malformed frontmatter (missing closing ---)")
        return fields, errors

    fm_text = parts[1].strip()
    lines = fm_text.split("\n")
    current_key: str | None = None
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
        if ":" in line and line[0] not in (" ", "\t"):
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


def find_referenced_files(text: str) -> list[str]:
    """Extract local file paths referenced in markdown links."""
    refs: list[str] = []
    for match in re.finditer(r"\[([^\]]*)\]\(([^)]+)\)", text):
        path = match.group(2)
        if not path.startswith("http") and not path.startswith("#"):
            refs.append(path)
    return refs


def validate_skill(skill_dir: Path) -> tuple[list[str], list[str]]:
    """Validate a single skill directory. Returns (errors, warnings)."""
    errors: list[str] = []
    warnings: list[str] = []
    skill_md = skill_dir / "SKILL.md"

    if not skill_md.exists():
        errors.append(f"{skill_dir.name}: missing SKILL.md")
        return errors, warnings

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
    if desc and len(desc) < 20:
        warnings.append(f"{skill_dir.name}: description is very short — consider adding trigger phrases")

    # Check for body content
    body = text.split("---", 2)[-1].strip() if text.startswith("---") else text
    if len(body) < 100:
        errors.append(f"{skill_dir.name}: SKILL.md body is too short (< 100 chars)")

    # Check for recommended sections
    body_lower = body.lower()
    for section in ["example", "troubleshoot"]:
        if section not in body_lower:
            warnings.append(f"{skill_dir.name}: no '{section}' section found (recommended)")

    # Check referenced files exist
    refs = find_referenced_files(body)
    for ref in refs:
        ref_path = skill_dir / ref
        if not ref_path.exists():
            errors.append(f"{skill_dir.name}: referenced file not found: {ref}")

    # Check scripts have execute permission
    scripts_dir = skill_dir / "scripts"
    if scripts_dir.exists():
        for script in scripts_dir.iterdir():
            if script.is_file() and script.suffix in (".py", ".sh", ".bash"):
                if not os.access(script, os.X_OK):
                    warnings.append(
                        f"{skill_dir.name}: script lacks execute permission: scripts/{script.name}"
                    )

    # Check for LICENSE
    if not (skill_dir / "LICENSE.txt").exists():
        warnings.append(f"{skill_dir.name}: no LICENSE.txt found (recommended)")

    # Check for README
    if not (skill_dir / "README.md").exists():
        warnings.append(f"{skill_dir.name}: no README.md found (recommended)")

    return errors, warnings


def print_results(skill_dirs: list[Path], all_errors: list[str], all_warnings: list[str]) -> None:
    """Print detailed results with error/warning breakdown."""
    for skill_dir in skill_dirs:
        prefix = f"{skill_dir.name}: "
        skill_errors = [e for e in all_errors if e.startswith(prefix) or e.startswith(f"{skill_dir.name}:")]
        skill_warnings = [w for w in all_warnings if w.startswith(prefix) or w.startswith(f"{skill_dir.name}:")]

        status = "PASS" if not skill_errors else "FAIL"
        print(f"\n{'='*60}")
        print(f"  {skill_dir.name}: {status}")
        print(f"{'='*60}")

        if skill_errors:
            print(f"  Errors ({len(skill_errors)}):")
            for e in skill_errors:
                # Strip prefix for cleaner output
                msg = e[len(f"{skill_dir.name}: "):] if e.startswith(f"{skill_dir.name}: ") else e
                print(f"    ✗ {msg}")

        if skill_warnings:
            print(f"  Warnings ({len(skill_warnings)}):")
            for w in skill_warnings:
                msg = w[len(f"{skill_dir.name}: "):] if w.startswith(f"{skill_dir.name}: ") else w
                print(f"    ⚠ {msg}")

        if not skill_errors and not skill_warnings:
            print("    ✓ All checks passed")

    print(f"\n{'─'*60}")
    print(f"  Summary: {len(skill_dirs)} skill(s), {len(all_errors)} error(s), {len(all_warnings)} warning(s)")
    print(f"{'─'*60}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate skills in the FlagOS skills repository",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="When called with no arguments, validates all skills under skills/.",
    )
    parser.add_argument(
        "path", nargs="?", default=None,
        help="Skill directory or parent directory (with --all). Defaults to skills/.",
    )
    parser.add_argument("--all", action="store_true", help="Validate all skills in directory")
    args = parser.parse_args()

    # Determine target directory and mode
    if args.path is None:
        # No arguments: validate all skills under SKILLS_DIR
        target = SKILLS_DIR
        scan_all = True
    elif args.all:
        target = Path(args.path)
        scan_all = True
    else:
        target = Path(args.path)
        scan_all = False

    if not target.exists():
        print(f"Error: path does not exist: {target}")
        return 1

    if scan_all:
        skill_dirs = [
            d
            for d in sorted(target.iterdir())
            if d.is_dir() and not d.name.startswith(".")
        ]
        if not skill_dirs:
            print("No skill directories found.")
            return 0
    else:
        skill_dirs = [target]

    all_errors: list[str] = []
    all_warnings: list[str] = []
    for skill_dir in skill_dirs:
        errors, warnings = validate_skill(skill_dir)
        all_errors.extend(errors)
        all_warnings.extend(warnings)

    print_results(skill_dirs, all_errors, all_warnings)
    return 1 if all_errors else 0


if __name__ == "__main__":
    sys.exit(main())
