#!/usr/bin/env python3
"""Update the Skills Catalog table in README.md.

Scans all skills in skills/ directory, parses their SKILL.md frontmatter,
and updates the table between <!-- BEGIN_SKILLS_TABLE --> and <!-- END_SKILLS_TABLE --> markers.

Behavior:
- Scans all skill directories and parses SKILL.md
- Updates/inserts skill rows in correct positions (grouped by category)
- Preserves manually maintained rows (PR links, *Planned* entries)
- Uses markdown table grouping: **Category** on first row, empty for continuation

Usage:
    python scripts/update_skills_catalog.py [--dry-run]
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

SKILLS_DIR = Path(__file__).parent.parent / "skills"
README_PATH = Path(__file__).parent.parent / "README.md"

# Category ordering for the table
CATEGORY_ORDER = [
    "Deployment & Release",
    "Benchmarking & Eval",
    "Kernel & Operator Development",
    "Multi-Chip Backend Onboarding",
    "Developer Tooling",
]


@dataclass
class SkillInfo:
    """Parsed skill information."""
    name: str
    description: str
    directory: str
    category: str = ""
    sub_category: str = ""


@dataclass
class TableRow:
    """Represents a row in the skills table."""
    category: str  # Raw category text (may be empty or **bold**)
    sub_category: str
    skill_link: str
    description: str
    is_skill_row: bool = False  # True if this is an actual skill (skills/xxx/)
    skill_dir: str = ""  # Directory name for skill rows
    inferred_category: str = ""  # Inferred category for sorting


def parse_frontmatter(text: str) -> dict[str, str]:
    """Parse YAML frontmatter from SKILL.md text. Returns fields dict."""
    fields: dict[str, str] = {}

    if not text.startswith("---"):
        return fields

    parts = text.split("---", 2)
    if len(parts) < 3:
        return fields

    fm_text = parts[1].strip()
    lines = fm_text.split("\n")
    current_key: str | None = None
    current_value_parts: list[str] = []

    def _flush():
        if current_key is not None:
            value = " ".join(current_value_parts).strip()
            if value in (">", "|", ">-", "|-"):
                value = ""
            fields[current_key] = value

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if line[0] in (" ", "\t") and current_key is not None:
            current_value_parts.append(stripped)
            continue

        if ":" in line and line[0] not in (" ", "\t"):
            _flush()
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            current_key = key
            if value in (">", "|", ">-", "|-"):
                current_value_parts = []
            else:
                current_value_parts = [value] if value else []
        else:
            if current_key and ":" in stripped:
                nested_key, _, nested_value = stripped.partition(":")
                nested_key = nested_key.strip()
                nested_value = nested_value.strip().strip('"').strip("'")
                fields[f"{current_key}.{nested_key}"] = nested_value

    _flush()
    return fields


def parse_skill(skill_dir: Path) -> SkillInfo | None:
    """Parse a skill directory and extract key info."""
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        return None

    text = skill_md.read_text(encoding="utf-8")
    fields = parse_frontmatter(text)

    name = fields.get("name", "")
    description = fields.get("description", "")
    category = fields.get("metadata.category", fields.get("category", ""))
    sub_category = fields.get("metadata.sub_category", fields.get("sub_category", ""))

    if not name or not description:
        return None

    return SkillInfo(
        name=name,
        description=description,
        directory=skill_dir.name,
        category=category,
        sub_category=sub_category,
    )


def categorize_skill(skill: SkillInfo) -> tuple[str, str]:
    """Determine category based on skill metadata or name patterns."""
    if skill.category and skill.sub_category:
        return skill.category, skill.sub_category

    name_lower = skill.name.lower()
    dir_lower = skill.directory.lower()

    if "model-migrate" in dir_lower:
        return "Deployment & Release", "Model Migration"
    elif "kernelgen" in dir_lower:
        return "Kernel & Operator Development", "Kernel Generation"
    elif "skill-creator" in dir_lower:
        return "Developer Tooling", "Skill Development"
    elif "install-stack" in dir_lower:
        return "Deployment & Release", "Stack Installation"
    elif "gpu-container" in dir_lower:
        return "Deployment & Release", "Base Image Selection"
    elif "model-verify" in dir_lower:
        return "Benchmarking & Eval", "Deployment A/B Verification"
    elif "perf-test" in dir_lower:
        return "Benchmarking & Eval", "Accuracy & Performance Test"
    elif "flagrelease" in dir_lower:
        return "Deployment & Release", "Release Pipeline"
    else:
        return "Developer Tooling", "General"


def scan_skills() -> dict[str, SkillInfo]:
    """Scan all skills and return dict keyed by directory name."""
    skills: dict[str, SkillInfo] = {}

    for skill_dir in sorted(SKILLS_DIR.iterdir()):
        if not skill_dir.is_dir() or skill_dir.name.startswith("."):
            continue

        info = parse_skill(skill_dir)
        if info:
            skills[info.directory] = info

    return skills


def parse_table_row(line: str, prev_category: str = "") -> TableRow | None:
    """Parse a markdown table row into TableRow."""
    if not line.startswith("|"):
        return None

    parts = [p.strip() for p in line.split("|")]
    if len(parts) < 5:
        return None

    category = parts[1]
    sub_category = parts[2]
    skill_link = parts[3]
    description = parts[4]

    # Extract bold category if present
    category_match = re.match(r"\*\*(.+?)\*\*", category)
    inferred_category = category_match.group(1) if category_match else prev_category

    # Check if this is an actual skill (links to skills/xxx/)
    skill_match = re.search(r"\[.*?\]\(skills/([^/]+)/?\)", skill_link)
    if skill_match:
        return TableRow(
            category=category,
            sub_category=sub_category,
            skill_link=skill_link,
            description=description,
            is_skill_row=True,
            skill_dir=skill_match.group(1),
            inferred_category=inferred_category,
        )

    return TableRow(
        category=category,
        sub_category=sub_category,
        skill_link=skill_link,
        description=description,
        is_skill_row=False,
        inferred_category=inferred_category,
    )


def escape_description(desc: str) -> str:
    """Escape description for markdown table."""
    escaped = desc.replace("\n", " ").replace("|", "\\|")
    escaped = re.sub(r"\s+", " ", escaped)
    return escaped.strip()


def get_category_index(category: str) -> int:
    """Get sort index for a category."""
    if category in CATEGORY_ORDER:
        return CATEGORY_ORDER.index(category)
    return len(CATEGORY_ORDER) + 1


def format_row(row: TableRow, is_first_in_category: bool) -> str:
    """Format a TableRow back to markdown with proper category grouping."""
    if is_first_in_category:
        cat_display = f"**{row.inferred_category}**"
    else:
        cat_display = ""
    return f"| {cat_display} | {row.sub_category} | {row.skill_link} | {row.description} |"


def parse_existing_table(readme_text: str) -> tuple[list[str], list[TableRow], list[str]]:
    """Parse existing table and return (header, rows, footer)."""
    begin_match = re.search(r"<!-- BEGIN_SKILLS_TABLE -->", readme_text)
    end_match = re.search(r"<!-- END_SKILLS_TABLE -->", readme_text)

    if not begin_match or not end_match:
        return [], [], []

    table_section = readme_text[begin_match.start():end_match.end()]
    lines = table_section.split("\n")

    header: list[str] = []
    rows: list[TableRow] = []
    footer: list[str] = []
    prev_category = ""

    for line in lines:
        stripped = line.strip()
        if stripped == "<!-- BEGIN_SKILLS_TABLE -->":
            header.append(line)
        elif stripped.startswith("| Category"):
            header.append(line)
        elif stripped.startswith("|--"):
            header.append(line)
        elif stripped == "<!-- END_SKILLS_TABLE -->":
            footer.append(line)
        elif line.startswith("|"):
            row = parse_table_row(line, prev_category)
            if row:
                # Update prev_category for next row
                if row.category and row.category.startswith("**"):
                    match = re.match(r"\*\*(.+?)\*\*", row.category)
                    if match:
                        prev_category = match.group(1)
                rows.append(row)

    return header, rows, footer


def build_new_table(
    existing_rows: list[TableRow],
    skills: dict[str, SkillInfo]
) -> list[TableRow]:
    """Build new table rows merging existing non-skill rows with scanned skills."""
    # Separate skill rows from non-skill rows
    existing_skill_dirs: set[str] = {
        r.skill_dir for r in existing_rows if r.is_skill_row
    }
    non_skill_rows = [r for r in existing_rows if not r.is_skill_row]

    # Group non-skill rows by inferred category
    non_skill_by_cat: dict[str, list[TableRow]] = {}
    for row in non_skill_rows:
        cat = row.inferred_category
        if cat not in non_skill_by_cat:
            non_skill_by_cat[cat] = []
        non_skill_by_cat[cat].append(row)

    # Build skill rows from scanned skills
    skill_rows_by_cat: dict[str, list[TableRow]] = {}
    for skill_dir, skill in skills.items():
        category, sub_category = categorize_skill(skill)
        row = TableRow(
            category="",  # Will be set to **Category** on first row
            sub_category=sub_category,
            skill_link=f"[`{skill.name}`](skills/{skill_dir}/)",
            description=escape_description(skill.description),
            is_skill_row=True,
            skill_dir=skill_dir,
            inferred_category=category,
        )
        if category not in skill_rows_by_cat:
            skill_rows_by_cat[category] = []
        skill_rows_by_cat[category].append(row)

    # Sort skill rows within each category by subcategory, then name
    for cat in skill_rows_by_cat:
        skill_rows_by_cat[cat].sort(key=lambda r: (r.sub_category, r.skill_dir))

    # Merge and build final result
    result: list[TableRow] = []

    for category in CATEGORY_ORDER:
        cat_rows: list[TableRow] = []

        # Add non-skill rows (PR links, Planned) for this category
        if category in non_skill_by_cat:
            cat_rows.extend(non_skill_by_cat[category])

        # Add skill rows for this category
        if category in skill_rows_by_cat:
            cat_rows.extend(skill_rows_by_cat[category])

        if cat_rows:
            # Sort: non-skill rows first (they have more specific sub_categories)
            # then skill rows
            cat_rows.sort(key=lambda r: (
                0 if not r.is_skill_row else 1,
                r.sub_category,
            ))
            result.extend(cat_rows)

    return result


def update_readme(skills: dict[str, SkillInfo], dry_run: bool = False) -> bool:
    """Update README.md with new table. Returns True if changed."""
    readme_text = README_PATH.read_text(encoding="utf-8")

    header, existing_rows, footer = parse_existing_table(readme_text)
    if not header:
        print("ERROR: Could not parse table markers in README.md", file=sys.stderr)
        return False

    # Build new rows
    new_rows = build_new_table(existing_rows, skills)

    # Build new table content
    new_table_lines = header[:]

    # Track which categories we've seen for proper **Category** formatting
    seen_categories: set[str] = set()

    for row in new_rows:
        is_first = row.inferred_category not in seen_categories
        seen_categories.add(row.inferred_category)
        new_table_lines.append(format_row(row, is_first))

    new_table_lines.extend(footer)
    new_table = "\n".join(new_table_lines)

    # Find and replace table in readme
    begin_match = re.search(r"<!-- BEGIN_SKILLS_TABLE -->", readme_text)
    end_match = re.search(r"<!-- END_SKILLS_TABLE -->", readme_text)

    new_readme = (
        readme_text[:begin_match.start()]
        + new_table
        + readme_text[end_match.end():]
    )

    # Compare normalized versions
    old_table = readme_text[begin_match.start():end_match.end()]
    old_normalized = old_table.replace("\r\n", "\n").strip()
    new_normalized = new_table.replace("\r\n", "\n").strip()

    if old_normalized == new_normalized:
        print("No changes needed - table is already up to date.")
        return False

    if dry_run:
        print("=== DRY RUN: Would update README.md with the following table ===")
        print(new_table)
        print("=== END DRY RUN ===")
        return False

    README_PATH.write_text(new_readme, encoding="utf-8")
    print("Updated README.md Skills Catalog table.")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Update Skills Catalog table in README.md"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying files",
    )
    args = parser.parse_args()

    print(f"Scanning skills in: {SKILLS_DIR}")
    skills = scan_skills()
    print(f"Found {len(skills)} skill(s)")

    for skill_dir, skill in sorted(skills.items()):
        print(f"  - {skill.name} ({skill_dir})")

    update_readme(skills, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
