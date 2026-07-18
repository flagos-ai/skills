#!/usr/bin/env python3

# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Update Skills Catalog table in README.md.

Logic:
- If skill directory exists → use skills/xxx/ link (removes PR links for same skill)
- If skill directory doesn't exist but PR link in table → keep PR link
- *Planned* and sub-skill (.md) links are preserved

Usage: python scripts/update_skills_catalog.py [--dry-run]
"""
import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

SKILLS_DIR = Path(__file__).parent.parent / "skills"
README_PATH = Path(__file__).parent.parent / "README.md"

CATEGORY_ORDER = [
    "Deployment & Release", "Benchmarking & Eval",
    "Kernel & Operator Development", "Multi-Chip Backend Onboarding", "Developer Tooling",
]


@dataclass
class Skill:
    name: str
    description: str
    directory: str
    category: str = ""
    sub_category: str = ""


def parse_frontmatter(text: str) -> dict:
    fields = {}
    if not text.startswith("---"):
        return fields
    parts = text.split("---", 2)
    if len(parts) < 3:
        return fields
    lines = parts[1].strip().split("\n")
    key, val_parts = None, []
    def flush():
        if key:
            v = " ".join(val_parts).strip()
            fields[key] = "" if v in (">", "|", ">-", "|-") else v
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        if ln[0] in " \t" and key:
            val_parts.append(s)
        elif ":" in ln and ln[0] not in " \t":
            flush()
            k, _, v = ln.partition(":")
            key = k.strip()
            v = v.strip().strip('"').strip("'")
            val_parts = [] if v in (">", "|", ">-", "|-") else [v] if v else []
        elif key and ":" in s:
            nk, _, nv = s.partition(":")
            fields[f"{key}.{nk.strip()}"] = nv.strip().strip('"').strip("'")
    flush()
    return fields


def parse_skill(d: Path):
    f = d / "SKILL.md"
    if not f.exists():
        return None
    fld = parse_frontmatter(f.read_text())
    n, desc = fld.get("name", ""), fld.get("description", "")
    if not n or not desc:
        return None
    return Skill(n, desc, d.name, fld.get("metadata.category", ""), fld.get("metadata.sub_category", ""))


def categorize(s: Skill):
    if s.category and s.sub_category:
        return s.category, s.sub_category
    d = s.directory.lower()
    if "model-migrate" in d: return "Deployment & Release", "Model Migration"
    if "kernelgen" in d: return "Kernel & Operator Development", "Kernel Generation"
    if "skill-creator" in d: return "Developer Tooling", "Skill Development"
    if "install-stack" in d: return "Deployment & Release", "Stack Installation"
    if "gpu-container" in d: return "Deployment & Release", "Base Image Selection"
    if "model-verify" in d: return "Benchmarking & Eval", "Deployment A/B Verification"
    if "perf-test" in d: return "Benchmarking & Eval", "Accuracy & Performance Test"
    if "flagrelease" in d: return "Deployment & Release", "Release Pipeline"
    return "Developer Tooling", "General"


def scan():
    return {d.name: s for d in sorted(SKILLS_DIR.iterdir()) if d.is_dir() and not d.name.startswith(".") and (s := parse_skill(d))}


def esc(d):
    return re.sub(r"\s+", " ", d.replace("\n", " ").replace("|", "\\|")).strip()


def cat_idx(c):
    return CATEGORY_ORDER.index(c) if c in CATEGORY_ORDER else 99


def update_readme(skills, dry):
    readme = README_PATH.read_text()
    m1, m2 = re.search(r"<!-- BEGIN_SKILLS_TABLE -->", readme), re.search(r"<!-- END_SKILLS_TABLE -->", readme)
    if not m1 or not m2:
        print("ERROR: Markers not found", file=sys.stderr)
        return False

    merged = set(skills.keys())
    rows = []
    prev_cat = ""
    for ln in readme[m1.end():m2.start()].split("\n"):
        if not ln.startswith("|") or "Category" in ln or "|--" in ln:
            continue
        p = [x.strip() for x in ln.split("|")]
        if len(p) < 5:
            continue
        cat_raw, sub, link, desc = p[1], p[2], p[3], p[4]
        cm = re.match(r"\*\*(.+?)\*\*", cat_raw)
        cat = cm.group(1) if cm else prev_cat
        if cat: prev_cat = cat

        # Detect row type
        main_m = re.search(r"skills/([^/]+)/?\)", link)  # skills/xxx/ or skills/xxx)
        pr_m = re.search(r"\[PR #\d+ `([^`]+)`\]", link)
        is_sub = ".md)" in link

        rows.append({
            "cat": cat, "sub": sub, "link": link, "desc": desc,
            "dir": main_m.group(1) if main_m else (pr_m.group(1) if pr_m else ""),
            "is_pr": bool(pr_m), "is_sub": is_sub, "is_plan": "*Planned*" in link,
        })

    # Build result: skip PR links for merged skills, keep everything else
    result = []
    seen_dirs = set()

    for r in rows:
        if r["is_pr"] and r["dir"] in merged:
            continue  # Skip PR link for merged skill
        if r["is_sub"] or r["is_plan"] or not r["dir"]:
            result.append(r)  # Keep sub-skills and planned
        elif r["dir"] in skills and r["dir"] not in seen_dirs:
            # Merge/update existing merged skill
            s = skills[r["dir"]]
            c, sc = categorize(s)
            result.append({"cat": c, "sub": sc, "link": f"[`{s.name}`](skills/{s.directory}/)", "desc": esc(s.description), "dir": s.directory, "is_pr": False})
            seen_dirs.add(r["dir"])

    # Add new skills not in table
    for d, s in skills.items():
        if d not in seen_dirs:
            c, sc = categorize(s)
            result.append({"cat": c, "sub": sc, "link": f"[`{s.name}`](skills/{d}/)", "desc": esc(s.description), "dir": d, "is_pr": False})

    # Keep PR links for unmerged skills
    for r in rows:
        if r["is_pr"] and r["dir"] not in merged:
            result.append(r)

    # Sort
    result.sort(key=lambda x: (cat_idx(x["cat"]), x["cat"], x["sub"], x["dir"]))

    # Dedup by dir (keep first occurrence of each merged skill)
    final, seen = [], set()
    for r in result:
        key = r["dir"] if r["dir"] and not r["is_pr"] else id(r)
        if key not in seen:
            final.append(r)
            seen.add(key)

    # Build table
    lines = ["<!-- BEGIN_SKILLS_TABLE -->", "| Category | Sub-category | Skill | Description |", "|----------|-------------|-------|-------------|"]
    seen_cat = set()
    for r in final:
        c = f"**{r['cat']}**" if r["cat"] not in seen_cat else ""
        seen_cat.add(r["cat"])
        lines.append(f"| {c} | {r['sub']} | {r['link']} | {r['desc']} |")
    lines.append("<!-- END_SKILLS_TABLE -->")

    new_table = "\n".join(lines)
    old = readme[m1.start():m2.end()].replace("\r\n", "\n").strip()
    new = new_table.replace("\r\n", "\n").strip()

    if old == new:
        print("No changes needed.")
        return False

    if dry:
        print("=== DRY RUN ===\n" + new_table + "\n=== END ===")
        return False

    README_PATH.write_text(readme[:m1.start()] + new_table + readme[m2.end():])
    print("Updated README.md")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(f"Scanning: {SKILLS_DIR}")
    skills = scan()
    print(f"Found {len(skills)} skills")
    for d, s in sorted(skills.items()):
        print(f"  - {s.name} ({d})")
    update_readme(skills, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
