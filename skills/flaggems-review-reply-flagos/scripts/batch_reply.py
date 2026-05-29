#!/usr/bin/env python3
"""Batch process and reply to review comments.

Usage:
    python batch_reply.py <pr_number> --plan       # Generate fix plan (dry-run)
    python batch_reply.py <pr_number> --execute    # Execute fixes and reply
    python batch_reply.py <pr_number> --reply-only # Reply without code changes

Orchestrates: fetch → classify → fix → verify → commit → reply.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

# Comment classification categories
CATEGORIES = {
    "code-fix": "Code needs modification",
    "style": "Code style/formatting issue",
    "naming": "Naming convention issue",
    "question": "Reviewer has a question",
    "suggestion": "Optional suggestion",
    "nit": "Minor issue",
    "invalid": "Not applicable / already correct",
}


def get_gh_repo() -> str:
    """Get owner/repo from git remote."""
    result = subprocess.run(
        ["gh", "repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("Error: cannot determine repo. Run from a git repo with gh configured.", file=sys.stderr)
        sys.exit(1)
    return result.stdout.strip()


def get_pr_author(pr_number: int) -> str:
    """Get PR author login."""
    result = subprocess.run(
        ["gh", "pr", "view", str(pr_number), "--json", "author", "-q", ".author.login"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def fetch_comments(pr_number: int) -> dict:
    """Fetch all review data."""
    repo = get_gh_repo()
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "fetch_reviews.py"),
        str(pr_number),
        "--format",
        "json",
        "--repo",
        repo,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error fetching reviews: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    return json.loads(result.stdout)


def get_pending_comments(data: dict, our_users: set[str] | None = None) -> list[dict]:
    """Filter to top-level comments (not replies) that haven't been addressed."""
    comments = data.get("comments", [])
    addressed_ids = set()
    for c in comments:
        parent = c.get("in_reply_to_id")
        if parent:
            if our_users is None or c["user"] in our_users:
                addressed_ids.add(parent)

    pending = []
    for c in comments:
        if c.get("in_reply_to_id"):
            continue  # skip replies
        if c["id"] in addressed_ids:
            continue  # already addressed by our user(s)
        pending.append(c)
    return pending


def generate_plan(pending: list[dict]) -> list[dict]:
    """Generate a fix plan for pending comments."""
    plan = []
    for c in pending:
        plan.append(
            {
                "comment_id": c["id"],
                "path": c["path"],
                "line": c["line"],
                "reviewer": c["user"],
                "body": c["body"][:300],
                "category": "TBD",  # To be classified by the model
                "action": "TBD",
            }
        )
    return plan


def main():
    parser = argparse.ArgumentParser(description="Batch process review comments")
    parser.add_argument("pr_number", type=int, help="PR number")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true", help="Generate fix plan (dry-run)")
    mode.add_argument("--execute", action="store_true", help="Execute fixes and reply")
    mode.add_argument("--reply-only", action="store_true", help="Reply without code changes")
    parser.add_argument("--repo-dir", default=".", help="Local repo path")
    parser.add_argument("--user", help="PR author username (auto-detected if omitted)")
    args = parser.parse_args()

    if not any([args.plan, args.execute, args.reply_only]):
        args.plan = True  # default to plan mode

    # Fetch review data
    print(f"Fetching reviews for PR #{args.pr_number}...")
    data = fetch_comments(args.pr_number)

    # Get pending comments — only consider comments addressed if the PR author replied
    pr_author = args.user or get_pr_author(args.pr_number)
    pr_author_users = {pr_author} if pr_author else None
    pending = get_pending_comments(data, our_users=pr_author_users)
    print(f"Found {len(pending)} pending review comments.")

    if not pending:
        print("No pending comments to address.")
        return

    if args.plan:
        plan = generate_plan(pending)
        print("\n=== Fix Plan ===")
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        print(f"\nTotal: {len(plan)} items to address.")
        print("Run with --execute to apply fixes, or handle manually.")
    elif args.execute:
        print(f"Execute mode: process each comment in {args.repo_dir}")
        print("Use the SKILL.md workflow (Phase 1-5) for each comment.")
        for i, c in enumerate(pending, 1):
            print(f"\n[{i}/{len(pending)}] {c['path']}:L{c['line']} @{c['user']}")
            print(f"  {c['body'][:200]}")
    elif args.reply_only:
        print("Reply-only mode: prepare replies without code changes.")
        for c in pending:
            print(f"\nComment {c['id']} ({c['path']}:L{c['line']}):")
            print(f"  {c['body'][:200]}")
            print("  → Reply: [TODO]")


if __name__ == "__main__":
    main()
