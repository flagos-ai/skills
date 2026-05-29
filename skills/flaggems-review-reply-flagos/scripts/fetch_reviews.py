#!/usr/bin/env python3
"""Fetch PR review comments from GitHub.

Usage:
    python fetch_reviews.py <pr_number> [--repo owner/repo] [--format json|summary]

Outputs structured review comments for analysis.
"""

import argparse
import json
import subprocess
import sys


def gh_api(endpoint: str) -> list:
    """Call gh api and return parsed JSON."""
    cmd = ["gh", "api", endpoint, "--paginate", "--jq", ".[]"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    items = []
    for line in result.stdout.strip().splitlines():
        if line:
            items.append(json.loads(line))
    return items


def fetch_review_comments(pr_number: int, repo: str) -> list[dict]:
    """Fetch inline review comments on a PR."""
    endpoint = f"repos/{repo}/pulls/{pr_number}/comments"
    raw = gh_api(endpoint)
    return [
        {
            "id": c["id"],
            "path": c["path"],
            "line": c.get("line") or c.get("original_line"),
            "body": c["body"],
            "user": c["user"]["login"],
            "created_at": c["created_at"],
            "in_reply_to_id": c.get("in_reply_to_id"),
        }
        for c in raw
    ]


def fetch_reviews(pr_number: int, repo: str) -> list[dict]:
    """Fetch top-level reviews (approve/request changes/comment)."""
    endpoint = f"repos/{repo}/pulls/{pr_number}/reviews"
    raw = gh_api(endpoint)
    return [
        {
            "id": r["id"],
            "state": r["state"],
            "body": r["body"],
            "user": r["user"]["login"],
            "submitted_at": r.get("submitted_at"),
        }
        for r in raw
        if r["body"]  # skip empty reviews
    ]


def format_summary(comments: list[dict], reviews: list[dict]) -> str:
    """Format as human-readable summary."""
    lines = []
    if reviews:
        lines.append("=== Reviews ===")
        for r in reviews:
            lines.append(f"[{r['state']}] {r['user']}: {r['body'][:200]}")
        lines.append("")

    if comments:
        lines.append("=== Inline Comments ===")
        # Group by file
        by_file: dict[str, list] = {}
        for c in comments:
            by_file.setdefault(c["path"], []).append(c)
        for path, file_comments in sorted(by_file.items()):
            lines.append(f"\n--- {path} ---")
            for c in sorted(file_comments, key=lambda x: x["line"] or 0):
                prefix = f"  L{c['line']}" if c["line"] else "  (file)"
                reply_marker = " [reply]" if c["in_reply_to_id"] else ""
                lines.append(f"{prefix} @{c['user']}{reply_marker}: {c['body'][:200]}")
    return "\n".join(lines)


def detect_repo() -> str:
    """Auto-detect owner/repo from git remote via gh."""
    result = subprocess.run(
        ["gh", "repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("Error: cannot detect repo. Pass --repo explicitly.", file=sys.stderr)
        sys.exit(1)
    return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser(description="Fetch PR review comments")
    parser.add_argument("pr_number", type=int, help="PR number")
    parser.add_argument("--repo", default=None, help="GitHub repo (owner/name). Auto-detected if omitted.")
    parser.add_argument(
        "--format",
        choices=["json", "summary"],
        default="summary",
        help="Output format",
    )
    args = parser.parse_args()

    repo = args.repo or detect_repo()
    comments = fetch_review_comments(args.pr_number, repo)
    reviews = fetch_reviews(args.pr_number, repo)

    if args.format == "json":
        output = {"reviews": reviews, "comments": comments}
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        print(format_summary(comments, reviews))


if __name__ == "__main__":
    main()
