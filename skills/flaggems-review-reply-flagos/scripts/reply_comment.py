#!/usr/bin/env python3
"""Reply to a PR review comment on GitHub.

Usage:
    python reply_comment.py <pr_number> <comment_id> <message>
    python reply_comment.py <pr_number> <comment_id> --file reply.md
    python reply_comment.py <pr_number> --review <review_id> <message>

Replies to inline review comments or top-level reviews.
"""

import argparse
import json
import subprocess
import sys


def gh_api_post(endpoint: str, data: dict) -> dict:
    """POST to gh api."""
    cmd = ["gh", "api", endpoint, "-X", "POST"]
    for key, value in data.items():
        cmd.extend(["-f", f"{key}={value}"])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    return json.loads(result.stdout) if result.stdout else {}


def reply_to_comment(pr_number: int, comment_id: int, body: str, repo: str):
    """Reply to an inline review comment."""
    endpoint = f"repos/{repo}/pulls/{pr_number}/comments/{comment_id}/replies"
    result = gh_api_post(endpoint, {"body": body})
    print(f"Replied to comment {comment_id}: {result.get('html_url', 'OK')}")


def reply_to_review(pr_number: int, review_id: int, body: str, repo: str):
    """Post a new review-level comment on a PR.

    GitHub API does not support replying to a specific review directly.
    This creates a new review with event=COMMENT as the closest alternative.
    The review_id is logged for traceability but not used in the API call.
    """
    endpoint = f"repos/{repo}/pulls/{pr_number}/reviews"
    result = gh_api_post(endpoint, {"body": body, "event": "COMMENT"})
    print(f"Posted review comment (ref review {review_id}): {result.get('html_url', 'OK')}")


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
    parser = argparse.ArgumentParser(description="Reply to PR review comments")
    parser.add_argument("pr_number", type=int, help="PR number")
    parser.add_argument("comment_id", type=int, nargs="?", help="Comment ID to reply to")
    parser.add_argument("message", nargs="?", help="Reply message")
    parser.add_argument("--review", type=int, help="Review ID (for top-level review reply)")
    parser.add_argument("--file", help="Read message from file")
    parser.add_argument("--repo", default=None, help="GitHub repo (owner/name). Auto-detected if omitted.")
    args = parser.parse_args()

    repo = args.repo or detect_repo()

    # Get message
    if args.file:
        with open(args.file) as f:
            body = f.read().strip()
    elif args.message:
        body = args.message
    else:
        if sys.stdin.isatty():
            print("Error: no message provided. Use positional arg, --file, or pipe via stdin.", file=sys.stderr)
            sys.exit(1)
        body = sys.stdin.read().strip()

    if not body:
        print("Error: empty message", file=sys.stderr)
        sys.exit(1)

    if args.review:
        reply_to_review(args.pr_number, args.review, body, repo)
    elif args.comment_id:
        reply_to_comment(args.pr_number, args.comment_id, body, repo)
    else:
        print("Error: must specify comment_id or --review", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
