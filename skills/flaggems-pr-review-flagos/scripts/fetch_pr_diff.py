#!/usr/bin/env python3
"""Fetch PR diff data for FlagGems PR review.

Supports two modes:
  - PR mode:    python fetch_pr_diff.py <pr_number> [--repo owner/repo]
  - Local mode: python fetch_pr_diff.py --local [--base upstream/master]

Outputs structured JSON to stdout with file diffs, contents, and commit messages.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional


def verify_environment() -> None:
    """Verify required environment before proceeding."""
    # Check GH_TOKEN (only for PR mode, but check early)
    if not os.getenv("GH_TOKEN"):
        print("ERROR: GH_TOKEN not set.", file=sys.stderr)
        print("Please run: gh auth login", file=sys.stderr)
        print("Or set: export GH_TOKEN=your_token", file=sys.stderr)
        sys.exit(1)

    # Check if in git repo or FLAGGEMS_REPO is set
    repo_path = os.getenv("FLAGGEMS_REPO")
    if repo_path:
        repo_path = Path(repo_path).resolve()
        if not (repo_path / ".git").is_dir():
            print(f"ERROR: FLAGGEMS_REPO={repo_path} is not a git repository", file=sys.stderr)
            sys.exit(1)
        try:
            os.chdir(repo_path)
        except OSError as e:
            print(f"ERROR: Cannot cd to FLAGGEMS_REPO={repo_path}: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # Check if current directory is in a git repo
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print("ERROR: Not in a git repository and FLAGGEMS_REPO not set", file=sys.stderr)
            print("", file=sys.stderr)
            print("Please either:", file=sys.stderr)
            print("  1. cd to FlagGems repository, or", file=sys.stderr)
            print("  2. export FLAGGEMS_REPO=/path/to/FlagGems", file=sys.stderr)
            sys.exit(1)


def run_cmd(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a shell command and return the result."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error running: {' '.join(cmd)}", file=sys.stderr)
        print(result.stderr.strip(), file=sys.stderr)
        sys.exit(1)
    return result


def detect_operator(files: list[dict[str, str]]) -> Optional[str]:
    """Auto-detect operator name from changed file paths.

    Looks for patterns like src/flag_gems/ops/<name>.py where name is not __init__.
    """
    pattern = re.compile(r"src/flag_gems/ops/([^/]+)\.py$")
    for f in files:
        match = pattern.search(f["path"])
        if match:
            name = match.group(1)
            if name != "__init__":
                return name
    return None


def detect_repo() -> str:
    """Auto-detect the repository via gh CLI. Falls back to flagos-ai/FlagGems if not in a git repo."""
    result = run_cmd(
        ["gh", "repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"],
        check=False,
    )
    repo = result.stdout.strip()
    if not repo:
        print("Warning: not in a git repo, pass --repo to specify. Trying flagos-ai/FlagGems.", file=sys.stderr)
        return "flagos-ai/FlagGems"
    return repo


def fetch_pr_head_branch(owner: str, repo: str, pr_number: int) -> str:
    """Fetch the head branch ref for a PR."""
    result = run_cmd([
        "gh", "api", f"repos/{owner}/{repo}/pulls/{pr_number}",
        "--jq", ".head.ref",
    ])
    return result.stdout.strip()


def fetch_pr_files(owner: str, repo: str, pr_number: int) -> list[dict[str, str]]:
    """Fetch the list of changed files in a PR with their patches."""
    result = run_cmd([
        "gh", "api", f"repos/{owner}/{repo}/pulls/{pr_number}/files",
        "--paginate", "--jq", ".[]",
    ])

    files: list[dict[str, str]] = []
    # --jq '.[]' produces NDJSON (one JSON object per line)
    for line in result.stdout.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        status_map = {
            "added": "added",
            "removed": "removed",
            "modified": "modified",
            "renamed": "renamed",
            "copied": "copied",
        }
        raw_status = obj.get("status", "modified")
        status = status_map.get(raw_status, raw_status)

        files.append({
            "path": obj.get("filename", ""),
            "status": status,
            "patch": obj.get("patch", ""),
            "content": "",
        })

    return files


def fetch_file_content(owner: str, repo: str, path: str, ref: str) -> str:
    """Fetch full file content from GitHub at a given ref."""
    result = run_cmd(
        ["gh", "api", f"repos/{owner}/{repo}/contents/{path}?ref={ref}", "--jq", ".content"],
        check=False,
    )
    if result.returncode != 0:
        # File may have been deleted or is binary
        return ""
    content_b64 = result.stdout.strip()
    if not content_b64:
        return ""
    try:
        return base64.b64decode(content_b64).decode("utf-8", errors="replace")
    except Exception:
        return ""


def fetch_pr_commits(owner: str, repo: str, pr_number: int) -> list[str]:
    """Fetch commit messages for a PR."""
    result = run_cmd([
        "gh", "api", f"repos/{owner}/{repo}/pulls/{pr_number}/commits",
        "--paginate", "--jq", ".[].commit.message",
    ])
    commits = [line for line in result.stdout.strip().splitlines() if line.strip()]
    return commits


def handle_pr_mode(pr_number: int, repo_arg: Optional[str]) -> dict:
    """Handle PR mode: fetch diff data from GitHub."""
    if repo_arg:
        full_repo = repo_arg
    else:
        full_repo = detect_repo()

    owner, repo = full_repo.split("/", 1)

    # Fetch head branch for content retrieval
    head_branch = fetch_pr_head_branch(owner, repo, pr_number)

    # Fetch changed files
    files = fetch_pr_files(owner, repo, pr_number)

    # Fetch full content for each file (skip removed files)
    for f in files:
        if f["status"] != "removed":
            f["content"] = fetch_file_content(owner, repo, f["path"], head_branch)

    # Fetch commits
    commits = fetch_pr_commits(owner, repo, pr_number)

    operator = detect_operator(files)

    return {
        "operator": operator,
        "mode": "pr",
        "pr_number": pr_number,
        "files": files,
        "commits": commits,
    }


def handle_local_mode(base: str) -> dict:
    """Handle local mode: fetch diff data from local git repo."""
    # Get changed files with status
    result = run_cmd(["git", "diff", "--unified=5", f"{base}...HEAD", "--name-status"])
    files: list[dict[str, str]] = []

    status_map = {
        "A": "added",
        "D": "removed",
        "M": "modified",
        "R": "renamed",
        "C": "copied",
    }

    for line in result.stdout.strip().splitlines():
        if not line.strip():
            continue
        parts = line.split("\t", 1)
        if len(parts) < 2:
            continue
        raw_status = parts[0].strip()
        # Handle statuses like R100 (renamed with similarity percentage)
        status_char = raw_status[0] if raw_status else "M"
        status = status_map.get(status_char, "modified")

        # For renames, path is the second tab-separated value
        path_parts = parts[1].split("\t")
        path = path_parts[-1].strip()

        files.append({
            "path": path,
            "status": status,
            "patch": "",
            "content": "",
        })

    # Get unified diff
    diff_result = run_cmd(["git", "diff", "--unified=5", f"{base}...HEAD"])
    full_diff = diff_result.stdout

    # Parse patches per file from the unified diff
    patch_map: dict[str, str] = {}
    current_file: Optional[str] = None
    current_patch_lines: list[str] = []

    for line in full_diff.splitlines(keepends=True):
        if line.startswith("diff --git"):
            # Save previous file's patch
            if current_file is not None:
                patch_map[current_file] = "".join(current_patch_lines)
            # Extract file path: diff --git a/path b/path
            match = re.search(r" b/(.+)$", line.rstrip())
            current_file = match.group(1) if match else None
            current_patch_lines = [line]
        else:
            current_patch_lines.append(line)

    if current_file is not None:
        patch_map[current_file] = "".join(current_patch_lines)

    # Assign patches and read file contents
    for f in files:
        f["patch"] = patch_map.get(f["path"], "")
        if f["status"] != "removed":
            file_path = Path(f["path"])
            if file_path.is_file():
                try:
                    f["content"] = file_path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    f["content"] = ""
            else:
                f["content"] = ""

    # Get commit messages between base and HEAD
    commit_result = run_cmd(
        ["git", "log", "--format=%s", f"{base}...HEAD"],
        check=False,
    )
    commits = [line for line in commit_result.stdout.strip().splitlines() if line.strip()]

    operator = detect_operator(files)

    return {
        "operator": operator,
        "mode": "local",
        "pr_number": None,
        "files": files,
        "commits": commits,
    }


def main() -> None:
    # Verify environment first
    verify_environment()

    parser = argparse.ArgumentParser(
        description="Fetch PR diff data for FlagGems PR review.",
    )
    parser.add_argument(
        "pr_number",
        nargs="?",
        type=int,
        help="Pull request number (required for PR mode)",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=None,
        help="Repository in owner/repo format (auto-detected if not provided)",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local mode instead of PR mode",
    )
    parser.add_argument(
        "--base",
        type=str,
        default="upstream/master",
        help="Base ref for local mode (default: upstream/master)",
    )

    args = parser.parse_args()

    if args.local:
        output = handle_local_mode(args.base)
    elif args.pr_number is not None:
        output = handle_pr_mode(args.pr_number, args.repo)
    else:
        parser.error("Either provide a PR number or use --local mode")

    json.dump(output, sys.stdout, indent=2, ensure_ascii=False)
    print()  # trailing newline


if __name__ == "__main__":
    main()
