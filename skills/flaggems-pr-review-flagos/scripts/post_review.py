#!/usr/bin/env python3
"""Post review findings as a GitHub PR review with inline comments."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from typing import Any


def run_gh(args: list[str], input_data: str | None = None) -> str:
    """Run a gh CLI command and return stdout."""
    result = subprocess.run(["gh"] + args, capture_output=True, text=True, input=input_data)
    if result.returncode != 0:
        raise RuntimeError(f"gh failed: {' '.join(args)}\n{result.stderr}")
    return result.stdout.strip()


def detect_repo() -> str:
    """Auto-detect owner/repo via gh. Falls back to flagos-ai/FlagGems if not in a git repo."""
    try:
        repo = run_gh(["repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"])
        if repo:
            return repo
    except RuntimeError:
        pass
    print("Warning: not in a git repo, pass --repo to specify. Trying flagos-ai/FlagGems.", file=sys.stderr)
    return "flagos-ai/FlagGems"


def load_findings(path: str) -> tuple[str | None, list[dict[str, Any]]]:
    """Load findings from JSON file, return (operator, findings)."""
    with open(path) as f:
        data = json.load(f)

    # Support both old format (array) and new format (dict with operator)
    if isinstance(data, list):
        return None, data
    elif isinstance(data, dict):
        return data.get("operator"), data.get("findings", [])
    else:
        raise ValueError("Invalid findings JSON format")


def get_diff_position_map(repo: str, pr: int) -> dict[str, dict[int, int]]:
    """Build mapping of file -> {line_number: diff_position} from PR files."""
    raw = run_gh(["api", f"repos/{repo}/pulls/{pr}/files", "--paginate", "--jq", ".[]"])
    position_map: dict[str, dict[int, int]] = {}
    for json_line in raw.splitlines():
        if not json_line.strip():
            continue
        f = json.loads(json_line)
        filename = f["filename"]
        patch = f.get("patch", "")
        if not patch:
            continue
        line_to_pos: dict[int, int] = {}
        position = 0
        current_line = 0
        for raw_line in patch.split("\n"):
            if raw_line.startswith("@@"):
                match = re.search(r"\+(\d+)", raw_line)
                if match:
                    current_line = int(match.group(1)) - 1
                else:
                    continue
                position += 1
                continue
            position += 1
            if raw_line.startswith("-"):
                continue
            current_line += 1
            line_to_pos[current_line] = position
        position_map[filename] = line_to_pos
    return position_map


def format_comment_body(findings: list[dict[str, Any]]) -> str:
    """Format multiple findings into a single comment body."""
    parts: list[str] = []

    severity_prefix = {
        "error": "Issue",
        "warning": "Suggestion",
        "info": "Note"
    }

    for f in findings:
        severity = f.get("severity", "info")
        rule_id = f.get("rule_id", "UNKNOWN")
        message = f.get("message", "")
        suggestion = f.get("suggestion", "")

        prefix = severity_prefix.get(severity, "Note")

        # More friendly format
        part = f"**{prefix}**: {message}"

        if suggestion:
            part += f"\n\n**Recommended fix:**\n{suggestion}"

        # Add rule reference at bottom in small text
        part += f"\n\n<sub>Rule: `{rule_id}`</sub>"

        parts.append(part)

    return "\n\n---\n\n".join(parts)


def build_review_body(operator: str | None, findings: list[dict[str, Any]],
                      non_inline: list[dict[str, Any]]) -> str:
    """Build the top-level review body markdown."""
    errors = sum(1 for f in findings if f.get("severity") == "error")
    warnings = sum(1 for f in findings if f.get("severity") == "warning")
    suggestions = len(findings) - errors - warnings

    op_display = f"`{operator}`" if operator else "unknown"
    verdict = "REQUEST_CHANGES" if errors > 0 else "COMMENT"

    body = f"""## FlagGems Automated Review

**Operator:** {op_display}
**Verdict:** {verdict}

### Summary
- Errors: {errors}
- Warnings: {warnings}
- Suggestions: {suggestions}"""

    if non_inline:
        body += "\n\n### Issues (no specific line)"
        for f in non_inline:
            severity = f.get("severity", "info").upper()
            rule_id = f.get("rule_id", "UNKNOWN")
            message = f.get("message", "")
            body += f"\n- **[{severity}]** `{rule_id}`: {message}"
            if f.get("suggestion"):
                body += f" — *Fix: {f['suggestion']}*"

    return body


def post_review(repo: str, pr: int, operator: str | None, findings: list[dict[str, Any]],
                event: str | None, dry_run: bool) -> None:
    """Build and post (or dry-run print) the PR review."""
    position_map = get_diff_position_map(repo, pr)

    # Separate inline vs non-inline findings
    inline_groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    non_inline: list[dict[str, Any]] = []

    for f in findings:
        file_path = f.get("file")
        line = f.get("line")
        if file_path and line and file_path in position_map and line in position_map[file_path]:
            inline_groups[(file_path, line)].append(f)
        else:
            non_inline.append(f)

    # Build inline comments
    comments: list[dict[str, Any]] = []
    for (file_path, line), group in inline_groups.items():
        comments.append({
            "path": file_path,
            "position": position_map[file_path][line],
            "body": format_comment_body(group),
        })

    # Determine event
    if event is None:
        has_error = any(f.get("severity") == "error" for f in findings)
        event = "REQUEST_CHANGES" if has_error else "COMMENT"

    body = build_review_body(operator, findings, non_inline)

    review_payload = {
        "event": event,
        "body": body,
        "comments": comments,
    }

    if dry_run:
        print(f"=== DRY RUN === Repo: {repo}, PR: #{pr}, Event: {event}")
        print(f"\n--- Body ---\n{body}")
        print(f"\n--- Inline comments ({len(comments)}) ---")
        for c in comments:
            print(f"  {c['path']}@pos{c['position']}: {c['body'][:100]}...")
        return

    # Post via gh api using temp file for input
    tmp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(review_payload, tmp_file)
    tmp_file.close()
    tmp_path = tmp_file.name

    try:
        result = run_gh(["api", f"repos/{repo}/pulls/{pr}/reviews",
                         "--method", "POST", "--input", tmp_path])
        print(f"Review posted successfully on {repo}#{pr}")
        if result:
            print(result)
    except RuntimeError as e:
        print(f"ERROR: Failed to post review: {e}", file=sys.stderr)
        print("\n--- Review content (not lost) ---", file=sys.stderr)
        print(json.dumps(review_payload, indent=2), file=sys.stderr)
        sys.exit(1)
    finally:
        os.unlink(tmp_path)

    time.sleep(1)  # Rate limit courtesy


def main() -> None:
    parser = argparse.ArgumentParser(description="Post review findings as GitHub PR review")
    parser.add_argument("pr_number", type=int, help="Pull request number")
    parser.add_argument("--findings", required=True, help="Path to findings JSON file")
    parser.add_argument("--repo", help="owner/repo (auto-detected if omitted)")
    parser.add_argument("--dry-run", action="store_true", help="Print review without posting")
    parser.add_argument("--event", choices=["COMMENT", "REQUEST_CHANGES", "APPROVE"],
                        help="Review event type (auto-selected if omitted)")
    args = parser.parse_args()

    repo = args.repo or detect_repo()
    operator, findings = load_findings(args.findings)

    if not findings:
        print("No findings to post.")
        return

    post_review(repo, args.pr_number, operator, findings, args.event, args.dry_run)


if __name__ == "__main__":
    main()
