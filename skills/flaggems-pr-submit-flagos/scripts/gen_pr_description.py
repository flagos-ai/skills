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

"""Generate PR description data for FlagGems operator PRs.

Usage:
  # Full mode: run benchmark + query domestic GPU
  python gen_pr_description.py <op_name> --repo /path/to/repo

  # Pipe mode (benchmark already run)
  pytest benchmark/test_<op>.py --level core -s | \
    python gen_pr_description.py <op_name> --nvidia-stdin

  # From log file
  python gen_pr_description.py <op_name> --nvidia-log <path>

  # Skip re-running benchmark (use cached data)
  python gen_pr_description.py <op_name> --repo /path/to/repo --skip-run

Output: JSON to stdout with nvidia_benchmark rows and domestic_gpu status.
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DOMESTIC_GPU_DIR = Path(os.environ.get("FLAGGEMS_DOMESTIC_DIR", "/workspace/国产GPU算子测试情况"))

# (directory_pattern, backend_name, try_aten_prefix)
# Order: prefer non-bak dirs, use glob to find latest
BACKEND_CONFIGS = [
    ("天数test_results_*", "tianshu", True),
    ("沐曦test_results_*", "muxi", True),
    ("华为test_*", "ascend", True),
    ("海光hygon_test_*", "hygon", True),
]

# Benchmark output line parser
# Format: SUCCESS  torch_ms  gems_ms  speedup  {detail} or [detail]
BENCH_RE = re.compile(
    r"SUCCESS\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(?:([\d.]+)\s+)?[\[{](.+?)[\]}]\s*$"
)
SHAPE_RE = re.compile(r"torch\.Size\(\[([^\]]+)\]\)")

# Error extraction patterns
# More-specific patterns first, generic FAILED/Error last
ERROR_PATTERNS = [
    r"\[MCR\]\[E\]\s*(.+)",
    r"mcError\w+[:\s]*(.+)",
    r"CompilationError[:\s]*(.+)",
    r"AssertionError[:\s]*(.+)",
    r"RuntimeError[:\s]*(.+)",
    r"NameError[:\s]*(.+)",
    r"(?:Error|ERROR)[:\s]\s*(.+)",
    r"(?:FAILED|failed)[:\s]*(.+)",
]
ERROR_RE = re.compile("|".join(ERROR_PATTERNS), re.IGNORECASE)


# ---------------------------------------------------------------------------
# Benchmark: run & parse
# ---------------------------------------------------------------------------


def find_benchmark_file(repo_dir: Path, op_name: str) -> Path | None:
    """Locate benchmark/test_<op>.py in repo."""
    bench_file = repo_dir / "benchmark" / f"test_{op_name}.py"
    if bench_file.exists():
        return bench_file
    # also try underscore-prefixed
    bare = op_name.lstrip("_")
    bench_file = repo_dir / "benchmark" / f"test_{bare}.py"
    if bench_file.exists():
        return bench_file
    # fallback: glob
    candidates = list(repo_dir.glob(f"benchmark/test_{bare}*.py"))
    if candidates:
        return candidates[0]
    return None


def run_nvidia_benchmark(repo_dir: Path, op_name: str, timeout: int = 300) -> str:
    """Run pytest benchmark with --level core, return stdout."""
    bench_file = find_benchmark_file(repo_dir, op_name)
    if not bench_file:
        raise FileNotFoundError(
            f"No benchmark file found for '{op_name}' in {repo_dir}/benchmark/"
        )
    rel = bench_file.relative_to(repo_dir)
    cmd = ["python", "-m", "pytest", str(rel), "--level", "core", "-s"]
    result = subprocess.run(
        cmd,
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.stdout


def parse_benchmark_output(text: str) -> list[dict]:
    """Extract performance rows from benchmark output."""
    rows = []
    for line in text.split("\n"):
        m = BENCH_RE.search(line)
        if not m:
            continue
        shape_raw = m.group(5).strip()
        shape_match = SHAPE_RE.findall(shape_raw)
        shape = shape_match[0] if shape_match else shape_raw
        rows.append(
            {
                "shape": shape,
                "torch_ms": float(m.group(1)),
                "gems_ms": float(m.group(2)),
                "speedup": float(m.group(3)),
                "tflops": float(m.group(4)) if m.group(4) else 0.0,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Domestic GPU: load & query
# ---------------------------------------------------------------------------


def _find_dirs(pattern: str) -> list[Path]:
    """Glob directories matching pattern, exclude _bak variants."""
    dirs = []
    for d in sorted(DOMESTIC_GPU_DIR.glob(pattern)):
        if d.is_dir() and "_bak" not in d.name and "备份" not in d.name:
            dirs.append(d)
    return dirs


def _load_summary(dir_path: Path) -> dict | None:
    """Load the summary JSON from a backend directory."""
    summaries = sorted(dir_path.glob("summary_*.json"))
    if not summaries:
        return None
    with open(summaries[-1]) as f:
        return json.load(f)


def _lookup_operator(
    op_name: str, data: dict, try_aten: bool = True
) -> dict | None:
    """Find operator in summary data, trying multiple key formats."""
    ops = data.get("operators", {})
    if not ops:
        return None
    # 1. Exact match
    if op_name in ops:
        return ops[op_name]
    bare = op_name.lstrip("_")
    if bare in ops:
        return ops[bare]
    # 2. aten:: prefix match (Ascend pattern)
    if try_aten:
        for prefix in ("aten::", "aten__"):
            if f"{prefix}{op_name}" in ops:
                return ops[f"{prefix}{op_name}"]
            if f"{prefix}{bare}" in ops:
                return ops[f"{prefix}{bare}"]
    # 3. Fuzzy: key contains op_name
    for k, v in ops.items():
        if op_name in k or bare in k.replace("::", "_").replace("_", " "):
            return v
    return None


MAX_ERROR_LEN = 400


def _tail_file(path: str, tail_lines: int = 100) -> str:
    """Read last N lines of a file, or the FAILURES section if present."""
    try:
        if not path or not os.path.exists(path):
            return ""
        with open(path) as f:
            content = f.read()

        # Prefer the FAILURES section (between "=== FAILURES ===" and next "===")
        failures_marker = "=" * 10 + " FAILURES " + "=" * 10
        idx = content.find(failures_marker)
        if idx >= 0:
            after = content[idx + len(failures_marker):]
            # Stop at next === divider, but not within traceback lines
            end_idx = after.find("\n=" * 10 + " ")
            if end_idx >= 0:
                return after[:end_idx]
            return after[:3000]

        # Fallback: last N lines
        lines = content.split("\n")
        return "\n".join(lines[-tail_lines:])
    except OSError:
        return ""


def _search_error(source: str) -> str | None:
    """Search for error patterns in text.

    Returns the most informative error line — prefers assertion messages
    and compilation errors over generic 'FAILED' lines.
    """
    if not source:
        return None

    # Find ALL matches, pick the most specific/meaningful one
    matches = list(ERROR_RE.finditer(source))
    if not matches:
        return None

    # Score: prefer patterns that contain specific error info
    best = None
    best_score = -1
    for m in matches:
        text = m.group(0).strip()
        score = 0
        # Prefer lines with actual error details
        if any(kw in text for kw in ("AssertionError", "CompilationError", "MCR")):
            score = 3
        elif any(kw in text for kw in ("RuntimeError", "NameError", "mcError")):
            score = 2
        elif "Error" in text:
            score = 1
        # Prefer longer, more informative messages (but not too long)
        if len(text) > 10:
            score += 1

        # Extend short CompilationError matches to include the next line with NameError
        if "CompilationError" in text and len(text) < 60:
            end = m.end()
            rest = source[end:end + 300]
            name_match = re.search(r"NameError\(.+?\)", rest, re.DOTALL)
            if name_match:
                text = text + " " + name_match.group(0).strip()

        if score > best_score:
            best_score = score
            best = text

    if best:
        return best[:MAX_ERROR_LEN] if len(best) > MAX_ERROR_LEN else best
    return None


def _resolve_log_file(summary_dir: Path, log_file_path: str, op_name: str, key: str) -> str:
    """Resolve a log file path from summary JSON to a local path.

    Summary JSONs typically contain original machine paths like
    /root/JudeWorkplace/test_results_*/op_test.log — those don't exist locally.
    The real files are in or near the summary directory.
    """
    if not log_file_path:
        return ""

    # Try the exact path first
    if os.path.exists(log_file_path):
        return log_file_path

    basename = os.path.basename(log_file_path)

    # Collect candidate directories to search
    search_dirs = [summary_dir]

    # Also search test_results_* subdirectories (Ascend pattern)
    for sub in sorted(summary_dir.glob("test_results_*")):
        if sub.is_dir():
            search_dirs.append(sub)

    for search_dir in search_dirs:
        # Try matching by basename
        local = search_dir / basename
        if local.exists():
            return str(local)

        # Try <op>_{test,bench}.log naming
        for name in (op_name, op_name.lstrip("_")):
            cand = search_dir / f"{name}_{key}.log"
            if cand.exists():
                return str(cand)

    return ""


def _extract_failure_reason(
    op_info: dict, op_name: str, summary_dir: Path, key: str = "test"
) -> str | None:
    """Extract failure reason from operator summary.

    Priority:
    1. stderr_tail field (if present and operator actually failed)
    2. Tail of the associated log file (resolved from summary dir)
    3. Return None if the operator passed (stderr noise is not an error)
    """
    passed_key = "benchmark_passed" if key == "benchmark" else "accuracy_passed"
    passed = op_info.get(passed_key)
    info = op_info.get(key, {}) or {}

    # If passed, don't extract errors
    if passed is not False:
        return None

    # Try stderr_tail first
    stderr = info.get("stderr_tail", "") or ""
    error = _search_error(stderr)
    if error:
        return error

    # Resolve log file path relative to summary dir
    log_file = _resolve_log_file(
        summary_dir, info.get("log_file", "") or "", op_name, key
    )
    if log_file:
        tail = _tail_file(log_file)
        error = _search_error(tail)
        if error:
            return error
        # Last resort: last meaningful line from log tail
        for line in reversed(tail.strip().split("\n")):
            line = line.strip()
            if line and len(line) > 10 and not line.startswith("#"):
                return line[:MAX_ERROR_LEN]

    return None


def _extract_bench_case_count(op_info: dict, summary_dir: Path, op_name: str) -> int | None:
    """Extract benchmark case count from operator info or log file."""
    bench_info = op_info.get("benchmark", {}) or {}
    data = bench_info.get("data", [])
    if data:
        return len(data)

    case_count = bench_info.get("case_count")
    if case_count is not None:
        return int(case_count)

    if op_info.get("benchmark_passed") is not None:
        log_file = _resolve_log_file(
            summary_dir, bench_info.get("log_file", "") or "", op_name, "benchmark"
        )
        if log_file and os.path.exists(log_file):
            try:
                with open(log_file) as f:
                    return f.read().count("SUCCESS")
            except OSError:
                pass
    return None


def _extract_bench_mean_speedup(op_info: dict) -> float | None:
    """Extract arithmetic mean speedup from benchmark data array."""
    bench_info = op_info.get("benchmark", {}) or {}
    data = bench_info.get("data", [])
    if not data:
        return None
    speedups = [r["speedup"] for r in data if "speedup" in r]
    if not speedups:
        return None
    return round(sum(speedups) / len(speedups), 3)


def _get_bench_command(backend_name: str, op_name: str) -> str:
    """Return the benchmark command template for a given backend."""
    bare = op_name.lstrip("_")
    shared_file_map = {
        "tianshu": f"pytest benchmark/test_unary_pointwise_perf.py -m {bare}",
        "muxi": f"pytest benchmark/test_unary_pointwise_perf.py -m {bare}",
        "ascend": f"pytest benchmark/test_unary_pointwise_perf.py -m {bare}",
        "hygon": f"pytest benchmark/test_unary_pointwise_perf.py -m {bare}",
    }
    return shared_file_map.get(backend_name, "")


def query_domestic_gpu(op_name: str) -> dict:
    """Query all domestic GPU backends for the given operator."""
    result = {}
    for pattern, backend_name, try_aten in BACKEND_CONFIGS:
        dirs = _find_dirs(pattern)
        if not dirs:
            result[backend_name] = {
            "found": False,
            "accuracy_passed": None,
            "benchmark_passed": None,
            "test_error": None,
            "bench_error": None,
            "bench_case_count": None,
            "bench_mean_speedup": None,
            "bench_command": _get_bench_command(backend_name, op_name),
        }
            continue

        dir_path = dirs[-1]  # use latest
        data = _load_summary(dir_path)
        if not data:
            result[backend_name] = {
                "found": False,
                "accuracy_passed": None,
                "benchmark_passed": None,
                "test_error": "summary JSON not found",
                "bench_error": None,
                "bench_case_count": None,
                "bench_mean_speedup": None,
                "bench_command": _get_bench_command(backend_name, op_name),
            }
            continue

        op = _lookup_operator(op_name, data, try_aten)
        if not op:
            result[backend_name] = {
                "found": False,
                "accuracy_passed": None,
                "benchmark_passed": None,
                "test_error": "operator not in summary",
                "bench_error": None,
                "bench_case_count": None,
                "bench_mean_speedup": None,
                "bench_command": _get_bench_command(backend_name, op_name),
            }
            continue

        result[backend_name] = {
            "found": True,
            "accuracy_passed": op.get("accuracy_passed"),
            "benchmark_passed": op.get("benchmark_passed"),
            "test_error": _extract_failure_reason(op, op_name, dir_path, "test"),
            "bench_error": _extract_failure_reason(op, op_name, dir_path, "benchmark"),
            "bench_case_count": _extract_bench_case_count(op, dir_path, op_name),
            "bench_mean_speedup": _extract_bench_mean_speedup(op),
            "bench_command": _get_bench_command(backend_name, op_name),
        }

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(
        description="Generate PR description data for FlagGems operator PRs"
    )
    p.add_argument("op_name", help="Operator name (e.g., special_bessel_j1)")
    p.add_argument(
        "--repo",
        default=os.environ.get("FLAGGEMS_REPO", "/workspace/FlagGems_minimax_2_7_pr"),
        help="Path to FlagGems repo",
    )
    p.add_argument("--nvidia-log", help="Path to Nvidia benchmark log file")
    p.add_argument(
        "--nvidia-stdin",
        action="store_true",
        help="Read Nvidia benchmark output from stdin",
    )
    p.add_argument(
        "--skip-run",
        action="store_true",
        help="Skip running benchmark (use domestic data only)",
    )
    p.add_argument(
        "--output", "-o",
        help="Output file (default: stdout)",
    )
    args = p.parse_args()

    op_name = args.op_name
    repo_dir = Path(args.repo).resolve()

    # --- Nvidia benchmark ---
    nvidia_rows = []
    nvidia_status = "skipped"
    nvidia_cmd = ""

    if args.nvidia_stdin:
        nvidia_rows = parse_benchmark_output(sys.stdin.read())
        nvidia_status = "parsed"
    elif args.nvidia_log:
        with open(args.nvidia_log) as f:
            nvidia_rows = parse_benchmark_output(f.read())
        nvidia_status = "parsed"
    elif not args.skip_run:
        try:
            raw = run_nvidia_benchmark(repo_dir, op_name)
            bench_file = find_benchmark_file(repo_dir, op_name)
            nvidia_cmd = (
                f"pytest {bench_file.relative_to(repo_dir)} --level core -s"
                if bench_file
                else ""
            )
            nvidia_rows = parse_benchmark_output(raw)
            nvidia_status = "success" if nvidia_rows else "no_data"
        except Exception as e:
            nvidia_status = f"error: {e}"

    # --- Compute stats ---
    am_speedup = 0.0
    if nvidia_rows:
        am_speedup = sum(r["speedup"] for r in nvidia_rows) / len(nvidia_rows)

    nvidia_case_count = len(nvidia_rows) if nvidia_rows else 0

    # --- Domestic GPU ---
    domestic = query_domestic_gpu(op_name)

    # --- Warnings for case count mismatches ---
    warnings = []
    for backend_name, backend_data in domestic.items():
        domestic_count = backend_data.get("bench_case_count")
        if domestic_count is not None and nvidia_case_count > 0:
            if domestic_count != nvidia_case_count:
                warnings.append(
                    f"Case count mismatch: Nvidia={nvidia_case_count}, "
                    f"{backend_name}={domestic_count}. "
                    f"Domestic GPU backends use legacy shared benchmark structure, "
                    f"case count may differ."
                )

    # --- Assemble output ---
    output = {
        "operator": op_name,
        "nvidia_benchmark": {
            "status": nvidia_status,
            "command": nvidia_cmd,
            "level": "core",
            "rows": nvidia_rows,
            "case_count": nvidia_case_count,
            "arithmetic_mean_speedup": round(am_speedup, 3),
        },
        "domestic_gpu": domestic,
        "warnings": warnings,
    }

    json_str = json.dumps(output, ensure_ascii=False, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(json_str)
        print(f"Written to {args.output}", file=sys.stderr)
    else:
        print(json_str)


if __name__ == "__main__":
    main()
