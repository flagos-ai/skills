#!/usr/bin/env python3
"""
Run all benchmark profiles and collect results.
Run INSIDE the container (requires vllm server already running).
Outputs JSON with all profile results and summary table.

Usage:
    python3 run_all_benchmarks.py --model <name> --tokenizer <path> \
        [--port 8000] [--output-dir /data/results/perf]
"""

import argparse
import json
import os
import subprocess
import sys

DEFAULT_PROFILES = [
    {"name": "short_prefill_short_decode", "input_len": 256,  "output_len": 128,  "num_prompts": 100},
    {"name": "long_prefill_short_decode",  "input_len": 4096, "output_len": 128,  "num_prompts": 100},
    {"name": "short_prefill_long_decode",  "input_len": 256,  "output_len": 1024, "num_prompts": 100},
    {"name": "long_prefill_long_decode",   "input_len": 4096, "output_len": 1024, "num_prompts": 100},
    {"name": "high_concurrency",           "input_len": 1024, "output_len": 512,  "num_prompts": 500},
]


def run_single_profile(profile, model, tokenizer, port, output_dir, extra_args=""):
    """Run a single benchmark profile using run_benchmark.py."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cmd = [
        sys.executable, os.path.join(script_dir, "run_benchmark.py"),
        "--model", model,
        "--tokenizer", tokenizer,
        "--input-len", str(profile["input_len"]),
        "--output-len", str(profile["output_len"]),
        "--num-prompts", str(profile["num_prompts"]),
        "--profile-name", profile["name"],
        "--port", str(port),
    ]
    if extra_args:
        cmd.extend(["--extra-args", extra_args])

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=660)
        result = json.loads(proc.stdout)
    except subprocess.TimeoutExpired:
        result = {"name": profile["name"], "status": "FAIL", "error": "Timed out"}
    except (json.JSONDecodeError, Exception) as e:
        result = {"name": profile["name"], "status": "FAIL", "error": str(e)}

    # Save individual result
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"{profile['name']}.json"), "w") as f:
            json.dump(result, f, indent=2)

    return result


def format_summary_table(results):
    """Format results as a markdown table."""
    header = "| Profile | Input | Output | Prompts | Req/s | Tok/s | TTFT(ms) | TPOT(ms) | P99(ms) | Status |"
    sep =    "|---------|-------|--------|---------|-------|-------|----------|----------|---------|--------|"
    rows = [header, sep]

    for r in results:
        m = r.get("metrics", {})
        rows.append(
            f"| {r['name']} "
            f"| {r.get('input_len', '-')} "
            f"| {r.get('output_len', '-')} "
            f"| {r.get('num_prompts', '-')} "
            f"| {m.get('throughput_req_s', '-')} "
            f"| {m.get('throughput_tok_s', '-')} "
            f"| {m.get('ttft_mean_ms', '-')} "
            f"| {m.get('tpot_mean_ms', '-')} "
            f"| {m.get('latency_p99_ms', '-')} "
            f"| {r.get('status', 'FAIL')} |"
        )
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description="Run all benchmark profiles")
    parser.add_argument("--model", required=True, help="Model name in vllm server")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer path")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output-dir", default="/data/results/perf")
    parser.add_argument("--extra-args", default="")
    args = parser.parse_args()

    results = []
    for profile in DEFAULT_PROFILES:
        print(f"=== Running profile: {profile['name']} ===", file=sys.stderr)
        result = run_single_profile(
            profile, args.model, args.tokenizer, args.port,
            args.output_dir, args.extra_args
        )
        results.append(result)
        print(f"=== Profile {profile['name']}: {result.get('status')} ===", file=sys.stderr)

    passed = sum(1 for r in results if r.get("status") == "PASS")
    total = len(results)

    report = {
        "status": "PASS" if passed == total else ("PARTIAL" if passed > 0 else "FAIL"),
        "profiles_passed": f"{passed}/{total}",
        "profiles": results,
        "summary_table": format_summary_table(results),
    }

    print(json.dumps(report, indent=2))
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
