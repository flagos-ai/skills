#!/usr/bin/env python3
"""
Run vllm bench serve for a single profile and parse results.
Run INSIDE the container. Outputs JSON with parsed metrics.

Usage:
    python3 run_benchmark.py --model <name> --tokenizer <path> \
        --input-len 256 --output-len 128 --num-prompts 100 \
        --profile-name short_prefill_short_decode \
        [--port 8000] [--extra-args '...']
"""

import argparse
import json
import re
import subprocess
import sys


def parse_bench_output(text):
    """Parse vllm bench serve output for key metrics."""
    metrics = {}

    patterns = {
        "throughput_req_s": r"Successful requests per second:\s*([\d.]+)",
        "throughput_tok_s": r"Output token throughput.*?:\s*([\d.]+)",
        "ttft_mean_ms": r"Mean TTFT.*?:\s*([\d.]+)",
        "ttft_p50_ms": r"P50 TTFT.*?:\s*([\d.]+)",
        "ttft_p99_ms": r"P99 TTFT.*?:\s*([\d.]+)",
        "tpot_mean_ms": r"Mean TPOT.*?:\s*([\d.]+)",
        "tpot_p50_ms": r"P50 TPOT.*?:\s*([\d.]+)",
        "tpot_p99_ms": r"P99 TPOT.*?:\s*([\d.]+)",
        "latency_mean_ms": r"Mean latency.*?:\s*([\d.]+)",
        "latency_p50_ms": r"P50 latency.*?:\s*([\d.]+)",
        "latency_p90_ms": r"P90 latency.*?:\s*([\d.]+)",
        "latency_p99_ms": r"P99 latency.*?:\s*([\d.]+)",
        "total_input_tokens": r"Total input tokens:\s*(\d+)",
        "total_output_tokens": r"Total generated tokens:\s*(\d+)",
        "completed_requests": r"Completed requests:\s*(\d+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            val = match.group(1)
            metrics[key] = float(val) if "." in val else int(val)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run single benchmark profile")
    parser.add_argument("--model", required=True, help="Model name (as registered in vllm)")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer path")
    parser.add_argument("--input-len", type=int, required=True)
    parser.add_argument("--output-len", type=int, required=True)
    parser.add_argument("--num-prompts", type=int, required=True)
    parser.add_argument("--profile-name", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--extra-args", default="")
    args = parser.parse_args()

    result = {
        "name": args.profile_name,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "num_prompts": args.num_prompts,
        "status": "FAIL",
        "metrics": {},
        "error": None,
        "raw_output": None,
    }

    cmd = [
        "vllm", "bench", "serve",
        "--host", "127.0.0.1",
        "--port", str(args.port),
        "--backend", "openai-chat",
        "--model", args.model,
        "--tokenizer", args.tokenizer,
        "--dataset-name", "random",
        "--endpoint", "/v1/chat/completions",
        "--ignore-eos",
        "--trust-remote-code",
        "--random-input-len", str(args.input_len),
        "--random-output-len", str(args.output_len),
        "--num-prompts", str(args.num_prompts),
    ]
    if args.extra_args:
        cmd.extend(args.extra_args.split())

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600
        )
        output = proc.stdout + proc.stderr
        result["raw_output"] = output[-2000:]  # last 2000 chars

        if proc.returncode == 0:
            metrics = parse_bench_output(output)
            if metrics:
                result["status"] = "PASS"
                result["metrics"] = metrics
            else:
                result["error"] = "Benchmark completed but no metrics found in output"
        else:
            result["error"] = f"Exit code {proc.returncode}"

    except subprocess.TimeoutExpired:
        result["error"] = "Benchmark timed out after 600s"
    except FileNotFoundError:
        result["error"] = "vllm bench serve command not found (vllm version issue?)"
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
