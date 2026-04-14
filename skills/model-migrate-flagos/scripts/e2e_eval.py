#!/usr/bin/env python3
"""E2E correctness verification: compare token-level output between GT and local vLLM servers.

Usage:
    # Minimal — just specify model, everything else has defaults
    python3 e2e_eval.py --model Qwen3.5-397B-A17B-Real --local-only

    # Full pipeline with CLI args (no config file needed)
    python3 e2e_eval.py --model MODEL_NAME --gt-host <GT_HOST_IP> --gt-port 8122 --local-port 8122

    # With config file (CLI args override config values)
    python3 e2e_eval.py --config e2e_config.json --model MODEL_NAME

    # Phases
    python3 e2e_eval.py --model MODEL_NAME --gt-only
    python3 e2e_eval.py --model MODEL_NAME --local-only
    python3 e2e_eval.py --model MODEL_NAME --compare-only

    # Prompt modes
    python3 e2e_eval.py --model MODEL_NAME --mode text          # 5 text prompts (default)
    python3 e2e_eval.py --model MODEL_NAME --mode multimodal    # 5 multimodal prompts
    python3 e2e_eval.py --model MODEL_NAME --mode all           # all 10 prompts
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PROMPTS = os.path.join(SCRIPT_DIR, "e2e_test_prompts.json")
DEFAULT_RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")


def load_config(path):
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def build_effective_config(args):
    """Build effective config by merging: defaults < config file < CLI args."""
    # Defaults
    cfg = {
        "gt_machine": {
            "host": None,
            "vllm_port": 8122,
        },
        "local": {
            "vllm_port": 8122,
        },
        "shared_storage": {
            "results_dir": DEFAULT_RESULTS_DIR,
        },
        "eval": {
            "max_tokens": 256,
            "token_match_count": 32,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 42,
        },
    }

    # Overlay config file if provided
    if args.config:
        file_cfg = load_config(args.config)
        for section in ("gt_machine", "local", "shared_storage", "eval"):
            if section in file_cfg:
                cfg[section].update(file_cfg[section])

    # CLI overrides (only if explicitly provided)
    if args.gt_host is not None:
        cfg["gt_machine"]["host"] = args.gt_host
    if args.gt_port is not None:
        cfg["gt_machine"]["vllm_port"] = args.gt_port
    if args.local_port is not None:
        cfg["local"]["vllm_port"] = args.local_port
    if args.results_dir is not None:
        cfg["shared_storage"]["results_dir"] = args.results_dir
    if args.max_tokens is not None:
        cfg["eval"]["max_tokens"] = args.max_tokens
    if args.token_match_count is not None:
        cfg["eval"]["token_match_count"] = args.token_match_count

    return cfg


def load_prompts(path, mode="text"):
    with open(path) as f:
        data = json.load(f)
    if mode == "all":
        return data.get("text", []) + data.get("multimodal", [])
    return data.get(mode, [])


def make_request(host, port, model_path, messages, eval_cfg):
    """Send a chat completion request and return token_ids + text."""
    url = f"http://{host}:{port}/v1/chat/completions"
    payload = {
        "model": model_path,
        "messages": messages,
        "max_tokens": eval_cfg["max_tokens"],
        "temperature": eval_cfg["temperature"],
        "top_p": eval_cfg["top_p"],
        "seed": eval_cfg["seed"],
        "logprobs": True,
        "top_logprobs": 1,
        "return_token_ids": True,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        return {"error": str(e), "token_ids": [], "tokens": [], "text": ""}
    except Exception as e:
        return {"error": str(e), "token_ids": [], "tokens": [], "text": ""}

    choice = result["choices"][0]
    text = choice["message"]["content"]

    # Extract token IDs from choice-level field (requires return_token_ids=True)
    token_ids = choice.get("token_ids") or []

    # Extract token strings from logprobs
    tokens = []
    logprobs_content = choice.get("logprobs", {})
    if logprobs_content and logprobs_content.get("content"):
        for entry in logprobs_content["content"]:
            tokens.append(entry.get("token", ""))

    return {"token_ids": token_ids, "tokens": tokens, "text": text}


def wait_for_server(host, port, timeout=600, interval=10):
    """Wait for vLLM server to be ready."""
    url = f"http://{host}:{port}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5):
                return True
        except Exception:
            print(f"  Waiting for server at {host}:{port}... ({int(time.time()-start)}s)")
            time.sleep(interval)
    return False


def run_inference(host, port, model_name, prompts, eval_cfg, label):
    """Run inference on all prompts and return results dict."""
    model_path = f"/models/{model_name}"
    results = {}
    total = len(prompts)
    for i, prompt in enumerate(prompts):
        pid = prompt["id"]
        print(f"  [{label}] ({i+1}/{total}) Running prompt {pid}...")
        result = make_request(host, port, model_path, prompt["messages"], eval_cfg)
        if result.get("error"):
            print(f"    ERROR: {result['error']}")
        else:
            n_tokens = len(result["tokens"])
            print(f"    Got {n_tokens} tokens")
        results[pid] = result
    return results


def save_results(results, results_dir, model_name, label):
    """Save results to JSON file."""
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"{model_name}_{label}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Results saved to {path}")
    return path


def load_results(results_dir, model_name, label):
    """Load previously saved results."""
    path = os.path.join(results_dir, f"{model_name}_{label}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def compare_results(gt_results, local_results, prompts, token_match_count):
    """Compare GT and local results, return detailed report."""
    report_lines = []
    passed = 0
    failed = 0
    errors = 0

    for prompt in prompts:
        pid = prompt["id"]
        msg = prompt["messages"][0]["content"]
        if isinstance(msg, list):
            msg = next((m["text"] for m in msg if isinstance(m, dict) and m.get("type") == "text"), str(msg))
        display_msg = msg[:60] + "..." if len(msg) > 60 else msg

        gt = gt_results.get(pid)
        local = local_results.get(pid)

        if not gt or gt.get("error"):
            report_lines.append(f'\n[{pid}] "{display_msg}"')
            report_lines.append(f"  Status: GT ERROR - {gt.get('error', 'missing')}")
            errors += 1
            continue

        if not local or local.get("error"):
            report_lines.append(f'\n[{pid}] "{display_msg}"')
            report_lines.append(f"  Status: LOCAL ERROR - {local.get('error', 'missing')}")
            errors += 1
            continue

        # Compare token IDs (primary) and token strings (secondary)
        gt_ids = gt["token_ids"] or []
        local_ids = local["token_ids"] or []
        gt_strs = gt["tokens"] or []
        local_strs = local["tokens"] or []

        # Use token_ids for comparison if available, otherwise fall back to strings
        has_ids = bool(gt_ids) and bool(local_ids)
        gt_seq = gt_ids if has_ids else gt_strs
        local_seq = local_ids if has_ids else local_strs

        n = min(token_match_count, len(gt_seq), len(local_seq))
        gt_head = gt_seq[:n]
        local_head = local_seq[:n]

        diverge_at = None
        for j in range(n):
            if gt_head[j] != local_head[j]:
                diverge_at = j
                break

        report_lines.append(f'\n[{pid}] "{display_msg}"')

        # Always show both token_ids and token_strs
        n_ids = min(token_match_count, len(gt_ids), len(local_ids)) if has_ids else 0
        n_strs = min(token_match_count, len(gt_strs), len(local_strs))
        if has_ids:
            report_lines.append(f"  GT token_ids (first {n_ids}):    {gt_ids[:n_ids]}")
            report_lines.append(f"  Local token_ids (first {n_ids}): {local_ids[:n_ids]}")
        report_lines.append(f"  GT token_strs (first {n_strs}):    {gt_strs[:n_strs]}")
        report_lines.append(f"  Local token_strs (first {n_strs}): {local_strs[:n_strs]}")

        if diverge_at is None:
            compare_type = "token_ids" if has_ids else "token_strs"
            report_lines.append(f"  Status: MATCH ({n}/{n} {compare_type} identical)")
            passed += 1
        else:
            report_lines.append(f"  Status: MISMATCH at token #{diverge_at}")
            gt_tok = gt_strs[diverge_at] if diverge_at < len(gt_strs) else "?"
            local_tok = local_strs[diverge_at] if diverge_at < len(local_strs) else "?"
            gt_id = gt_ids[diverge_at] if has_ids and diverge_at < len(gt_ids) else "n/a"
            local_id = local_ids[diverge_at] if has_ids and diverge_at < len(local_ids) else "n/a"
            report_lines.append(f'  GT     token #{diverge_at}: "{gt_tok}" (id={gt_id})')
            report_lines.append(f'  Local  token #{diverge_at}: "{local_tok}" (id={local_id})')
            failed += 1

        gt_raw = gt.get("text") or ""
        local_raw = local.get("text") or ""
        gt_text = gt_raw[:120] + "..." if len(gt_raw) > 120 else gt_raw
        local_text = local_raw[:120] + "..." if len(local_raw) > 120 else local_raw
        report_lines.append(f"  GT text:    {gt_text}")
        report_lines.append(f"  Local text: {local_text}")

    total = passed + failed + errors
    report_lines.insert(0, "")
    report_lines.append("")
    report_lines.append("=" * 40)
    report_lines.append(f"  Passed: {passed}/{total} | Failed: {failed}/{total} | Errors: {errors}/{total}")

    if failed == 0 and errors == 0:
        report_lines.append("  VERDICT: PASS - migration correctness verified")
        verdict = True
    else:
        report_lines.append("  VERDICT: FAIL - migration needs investigation")
        verdict = False

    report_lines.append("=" * 40)

    return "\n".join(report_lines), verdict, {"passed": passed, "failed": failed, "errors": errors}


def main():
    parser = argparse.ArgumentParser(description="E2E correctness verification for vLLM model migration")
    parser.add_argument("--model", required=True, help="Model display name (e.g. Qwen3.5-397B-A17B-Real)")

    # Config file (optional — CLI args can replace it entirely)
    parser.add_argument("--config", default=None, help="Path to e2e_config.json (optional, CLI args override)")
    parser.add_argument("--prompts", default=DEFAULT_PROMPTS, help="Path to test prompts JSON")

    # Server endpoints
    parser.add_argument("--gt-host", default=None, help="GT server host (required, or set gt_machine.host in config file)")
    parser.add_argument("--gt-port", type=int, default=None, help="GT server port (default: 8122)")
    parser.add_argument("--local-port", type=int, default=None, help="Local server port (default: 8122)")

    # Eval params
    parser.add_argument("--max-tokens", type=int, default=None, help="Max tokens to generate (default: 256)")
    parser.add_argument("--token-match-count", type=int, default=None, help="Number of tokens to compare (default: 32)")
    parser.add_argument("--results-dir", default=None, help="Directory to save results")

    # Mode
    parser.add_argument("--mode", choices=["text", "multimodal", "all"], default="text",
                        help="Which prompts to use (default: text)")

    # Phase control
    parser.add_argument("--gt-only", action="store_true", help="Only generate GT results")
    parser.add_argument("--local-only", action="store_true", help="Only generate local results")
    parser.add_argument("--compare-only", action="store_true", help="Only compare existing results")

    args = parser.parse_args()

    cfg = build_effective_config(args)
    prompts = load_prompts(args.prompts, args.mode)
    if not prompts:
        print(f"ERROR: No prompts found for mode '{args.mode}'")
        sys.exit(1)

    gt_cfg = cfg["gt_machine"]
    local_cfg = cfg["local"]
    eval_cfg = cfg["eval"]
    results_dir = cfg["shared_storage"]["results_dir"]

    print("=" * 50)
    print("  E2E Correctness Verification")
    print(f"  Model: {args.model}")
    print(f"  Mode: {args.mode} ({len(prompts)} prompts)")
    print(f"  Token match threshold: first {eval_cfg['token_match_count']} tokens")
    print(f"  GT server: {gt_cfg['host']}:{gt_cfg['vllm_port']}")
    print(f"  Local server: localhost:{local_cfg['vllm_port']}")
    print(f"  Results dir: {results_dir}")
    print("=" * 50)

    gt_results = None
    local_results = None

    # --- GT inference ---
    if not args.local_only and not args.compare_only:
        if not gt_cfg["host"] or str(gt_cfg["host"]).startswith("<"):
            print("ERROR: GT host not configured.")
            print("  Use --gt-host <IP> or set gt_machine.host in config file.")
            print("  Example: python3 e2e_eval.py --model MODEL --gt-host 10.0.1.100")
            sys.exit(1)
        print(f"\n[Phase 1] Generating GT results on {gt_cfg['host']}...")
        if not wait_for_server(gt_cfg["host"], gt_cfg["vllm_port"], timeout=600):
            print(f"ERROR: GT server at {gt_cfg['host']}:{gt_cfg['vllm_port']} not reachable")
            sys.exit(1)
        gt_results = run_inference(gt_cfg["host"], gt_cfg["vllm_port"], args.model, prompts, eval_cfg, "GT")
        save_results(gt_results, results_dir, args.model, "gt")

    if args.gt_only:
        print("\n[GT-only mode] Done. Run again without --gt-only to continue.")
        sys.exit(0)

    # --- Local inference ---
    if not args.gt_only and not args.compare_only:
        print(f"\n[Phase 2] Generating local results on localhost:{local_cfg['vllm_port']}...")
        if not wait_for_server("localhost", local_cfg["vllm_port"], timeout=600):
            print(f"ERROR: Local server at localhost:{local_cfg['vllm_port']} not reachable")
            sys.exit(1)
        local_results = run_inference("localhost", local_cfg["vllm_port"], args.model, prompts, eval_cfg, "LOCAL")
        save_results(local_results, results_dir, args.model, "local")

    if args.local_only:
        print("\n[Local-only mode] Done. Run again without --local-only to continue.")
        sys.exit(0)

    # --- Compare ---
    print("\n[Phase 3] Comparing results...")
    if gt_results is None:
        gt_results = load_results(results_dir, args.model, "gt")
        if gt_results is None:
            print(f"ERROR: GT results not found at {results_dir}/{args.model}_gt.json")
            sys.exit(1)
    if local_results is None:
        local_results = load_results(results_dir, args.model, "local")
        if local_results is None:
            print(f"ERROR: Local results not found at {results_dir}/{args.model}_local.json")
            sys.exit(1)

    report, verdict, stats = compare_results(gt_results, local_results, prompts, eval_cfg["token_match_count"])

    print("\n" + "=" * 50)
    print("  E2E Correctness Report")
    print("=" * 50)
    print(report)

    # Save report
    os.makedirs(results_dir, exist_ok=True)
    report_path = os.path.join(results_dir, f"{args.model}_e2e_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")

    sys.exit(0 if verdict else 1)


if __name__ == "__main__":
    main()
