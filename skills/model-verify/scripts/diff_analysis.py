#!/usr/bin/env python3
"""
Compare results from two verification runs (without vs with multi-chip stack)
and produce a diff analysis report.

Usage:
    python3 diff_analysis.py --run-a run_a.json --run-b run_b.json

Input: JSON files from run_offline_inference.py / test_serve_mode.py
Output: JSON diff analysis with layer classification.
"""

import argparse
import json
import sys


MULTICHIP_ERROR_PATTERNS = [
    ("FlagGems", ["flag_gems", "FlagGems", "flaggems"]),
    ("FlagTree", ["triton", "Triton", "compilation error", "flagtree"]),
    ("FlagCX", ["flagcx", "FlagCX", "all_reduce", "all_gather", "broadcast"]),
    ("vllm-plugin-FL", ["vllm_fl", "dispatch error", "platform_plugins"]),
    ("Numerical", ["NaN", "nan", "inf", "numerical", "mismatch"]),
]


def classify_multichip_error(error_text):
    """Classify a multi-chip error by component."""
    if not error_text:
        return "unknown"
    for component, patterns in MULTICHIP_ERROR_PATTERNS:
        if any(p in error_text for p in patterns):
            return component
    return "unknown"


def diff_runs(run_a, run_b):
    """Compare two runs and produce analysis."""
    a_pass = run_a.get("status") == "PASS"
    b_pass = run_b.get("status") == "PASS"

    if a_pass and b_pass:
        conclusion = "BOTH_PASS"
        detail = "Full multi-chip stack works end-to-end"
        recommended_stack = "full"
    elif a_pass and not b_pass:
        conclusion = "MULTICHIP_ERROR"
        error_text = run_b.get("error", "") or run_b.get("traceback", "")
        component = classify_multichip_error(error_text)
        detail = (f"Base stack works, multi-chip stack fails. "
                  f"Error component: {component}")
        recommended_stack = "base"
    elif not a_pass and not b_pass:
        a_err = run_a.get("error", "")
        b_err = run_b.get("error", "")
        if a_err == b_err:
            conclusion = "SAME_ERROR"
            detail = "Same error in both runs — NOT a multi-chip issue"
        else:
            conclusion = "DIFFERENT_ERRORS"
            detail = "Different errors — base issue AND multi-chip issue"
        recommended_stack = "none"
    else:  # not a_pass and b_pass
        conclusion = "UNEXPECTED"
        detail = "Base stack fails but multi-chip works — unusual, investigate"
        recommended_stack = "full"

    return {
        "conclusion": conclusion,
        "detail": detail,
        "recommended_stack": recommended_stack,
        "run_a_status": run_a.get("status"),
        "run_b_status": run_b.get("status"),
        "run_a_error": run_a.get("error"),
        "run_b_error": run_b.get("error"),
        "multichip_component": (
            classify_multichip_error(run_b.get("error", "") or run_b.get("traceback", ""))
            if a_pass and not b_pass else None
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Diff two verification runs")
    parser.add_argument("--run-a", required=True, help="Run A (base stack) JSON file")
    parser.add_argument("--run-b", required=True, help="Run B (full stack) JSON file")
    args = parser.parse_args()

    with open(args.run_a) as f:
        run_a = json.load(f)
    with open(args.run_b) as f:
        run_b = json.load(f)

    result = diff_runs(run_a, run_b)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
