#!/usr/bin/env python3
"""
Select the correct FlagTree wheel specifier based on GPU vendor, Python version,
and glibc version. Run anywhere (host or container).

Usage:
    python3 select_flagtree_wheel.py --vendor ascend --python 3.11 --glibc 2.34

Outputs JSON with the wheel specifier or an error report.
"""

import argparse
import json
import sys

# Wheel selection matrix: (vendor, python_versions, min_glibc, wheel_specifier)
WHEEL_MATRIX = [
    ("nvidia",     ["3.10", "3.11", "3.12"], "2.30", "flagtree==0.4.0"),
    ("nvidia_3.2", ["3.10", "3.11", "3.12"], "2.30", "flagtree==0.4.0+3.2"),
    ("nvidia_3.3", ["3.10", "3.11", "3.12"], "2.30", "flagtree==0.4.0+3.3"),
    ("nvidia_3.5", ["3.12"],                  "2.39", "flagtree==0.4.1+3.5"),
    ("iluvatar",   ["3.10"],                  "2.35", "flagtree==0.4.0+iluvatar3.1"),
    ("mthreads",   ["3.10"],                  "2.35", "flagtree==0.4.0+mthreads3.1"),
    ("metax",      ["3.10"],                  "2.39", "flagtree==0.4.0rc1+metax3.1"),
    ("ascend",     ["3.11"],                  "2.34", "flagtree==0.4.1+ascend3.2"),
    ("hcu",        ["3.10"],                  "2.35", "flagtree==0.4.0+hcu3.0"),
    ("enflame",    ["3.10"],                  "2.35", "flagtree==0.4.0+enflame3.3"),
    ("tsingmicro", ["3.10"],                  "2.30", "flagtree==0.4.0+tsingmicro3.3"),
    ("sunrise",    ["3.10"],                  "2.39", "flagtree==0.4.0+sunrise3.4"),
]

FLAGOS_PYPI = "--index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple --trusted-host=resource.flagos.net"


def version_ge(actual, required):
    """Check if actual version >= required version."""
    a = tuple(int(x) for x in actual.split("."))
    r = tuple(int(x) for x in required.split("."))
    return a >= r


def select_wheel(vendor, python_ver, glibc_ver):
    """Find matching wheel. Returns (wheel_specifier, pip_extra_args) or None."""
    candidates = []
    for row_vendor, py_versions, min_glibc, specifier in WHEEL_MATRIX:
        # Match vendor (nvidia matches nvidia, nvidia_3.2, nvidia_3.3, etc.)
        if row_vendor == vendor or row_vendor.startswith(vendor + "_"):
            if python_ver in py_versions and version_ge(glibc_ver, min_glibc):
                candidates.append({
                    "vendor_variant": row_vendor,
                    "specifier": specifier,
                    "min_glibc": min_glibc,
                })

    if not candidates:
        return None

    # For nvidia, prefer the base variant (no suffix) unless specific variant requested
    # Return the last match (most specific / highest version)
    return candidates[-1]


def main():
    parser = argparse.ArgumentParser(description="Select FlagTree wheel")
    parser.add_argument("--vendor", required=True, help="GPU vendor name")
    parser.add_argument("--python", required=True, help="Python version (e.g. 3.11)")
    parser.add_argument("--glibc", required=True, help="glibc version (e.g. 2.34)")
    args = parser.parse_args()

    match = select_wheel(args.vendor, args.python, args.glibc)

    if match:
        result = {
            "status": "FOUND",
            "specifier": match["specifier"],
            "pip_args": FLAGOS_PYPI,
            "install_cmd": f"python3 -m pip install {match['specifier']} {FLAGOS_PYPI}",
        }
    else:
        # Collect available options for this vendor
        available = [
            {"specifier": s, "python": pv, "min_glibc": mg}
            for v, pv, mg, s in WHEEL_MATRIX
            if v == args.vendor or v.startswith(args.vendor + "_")
        ]
        result = {
            "status": "NOT_FOUND",
            "vendor": args.vendor,
            "python_version": args.python,
            "glibc_version": args.glibc,
            "available_for_vendor": available,
            "suggestion": "No pre-compiled wheel matches this combination. "
                          "Build from source or request a wheel.",
        }

    print(json.dumps(result, indent=2))
    return 0 if match else 1


if __name__ == "__main__":
    sys.exit(main())
