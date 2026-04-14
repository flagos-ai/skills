#!/usr/bin/env python3
"""
Validate all installed packages and produce a structured report.
Run INSIDE the container. Outputs JSON with per-package status.
"""

import importlib
import json
import subprocess
import sys


def check_import(module_name):
    """Try importing a module. Returns (success, version_or_error)."""
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, "__version__", None)
        if version is None:
            # Try pip show
            r = subprocess.run(
                [sys.executable, "-m", "pip", "show", module_name],
                capture_output=True, text=True, timeout=10
            )
            for line in r.stdout.split("\n"):
                if line.startswith("Version:"):
                    version = line.split(":", 1)[1].strip()
                    break
        return True, version
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def check_vllm():
    ok, ver = check_import("vllm")
    result = {"importable": ok, "version": ver}
    if ok:
        result["version_match"] = ver == "0.13.0"
    return result


def check_flagtree():
    """FlagTree replaces triton — check via triton import path."""
    result = {"importable": False, "version": None}
    try:
        import triton
        path = str(triton.__path__)
        result["importable"] = True
        result["triton_path"] = path
        result["is_flagtree"] = "flagtree" in path.lower() or "flag_tree" in path.lower()
        # Get version from pip
        r = subprocess.run(
            [sys.executable, "-m", "pip", "show", "flagtree"],
            capture_output=True, text=True, timeout=10
        )
        for line in r.stdout.split("\n"):
            if line.startswith("Version:"):
                result["version"] = line.split(":", 1)[1].strip()
                break
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
    return result


def check_flaggems():
    ok, ver = check_import("flag_gems")
    return {"importable": ok, "version": ver if ok else None, "error": ver if not ok else None}


def check_flagcx():
    ok, ver = check_import("flagcx")
    return {"importable": ok, "version": ver if ok else None, "error": ver if not ok else None}


def check_vllm_plugin_fl():
    result = {"importable": False, "registered": False}
    try:
        result["importable"] = True

        from importlib.metadata import entry_points
        eps = entry_points()
        platform_eps = [e for e in eps.get("vllm.platform_plugins", []) if e.name == "fl"]
        general_eps = [e for e in eps.get("vllm.general_plugins", []) if e.name == "fl"]
        result["registered"] = bool(platform_eps and general_eps)
        result["platform_plugins"] = len(platform_eps)
        result["general_plugins"] = len(general_eps)
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
    return result


def main():
    report = {
        "vllm": check_vllm(),
        "flagtree": check_flagtree(),
        "flaggems": check_flaggems(),
        "flagcx": check_flagcx(),
        "vllm_plugin_fl": check_vllm_plugin_fl(),
    }

    # Determine overall status
    gate_ok = (
        report["vllm"].get("importable") and
        report["vllm_plugin_fl"].get("importable") and
        report["vllm_plugin_fl"].get("registered")
    )
    all_ok = gate_ok and all(
        report[pkg].get("importable")
        for pkg in ["flagtree", "flaggems", "flagcx"]
    )

    report["overall"] = {
        "status": "PASS" if all_ok else ("PARTIAL" if gate_ok else "FAIL"),
        "gate_passed": gate_ok,
    }

    print(json.dumps(report, indent=2))
    return 0 if gate_ok else 1


if __name__ == "__main__":
    sys.exit(main())
