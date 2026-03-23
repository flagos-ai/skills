#!/usr/bin/env python3
"""
Collect environment info needed for package installation decisions.
Run INSIDE the container. Outputs JSON with python version, glibc version,
architecture, and GPU vendor (if detectable).
"""

import json
import os
import platform
import subprocess
import sys


def get_python_version():
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def get_glibc_version():
    try:
        result = subprocess.run(
            ["ldd", "--version"], capture_output=True, text=True, timeout=5
        )
        output = result.stdout + result.stderr
        for line in output.split("\n"):
            if "ldd" in line.lower() or "glibc" in line.lower():
                parts = line.strip().split()
                for part in reversed(parts):
                    try:
                        float(part)
                        return part
                    except ValueError:
                        continue
    except Exception:
        pass
    return None


def get_arch():
    return platform.machine()


def detect_vendor():
    """Quick GPU vendor detection inside container."""
    checks = [
        ("nvidia-smi", "nvidia"),
        ("npu-smi info -l", "ascend"),
        ("mx-smi -L", "metax"),
        ("ixsmi -L", "iluvatar"),
    ]
    for cmd, vendor in checks:
        try:
            r = subprocess.run(cmd, shell=True, capture_output=True, timeout=5)
            if r.returncode == 0 and r.stdout.strip():
                return vendor
        except Exception:
            continue

    # Check for Hygon DCU
    import glob
    dtk_paths = glob.glob("/opt/dtk-*/bin/rocm-smi")
    if dtk_paths:
        return "hcu"

    # Check for AMD ROCm (non-Hygon)
    if os.path.exists("/opt/rocm/bin/rocm-smi"):
        return "amd"

    return "unknown"


def get_disk_free_gb(path="/"):
    """Get free disk space in GB."""
    try:
        st = os.statvfs(path)
        return round(st.f_bavail * st.f_frsize / (1024**3), 1)
    except Exception:
        return None


def main():
    result = {
        "python_version": get_python_version(),
        "glibc_version": get_glibc_version(),
        "arch": get_arch(),
        "vendor": detect_vendor(),
        "disk_free_gb": get_disk_free_gb(),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
