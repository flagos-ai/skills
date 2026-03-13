#!/usr/bin/env python3
"""
GPU Vendor Detection Script
Detects GPU vendor and device information for multi-vendor container setup.
"""

import json
import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, timeout=10):
    """Run a command and return (success, output)."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return result.returncode == 0, result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False, ""


def detect_nvidia():
    """Detect NVIDIA GPUs using nvidia-smi."""
    success, output = run_command("nvidia-smi --query-gpu=name --format=csv,noheader")
    if success and output:
        devices = [line.strip() for line in output.split("\n") if line.strip()]
        return {"vendor": "nvidia", "devices": devices, "count": len(devices)}
    return None


def detect_amd():
    """Detect AMD GPUs using rocm-smi or /opt/rocm."""
    # Check rocm-smi first
    success, output = run_command("rocm-smi --showproductname")
    if success:
        devices = []
        for line in output.split("\n"):
            if "GPU" in line and ":" in line:
                devices.append(line.split(":")[-1].strip())
        if devices:
            return {"vendor": "amd", "devices": devices, "count": len(devices)}

    # Check /opt/rocm directory
    if Path("/opt/rocm").exists():
        # Try to get device count from /sys
        gpu_count = len(list(Path("/sys/class/drm").glob("card[0-9]*")))
        if gpu_count > 0:
            return {"vendor": "amd", "devices": ["ROCm GPU"] * gpu_count, "count": gpu_count}

    return None


def detect_ascend():
    """Detect Huawei Ascend NPUs using npu-smi or /usr/local/Ascend."""
    # Check npu-smi first
    success, output = run_command("npu-smi info -l")
    if success and output:
        devices = []
        count = 0
        for line in output.split("\n"):
            if "NPU" in line:
                count += 1
            if "Name" in line and ":" in line:
                devices.append(line.split(":")[-1].strip())
        if count > 0:
            if not devices:
                devices = ["Ascend NPU"] * count
            return {"vendor": "ascend", "devices": devices, "count": count}

    # Check /usr/local/Ascend directory
    if Path("/usr/local/Ascend").exists():
        # Try to detect device count from /dev/davinci*
        davinci_devices = list(Path("/dev").glob("davinci[0-9]*"))
        count = len(davinci_devices) if davinci_devices else 1
        return {"vendor": "ascend", "devices": ["Ascend NPU"] * count, "count": count}

    return None


def detect_metax():
    """Detect Metax GPUs using mx-smi or /opt/metax."""
    # Check mx-smi first
    success, output = run_command("mx-smi -L")
    if success and output:
        devices = []
        for line in output.split("\n"):
            if "GPU" in line or "MX" in line:
                devices.append(line.strip())
        if devices:
            return {"vendor": "metax", "devices": devices, "count": len(devices)}

    # Check /opt/metax directory
    if Path("/opt/metax").exists():
        # Try to detect device count from /dev/mx*
        mx_devices = list(Path("/dev").glob("mx[0-9]*"))
        count = len(mx_devices) if mx_devices else 1
        return {"vendor": "metax", "devices": ["Metax GPU"] * count, "count": count}

    return None


def detect_iluvatar():
    """Detect Iluvatar GPUs using ixsmi or /opt/iluvatar."""
    # Check ixsmi first
    success, output = run_command("ixsmi -L")
    if success and output:
        devices = []
        for line in output.split("\n"):
            if "GPU" in line or "BI" in line or "Iluvatar" in line:
                devices.append(line.strip())
        if devices:
            return {"vendor": "iluvatar", "devices": devices, "count": len(devices)}

    # Check /opt/iluvatar directory
    if Path("/opt/iluvatar").exists():
        # Try to detect device count from /dev/bi*
        bi_devices = list(Path("/dev").glob("bi[0-9]*"))
        count = len(bi_devices) if bi_devices else 1
        return {"vendor": "iluvatar", "devices": ["Iluvatar GPU"] * count, "count": count}

    return None


def detect_gpu():
    """
    Detect GPU vendor in priority order.
    Returns dict with vendor, devices list, and count.
    """
    # Detection order: NVIDIA (most common) -> AMD -> Ascend -> Metax -> Iluvatar
    detectors = [
        ("nvidia", detect_nvidia),
        ("amd", detect_amd),
        ("ascend", detect_ascend),
        ("metax", detect_metax),
        ("iluvatar", detect_iluvatar),
    ]

    for vendor_name, detector in detectors:
        result = detector()
        if result:
            return result

    return {"vendor": "unknown", "devices": [], "count": 0}


def main():
    """Main entry point."""
    result = detect_gpu()
    print(json.dumps(result, indent=2))
    return 0 if result["vendor"] != "unknown" else 1


if __name__ == "__main__":
    sys.exit(main())
