#!/usr/bin/env python3
"""
Data Disk Detection Script
Finds the most appropriate data disk mount point for container volumes.
"""

import json
import subprocess
import sys
from pathlib import Path


def parse_df_output():
    """Parse df -h output to get filesystem information."""
    try:
        result = subprocess.run(
            ["df", "-h"], capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return []

        lines = result.stdout.strip().split("\n")
        if len(lines) < 2:
            return []

        filesystems = []
        for line in lines[1:]:  # Skip header
            parts = line.split()
            if len(parts) >= 6:
                # Format: Filesystem Size Used Avail Use% Mounted
                fs = {
                    "filesystem": parts[0],
                    "size": parts[1],
                    "used": parts[2],
                    "avail": parts[3],
                    "use_percent": parts[4],
                    "mount": parts[5],
                }
                filesystems.append(fs)

        return filesystems
    except Exception:
        return []


def parse_size_to_gb(size_str):
    """Convert size string (e.g., '500G', '1.5T') to GB."""
    size_str = size_str.upper().strip()
    try:
        if size_str.endswith("T"):
            return float(size_str[:-1]) * 1024
        elif size_str.endswith("G"):
            return float(size_str[:-1])
        elif size_str.endswith("M"):
            return float(size_str[:-1]) / 1024
        elif size_str.endswith("K"):
            return float(size_str[:-1]) / (1024 * 1024)
        else:
            return float(size_str) / (1024 * 1024 * 1024)  # Assume bytes
    except ValueError:
        return 0


def is_excluded_mount(mount_point, filesystem):
    """Check if mount point should be excluded."""
    excluded_mounts = {"/", "/boot", "/boot/efi", "/tmp", "/var", "/run"}
    excluded_fs_types = {"tmpfs", "devtmpfs", "overlay", "shm"}

    # Exclude system mounts
    if mount_point in excluded_mounts:
        return True

    # Exclude special filesystems
    if filesystem in excluded_fs_types or filesystem.startswith("tmpfs"):
        return True

    # Exclude /sys, /proc, /dev paths
    if mount_point.startswith(("/sys", "/proc", "/dev")):
        return True

    return False


def score_mount_point(fs_info):
    """
    Score a mount point for suitability as data disk.
    Higher score = better candidate.
    """
    mount = fs_info["mount"]
    avail_gb = parse_size_to_gb(fs_info["avail"])

    score = 0

    # Preferred paths get bonus points
    if mount == "/data":
        score += 100
    elif mount.startswith("/data"):
        score += 80
    elif mount.startswith("/mnt"):
        score += 60
    elif mount.startswith("/home"):
        score += 40
    elif mount.startswith("/opt"):
        score += 20

    # Size-based scoring (prefer larger disks)
    if avail_gb >= 500:
        score += 50
    elif avail_gb >= 200:
        score += 30
    elif avail_gb >= 100:
        score += 20
    elif avail_gb >= 50:
        score += 10

    # Penalize nearly full disks
    try:
        use_pct = int(fs_info["use_percent"].rstrip("%"))
        if use_pct > 90:
            score -= 50
        elif use_pct > 80:
            score -= 20
    except ValueError:
        pass

    return score


def find_data_disk():
    """
    Find the most suitable data disk mount point.
    Returns the mount path or None if no suitable disk found.
    """
    filesystems = parse_df_output()
    if not filesystems:
        return None

    candidates = []
    for fs in filesystems:
        if is_excluded_mount(fs["mount"], fs["filesystem"]):
            continue

        avail_gb = parse_size_to_gb(fs["avail"])
        # Require at least 10GB available (lower threshold for flexibility)
        if avail_gb < 10:
            continue

        score = score_mount_point(fs)
        candidates.append((score, fs))

    if not candidates:
        # Fallback: check if common data directories exist
        fallback_paths = ["/data", "/mnt/data", "/home/data", "/opt/data"]
        for path in fallback_paths:
            if Path(path).exists() and Path(path).is_dir():
                return path
        return None

    # Sort by score (descending) and return best match
    candidates.sort(key=lambda x: x[0], reverse=True)
    best_match = candidates[0][1]

    return best_match["mount"]


def main():
    """Main entry point."""
    result = find_data_disk()

    output = {
        "data_disk": result,
        "found": result is not None,
    }

    # Add additional info if found
    if result:
        filesystems = parse_df_output()
        for fs in filesystems:
            if fs["mount"] == result:
                output["size"] = fs["size"]
                output["available"] = fs["avail"]
                output["use_percent"] = fs["use_percent"]
                break

    print(json.dumps(output, indent=2))
    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())
