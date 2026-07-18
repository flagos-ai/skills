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

"""
Detect network environment and determine whether GitHub/PyPI mirrors are needed.
Run INSIDE the container. Outputs JSON with mirror configuration.
"""

import json
import subprocess
import sys


def probe_url(url, timeout=5):
    """Test if a URL is reachable within timeout. Returns (reachable, latency_ms)."""
    try:
        result = subprocess.run(
            ["curl", "-s", "-o", "/dev/null", "-w", "%{time_total}",
             "--connect-timeout", str(timeout), url],
            capture_output=True, text=True, timeout=timeout + 2
        )
        if result.returncode == 0:
            latency = float(result.stdout.strip()) * 1000
            return True, latency
        return False, -1
    except (subprocess.TimeoutExpired, Exception):
        return False, -1


def detect_network():
    """Detect network environment and return mirror config."""
    github_ok, github_ms = probe_url("https://github.com")
    pypi_ok, pypi_ms = probe_url("https://pypi.org/simple/")

    # Use mirrors if unreachable or slow (>3s)
    need_github_mirror = not github_ok or github_ms > 3000
    need_pypi_mirror = not pypi_ok or pypi_ms > 3000

    result = {
        "github": {
            "direct_reachable": github_ok,
            "latency_ms": round(github_ms, 1) if github_ms >= 0 else None,
            "use_mirror": need_github_mirror,
            "prefix": "https://ghfast.top/https://github.com" if need_github_mirror else "https://github.com",
        },
        "pypi": {
            "direct_reachable": pypi_ok,
            "latency_ms": round(pypi_ms, 1) if pypi_ms >= 0 else None,
            "use_mirror": need_pypi_mirror,
            "index_flag": "-i https://pypi.tuna.tsinghua.edu.cn/simple" if need_pypi_mirror else "",
        },
    }
    return result


def main():
    result = detect_network()
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
