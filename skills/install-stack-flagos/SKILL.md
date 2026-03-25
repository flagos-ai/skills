---
name: install-stack-flagos
description: |
  Install the 5-package multi-chip software stack (vLLM, FlagTree, FlagGems, FlagCX,
  vllm-plugin-FL) inside a GPU container. Handles network mirror detection, dependency
  ordering, wheel selection, and per-package validation. Use after gpu-container-setup
  has produced a running container with PyTorch + GPU access.
user-invokable: true
allowed-tools: "Bash(*) Read Edit Write Glob Grep WebSearch WebFetch AskUserQuestion"
---

# Install Multi-Chip Software Stack

Install 5 packages inside a running GPU container, in dependency order, with
per-package validation and structured error reporting.

## Skill Components

```
install-stack/
├── SKILL.md                           # This file — execution flow
├── scripts/
│   ├── detect_network.py              # Probe GitHub/PyPI, return mirror config (JSON)
│   ├── collect_env_info.py            # Python/glibc/arch/vendor/disk info (JSON)
│   ├── select_flagtree_wheel.py       # Match vendor+python+glibc → wheel specifier (JSON)
│   └── validate_packages.py           # Import-test all 5 packages, report status (JSON)
└── references/
    ├── vendor-mappings.md             # FlagCX make flags, adaptor names, dependency chain
    └── network-mirrors.md             # GitHub/PyPI mirror config rules
```

## Prerequisites

- A running Docker container with PyTorch + GPU access (from `/gpu-container-setup`)
- Know the **container name** and **GPU vendor**

If invoked standalone, ask the user for container name and GPU vendor.
If invoked from `/flagrelease` orchestrator, these are passed as context.

## Execution Flow

### Step 0: Resolve Container & Vendor

Verify the container is running:

```bash
docker inspect --format='{{.State.Status}}' <CONTAINER> | grep -q running
```

Copy and run `scripts/collect_env_info.py` inside the container to get vendor,
Python version, glibc version, architecture, and free disk space:

```bash
docker cp <SKILL_DIR>/scripts/collect_env_info.py <CONTAINER>:/tmp/
docker exec <CONTAINER> python3 /tmp/collect_env_info.py
```

If vendor is `unknown` and user didn't provide `--vendor`, ask the user.

### Step 1: Detect Network Environment

Copy and run `scripts/detect_network.py` inside the container:

```bash
docker cp <SKILL_DIR>/scripts/detect_network.py <CONTAINER>:/tmp/
docker exec <CONTAINER> python3 /tmp/detect_network.py
```

Parse the JSON output to get `GITHUB_PREFIX` and `PIP_INDEX` for all subsequent
commands. See `references/network-mirrors.md` for fallback rules.

### Step 2: Check Disk Space

From `collect_env_info.py` output, verify at least 10GB free. If not, warn user
and ask whether to proceed.

### Step 3: Install Packages (in dependency order)

See `references/vendor-mappings.md` for dependency chain and install order:
**vLLM → FlagTree → FlagGems → FlagCX → vllm-plugin-FL**

---

#### 3.1: vLLM 0.13.0

```bash
docker exec <CONTAINER> pip install ${PIP_INDEX} vllm==0.13.0
```

Quick validate:
```bash
docker exec <CONTAINER> python3 -c "import vllm; assert vllm.__version__ == '0.13.0'"
```

**GATE:** If vLLM install fails → record error and **EXIT** the skill.

---

#### 3.2: FlagTree (pre-compiled wheel)

Run `scripts/select_flagtree_wheel.py` to find the correct wheel:

```bash
python3 <SKILL_DIR>/scripts/select_flagtree_wheel.py \
    --vendor <VENDOR> --python <PY_VER> --glibc <GLIBC_VER>
```

If status is `FOUND`, uninstall stock triton and install the wheel:

```bash
docker exec <CONTAINER> bash -c '
python3 -m pip uninstall -y triton
python3 -m pip uninstall -y triton
python3 -m pip install <SPECIFIER> <PIP_ARGS>
'
```

If status is `NOT_FOUND`, record the mismatch and **continue** (do not exit).

---

#### 3.3: FlagGems

```bash
docker exec <CONTAINER> bash -c "
cd /tmp && git clone ${GITHUB_PREFIX}/FlagOpen/FlagGems
cd FlagGems && pip install ${PIP_INDEX} -e .
"
```

Failure → record and continue.

---

#### 3.4: FlagCX (two-phase build)

Read `references/vendor-mappings.md` to look up the correct Make flag and
FLAGCX_ADAPTOR for the detected vendor.

**Phase 1:** Build C++ library:
```bash
docker exec <CONTAINER> bash -c "
cd /tmp && git clone ${GITHUB_PREFIX}/flagos-ai/FlagCX
cd FlagCX && git submodule update --init --recursive
make <MAKE_FLAG> -j\$(nproc)
"
```

**Phase 2:** Install PyTorch plugin:
```bash
docker exec <CONTAINER> bash -c "
cd /tmp/FlagCX/plugin/torch
FLAGCX_ADAPTOR=<ADAPTOR> pip install -e . --no-build-isolation
"
```

Failure → record and continue.

---

#### 3.5: vllm-plugin-FL

```bash
docker exec <CONTAINER> bash -c "
cd /tmp && git clone ${GITHUB_PREFIX}/flagos-ai/vllm-plugin-FL
cd vllm-plugin-FL
pip install ${PIP_INDEX} -r requirements.txt
pip install --no-build-isolation -e .
"
```

On **Iluvatar**, if `requirements.txt` fails, retry with `requirements_iluvatar.txt`.

**GATE:** If vllm-plugin-FL fails → record error and **EXIT** the skill.

---

### Step 4: Validate All Packages

Copy and run `scripts/validate_packages.py` inside the container:

```bash
docker cp <SKILL_DIR>/scripts/validate_packages.py <CONTAINER>:/tmp/
docker exec <CONTAINER> python3 /tmp/validate_packages.py
```

This produces a comprehensive JSON report of all 5 packages with import status,
versions, and gate check.

### Step 5: Set Runtime Environment

If FlagCX installed successfully, persist FLAGCX_PATH:

```bash
docker exec <CONTAINER> bash -c "echo 'export FLAGCX_PATH=/tmp/FlagCX' >> ~/.bashrc"
```

### Step 6: Produce Final Report

Combine all results into structured output:

```json
{
  "status": "PASS | PARTIAL | FAIL",
  "stage": "install-stack",
  "container": "<name>",
  "vendor": "<vendor>",
  "network": {"github_mirror": true, "pypi_mirror": true},
  "python_version": "3.11",
  "glibc_version": "2.34",
  "packages": {
    "vllm": {"status": "PASS", "version": "0.13.0"},
    "flagtree": {"status": "PASS", "version": "0.4.1+ascend3.2"},
    "flaggems": {"status": "PASS", "version": "..."},
    "flagcx": {"status": "PASS", "version": "0.10.0"},
    "vllm_plugin_fl": {"status": "PASS", "version": "..."}
  },
  "flagcx_path": "/tmp/FlagCX",
  "gate_passed": true,
  "errors": []
}
```

**Status logic:**
- `PASS` — all 5 packages installed and validated
- `PARTIAL` — vLLM + plugin installed, some of FlagTree/FlagGems/FlagCX failed
- `FAIL` — vLLM or plugin failed (gate failed)

## Error Handling

| Failure | Behavior |
|---------|----------|
| Container not running | Report error, exit |
| Network unreachable (both direct and mirror) | Report, exit |
| pip install fails | Report package, full error, continue (unless gate) |
| Build from source fails | Report compiler error, continue |
| No matching FlagTree wheel | Report vendor/python/glibc mismatch, continue |
| Import fails after install | Report traceback, continue |
| Disk space < 10GB | Warn user, ask whether to proceed |
| Timeout (any command) | All commands use timeout; report which step |

## Timeout Rules

| Operation | Timeout |
|-----------|---------|
| pip install (per package) | 300s |
| git clone | 120s |
| make (FlagCX) | 300s |
| Network probe | 5s |
| Validation (import) | 30s |
