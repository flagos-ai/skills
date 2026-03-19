---
name: gpu-container-setup
description: |
  Automatically detect GPU vendor, find appropriate PyTorch container image,
  launch with correct mounts, and validate GPU functionality. Supports NVIDIA,
  Ascend, Metax, Iluvatar, and AMD/ROCm. Use when user says "setup container",
  "start pytorch container", or invokes /gpu-container-setup.
user-invokable: true
allowed-tools: "Bash(*) Read Edit Write Glob Grep WebSearch WebFetch AskUserQuestion"
---

# GPU Container Setup Skill

This skill automates multi-vendor GPU container setup for PyTorch workloads.

## Supported GPU Vendors

| Vendor | PyTorch Backend | Detection |
|--------|-----------------|-----------|
| NVIDIA | CUDA | `nvidia-smi` |
| AMD | ROCm (HIP) | `rocm-smi`, `/opt/rocm` |
| Ascend | torch_npu | `npu-smi`, `/usr/local/Ascend` |
| Metax | torch_musa | `mx-smi`, `/opt/metax` |
| Iluvatar | torch_corex | `ixsmi`, `/opt/iluvatar` |

## Execution Flow

When invoked, follow these steps:

### Step 1: Parse Arguments

Check if user provided:
- `--vendor <name>` - Force specific vendor (skip detection)
- `--image <image>` - Force specific container image
- `--data <path>` - Force specific data mount path
- `--name <name>` - Container name (default: `pytorch-gpu`)

### Step 2: Detect GPU Vendor

Run the detection script:

```bash
python3 .claude/skills/gpu-container-setup/scripts/detect_gpu.py
```

Expected output:
```json
{"vendor": "ascend", "devices": ["Ascend 910B"], "count": 8}
```

If detection fails and no `--vendor` flag provided, ask user which vendor to use.

### Step 3: Find Data Disk

Run the data disk detection:

```bash
python3 .claude/skills/gpu-container-setup/scripts/find_data_disk.py
```

Expected output:
```json
{"data_disk": "/mnt/data", "found": true, "size": "2.0T", "available": "1.5T"}
```

If no suitable disk found, ask user for data mount path.

### Step 4: Find Container Image

Follow strict priority order (only proceed to next if current fails):

```
1. Primary Vendor Hub (hardcoded) → 2. BAAI Harbor → 3. Web Search → 4. Local Images → 5. Ask User
```

#### Step 4.1: Primary Vendor Hub (hardcoded URLs)

| Vendor | Registry | API/Query |
|--------|----------|-----------|
| NVIDIA | `nvcr.io` | `https://api.ngc.nvidia.com/v2/repos/nvidia/pytorch/tags` |
| Ascend | `ascendhub.huawei.com` | Portal: https://ascendhub.huawei.com |
| Metax | `registry.metax-tech.com` | `https://registry.metax-tech.com/v2/pytorch/metax-pytorch/tags/list` |
| Iluvatar | `hub.iluvatar.com` | `https://hub.iluvatar.com/v2/pytorch/iluvatar-pytorch/tags/list` |
| AMD | `docker.io` (rocm/pytorch) | `https://hub.docker.com/v2/repositories/rocm/pytorch/tags` |

```bash
# Example: Query NGC for latest NVIDIA PyTorch
TAG=$(curl -s "https://api.ngc.nvidia.com/v2/repos/nvidia/pytorch/tags" | jq -r '.tags[].name' | grep -E '^[0-9]{2}\.[0-9]{2}-py3$' | sort -rV | head -1)
IMAGE="nvcr.io/nvidia/pytorch:${TAG}"
```

#### Step 4.2: BAAI Harbor (fallback)

Only if Step 4.1 fails (unreachable, no image, pull fails).

```bash
# Query BAAI Harbor
curl -s "https://harbor.baai.ac.cn/api/v2.0/projects/flagrelease-public/repositories?page_size=100" | jq -r '.[].name' | grep "flagrelease-<vendor>"
```

#### Step 4.3: Web Search (fallback)

Only if Steps 4.1 and 4.2 fail. Search for `"<vendor> pytorch docker official"`.

#### Step 4.4: Local Images (fallback)

Only if Steps 4.1-4.3 fail. Check `docker images | grep pytorch`.

#### Test Before Use

```bash
docker pull "${IMAGE}" && docker run --rm "${IMAGE}" python -c "import torch; print(torch.__version__)"
```

If test fails, try next source. If all fail, ask user for image.

#### Step 4.5: Update Skill (self-improvement)

**IMPORTANT**: If image found via Web Search (Step 4.3) passes all tests, update `references/image-sources.md` to add the newly discovered vendor hub as a primary source. This makes future lookups faster.

```bash
# After successful web search discovery:
# 1. Verify image works (pull + pytorch test + GPU test)
# 2. Extract registry URL pattern
# 3. Update references/image-sources.md Step 1 section with new vendor hub
```

### Step 5: Build Docker Command

Refer to `references/mount-requirements.md` for vendor-specific requirements.

**NVIDIA:**
```bash
docker run -d --gpus all \
  --name pytorch-gpu \
  --shm-size=16g \
  -v <data_disk>:/data \
  <image> sleep infinity
```

**AMD/ROCm:**
```bash
docker run -d \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  --name pytorch-gpu \
  --shm-size=16g \
  -v <data_disk>:/data \
  <image> sleep infinity
```

**Ascend:**
```bash
docker run -d \
  --device=/dev/davinci0 --device=/dev/davinci1 ... \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/Ascend:/usr/local/Ascend:ro \
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro \
  --name pytorch-gpu \
  --shm-size=16g \
  -v <data_disk>:/data \
  <image> sleep infinity
```

**Metax:**
```bash
docker run -d \
  --device=/dev/mx0 --device=/dev/mx1 ... \
  -v /opt/metax:/opt/metax:ro \
  --name pytorch-gpu \
  --shm-size=16g \
  -v <data_disk>:/data \
  <image> sleep infinity
```

**Iluvatar:**
```bash
docker run -d \
  --device=/dev/bi0 --device=/dev/bi1 ... \
  -v /opt/iluvatar:/opt/iluvatar:ro \
  --name pytorch-gpu \
  --shm-size=16g \
  -v <data_disk>:/data \
  <image> sleep infinity
```

### Step 6: Start Container

Execute the docker run command. If container with same name exists:
1. Check if it's running - offer to use existing or replace
2. If stopped - offer to restart or replace

### Step 7: Validate PyTorch GPU

Copy and run validation script inside container:

```bash
docker cp .claude/skills/gpu-container-setup/scripts/validate_pytorch.py pytorch-gpu:/tmp/
docker exec pytorch-gpu python3 /tmp/validate_pytorch.py
```

Expected output:
```json
{
  "status": "PASS",
  "backend": "npu",
  "device_count": 8,
  "device_names": ["Ascend 910B", ...],
  "tests": {
    "device_detection": true,
    "tensor_creation": true,
    "matrix_multiply": true,
    "gpu_to_cpu_transfer": true
  }
}
```

### Step 8: Report Results

Summarize to user:
- GPU vendor and devices detected
- Container name and image used
- Data mount path
- Validation status
- How to access: `docker exec -it pytorch-gpu bash`

## Error Handling

| Error | Action |
|-------|--------|
| No GPU detected | Ask user for vendor or check drivers |
| Image pull fails | Try alternative registry or web search |
| Container start fails | Check device permissions, show error |
| Validation fails | Show detailed error, suggest fixes |

## Reference Files

- `references/gpu-detection.md` - Detection methods by vendor
- `references/image-sources.md` - Image discovery guide (registry APIs, priority order, selection criteria)
- `references/mount-requirements.md` - Vendor mount specifications

## Example Usage

```
User: /gpu-container-setup
User: setup a pytorch container
User: start container with ascend GPU
User: /gpu-container-setup --image nvcr.io/nvidia/pytorch:24.01-py3
User: /gpu-container-setup --image harbor.baai.ac.cn/flagrelease-public/ngctorch:2601
```
