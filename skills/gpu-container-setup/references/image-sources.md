# Container Image Discovery Guide

This document describes how to discover and select PyTorch container images.

## Discovery Priority Order (MUST follow in sequence)

```
1. Primary Vendor Hub (hardcoded) → 2. BAAI Harbor → 3. Web Search → 4. Local Images
```

**IMPORTANT**: Only proceed to next step if current step fails completely.

---

## Step 1: Primary Vendor Hub (Hardcoded)

These are the official vendor registries. Try these FIRST.

### NVIDIA - NGC Registry

```
Registry: nvcr.io
Base URL: https://api.ngc.nvidia.com
```

```bash
# Query available PyTorch tags
curl -s "https://api.ngc.nvidia.com/v2/repos/nvidia/pytorch/tags" | jq -r '.tags[].name' | head -20

# Select latest stable (format: YY.MM-py3, avoid rc/beta)
TAG=$(curl -s "https://api.ngc.nvidia.com/v2/repos/nvidia/pytorch/tags" | jq -r '.tags[].name' | grep -E '^[0-9]{2}\.[0-9]{2}-py3$' | sort -rV | head -1)

# Full image
IMAGE="nvcr.io/nvidia/pytorch:${TAG}"
```

### Huawei Ascend - Ascend Hub

```
Registry: ascendhub.huawei.com
Portal: https://ascendhub.huawei.com
```

```bash
# List available images (may require login for API)
# Check portal for latest tags: https://ascendhub.huawei.com/public-ascendhub/pytorch-modelzoo

# Common image patterns:
IMAGE="ascendhub.huawei.com/public-ascendhub/pytorch-modelzoo:<tag>"
IMAGE="ascendhub.huawei.com/public-ascendhub/ascend-pytorch:<tag>"
```

### Metax - Metax Registry

```
Registry: cr.metax-tech.com (PRIMARY - confirmed working)
Fallback: registry.metax-tech.com (often unreachable)
```

```bash
# Query tags from cr.metax-tech.com (PRIMARY)
curl -s "https://cr.metax-tech.com/v2/public-ai-release/maca/vllm-metax/tags/list" | jq -r '.tags[]'
curl -s "https://cr.metax-tech.com/v2/public-library/maca-pytorch/tags/list" | jq -r '.tags[]'

# Recommended images (tested working):
# vLLM + PyTorch (includes torch_musa):
IMAGE="cr.metax-tech.com/public-ai-release/maca/vllm-metax:0.13.0-maca.ai3.3.0.303-torch2.8-py312-ubuntu22.04-amd64"

# Base PyTorch only:
IMAGE="cr.metax-tech.com/public-library/maca-pytorch:3.3.0.4-torch2.8-py312-ubuntu24.04-amd64"

# Fallback (legacy URL, often unreachable)
curl -s "https://registry.metax-tech.com/v2/pytorch/metax-pytorch/tags/list" | jq -r '.tags[]'
IMAGE="registry.metax-tech.com/pytorch/metax-pytorch:<tag>"
```

**IMPORTANT Notes for Metax:**
- Use conda Python: `/opt/conda/bin/python3` (not system python)
- Do NOT mount host's `/opt/maca` - causes LLVM version mismatch errors
- The container has its own MACA libraries built-in
- Metax uses `torch.cuda` API (not `torch.musa`) for GPU operations

### Iluvatar - Iluvatar Hub

```
Registry: hub.iluvatar.com
Portal: https://hub.iluvatar.com
```

```bash
# Query tags
curl -s "https://hub.iluvatar.com/v2/pytorch/iluvatar-pytorch/tags/list" | jq -r '.tags[]'

# Full image
IMAGE="hub.iluvatar.com/pytorch/iluvatar-pytorch:<tag>"
```

### AMD - Docker Hub ROCm

```
Registry: docker.io
```

```bash
# Query tags
curl -s "https://hub.docker.com/v2/repositories/rocm/pytorch/tags?page_size=50" | jq -r '.results[].name'

# Full image
IMAGE="rocm/pytorch:<tag>"
```

### MThreads - MThreads Registry

```
Registry: registry.mthreads.com
```

```bash
# Query tags
curl -s "https://registry.mthreads.com/v2/pytorch/mthreads-pytorch/tags/list" | jq -r '.tags[]'

# Full image
IMAGE="registry.mthreads.com/pytorch/mthreads-pytorch:<tag>"
```

### Hygon DCU - SourceFind Registry & BAAI Harbor

```
Primary Registry: harbor.sourcefind.cn:5443
BAAI Harbor: harbor.baai.ac.cn (verified working images)
```

```bash
# Query SourceFind for DTK PyTorch images
curl -s "https://harbor.sourcefind.cn:5443/v2/dcu/admin/base/pytorch/tags/list" | jq -r '.tags[]'

# Recommended images (tested working):
# BAAI Harbor (verified):
IMAGE="harbor.baai.ac.cn/flagrelease-public/hygon-pytorch:2.5.1-dtk25.04-driver6.3.28"

# SourceFind (base images):
IMAGE="harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.5.1-ubuntu22.04-dtk25.04.4-1230-py3.10-20251230"
```

**CRITICAL Mount Requirements for Hygon DCU:**
- Mount host HYHAL: `-v /opt/hyhal:/opt/hyhal:ro` (symlink to `/usr/local/hyhal`)
- Do NOT mount host DTK (`/opt/dtk-*`) - causes library conflicts
- Set `HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` for all GPUs
- Container uses ROCm/HIP backend via `torch.cuda` API

---

## Step 2: BAAI Harbor (Fallback)

Use ONLY if Step 1 fails (registry unreachable, no suitable image, pull fails).

```
Registry: harbor.baai.ac.cn
Project: flagrelease-public
```

### Query Available Images

```bash
# List all repositories (paginated, max 100 per page)
curl -s "https://harbor.baai.ac.cn/api/v2.0/projects/flagrelease-public/repositories?page_size=100&page=1" | jq -r '.[].name'
curl -s "https://harbor.baai.ac.cn/api/v2.0/projects/flagrelease-public/repositories?page_size=100&page=2" | jq -r '.[].name'

# Filter by vendor
VENDOR="ascend"  # or nvidia, metax, iluvatar, hygon, mthreads
curl -s "https://harbor.baai.ac.cn/api/v2.0/projects/flagrelease-public/repositories?page_size=100" | jq -r '.[].name' | grep "flagrelease-${VENDOR}"

# Get tags for specific repository
REPO="flagrelease-ascend-release-model_xxx"
curl -s "https://harbor.baai.ac.cn/api/v2.0/projects/flagrelease-public/repositories/${REPO}/artifacts" | jq -r '.[].tags[]?.name'
```

### Image Naming Convention

```
flagrelease-<vendor>-release-model_<model>-tree_<ver>-gems_<ver>-scale_<ver>-cx_<ver>-python_<ver>-torch_<ver>-pcp_<platform>-gpu_<gpu>-arc_<arch>-driver_<ver>
```

### Selection Criteria for BAAI Harbor

1. Filter by vendor name
2. Filter by architecture (`arc_amd64` or `arc_arm64`)
3. Prefer images with actual PyTorch version (not `torch_none`)
4. Prefer `latest` tag if available

---

## Step 3: Web Search (Fallback)

Use ONLY if Step 1 and Step 2 both fail.

### Search Queries

```
"<vendor> pytorch docker image official"
"<vendor> container registry pytorch"
"<vendor> deep learning container download"
"<vendor>hub pytorch"
```

### Expected Results

Look for:
- Official vendor documentation with registry URL
- GitHub repos with Dockerfile
- Vendor developer portal with container downloads

### Extract and Verify

From search results:
1. Find registry URL pattern
2. Test API endpoint
3. Query available tags
4. Select and pull image

### IMPORTANT: Update Skill After Success

If web search finds a working image that passes all tests (pull + PyTorch import + GPU detection), **you MUST update this file** to add the newly discovered registry as a primary vendor hub.

**Update Steps:**
1. Confirm image passes all tests
2. Extract the registry base URL (e.g., `hub.newvendor.com`)
3. Identify the API pattern for querying tags
4. Add new entry to "Step 1: Primary Vendor Hub" section with:
   - Registry URL
   - API endpoint
   - Example query command
   - Image path pattern

This makes the skill self-improving - future invocations will find images faster.

---

## Step 4: Local Images (Last Resort)

Use ONLY if Steps 1-3 all fail.

### Check Existing Local Images

```bash
# List local images with pytorch
docker images | grep -i pytorch

# List local images by vendor
docker images | grep -iE "(ascend|metax|iluvatar|nvidia|rocm|hygon|mthreads)"

# Check if image has working PyTorch
docker run --rm <local_image> python -c "import torch; print(torch.__version__)"
```

### Try to Launch and Test

```bash
# For each candidate local image:
IMAGE="<local_image>"

# Test PyTorch import
docker run --rm "${IMAGE}" python -c "import torch; print(torch.__version__)" && echo "PASS" || echo "FAIL"
```

---

## Image Selection Criteria

When multiple images available, prefer:

1. **Architecture match** - MUST match system (x86_64 → amd64, aarch64 → arm64)
2. **Platform version** - SDK version should be compatible with host drivers
3. **Stability** - Prefer release over rc/beta/nightly
4. **Recency** - Newer versions for latest features
5. **Python version** - Prefer 3.10+

---

## Test Before Use

After selecting image from ANY step:

```bash
# 1. Pull image
docker pull "${IMAGE}" || { echo "Pull failed"; exit 1; }

# 2. Test PyTorch import
docker run --rm "${IMAGE}" python -c "import torch; print(torch.__version__)" || { echo "Import failed"; exit 1; }

# 3. Test GPU detection (with appropriate device mounts)
# NVIDIA:
docker run --rm --gpus all "${IMAGE}" python -c "import torch; assert torch.cuda.is_available()"

# Ascend:
docker run --rm <device-mounts> "${IMAGE}" python -c "import torch_npu; assert torch_npu.npu.is_available()"

# Metax:
docker run --rm <device-mounts> "${IMAGE}" python -c "import torch; assert torch.musa.is_available()"

# Iluvatar:
docker run --rm <device-mounts> "${IMAGE}" python -c "import torch; assert torch.cuda.is_available()"
```

If test fails, go back and try next option in current step, or proceed to next step.

---

## Summary Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: Query Primary Vendor Hub (hardcoded URL)                │
│         ↓ success → use image                                   │
│         ↓ fail → Step 2                                         │
├─────────────────────────────────────────────────────────────────┤
│ Step 2: Query BAAI Harbor API                                   │
│         ↓ success → use image                                   │
│         ↓ fail → Step 3                                         │
├─────────────────────────────────────────────────────────────────┤
│ Step 3: Web Search for vendor registry                          │
│         ↓ success → use image → UPDATE SKILL with new hub       │
│         ↓ fail → Step 4                                         │
├─────────────────────────────────────────────────────────────────┤
│ Step 4: Check local images, test PyTorch                        │
│         ↓ success → use image                                   │
│         ↓ fail → Ask user for image                             │
└─────────────────────────────────────────────────────────────────┘

** SELF-IMPROVEMENT RULE **
If Step 3 (Web Search) finds a working image, after all tests pass:
  → Edit this file to add the discovered registry to Step 1
  → Future invocations will use the new primary hub directly
```
