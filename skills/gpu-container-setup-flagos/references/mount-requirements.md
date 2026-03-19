# Vendor-Specific Mount Requirements

This document specifies the required mounts and Docker flags for each GPU vendor.

## Quick Reference

| Vendor | Docker GPU Flag | Device Mounts | Directory Mounts |
|--------|----------------|---------------|------------------|
| NVIDIA | `--gpus all` | (automatic) | None required |
| AMD | (none) | `/dev/kfd`, `/dev/dri` | `/opt/rocm` (optional) |
| Ascend | (none) | `/dev/davinci*`, `/dev/devmm_svm`, `/dev/hisi_hdc` | `/usr/local/Ascend/driver`, `/usr/local/Ascend/ascend-toolkit`, `/usr/local/sbin/npu-smi` + `LD_LIBRARY_PATH` |
| Metax | (none) | `/dev/mxcd`, `/dev/dri` | None (use container's built-in MACA) |
| Iluvatar | (none) | `/dev/bi*` | `/opt/iluvatar` |
| Hygon DCU | (none) | `/dev/kfd`, `/dev/dri` | `/opt/hyhal` (CRITICAL) |

## Detailed Mount Specifications

### NVIDIA

NVIDIA uses the NVIDIA Container Toolkit which handles mounts automatically.

```bash
docker run --gpus all \
  -v /path/to/data:/data \
  nvcr.io/nvidia/pytorch:24.01-py3
```

**Required:**
- Docker flag: `--gpus all` (or `--gpus '"device=0,1"'` for specific GPUs)

**Optional environment variables:**
- `NVIDIA_VISIBLE_DEVICES` - Select specific GPUs
- `NVIDIA_DRIVER_CAPABILITIES` - Limit driver capabilities

### AMD/ROCm

```bash
docker run \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --group-add render \
  -v /path/to/data:/data \
  rocm/pytorch:latest
```

**Required devices:**
- `/dev/kfd` - Kernel Fusion Driver
- `/dev/dri` - Direct Rendering Infrastructure

**Required groups:**
- `video`
- `render`

**Optional mounts:**
- `/opt/rocm:/opt/rocm:ro` - If host has ROCm installed

### Huawei Ascend

```bash
docker run \
  --device=/dev/davinci0 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/Ascend/ascend-toolkit:/usr/local/Ascend/ascend-toolkit:ro \
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro \
  -e LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64:/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64 \
  -e ASCEND_HOME=/usr/local/Ascend \
  --entrypoint "" \
  -v /path/to/data:/data \
  <image> \
  sleep infinity
```

**Required devices:**
- `/dev/davinci[0-N]` - NPU devices (one per device, can be 0-15 or more)
- `/dev/davinci_manager` - Device manager
- `/dev/devmm_svm` - Shared virtual memory
- `/dev/hisi_hdc` - HiSilicon data channel

**Required mounts (mount separately to avoid conflicts):**
- `/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro` - NPU driver
- `/usr/local/Ascend/ascend-toolkit:/usr/local/Ascend/ascend-toolkit:ro` - CANN toolkit
- `/usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro` - NPU management tool

**Critical: Do NOT mount `/usr/local/Ascend` as a whole** - this can cause CANN version conflicts between host and container.

**Required environment variables:**
- `LD_LIBRARY_PATH` - Must include these paths (order matters):
  - `/usr/local/Ascend/driver/lib64`
  - `/usr/local/Ascend/driver/lib64/driver` (contains `libascend_hal.so`)
  - `/usr/local/Ascend/ascend-toolkit/latest/lib64`
  - `/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64`
  - `/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64`
- `ASCEND_HOME=/usr/local/Ascend`

**Optional environment variables:**
- `ASCEND_VISIBLE_DEVICES` - Select specific NPUs

**Entrypoint override:**
- Use `--entrypoint ""` if container fails with CANN path errors (e.g., "set_env.sh: No such file")
- Many Ascend images have entrypoints that expect specific CANN versions

**Persisting environment in container:**
```bash
docker exec <container> bash -c 'cat >> ~/.bashrc << "EOF"
export ASCEND_HOME=/usr/local/Ascend
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64:/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64:$LD_LIBRARY_PATH
EOF'
```

### Metax

```bash
docker run \
  --device=/dev/mxcd \
  --device=/dev/dri \
  --group-add video \
  --shm-size=16g \
  --ipc=host \
  -v /path/to/data:/data \
  cr.metax-tech.com/public-ai-release/maca/vllm-metax:latest
```

**Required devices:**
- `/dev/mxcd` - Metax control device
- `/dev/dri` - DRI devices (card0-N, renderD128-N)

**Required groups:**
- `video`

**DO NOT mount host MACA:**
- Avoid `-v /opt/maca:/opt/maca:ro` - causes LLVM version mismatch
- Container images include their own MACA libraries

**Python path:**
- Use `/opt/conda/bin/python3` (not system python)
- Or activate conda: `source /opt/conda/etc/profile.d/conda.sh && conda activate base`

**GPU API:**
- Metax uses `torch.cuda` API (not `torch.musa`)
- `torch.cuda.is_available()` returns True when GPUs are accessible

**Environment variables:**
- `MUSA_VISIBLE_DEVICES` - Select specific GPUs

### Iluvatar

```bash
docker run \
  --device=/dev/bi0 \
  --device=/dev/bi1 \
  -v /opt/iluvatar:/opt/iluvatar:ro \
  -v /path/to/data:/data \
  hub.iluvatar.com/pytorch/iluvatar-pytorch:latest
```

**Required devices:**
- `/dev/bi[0-N]` - Iluvatar BI-series devices

**Required mounts:**
- `/opt/iluvatar:/opt/iluvatar:ro` - Iluvatar CoreX stack

**Environment variables:**
- `COREX_VISIBLE_DEVICES` - Select specific GPUs

### Hygon DCU (HCU/BW series)

```bash
docker run \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --group-add render \
  --security-opt seccomp=unconfined \
  -v /opt/hyhal:/opt/hyhal:ro \
  -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  --shm-size=16g \
  -v /path/to/data:/data \
  harbor.baai.ac.cn/flagrelease-public/hygon-pytorch:2.5.1-dtk25.04-driver6.3.28
```

**Required devices:**
- `/dev/kfd` - Kernel Fusion Driver (ROCm/HIP compatible)
- `/dev/dri` - Direct Rendering Infrastructure (card0-N, renderD128-N)

**Required groups:**
- `video`
- `render`

**CRITICAL mount:**
- `/opt/hyhal:/opt/hyhal:ro` - HYHAL libraries (host symlink to `/usr/local/hyhal`)
- **DO NOT mount host DTK** (`/opt/dtk-*`) - causes NCCL/library version conflicts

**Required environment variables:**
- `HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` - Specify visible GPUs (required for HIP init)

**GPU API:**
- Hygon DCU uses ROCm/HIP backend via `torch.cuda` API
- `torch.cuda.is_available()` returns True when GPUs accessible
- `torch.cuda.device_count()` returns number of DCU devices

**Detection commands:**
```bash
# Host detection
/opt/dtk-*/bin/rocm-smi --showproductname  # Shows BW200/BW3000 etc.

# Inside container
python -c "import torch; print(torch.cuda.device_count(), torch.cuda.get_device_name(0))"
```

**Troubleshooting:**
- "No HIP GPUs are available" with 8 devices detected: Missing `/opt/hyhal` mount
- "ncclCommRegister undefined symbol": Host DTK mounted causing version conflict
- "libhsa-runtime64.so error": HYHAL libraries not in LD_LIBRARY_PATH

## Common Additional Mounts

For all vendors, consider these common mounts:

```bash
# Data directory
-v /data:/data

# Model cache (for Hugging Face, etc.)
-v /path/to/cache:/root/.cache

# Shared memory (increase for multi-GPU training)
--shm-size=16g

# Host networking (optional, for distributed training)
--network=host

# IPC for multi-process communication
--ipc=host
```

## Security Considerations

- Use `:ro` (read-only) for driver directories when possible
- Avoid running containers as root when not necessary
- Use `--security-opt seccomp=unconfined` only if required for debugging
- Consider `--cap-add` instead of `--privileged` when specific capabilities are needed

## Troubleshooting

### Device Permission Denied

```bash
# Check device permissions
ls -la /dev/davinci* /dev/mx* /dev/bi* /dev/nvidia*

# Add user to appropriate group
sudo usermod -aG video $USER
```

### Mount Not Found

```bash
# Verify host paths exist
ls -la /usr/local/Ascend /opt/metax /opt/iluvatar /opt/rocm
```

### Container Can't Access GPU

1. Verify host driver is working: run vendor's SMI tool
2. Check if container runtime supports the GPU
3. Verify all required devices are mounted
4. Check environment variables inside container
