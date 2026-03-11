# GPU Vendor Detection Methods

This document describes how to detect different GPU vendors on Linux systems.

## Detection Priority Order

1. **NVIDIA** - Most common, check first
2. **AMD/ROCm** - Growing in HPC/ML
3. **Ascend** - Huawei NPUs
4. **Metax** - Chinese GPU vendor
5. **Iluvatar** - Chinese GPU vendor
6. **Hygon DCU** - Chinese GPU vendor (ROCm-compatible)

## Vendor Detection Methods

### NVIDIA

| Method | Command/Path | Notes |
|--------|-------------|-------|
| Primary | `nvidia-smi` | Standard NVIDIA driver tool |
| Fallback | `/dev/nvidia*` | Device files exist when driver loaded |
| Library | `libnvidia-ml.so` | NVML library presence |

```bash
# Check NVIDIA GPU
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

### AMD/ROCm

| Method | Command/Path | Notes |
|--------|-------------|-------|
| Primary | `rocm-smi` | ROCm SMI tool |
| Fallback | `/opt/rocm` | ROCm installation directory |
| Device | `/dev/kfd`, `/dev/dri` | Kernel fusion driver and DRI |

```bash
# Check AMD GPU
rocm-smi --showproductname
rocm-smi --showmeminfo vram
```

### Huawei Ascend NPU

| Method | Command/Path | Notes |
|--------|-------------|-------|
| Primary | `npu-smi` | Ascend management tool |
| Fallback | `/usr/local/Ascend` | CANN toolkit directory |
| Device | `/dev/davinci*` | Da Vinci AI core devices |
| Env | `ASCEND_HOME` | Environment variable |

```bash
# Check Ascend NPU
npu-smi info -l
npu-smi info -t board -i 0
```

### Metax

| Method | Command/Path | Notes |
|--------|-------------|-------|
| Primary | `mx-smi` | Metax management tool |
| Fallback | `/opt/metax` | Metax software directory |
| Device | `/dev/mx*` | Metax device files |

```bash
# Check Metax GPU
mx-smi -L
mx-smi -q
```

### Iluvatar

| Method | Command/Path | Notes |
|--------|-------------|-------|
| Primary | `ixsmi` | Iluvatar management tool |
| Fallback | `/opt/iluvatar` | Iluvatar software directory |
| Device | `/dev/bi*` | BI-series device files |

```bash
# Check Iluvatar GPU
ixsmi -L
ixsmi -q
```

### Hygon DCU (HCU/BW series)

| Method | Command/Path | Notes |
|--------|-------------|-------|
| Primary | `rocm-smi` in `/opt/dtk-*/bin/` | DTK's ROCm-compatible SMI |
| Fallback | `/opt/dtk-*` | DTK (DCU Toolkit) directory |
| Device | `/dev/kfd`, `/dev/dri` | Kernel fusion driver + DRI |
| Library | `/opt/hyhal`, `/usr/local/hyhal` | HYHAL runtime libraries |

```bash
# Check Hygon DCU GPU
/opt/dtk-*/bin/rocm-smi --showproductname  # Shows BW200, BW3000, etc.
/opt/dtk-*/bin/rocm-smi --showdriverversion

# Output example:
# HCU[0]: Card Series: BW200, UBB BW1000
# HCU[0]: Card Vendor: C-3000 IC Design Co., Ltd.
```

**Key identifiers for Hygon DCU:**
- `rocm-smi` exists in `/opt/dtk-*/bin/` (not `/opt/rocm/bin/`)
- Output shows "HCU" (Hygon Compute Unit) instead of "GPU"
- Card series: BW200, BW3000, K100, etc.
- Vendor: "C-3000 IC Design" or "Chengdu C-3000"

**Distinguishing Hygon DCU from AMD ROCm:**
- Hygon: `rocm-smi` at `/opt/dtk-*/bin/rocm-smi`, shows "HCU" in output
- AMD: `rocm-smi` at `/opt/rocm/bin/rocm-smi`, shows "GPU" in output

## Detection Script Output Format

The `detect_gpu.py` script outputs JSON:

```json
{
  "vendor": "ascend",
  "devices": ["Ascend 910B", "Ascend 910B"],
  "count": 2
}
```

## Environment Variables

Some vendors require specific environment variables:

| Vendor | Variables |
|--------|-----------|
| NVIDIA | `CUDA_VISIBLE_DEVICES`, `NVIDIA_VISIBLE_DEVICES` |
| AMD | `HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES` |
| Ascend | `ASCEND_DEVICE_ID`, `ASCEND_HOME` |
| Metax | `MUSA_VISIBLE_DEVICES` |
| Iluvatar | `COREX_VISIBLE_DEVICES` |
| Hygon DCU | `HIP_VISIBLE_DEVICES` (required for GPU init) |

## Troubleshooting

### No GPU Detected

1. Check if driver is loaded: `lsmod | grep -E "nvidia|amdgpu|drv_davinci"`
2. Check device files: `ls -la /dev/nvidia* /dev/davinci* /dev/mx* /dev/bi*`
3. Check dmesg for errors: `dmesg | grep -i gpu`
4. Verify software installation in vendor directories

### Permission Issues

GPU device files typically require root or membership in specific groups:

- NVIDIA: `video` group
- AMD: `video`, `render` groups
- Ascend: `HwHiAiUser` group
- Metax/Iluvatar: Vendor-specific groups or root
