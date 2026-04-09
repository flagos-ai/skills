#!/usr/bin/env python3
"""
PyTorch GPU Validation Script
Validates GPU functionality with basic PyTorch operations.
"""

import json
import sys
import traceback


def get_gpu_backend():
    """Detect which GPU backend is available."""
    # Try NVIDIA CUDA first
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", torch
    except ImportError:
        pass

    # Try Ascend NPU (torch_npu)
    try:
        import torch
        import torch_npu
        if torch_npu.npu.is_available():
            return "npu", torch
    except ImportError:
        pass

    # Try Metax MUSA (torch_musa)
    try:
        import torch
        import torch_musa
        if torch_musa.musa.is_available():
            return "musa", torch
    except ImportError:
        pass

    # Try Iluvatar CoreX (torch_corex)
    try:
        import torch
        import torch_corex  # noqa: F401  # side-effect import registers backend
        if hasattr(torch, 'corex') and torch.corex.is_available():
            return "corex", torch
    except ImportError:
        pass

    # Try AMD ROCm (uses CUDA API)
    try:
        import torch
        if torch.cuda.is_available():
            # ROCm uses HIP but exposes via CUDA API
            device_name = torch.cuda.get_device_name(0).lower()
            if "amd" in device_name or "radeon" in device_name or "mi" in device_name:
                return "rocm", torch
            return "cuda", torch
    except ImportError:
        pass

    # Fallback to checking if torch is available at all
    try:
        import torch
        return "cpu_only", torch
    except ImportError:
        return None, None


def validate_gpu():
    """
    Validate GPU functionality with PyTorch operations.
    Returns dict with validation results.
    """
    result = {
        "status": "FAIL",
        "backend": None,
        "device_count": 0,
        "device_names": [],
        "tests": {},
        "error": None,
    }

    backend, torch = get_gpu_backend()

    if torch is None:
        result["error"] = "PyTorch not installed"
        return result

    result["backend"] = backend

    if backend == "cpu_only":
        result["error"] = "No GPU backend available, CPU only"
        return result

    try:
        # Get device function based on backend
        if backend == "cuda" or backend == "rocm":
            device_type = "cuda"
            get_device_count = torch.cuda.device_count
            get_device_name = torch.cuda.get_device_name
            synchronize = torch.cuda.synchronize
        elif backend == "npu":
            import torch_npu
            device_type = "npu"
            get_device_count = torch_npu.npu.device_count
            def get_device_name(i):
                return f"Ascend NPU {i}"
            synchronize = torch_npu.npu.synchronize
        elif backend == "musa":
            import torch_musa
            device_type = "musa"
            get_device_count = torch_musa.musa.device_count
            get_device_name = torch_musa.musa.get_device_name
            synchronize = torch_musa.musa.synchronize
        elif backend == "corex":
            device_type = "corex"
            get_device_count = torch.corex.device_count
            get_device_name = torch.corex.get_device_name
            synchronize = torch.corex.synchronize
        else:
            result["error"] = f"Unknown backend: {backend}"
            return result

        # Test 1: Device count
        device_count = get_device_count()
        result["device_count"] = device_count
        result["tests"]["device_detection"] = device_count > 0

        if device_count == 0:
            result["error"] = "No GPU devices found"
            return result

        # Get device names
        for i in range(device_count):
            try:
                name = get_device_name(i)
                result["device_names"].append(name)
            except Exception:
                result["device_names"].append(f"Device {i}")

        # Test 2: Tensor creation on GPU
        device = torch.device(f"{device_type}:0")
        x = torch.randn(100, 100, device=device)
        result["tests"]["tensor_creation"] = x.device.type == device_type

        # Test 3: Basic computation (matrix multiplication)
        y = torch.randn(100, 100, device=device)
        z = torch.matmul(x, y)
        synchronize()
        result["tests"]["matrix_multiply"] = z.shape == (100, 100)

        # Test 4: Transfer back to CPU
        z_cpu = z.cpu()
        result["tests"]["gpu_to_cpu_transfer"] = z_cpu.device.type == "cpu"

        # Test 5: Memory allocation check
        if device_type == "cuda":
            mem_allocated = torch.cuda.memory_allocated(0)
            result["tests"]["memory_allocation"] = mem_allocated > 0
            result["memory_allocated_mb"] = mem_allocated / (1024 * 1024)
        else:
            result["tests"]["memory_allocation"] = True  # Skip for non-CUDA

        # All tests passed
        all_passed = all(result["tests"].values())
        result["status"] = "PASS" if all_passed else "PARTIAL"

    except Exception as e:
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()

    return result


def main():
    """Main entry point."""
    result = validate_gpu()
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
