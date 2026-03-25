"""
M4 Pro Hardware Optimization Utility.

Provides centralized detection and configuration for Apple Silicon hardware
optimizations across the ColdPath ML pipeline.

M4 Pro with 48GB RAM Optimizations:
- MPS (Metal Performance Shaders) for PyTorch acceleration
- Unified memory architecture allows larger batch sizes
- All 10+ performance cores for parallel training
- Optimized thread counts for tree-based models (LightGBM, XGBoost)
"""

from __future__ import annotations

import logging
import multiprocessing
import os
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


class ComputeDevice(Enum):
    """Available compute devices."""

    CUDA = "cuda"  # NVIDIA GPU
    MPS = "mps"  # Apple Silicon GPU
    CPU = "cpu"  # Fallback CPU


@dataclass
class HardwareProfile:
    """Detected hardware profile."""

    device: ComputeDevice
    device_name: str
    cpu_cores: int
    optimal_threads: int
    memory_gb: float
    recommended_batch_size: int
    supports_mixed_precision: bool

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "device": self.device.value,
            "device_name": self.device_name,
            "cpu_cores": self.cpu_cores,
            "optimal_threads": self.optimal_threads,
            "memory_gb": self.memory_gb,
            "recommended_batch_size": self.recommended_batch_size,
            "supports_mixed_precision": self.supports_mixed_precision,
        }


def detect_cuda() -> bool:
    """Detect CUDA GPU availability."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return bool(os.environ.get("CUDA_VISIBLE_DEVICES"))


def detect_mps() -> bool:
    """
    Detect Apple Silicon MPS availability for PyTorch acceleration.

    MPS (Metal Performance Shaders) provides GPU acceleration on Apple Silicon.
    Requires:
    - PyTorch 2.0+ with MPS support
    - macOS 12.3+ (Monterey) or macOS 13+ (Ventura) recommended
    - Apple Silicon Mac (M1/M2/M3/M4 series)
    """
    try:
        import torch

        return (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
            and torch.backends.mps.is_built()
        )
    except ImportError:
        return False


def detect_optimal_device() -> ComputeDevice:
    """
    Detect the best available compute device.

    Priority: CUDA > MPS > CPU
    """
    if detect_cuda():
        return ComputeDevice.CUDA
    if detect_mps():
        return ComputeDevice.MPS
    return ComputeDevice.CPU


def get_optimal_thread_count(device: ComputeDevice | None = None) -> int:
    """
    Get optimal thread count for CPU-bound operations.

    M4 Pro with 10+ performance cores benefits from full utilization
    due to unified memory architecture (no NUMA concerns).

    Args:
        device: Current compute device (auto-detected if None)

    Returns:
        Optimal number of threads for parallel operations
    """
    # Check for explicit override
    explicit = os.environ.get("ML_NUM_THREADS")
    if explicit:
        return int(explicit)

    if device is None:
        device = detect_optimal_device()

    cpu_count = multiprocessing.cpu_count()

    # Apple Silicon: use all cores (unified memory, no NUMA issues)
    if device == ComputeDevice.MPS:
        return cpu_count  # M4 Pro: all 10+ cores

    # CUDA: leave some CPU cores for data loading
    if device == ComputeDevice.CUDA:
        return max(1, cpu_count - 2)

    # CPU-only: leave one core for system
    return max(1, cpu_count - 1)


def get_recommended_batch_size(
    device: ComputeDevice | None = None, memory_gb: float | None = None
) -> int:
    """
    Get recommended batch size based on available memory.

    M4 Pro with 48GB unified memory can handle larger batches.

    Args:
        device: Current compute device (auto-detected if None)
        memory_gb: Available memory in GB (auto-detected if None)

    Returns:
        Recommended batch size for training
    """
    if device is None:
        device = detect_optimal_device()

    if memory_gb is None:
        memory_gb = get_available_memory_gb()

    # Check for explicit override
    explicit = os.environ.get("ML_BATCH_SIZE")
    if explicit:
        return int(explicit)

    # Scale batch size based on available memory
    # Base: 32 samples per 16GB
    base_batch_per_16gb = 32

    if device == ComputeDevice.MPS:
        # Apple Silicon unified memory - can use more aggressively
        # M4 Pro 48GB can handle ~3x base
        return int(base_batch_per_16gb * (memory_gb / 16) * 0.75)
    elif device == ComputeDevice.CUDA:
        # Dedicated GPU memory - conservative scaling
        return int(base_batch_per_16gb * (memory_gb / 16) * 0.5)
    else:
        # CPU - very conservative
        return int(base_batch_per_16gb * (memory_gb / 16) * 0.25)


def get_available_memory_gb() -> float:
    """
    Get available system memory in GB.

    Returns:
        Available memory in GB (approximate)
    """
    try:
        import psutil

        return psutil.virtual_memory().available / (1024**3)
    except ImportError:
        # Fallback: assume 16GB if psutil not available
        logger.warning("psutil not available, assuming 16GB memory")
        return 16.0


def get_device_name(device: ComputeDevice | None = None) -> str:
    """
    Get human-readable device name.

    Args:
        device: Compute device (auto-detected if None)

    Returns:
        Device name string (e.g., "Apple M4 Pro", "NVIDIA RTX 3090")
    """
    if device is None:
        device = detect_optimal_device()

    if device == ComputeDevice.CUDA:
        try:
            import torch

            return torch.cuda.get_device_name(0)
        except (ImportError, Exception):
            return "NVIDIA GPU"

    if device == ComputeDevice.MPS:
        try:
            import subprocess

            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return "Apple Silicon"

    return "CPU"


def get_hardware_profile() -> HardwareProfile:
    """
    Get complete hardware profile for ML optimization.

    Returns:
        HardwareProfile with all detected settings
    """
    device = detect_optimal_device()
    cpu_cores = multiprocessing.cpu_count()
    memory_gb = get_available_memory_gb()

    return HardwareProfile(
        device=device,
        device_name=get_device_name(device),
        cpu_cores=cpu_cores,
        optimal_threads=get_optimal_thread_count(device),
        memory_gb=memory_gb,
        recommended_batch_size=get_recommended_batch_size(device, memory_gb),
        supports_mixed_precision=(device == ComputeDevice.CUDA),
    )


def get_torch_device() -> torch.device:
    """
    Get PyTorch device object for the optimal compute device.

    Returns:
        torch.device for CUDA, MPS, or CPU
    """
    try:
        import torch
    except ImportError:
        raise RuntimeError("PyTorch is required for get_torch_device()") from None

    device = detect_optimal_device()

    if device == ComputeDevice.CUDA:
        return torch.device("cuda")
    elif device == ComputeDevice.MPS:
        return torch.device("mps")
    else:
        return torch.device("cpu")


def configure_torch_for_m4_pro() -> None:
    """
    Configure PyTorch for optimal M4 Pro performance.

    Sets:
    - MPS fallback for unsupported operations
    - Optimal thread count for CPU operations
    - Memory allocation settings
    """
    try:
        import torch
    except ImportError:
        logger.warning("PyTorch not available, skipping M4 Pro configuration")
        return

    device = detect_optimal_device()

    if device == ComputeDevice.MPS:
        # Enable MPS fallback for unsupported operations
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

        # Set optimal thread count for CPU operations
        threads = get_optimal_thread_count(device)
        torch.set_num_threads(threads)

        logger.info(f"Configured PyTorch for Apple Silicon: {threads} threads, MPS enabled")

    elif device == ComputeDevice.CUDA:
        # CUDA optimizations
        threads = get_optimal_thread_count(device)
        torch.set_num_threads(threads)

        # Enable TF32 for faster matmul on Ampere+
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        logger.info(f"Configured PyTorch for CUDA: {threads} threads")

    else:
        # CPU-only optimizations
        threads = get_optimal_thread_count(device)
        torch.set_num_threads(threads)

        logger.info(f"Configured PyTorch for CPU: {threads} threads")


def configure_sklearn_for_m4_pro() -> None:
    """
    Configure scikit-learn for optimal M4 Pro performance.

    Sets environment variables for parallel processing.
    """
    device = detect_optimal_device()
    threads = get_optimal_thread_count(device)

    # Set thread counts for various BLAS implementations
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(threads)

    logger.info(f"Configured scikit-learn for {threads} threads")


def configure_lightgbm_for_m4_pro() -> dict:
    """
    Get LightGBM parameters optimized for M4 Pro.

    Returns:
        Dictionary of LightGBM parameters
    """
    device = detect_optimal_device()
    threads = get_optimal_thread_count(device)

    params = {
        "n_jobs": threads,
        "force_col_wise": (device == ComputeDevice.MPS),  # Better cache efficiency
        "max_bin": 255,  # Balanced for unified memory
        "device": "gpu" if device == ComputeDevice.CUDA else "cpu",
    }

    return params


def configure_xgboost_for_m4_pro() -> dict:
    """
    Get XGBoost parameters optimized for M4 Pro.

    Returns:
        Dictionary of XGBoost parameters
    """
    device = detect_optimal_device()
    threads = get_optimal_thread_count(device)

    params = {
        "n_jobs": threads,
        "tree_method": "gpu_hist" if device == ComputeDevice.CUDA else "hist",
        "device": "cuda" if device == ComputeDevice.CUDA else "cpu",
    }

    return params


# Module-level cached values for performance
_CACHED_DEVICE: ComputeDevice | None = None
_CACHED_PROFILE: HardwareProfile | None = None


def get_cached_device() -> ComputeDevice:
    """Get cached compute device (faster than re-detecting)."""
    global _CACHED_DEVICE
    if _CACHED_DEVICE is None:
        _CACHED_DEVICE = detect_optimal_device()
    return _CACHED_DEVICE


def get_cached_profile() -> HardwareProfile:
    """Get cached hardware profile (faster than re-detecting)."""
    global _CACHED_PROFILE
    if _CACHED_PROFILE is None:
        _CACHED_PROFILE = get_hardware_profile()
    return _CACHED_PROFILE


# Convenience aliases for backward compatibility
_detect_cuda = detect_cuda
_detect_mps = detect_mps
_detect_optimal_device = detect_optimal_device
_get_optimal_thread_count = get_optimal_thread_count
