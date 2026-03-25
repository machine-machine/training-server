"""
ColdPath utility helpers.

Components:
- m4_pro_optimizer: Apple Silicon M4 Pro hardware detection and optimization
- gpu: GPU detection and configuration utilities
- fee_calculator: Trading fee calculation utilities
"""

from .m4_pro_optimizer import (
    # Device detection
    ComputeDevice,
    HardwareProfile,
    configure_lightgbm_for_m4_pro,
    configure_sklearn_for_m4_pro,
    configure_torch_for_m4_pro,
    configure_xgboost_for_m4_pro,
    detect_cuda,
    detect_mps,
    detect_optimal_device,
    get_available_memory_gb,
    get_cached_device,
    get_cached_profile,
    get_device_name,
    get_hardware_profile,
    # Configuration
    get_optimal_thread_count,
    get_recommended_batch_size,
    get_torch_device,
)

__all__ = [
    # Device detection
    "ComputeDevice",
    "HardwareProfile",
    "detect_cuda",
    "detect_mps",
    "detect_optimal_device",
    "get_cached_device",
    "get_cached_profile",
    "get_hardware_profile",
    "get_torch_device",
    # Configuration
    "get_optimal_thread_count",
    "get_recommended_batch_size",
    "get_available_memory_gb",
    "get_device_name",
    "configure_torch_for_m4_pro",
    "configure_sklearn_for_m4_pro",
    "configure_lightgbm_for_m4_pro",
    "configure_xgboost_for_m4_pro",
]
