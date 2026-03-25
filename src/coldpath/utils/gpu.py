"""GPU detection utilities shared across ColdPath training modules."""

import os


def detect_cuda() -> bool:
    """Detect CUDA GPU availability for tree-based and PyTorch acceleration.

    Checks torch.cuda first (authoritative), falls back to CUDA_VISIBLE_DEVICES
    env var when PyTorch is not installed.
    """
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return bool(os.environ.get("CUDA_VISIBLE_DEVICES"))


CUDA_AVAILABLE: bool = detect_cuda()
