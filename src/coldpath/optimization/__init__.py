"""
Optimization module.

Provides resource management and safety guards for AI-driven optimization.
"""

from .resource_guard import (
    ResourceGuard,
    ResourceStatus,
    get_resource_guard,
)

__all__ = [
    "ResourceGuard",
    "ResourceStatus",
    "get_resource_guard",
]
