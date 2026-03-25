"""IPC module for training-server communication with HotPath."""

from .hotpath_client import HotPathClient, push_artifact_sync

__all__ = ["HotPathClient", "push_artifact_sync"]
