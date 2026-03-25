"""Secure pickle deserialization with HMAC-SHA256 verification."""

import hashlib
import hmac
import pickle
from typing import Any

PICKLE_HMAC_KEY = b"coldpath-secure-pickle-v1"  # In production, load from env


class PickleSignatureError(Exception):
    """Raised when pickle data signature verification fails."""

    pass


def sign_data(data: bytes) -> str:
    """Generate HMAC-SHA256 signature for data."""
    return hmac.new(PICKLE_HMAC_KEY, data, hashlib.sha256).hexdigest()


def secure_loads(data: bytes, expected_hash: str | None = None) -> Any:
    """
    Deserialize pickle data with optional signature verification.

    Args:
        data: Pickle serialized bytes
        expected_hash: Optional expected HMAC-SHA256 hex digest

    Returns:
        Deserialized object

    Raises:
        PickleSignatureError: If expected_hash provided and verification fails
    """
    if expected_hash is not None:
        computed = sign_data(data)
        if not hmac.compare_digest(computed, expected_hash):
            raise PickleSignatureError(
                f"Signature mismatch: expected {expected_hash[:16]}..., got {computed[:16]}..."
            )
    return pickle.loads(data)


def secure_load(file, expected_hash: str | None = None) -> Any:
    """
    Deserialize pickle from file with optional signature verification.

    Args:
        file: File-like object opened in binary mode
        expected_hash: Optional expected HMAC-SHA256 hex digest

    Returns:
        Deserialized object
    """
    data = file.read()
    return secure_loads(data, expected_hash)
