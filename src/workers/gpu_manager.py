"""GPU allocation manager for 2x RTX 3090.

State is stored in Redis so all Celery prefork workers share the same
view of GPU allocations — an in-process threading.Lock does not survive
across forked worker processes.
"""

import os
import subprocess

import redis


def _get_redis() -> redis.Redis:
    url = os.environ.get("REDIS_URL", os.environ.get("CELERY_BROKER_URL", "redis://redis:6379/0"))
    return redis.from_url(url, decode_responses=True)


class GPUManager:
    """Track GPU availability and allocate devices to training jobs.

    Uses Redis SET NX / DEL for atomic cross-process allocation so that
    multiple Celery prefork workers can't assign the same GPU concurrently.
    """

    _KEY_PREFIX = "dexy:gpu:alloc:"
    _TTL_SECONDS = 7200  # 2 h — matches task time_limit; auto-expires stale locks

    def __init__(self, device_count: int = 2):
        self._device_count = device_count

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def allocate(self, job_id: str, preferred_gpu: int | None = None) -> int | None:
        """Atomically allocate a GPU. Returns gpu_id or None if all busy."""
        r = _get_redis()
        candidates = (
            [preferred_gpu]
            if preferred_gpu is not None and 0 <= preferred_gpu < self._device_count
            else []
        )
        candidates += [i for i in range(self._device_count) if i != preferred_gpu]

        for gpu_id in candidates:
            key = f"{self._KEY_PREFIX}{gpu_id}"
            if r.set(key, job_id, nx=True, ex=self._TTL_SECONDS):
                return gpu_id

        return None

    def release(self, gpu_id: int) -> None:
        _get_redis().delete(f"{self._KEY_PREFIX}{gpu_id}")

    @property
    def _allocated(self) -> dict[int, str]:
        """Return current allocations for status reporting."""
        r = _get_redis()
        result = {}
        for gpu_id in range(self._device_count):
            val = r.get(f"{self._KEY_PREFIX}{gpu_id}")
            if val:
                result[gpu_id] = val
        return result

    def get_free_memory_mb(self, gpu_id: int) -> int:
        """Query free VRAM for a specific GPU via nvidia-smi."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={gpu_id}",
                    "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return int(result.stdout.strip())
        except Exception:
            return 0


# Singleton — state lives in Redis, so this is safe across forked workers
gpu_manager = GPUManager()
