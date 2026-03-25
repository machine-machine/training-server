"""Health check endpoint — no auth required."""

import psutil
from fastapi import APIRouter

router = APIRouter()


def _worker_gpu_info() -> dict:
    """Query GPU status from the celery-worker via a Celery task.

    The API container has no GPU access — only the celery-worker does.
    We dispatch a lightweight task and wait up to 8s for the result.
    """
    try:
        from src.workers.celery_app import app

        # Check if any workers are alive first (fast, no task dispatch)
        inspect = app.control.inspect(timeout=3)
        ping = inspect.ping()
        if not ping:
            return {"gpus": [], "workers_online": 0, "error": "No Celery workers responded to ping"}

        # Workers are alive — ask one for GPU info
        from src.workers.training_tasks import get_gpu_status

        result = get_gpu_status.apply_async()
        gpu_data = result.get(timeout=8)
        return {**gpu_data, "workers_online": len(ping)}
    except Exception as exc:
        return {"gpus": [], "workers_online": 0, "error": str(exc)}


def _active_job_count() -> int:
    """Count currently running Celery tasks."""
    try:
        from src.workers.celery_app import app

        inspect = app.control.inspect(timeout=3)
        active = inspect.active()
        if not active:
            return 0
        return sum(len(tasks) for tasks in active.values())
    except Exception:
        return 0


@router.get(
    "/health",
    summary="Server health",
    description=(
        "Returns current server health including GPU availability, number of online Celery "
        "workers, active training jobs, and host CPU/memory usage. No authentication required."
    ),
    response_description="Health status object",
)
async def health_check():
    worker_info = _worker_gpu_info()
    gpus = worker_info.get("gpus", [])
    active_jobs = _active_job_count()
    gpu_probe_error = worker_info.get("error")

    return {
        "status": "healthy",
        "gpu_available": len(gpus) > 0,
        "gpu_count": len(gpus),
        "gpus": gpus,
        "workers_online": worker_info.get("workers_online", 0),
        "gpu_allocated": worker_info.get("allocated", {}),
        "gpu_probe_error": gpu_probe_error,
        "gpu_probe_source": "celery_worker_task",
        "active_jobs": active_jobs,
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
    }
