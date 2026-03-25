"""Celery application factory."""

import os

from celery import Celery

BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://redis:6379/0")
RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", "redis://redis:6379/0")

app = Celery(
    "dexy_training",
    broker=BROKER_URL,
    backend=RESULT_BACKEND,
    include=["src.workers.training_tasks"],
)

app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,  # One task at a time per worker (GPU-bound)
    result_expires=86400,  # 24h
)
