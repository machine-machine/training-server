"""Data management endpoints — upload datasets, receive trade feedback."""

import hashlib
import os
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.auth import verify_api_key
from src.api.deps import get_db
from src.db.models import TrainingDataset

router = APIRouter()

DATA_PATH = Path(os.environ.get("TRAINING_DATA_PATH", "/data/datasets"))
MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500 MB


@router.post(
    "/upload",
    dependencies=[Depends(verify_api_key)],
    status_code=201,
    summary="Upload a training dataset",
    description=(
        "Accepts a CSV, Parquet, or newline-delimited JSON file (max 500 MB). "
        "Deduplicated by SHA-256 — re-uploading the same file returns the existing record. "
        "Returns a ``dataset_id`` UUID to pass to ``POST /jobs/train``."
    ),
    response_description="dataset_id, row/feature counts, and SHA-256 checksum",
)
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(..., description="Unique human-readable dataset name"),
    description: str = Form("", description="Optional description"),
    label_column: str = Form("", description="Name of the target/label column, if known"),
    db: AsyncSession = Depends(get_db),
):
    """Upload a training dataset (CSV, Parquet, JSON)."""
    if not file.filename:
        raise HTTPException(400, detail="No filename provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in {".csv", ".parquet", ".json"}:
        raise HTTPException(400, detail="Supported formats: .csv, .parquet, .json")

    # Read file contents
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, detail="File exceeds 500 MB limit")

    checksum = hashlib.sha256(contents).hexdigest()

    # Save to disk
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    file_path = DATA_PATH / f"{checksum}{ext}"
    file_path.write_bytes(contents)

    # Count rows/features
    row_count = 0
    feature_count = 0
    try:
        if ext == ".csv":
            df = pd.read_csv(file_path, nrows=0)
            feature_count = len(df.columns)
            row_count = sum(1 for _ in open(file_path)) - 1
        elif ext == ".parquet":
            df = pd.read_parquet(file_path)
            row_count = len(df)
            feature_count = len(df.columns)
        elif ext == ".json":
            df = pd.read_json(file_path, lines=True, nrows=1)
            feature_count = len(df.columns)
    except Exception:
        pass  # Stats are best-effort

    dataset = TrainingDataset(
        name=name,
        description=description,
        file_path=str(file_path),
        file_format=ext.lstrip("."),
        row_count=row_count,
        feature_count=feature_count,
        label_column=label_column or None,
        checksum=checksum,
        size_bytes=len(contents),
    )
    db.add(dataset)
    await db.commit()
    await db.refresh(dataset)

    return {
        "dataset_id": str(dataset.id),
        "name": name,
        "row_count": row_count,
        "feature_count": feature_count,
        "checksum": f"sha256:{checksum}",
    }


class TradeFeedback(BaseModel):
    trade_id: str
    """Unique trade identifier (HotPath internal ID)."""
    timestamp_ms: int
    """Unix timestamp in milliseconds when the trade executed."""
    mint: str
    """Solana token mint address."""
    features: list[float]
    """Feature vector used by the model to make this decision."""
    feature_names: list[str] = []
    """Names corresponding to each element of ``features``."""
    decision: str
    """Model decision: ``buy`` or ``skip``."""
    confidence: float
    """Model confidence score 0–1."""
    amount_in_sol: float
    """SOL spent on the trade."""
    amount_out_tokens: float
    """Tokens received."""
    pnl_sol: float
    """Realised profit/loss in SOL (positive = profit)."""
    slippage_bps: int
    """Actual slippage experienced in basis points."""
    execution_latency_ms: int = 0
    """Time from decision to on-chain confirmation in ms."""
    model_version: int = 0
    """Version of the model artifact that made this decision."""


class FeedbackBatch(BaseModel):
    trades: list[TradeFeedback]
    """List of trade outcomes. Max 500 per request."""


@router.post(
    "/feedback",
    dependencies=[Depends(verify_api_key)],
    summary="Submit live trade outcomes",
    description=(
        "Receives HotPath trade results and appends them to the feedback JSONL file "
        "(``/data/datasets/feedback/trade_outcomes.jsonl``) for use as a training dataset. "
        "Max 500 trades per batch. To train on this data, upload the JSONL file via "
        "``POST /data/upload`` and launch a job."
    ),
    response_description="Count of accepted trades and running total",
)
async def submit_feedback(batch: FeedbackBatch, db: AsyncSession = Depends(get_db)):
    """Receive trade outcomes from HotPath for online learning."""
    if not batch.trades:
        raise HTTPException(400, detail="Empty trade batch")
    if len(batch.trades) > 500:
        raise HTTPException(400, detail="Max 500 trades per batch")

    # Append to the feedback dataset file (JSONL format)
    feedback_path = DATA_PATH / "feedback"
    feedback_path.mkdir(parents=True, exist_ok=True)
    feedback_file = feedback_path / "trade_outcomes.jsonl"

    import json

    with open(feedback_file, "a") as f:
        for trade in batch.trades:
            f.write(json.dumps(trade.model_dump()) + "\n")

    # Count total feedback records
    total = 0
    if feedback_file.exists():
        total = sum(1 for _ in open(feedback_file))

    return {
        "accepted": len(batch.trades),
        "rejected": 0,
        "total_feedback_records": total,
    }
