"""
ColdPath gRPC servicer implementations.

This module provides working implementations of the gRPC servicers
that HotPath calls to submit jobs and request data from ColdPath.
"""

import json
import logging
import math
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

import grpc

from . import coldpath_pb2, coldpath_pb2_grpc

if TYPE_CHECKING:
    from ..learning.outcome_tracker import OutcomeTracker
    from ..storage import DatabaseManager

logger = logging.getLogger(__name__)


def validate_trade_record(trade) -> str | None:
    """Validate a trade record for training data quality.

    Returns None if valid, or an error message string if invalid.

    Quality Checks:
    - mint must not be empty
    - entry_price must be positive and finite (if > 0)
    - exit_price must be non-negative and finite (if > 0)
    - slippage values must be non-negative and finite
    - No NaN or Infinity in numeric fields
    """
    # Check required fields
    if not trade.mint or not trade.mint.strip():
        return "mint is empty"

    # Check entry_price
    if trade.entry_price > 0:
        if not math.isfinite(trade.entry_price):
            return f"entry_price is not finite: {trade.entry_price}"

    # Check exit_price
    if trade.exit_price > 0:
        if not math.isfinite(trade.exit_price):
            return f"exit_price is not finite: {trade.exit_price}"

    # Check slippage values
    if not math.isfinite(trade.quoted_slippage_bps):
        return f"quoted_slippage_bps is not finite: {trade.quoted_slippage_bps}"
    if trade.quoted_slippage_bps < 0:
        return f"quoted_slippage_bps must be non-negative: {trade.quoted_slippage_bps}"

    if not math.isfinite(trade.realized_slippage_bps):
        return f"realized_slippage_bps is not finite: {trade.realized_slippage_bps}"
    if trade.realized_slippage_bps < 0:
        return f"realized_slippage_bps must be non-negative: {trade.realized_slippage_bps}"

    # Check latencies
    if not math.isfinite(trade.detection_latency_ms):
        return f"detection_latency_ms is not finite: {trade.detection_latency_ms}"
    if trade.detection_latency_ms < 0:
        return f"detection_latency_ms must be non-negative: {trade.detection_latency_ms}"

    if not math.isfinite(trade.execution_latency_ms):
        return f"execution_latency_ms is not finite: {trade.execution_latency_ms}"
    if trade.execution_latency_ms < 0:
        return f"execution_latency_ms must be non-negative: {trade.execution_latency_ms}"

    return None  # Valid


class ColdPathJobsServicerImpl(coldpath_pb2_grpc.ColdPathJobsServicer):
    """Implementation of ColdPathJobs service for HotPath -> ColdPath calls.

    Handles training data submission, calibration requests, and backtest data.
    """

    def __init__(self, db: "DatabaseManager", outcome_tracker: Optional["OutcomeTracker"] = None):
        """Initialize the servicer with database and optional outcome tracker.

        Args:
            db: DatabaseManager instance for data persistence
            outcome_tracker: Optional OutcomeTracker for recording trade outcomes
        """
        self.db = db
        self.outcome_tracker = outcome_tracker

    def SubmitTrainingData(self, request, context):
        """Receive training data from HotPath and store for ML training.

        Transforms TradeRecord messages from HotPath into scan_outcomes records
        that can be used for profitability model training.

        Performs data quality validation on each trade record and rejects
        invalid samples.

        IMPORTANT: Live mode training data requires special handling due to:
        - Real financial risk
        - Slippage uncertainty until blockchain confirmation
        - Potential for data quality issues

        Live mode samples are marked with is_live_mode=true in features_json.

        Args:
            request: TrainingDataBatch containing TradeRecord messages
            context: gRPC context

        Returns:
            JobAck with job_id, accepted status, and optional error message
        """
        job_id = str(uuid.uuid4())

        try:
            count = 0
            rejected = 0
            live_count = 0
            timestamp_ms = request.batch_timestamp or int(datetime.now().timestamp() * 1000)
            source_mode = request.source_mode or "UNKNOWN"

            # Warn if receiving live mode data
            if "LIVE" in source_mode.upper():
                logger.warning(
                    "⚠️  RECEIVING LIVE MODE TRAINING DATA - Use with caution! "
                    "Live trades involve real financial risk."
                )

            for trade in request.trades:
                # Data quality validation
                validation_error = validate_trade_record(trade)
                if validation_error:
                    logger.warning(
                        f"Rejected invalid trade record (mint={trade.mint[:16]}...): "
                        f"{validation_error}"
                    )
                    rejected += 1
                    continue

                # Track if this is live mode data
                is_live_mode = "LIVE" in trade.mode.upper() or "LIVE" in source_mode.upper()
                if is_live_mode:
                    live_count += 1

                # Calculate PnL percentage
                pnl_pct = None
                if trade.entry_price and trade.entry_price > 0 and trade.exit_price:
                    pnl_pct = ((trade.exit_price - trade.entry_price) / trade.entry_price) * 100

                # Calculate PnL in SOL (approximate, would need position size)
                # For now, we just record the percentage
                pnl_sol = None

                # Calculate hold duration
                hold_duration_ms = None
                if trade.exit_slot and trade.entry_slot:
                    # Slots are ~400ms apart on Solana
                    hold_duration_ms = (trade.exit_slot - trade.entry_slot) * 400

                # Build features JSON from optional fields
                # In proto3, optional scalar fields can be checked with HasField for presence
                # Non-optional fields with default values should just check non-zero
                features = {}

                # Optional double fields (field numbers 20-24 in TradeRecord)
                if trade.HasField("liquidity_usd"):
                    features["liquidity_usd"] = trade.liquidity_usd
                if trade.HasField("fdv_usd"):
                    features["fdv_usd"] = trade.fdv_usd
                if trade.HasField("volume_1h"):
                    features["volume_1h"] = trade.volume_1h
                if trade.HasField("holder_count"):
                    features["holder_count"] = trade.holder_count
                if trade.HasField("volatility"):
                    features["volatility"] = trade.volatility

                # Add execution metadata
                features["quoted_slippage_bps"] = trade.quoted_slippage_bps
                features["realized_slippage_bps"] = trade.realized_slippage_bps
                features["detection_latency_ms"] = trade.detection_latency_ms
                features["execution_latency_ms"] = trade.execution_latency_ms
                features["included"] = trade.included
                features["mode"] = trade.mode
                features["source_mode"] = source_mode

                # SAFEGUARD: Mark live mode samples clearly for special handling in training
                features["is_live_mode"] = is_live_mode
                if is_live_mode:
                    features["live_mode_warning"] = (
                        "Real financial trade - use with caution in training"
                    )

                # Determine outcome type
                if trade.included:
                    outcome_type = "traded"
                else:
                    outcome_type = "failed"

                # Prepare scan_outcome record
                outcome_data = {
                    "mint": trade.mint,
                    "pool": "",  # Not provided in TradeRecord
                    "timestamp_ms": trade.entry_slot * 400 if trade.entry_slot else timestamp_ms,
                    "outcome_type": outcome_type,
                    "skip_reason": None,
                    "features_json": json.dumps(features),
                    "profitability_score": None,
                    "confidence": None,
                    "expected_return_pct": None,
                    "model_version": 0,
                    "source": f"hotpath_{source_mode.lower()}",
                    "entry_price": trade.entry_price if trade.entry_price else None,
                    "exit_price": trade.exit_price if trade.exit_price else None,
                    "pnl_pct": pnl_pct,
                    "pnl_sol": pnl_sol,
                    "hold_duration_ms": hold_duration_ms,
                }

                # Insert into database (sync wrapper for async method)
                # Note: In a proper async gRPC setup, we'd use async handlers
                import asyncio

                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # We're in an async context, schedule the coroutine
                        asyncio.create_task(self.db.insert_scan_outcome(outcome_data))
                    else:
                        # We're in a sync context, run the coroutine
                        loop.run_until_complete(self.db.insert_scan_outcome(outcome_data))
                except RuntimeError:
                    # No event loop, create one
                    asyncio.run(self.db.insert_scan_outcome(outcome_data))

                count += 1

            # Log summary with live mode warning
            log_msg = (
                f"Received {count} training samples from HotPath "
                f"(job_id={job_id}, mode={source_mode}, rejected={rejected})"
            )
            if live_count > 0:
                log_msg += f" ⚠️ LIVE_MODE_COUNT={live_count}"
                logger.warning(log_msg)
            else:
                logger.info(log_msg)

            return coldpath_pb2.JobAck(
                job_id=job_id,
                accepted=True,
            )

        except Exception as e:
            logger.error(f"SubmitTrainingData failed: {e}", exc_info=True)
            return coldpath_pb2.JobAck(
                job_id=job_id,
                accepted=False,
                error=str(e),
            )

    def RequestCalibration(self, request, context):
        """Request calibration of a model type.

        Args:
            request: CalibrationRequest with model_type and lookback_hours
            context: gRPC context

        Returns:
            CalibrationResult with success status and optional updated params
        """
        # TODO: Implement calibration request handling
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Calibration request not yet implemented")
        return coldpath_pb2.CalibrationResult(
            model_type=request.model_type,
            success=False,
            error="Calibration request not yet implemented",
        )

    def GetModelParams(self, request, context):
        """Get model parameters for requested model types.

        Args:
            request: ParamRequest with list of model types
            context: gRPC context

        Returns:
            ModelParams with current parameter values
        """
        # TODO: Implement model params retrieval
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Model params retrieval not yet implemented")
        return coldpath_pb2.ModelParams()

    def GetBacktestData(self, request, context):
        """Stream backtest data for the requested period.

        Args:
            request: BacktestDataRequest with data source and time range
            context: gRPC context

        Yields:
            BacktestEvent messages with historical data
        """
        # TODO: Implement backtest data streaming
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Backtest data streaming not yet implemented")
        return
        yield  # Make this a generator


class AnnouncementProcessorServicerImpl(coldpath_pb2_grpc.AnnouncementProcessorServicer):
    """Implementation of AnnouncementProcessor service for HotPath -> ColdPath calls.

    Handles announcement classification, entity extraction, and verification.
    """

    def __init__(self, db: "DatabaseManager"):
        """Initialize the servicer with database.

        Args:
            db: DatabaseManager instance for data persistence
        """
        self.db = db

    def ProcessBatch(self, request, context):
        """Process a batch of announcements through full pipeline."""
        # TODO: Implement batch processing
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Batch processing not yet implemented")
        return coldpath_pb2.ProcessedBatch()

    def ClassifyBatch(self, request, context):
        """Classify announcements only (fast path)."""
        # TODO: Implement classification
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Classification not yet implemented")
        return coldpath_pb2.ClassificationBatch()

    def ExtractEntities(self, request, context):
        """Extract entities from a single announcement."""
        # TODO: Implement entity extraction
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Entity extraction not yet implemented")
        return coldpath_pb2.ExtractedEntities()

    def VerifyOnChain(self, request, context):
        """Verify token on-chain."""
        # TODO: Implement on-chain verification
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("On-chain verification not yet implemented")
        return coldpath_pb2.VerificationResult()

    def GetSourceReputation(self, request, context):
        """Get source reputation."""
        # TODO: Implement source reputation lookup
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Source reputation lookup not yet implemented")
        return coldpath_pb2.SourceReputationResponse()

    def RecordOutcome(self, request, context):
        """Record outcome for learning.

        Args:
            request: OutcomeRecord with announcement outcome data
            context: gRPC context

        Returns:
            JobAck with acceptance status
        """
        job_id = str(uuid.uuid4())

        try:
            # Store outcome for learning
            import asyncio

            outcome_data = {
                "mint": request.mint,
                "pool": "",
                "timestamp_ms": request.recorded_at_ms or int(datetime.now().timestamp() * 1000),
                "outcome_type": "traded" if request.was_traded else "skipped",
                "skip_reason": None,
                "features_json": json.dumps(
                    {
                        "was_traded": request.was_traded,
                        "was_profitable": request.was_profitable,
                        "was_scam": request.was_scam,
                        "lead_time_ms": request.lead_time_ms,
                    }
                ),
                "pnl_sol": request.pnl_sol if request.HasField("pnl_sol") else None,
                "pnl_pct": None,  # Would need to calculate from pnl_sol and position size
                "model_version": 0,
                "source": "announcement_outcome",
            }

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self.db.insert_scan_outcome(outcome_data))
                else:
                    loop.run_until_complete(self.db.insert_scan_outcome(outcome_data))
            except RuntimeError:
                asyncio.run(self.db.insert_scan_outcome(outcome_data))

            logger.info(f"Recorded announcement outcome for {request.mint} (job_id={job_id})")

            return coldpath_pb2.JobAck(
                job_id=job_id,
                accepted=True,
            )

        except Exception as e:
            logger.error(f"RecordOutcome failed: {e}", exc_info=True)
            return coldpath_pb2.JobAck(
                job_id=job_id,
                accepted=False,
                error=str(e),
            )
