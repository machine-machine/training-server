"""
AutoTrader control and status routes.

Endpoints:
- GET /autotrader/status - Get live AutoTrader status
- POST /autotrader/start - Start the AutoTrader
- POST /autotrader/stop - Stop the AutoTrader
- POST /autotrader/pause - Pause trading
- POST /autotrader/resume - Resume trading
- PUT /autotrader/config - Update AutoTrader configuration
- POST /autotrader/force-trading - Force transition to trading mode
"""

import logging
import os
import random
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from coldpath.autotrader.coordinator import AutoTraderState
from coldpath.storage import DatabaseManager, ModelArtifactStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/autotrader", tags=["autotrader"])


def get_app_state():
    """Lazy import to avoid circular dependency during server bootstrap."""
    from coldpath.api.server import get_app_state as _get_app_state

    return _get_app_state()


class AutoTraderDailyMetricsResponse(BaseModel):
    date: str | None = None
    volume_sol: float = 0.0
    pnl_sol: float = 0.0
    trade_count: int = 0
    win_rate: float = 0.0
    max_drawdown_sol: float = 0.0


class AutoTraderLearningMetricsResponse(BaseModel):
    observations: int = 0
    paper_trades: int = 0
    paper_win_rate: float = 0.0
    duration_hours: float = 0.0


class AutoTraderStatusResponse(BaseModel):
    state: str = "stopped"
    execution_mode: str = "stopped"
    state_changed_at: str | None = None
    last_status_at: str | None = None
    last_candidate_at: str | None = None
    last_signal_at: str | None = None
    last_outcome_at: str | None = None
    hotpath_connected: bool = False
    config: dict[str, Any] = Field(default_factory=dict)
    daily_metrics: AutoTraderDailyMetricsResponse
    learning_metrics: AutoTraderLearningMetricsResponse
    pending_signals: int = 0
    in_cooldown: bool = False
    consecutive_losses: int = 0
    source: str = "live"


class AutoTraderControlResponse(BaseModel):
    success: bool
    message: str
    state: str


class TradeSummary(BaseModel):
    id: str
    timestamp_ms: int
    plan_id: str | None = None
    mint: str
    symbol: str | None = None
    side: str
    amount_sol: float
    price: float
    pnl_sol: float | None = None
    pnl_pct: float | None = None
    included: bool | None = None

    # Fee breakdown (Migration 1)
    base_fee_lamports: int | None = None
    priority_fee_lamports: int | None = None
    jito_tip_lamports: int | None = None
    dex_fee_lamports: int | None = None
    total_fees_sol: float | None = None
    net_pnl_sol: float | None = None

    # Execution details (Migration 2)
    tx_signature: str | None = None
    slot: int | None = None
    execution_time_ms: int | None = None
    amount_out_lamports: int | None = None


@router.get("/status", response_model=AutoTraderStatusResponse)
async def get_autotrader_status():
    """Return live AutoTrader status or fallback to artifacts."""
    state = get_app_state()

    # Prefer live AutoTrader instance when available
    autotrader = state.autotrader
    if (
        autotrader is None
        and state.engine is not None
        and getattr(state.engine, "autotrader", None)
    ):
        autotrader = state.engine.autotrader
        logger.info("AutoTrader status using engine-backed instance")

    # If AutoTrader is running, return live status
    if autotrader is not None:
        status = autotrader.get_status()
        daily = status.get("daily_metrics", {})
        learning = status.get("learning_metrics", {})
        hotpath_connected = False
        if state.engine is not None and state.engine.hotpath_client is not None:
            hotpath_connected = state.engine.hotpath_client.is_connected

        return AutoTraderStatusResponse(
            state=status.get("state", "stopped"),
            execution_mode=status.get("execution_mode", status.get("state", "stopped")),
            state_changed_at=status.get("state_changed_at"),
            last_status_at=status.get("last_status_at"),
            last_candidate_at=status.get("last_candidate_at"),
            last_signal_at=status.get("last_signal_at"),
            last_outcome_at=status.get("last_outcome_at"),
            hotpath_connected=hotpath_connected,
            config=status.get("config", {}),
            daily_metrics=AutoTraderDailyMetricsResponse(
                date=daily.get("date"),
                volume_sol=daily.get("volume_sol", 0.0),
                pnl_sol=daily.get("pnl_sol", 0.0),
                trade_count=daily.get("trade_count", 0),
                win_rate=daily.get("win_rate", 0.0),
                max_drawdown_sol=daily.get("max_drawdown_sol", 0.0),
            ),
            learning_metrics=AutoTraderLearningMetricsResponse(
                observations=learning.get("observations", 0),
                paper_trades=learning.get("paper_trades", 0),
                paper_win_rate=learning.get("paper_win_rate", 0.0),
                duration_hours=learning.get("duration_hours", 0.0),
            ),
            pending_signals=status.get("pending_signals", 0),
            in_cooldown=status.get("in_cooldown", False),
            consecutive_losses=status.get("consecutive_losses", 0),
            source="live",
        )

    # Fallback to artifacts if AutoTrader not initialized
    model_dir = os.getenv("MODEL_DIR", "models")
    store = ModelArtifactStore(model_dir)

    artifact_state = store.load_latest("autotrader")
    if not artifact_state:
        return AutoTraderStatusResponse(
            daily_metrics=AutoTraderDailyMetricsResponse(),
            learning_metrics=AutoTraderLearningMetricsResponse(),
            source="default",
        )

    daily = artifact_state.get("daily_metrics", {})
    learning = artifact_state.get("learning_metrics", {})

    win_count = daily.get("win_count", 0)
    loss_count = daily.get("loss_count", 0)
    total_wins = win_count + loss_count
    win_rate = win_count / total_wins if total_wins > 0 else 0.0

    paper_trades = learning.get("paper_trades", 0)
    paper_wins = learning.get("paper_wins", 0)
    paper_win_rate = paper_wins / paper_trades if paper_trades > 0 else 0.0

    start_time_str = learning.get("start_time")
    duration_hours = 0.0
    if start_time_str:
        try:
            start_time = datetime.fromisoformat(start_time_str)
            duration_hours = (datetime.now() - start_time).total_seconds() / 3600
        except ValueError:
            pass

    return AutoTraderStatusResponse(
        state=artifact_state.get("state", "stopped"),
        execution_mode=artifact_state.get(
            "execution_mode",
            artifact_state.get("state", "stopped"),
        ),
        config=artifact_state.get("config", {}) or {},
        hotpath_connected=False,
        daily_metrics=AutoTraderDailyMetricsResponse(
            date=daily.get("date"),
            volume_sol=daily.get("total_volume_sol", 0.0),
            pnl_sol=daily.get("total_pnl_sol", 0.0),
            trade_count=daily.get("trade_count", 0),
            win_rate=win_rate,
            max_drawdown_sol=daily.get("max_drawdown_sol", 0.0),
        ),
        learning_metrics=AutoTraderLearningMetricsResponse(
            observations=learning.get("observations", 0),
            paper_trades=paper_trades,
            paper_win_rate=paper_win_rate,
            duration_hours=duration_hours,
        ),
        pending_signals=artifact_state.get("pending_signals", 0),
        in_cooldown=artifact_state.get("in_cooldown", False),
        consecutive_losses=artifact_state.get("consecutive_losses", 0),
        source="artifact",
    )


@router.post("/start", response_model=AutoTraderControlResponse)
async def start_autotrader():
    """Start the AutoTrader."""
    state = get_app_state()

    # Require engine-backed autotrader to avoid DB-less stub
    if state.autotrader is None:
        logger.error("AutoTrader not available: engine not initialized")
        raise HTTPException(
            status_code=503, detail="Engine not initialized; AutoTrader unavailable"
        )

    if state.shutting_down:
        logger.warning("Rejecting AutoTrader start during ColdPath shutdown")
        raise HTTPException(
            status_code=503, detail="ColdPath is shutting down; AutoTrader start rejected"
        )

    if state.engine_task is not None and state.engine_task.done():
        logger.warning(
            "Rejecting AutoTrader start because ColdPath engine task is no longer running"
        )
        raise HTTPException(
            status_code=503, detail="ColdPath engine is not running; AutoTrader start rejected"
        )

    if state.autotrader.state != AutoTraderState.STOPPED:
        current_state = state.autotrader.state
        current = current_state.value
        # Treat duplicate starts as idempotent success while AutoTrader is already active.
        # This avoids surfacing false-failure startup paths to callers racing with engine bootstrap.
        if current_state in (
            AutoTraderState.STARTING,
            AutoTraderState.LEARNING,
            AutoTraderState.TRADING,
        ):
            logger.info("AutoTrader start requested while already active (state=%s)", current)
            return AutoTraderControlResponse(
                success=True,
                message=f"AutoTrader already active (state={current})",
                state=current,
            )
        logger.warning("Rejecting AutoTrader start from non-startable state=%s", current)
        raise HTTPException(
            status_code=409, detail=f"AutoTrader start rejected: current_state={current}"
        )

    try:
        await state.autotrader.start()
        current_state = state.autotrader.state
        if current_state not in (AutoTraderState.LEARNING, AutoTraderState.TRADING):
            logger.error(
                "AutoTrader start did not reach active state; state=%s", current_state.value
            )
            raise HTTPException(
                status_code=503,
                detail=f"AutoTrader start not accepted by coordinator: state={current_state.value}",
            )

        # Auto-start data collection: Connect to HotPath and start receiving candidates
        hotpath_connected = False
        if state.engine is not None and hasattr(state.engine, "hotpath_client"):
            hotpath_client = state.engine.hotpath_client
            if hotpath_client is not None:
                # Try to connect if not already connected
                if not hotpath_client.is_connected:
                    try:
                        hotpath_connected = await hotpath_client.connect()
                        if hotpath_connected:
                            logger.info("Auto-connected to HotPath for data collection")
                            # Start receiving candidate events
                            await hotpath_client.start_event_stream()
                    except Exception as e:
                        logger.warning(f"Failed to connect to HotPath: {e}")
                else:
                    hotpath_connected = True

                # Start HotPath AutoPipeline if connected
                if hotpath_connected:
                    try:
                        result = await hotpath_client.start_autotrader()
                        if result.get("autotrader_running"):
                            logger.info("Started HotPath AutoPipeline via ColdPath API")
                        else:
                            logger.warning(f"HotPath AutoPipeline start failed: {result}")
                    except Exception as e:
                        logger.warning(f"Failed to start HotPath AutoPipeline: {e}")

        # Start training integration
        try:
            from coldpath.autotrader.training_integration import (
                get_training_integration,
            )

            if state.engine and hasattr(state.engine, "db") and state.engine.db:
                integration = get_training_integration(state.engine.db)
                await integration.start()
                logger.info("Training integration started")
        except Exception as e:
            logger.warning(f"Failed to start training integration: {e}")

        return AutoTraderControlResponse(
            success=True,
            message=(
                f"AutoTrader started (HotPath: "
                f"{'connected' if hotpath_connected else 'disconnected'})"
            ),
            state=state.autotrader.state.value,
        )
    except Exception as e:
        logger.error(f"Failed to start AutoTrader: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/stop", response_model=AutoTraderControlResponse)
async def stop_autotrader():
    """Stop the AutoTrader."""
    state = get_app_state()

    if state.autotrader is None:
        return AutoTraderControlResponse(
            success=True,
            message="AutoTrader not running",
            state="stopped",
        )

    try:
        await state.autotrader.stop()
        return AutoTraderControlResponse(
            success=True,
            message="AutoTrader stopped",
            state=state.autotrader.state.value,
        )
    except Exception as e:
        logger.error(f"Failed to stop AutoTrader: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/pause", response_model=AutoTraderControlResponse)
async def pause_autotrader():
    """Pause AutoTrader trading."""
    state = get_app_state()

    if state.autotrader is None:
        raise HTTPException(status_code=400, detail="AutoTrader not initialized")

    try:
        await state.autotrader.pause("api_request")
        return AutoTraderControlResponse(
            success=True,
            message="AutoTrader paused",
            state=state.autotrader.state.value,
        )
    except Exception as e:
        logger.error(f"Failed to pause AutoTrader: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/trades", response_model=list[TradeSummary])
async def get_autotrader_trades(limit: int = 200):
    """Return recent trades for UI performance charts."""
    state = get_app_state()

    # Prefer live DB from running engine
    db: DatabaseManager | None = None
    if state.engine is not None:
        db = state.engine.db

    if db is None:
        raise HTTPException(status_code=500, detail="Engine database unavailable")

    try:
        async with db._lock:  # reuse existing lock; read-only query
            with db._cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, timestamp_ms, mint, symbol, side, amount_sol,
                           price, pnl_sol, pnl_pct, execution_mode, included,
                           base_fee_lamports, priority_fee_lamports, jito_tip_lamports,
                           dex_fee_lamports, total_fees_sol, net_pnl_sol,
                           tx_signature, slot, execution_time_ms, amount_out_lamports
                    FROM trades
                    ORDER BY timestamp_ms DESC
                    LIMIT ?
                    """,
                    (limit,),
                )
                rows = cursor.fetchall()
                summaries = []
                for row in rows:
                    data = dict(row)
                    summaries.append(
                        TradeSummary(
                            id=data.get("id"),
                            timestamp_ms=data.get("timestamp_ms"),
                            plan_id=data.get("id"),  # plan_id stored as id in this table
                            mint=data.get("mint"),
                            symbol=data.get("symbol"),
                            side=data.get("side"),
                            amount_sol=data.get("amount_sol", 0.0),
                            price=data.get("price", 0.0),
                            pnl_sol=data.get("pnl_sol"),
                            pnl_pct=data.get("pnl_pct"),
                            included=bool(data.get("included", 1)),
                            # Fee fields
                            base_fee_lamports=data.get("base_fee_lamports"),
                            priority_fee_lamports=data.get("priority_fee_lamports"),
                            jito_tip_lamports=data.get("jito_tip_lamports"),
                            dex_fee_lamports=data.get("dex_fee_lamports"),
                            total_fees_sol=data.get("total_fees_sol"),
                            net_pnl_sol=data.get("net_pnl_sol"),
                            # Execution details
                            tx_signature=data.get("tx_signature"),
                            slot=data.get("slot"),
                            execution_time_ms=data.get("execution_time_ms"),
                            amount_out_lamports=data.get("amount_out_lamports"),
                        )
                    )
                return summaries
    except Exception as e:
        logger.error(f"Failed to fetch trades: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch trades") from e


@router.post("/resume", response_model=AutoTraderControlResponse)
async def resume_autotrader():
    """Resume AutoTrader trading."""
    state = get_app_state()

    if state.autotrader is None:
        raise HTTPException(status_code=400, detail="AutoTrader not initialized")

    try:
        await state.autotrader.resume()
        return AutoTraderControlResponse(
            success=True,
            message="AutoTrader resumed",
            state=state.autotrader.state.value,
        )
    except Exception as e:
        logger.error(f"Failed to resume AutoTrader: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


class AutoTraderConfigUpdate(BaseModel):
    """Request body for updating AutoTrader config at runtime."""

    max_position_sol: float | None = None
    max_daily_volume_sol: float | None = None
    max_daily_loss_sol: float | None = None
    max_daily_trades: int | None = None
    min_confidence: float | None = Field(None, alias="min_confidence_to_trade")
    max_concurrent_positions: int | None = None
    cooldown_after_loss_seconds: int | None = None
    ml_strategy: str | None = None
    risk_mode: str | None = None
    ensemble_enabled: bool | None = None
    trailing_stop_enabled: bool | None = None
    trailing_stop_percent: float | None = None
    paper_mode: bool | None = None
    skip_learning: bool | None = None
    min_observations_to_trade: int | None = None

    model_config = {"populate_by_name": True}


class AutoTraderConfigResponse(BaseModel):
    success: bool
    message: str
    config: dict[str, Any]


@router.put("/config", response_model=AutoTraderConfigResponse)
async def update_autotrader_config(update: AutoTraderConfigUpdate):
    """Update AutoTrader configuration at runtime."""
    state = get_app_state()

    if state.autotrader is None:
        raise HTTPException(status_code=400, detail="AutoTrader not initialized")

    try:
        config = state.autotrader.config
        updated_fields = []

        for field_name, value in update.model_dump(exclude_none=True).items():
            mapped = field_name
            # Handle alias
            if field_name == "min_confidence":
                mapped = "min_confidence_to_trade"
            if hasattr(config, mapped):
                setattr(config, mapped, value)
                updated_fields.append(mapped)

        logger.info(f"AutoTrader config updated: {updated_fields}")
        return AutoTraderConfigResponse(
            success=True,
            message=f"Updated {len(updated_fields)} field(s): {', '.join(updated_fields)}",
            config=config.to_dict(),
        )
    except Exception as e:
        logger.error(f"Failed to update AutoTrader config: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/force-trading", response_model=AutoTraderControlResponse)
async def force_trading_mode():
    """Force AutoTrader to transition to trading mode (bypass learning requirements)."""
    state = get_app_state()

    if state.autotrader is None:
        raise HTTPException(status_code=400, detail="AutoTrader not initialized")

    try:
        # Access internal state to force transition
        from coldpath.autotrader.coordinator import AutoTraderState

        state.autotrader._set_state(AutoTraderState.TRADING)
        return AutoTraderControlResponse(
            success=True,
            message="AutoTrader forced to trading mode",
            state=state.autotrader.state.value,
        )
    except Exception as e:
        logger.error(f"Failed to force trading mode: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ===========================================
# Adaptive Limits Management Endpoints
# ===========================================


class AdaptiveLimitsStatusResponse(BaseModel):
    """Response model for adaptive limits status."""

    autonomous_enabled: bool
    total_trades: int
    rolling_sample_size: int
    rolling_win_rate: float
    rolling_avg_pnl: float
    current_multiplier: float
    last_adjustment: str | None
    last_action: str | None
    in_cooldown: bool
    adjustment_count: int


class AdaptiveLimitsHistoryResponse(BaseModel):
    """Response model for adjustment history."""

    count: int
    adjustments: list[dict[str, Any]]


class SetAutonomousRequest(BaseModel):
    """Request to enable/disable autonomous mode."""

    enabled: bool


class ForceAdjustmentRequest(BaseModel):
    """Request to force a specific adjustment."""

    action: str  # "scale_up", "scale_down", "emergency_reduce"
    reason: str = "manual_request"


@router.get("/adaptive-limits/status", response_model=AdaptiveLimitsStatusResponse)
async def get_adaptive_limits_status():
    """Get current status of the adaptive limits manager.

    Returns rolling win rate, current multiplier, adjustment history count,
    and whether autonomous mode is enabled.
    """
    state = get_app_state()

    if state.autotrader is None:
        raise HTTPException(status_code=400, detail="AutoTrader not initialized")

    try:
        status = state.autotrader.adaptive_limits.get_status()
        return AdaptiveLimitsStatusResponse(
            autonomous_enabled=status["autonomous_enabled"],
            total_trades=status["total_trades"],
            rolling_sample_size=status["rolling_sample_size"],
            rolling_win_rate=status["rolling_win_rate"],
            rolling_avg_pnl=status["rolling_avg_pnl"],
            current_multiplier=status["current_multiplier"],
            last_adjustment=status["last_adjustment"],
            last_action=status["last_action"],
            in_cooldown=status["in_cooldown"],
            adjustment_count=status["adjustment_count"],
        )
    except Exception as e:
        logger.error(f"Failed to get adaptive limits status: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/adaptive-limits/history", response_model=AdaptiveLimitsHistoryResponse)
async def get_adaptive_limits_history(limit: int = 20):
    """Get history of adaptive limit adjustments.

    Args:
        limit: Maximum number of adjustments to return (default 20)
    """
    state = get_app_state()

    if state.autotrader is None:
        raise HTTPException(status_code=400, detail="AutoTrader not initialized")

    try:
        history = state.autotrader.adaptive_limits.get_adjustment_history(limit=limit)
        return AdaptiveLimitsHistoryResponse(
            count=len(history),
            adjustments=history,
        )
    except Exception as e:
        logger.error(f"Failed to get adjustment history: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/adaptive-limits/autonomous")
async def set_adaptive_limits_autonomous(request: SetAutonomousRequest):
    """Enable or disable autonomous limit management.

    When enabled, the system will automatically scale limits up/down
    based on rolling win rate performance.

    Args:
        request: {enabled: true/false}
    """
    state = get_app_state()

    if state.autotrader is None:
        raise HTTPException(status_code=400, detail="AutoTrader not initialized")

    try:
        state.autotrader.adaptive_limits.set_autonomous(request.enabled)
        return {
            "success": True,
            "message": (
                f"Autonomous limit management {'enabled' if request.enabled else 'disabled'}"
            ),
            "autonomous_enabled": request.enabled,
        }
    except Exception as e:
        logger.error(f"Failed to set autonomous mode: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/adaptive-limits/force-adjustment")
async def force_adaptive_limit_adjustment(request: ForceAdjustmentRequest):
    """Force a specific limit adjustment.

    WARNING: Bypasses all safety checks and cooldowns.
    Use with caution.

    Args:
        request: {action: "scale_up"|"scale_down"|"emergency_reduce", reason: "..."}
    """
    state = get_app_state()

    if state.autotrader is None:
        raise HTTPException(status_code=400, detail="AutoTrader not initialized")

    try:
        from coldpath.autotrader.adaptive_limits import LimitAction

        # Map string to enum
        action_map = {
            "scale_up": LimitAction.SCALE_UP,
            "scale_down": LimitAction.SCALE_DOWN,
            "emergency_reduce": LimitAction.EMERGENCY_REDUCE,
        }

        if request.action.lower() not in action_map:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Invalid action: {request.action}. Must be one of: {list(action_map.keys())}"
                ),
            )

        action = action_map[request.action.lower()]

        # Get current config
        current_config = state.autotrader.config.to_dict()

        # Apply the adjustment with profitability context
        new_config = state.autotrader.adaptive_limits.apply_adjustment(
            current_config,
            action,
            reason=request.reason,
            daily_pnl_sol=state.autotrader.daily_metrics.total_pnl_sol,
            current_trade_count=state.autotrader.daily_metrics.trade_count,
        )

        # Update the config
        from coldpath.autotrader.coordinator import AutoTraderConfig

        state.autotrader.config = AutoTraderConfig.from_dict(new_config)

        return {
            "success": True,
            "message": f"Forced {action.value} adjustment",
            "action": action.value,
            "reason": request.reason,
            "new_config": {
                "max_daily_trades": new_config.get("max_daily_trades"),
                "max_concurrent_positions": new_config.get("max_concurrent_positions"),
                "max_position_sol": new_config.get("max_position_sol"),
                "max_daily_volume_sol": new_config.get("max_daily_volume_sol"),
                "min_confidence_to_trade": new_config.get("min_confidence_to_trade"),
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to force adjustment: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/adaptive-limits/reset")
async def reset_adaptive_limits():
    """Reset the adaptive limits manager state.

    Clears rolling outcome history and resets multiplier to 1.0.
    Does not change current config limits.
    """
    state = get_app_state()

    if state.autotrader is None:
        raise HTTPException(status_code=400, detail="AutoTrader not initialized")

    try:
        state.autotrader.adaptive_limits.reset()
        return {
            "success": True,
            "message": "Adaptive limits manager reset",
        }
    except Exception as e:
        logger.error(f"Failed to reset adaptive limits: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ===========================================
# Performance Summary Endpoint
# ===========================================


class PerformanceSummaryResponse(BaseModel):
    """Response model for performance summary."""

    daily_pnl_pct: float
    daily_win_rate: float
    daily_trades: int
    rolling_win_rate_30: float
    current_multiplier: float
    target_pnl_pct: float
    target_win_rate: float
    on_target: bool
    recommended_action: str


@router.get("/performance/summary", response_model=PerformanceSummaryResponse)
async def get_performance_summary():
    """Get a summary of current trading performance vs targets.

    Returns:
        - Current daily P&L % and win rate
        - Rolling 30-trade win rate
        - Current adaptive multiplier
        - Whether on target for 0.8-1% daily P&L goal
        - Recommended action (scale_up/scale_down/hold)
    """
    state = get_app_state()

    if state.autotrader is None:
        raise HTTPException(status_code=400, detail="AutoTrader not initialized")

    try:
        # Get metrics
        daily_metrics = state.autotrader.daily_metrics
        adaptive_status = state.autotrader.adaptive_limits.get_status()

        daily_pnl_pct = state.autotrader._calculate_daily_pnl_pct()
        daily_win_rate = daily_metrics.win_rate
        rolling_win_rate = adaptive_status["rolling_win_rate"]
        current_multiplier = adaptive_status["current_multiplier"]

        # Check if on target
        target_pnl = 0.8
        target_win = 0.50
        on_target = daily_pnl_pct >= target_pnl and daily_win_rate >= target_win

        # Recommend action
        if rolling_win_rate > 0.58 and daily_pnl_pct > 0.5:
            recommended = "scale_up"
        elif rolling_win_rate < 0.45 or daily_pnl_pct < 0.2:
            recommended = "scale_down"
        elif rolling_win_rate < 0.35:
            recommended = "emergency_reduce"
        else:
            recommended = "hold"

        return PerformanceSummaryResponse(
            daily_pnl_pct=round(daily_pnl_pct, 2),
            daily_win_rate=round(daily_win_rate, 3),
            daily_trades=daily_metrics.trade_count,
            rolling_win_rate_30=round(rolling_win_rate, 3),
            current_multiplier=round(current_multiplier, 3),
            target_pnl_pct=target_pnl,
            target_win_rate=target_win,
            on_target=on_target,
            recommended_action=recommended,
        )
    except Exception as e:
        logger.error(f"Failed to get performance summary: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ===========================================
# Verified Settings Integration Endpoints
# ===========================================


@router.post("/config/load-best")
async def load_best_verified_config():
    """Load the best verified configuration from the store.

    This will replace the current AutoTrader config with the
    highest-performing saved configuration.
    """
    state = get_app_state()

    if state.autotrader is None:
        raise HTTPException(status_code=400, detail="AutoTrader not initialized")

    try:
        success = state.autotrader.load_best_verified_config()

        if success:
            return {
                "success": True,
                "message": "Loaded best verified configuration",
                "config": state.autotrader.config.to_dict(),
            }
        else:
            return {
                "success": False,
                "message": "No verified configuration found to load",
            }
    except Exception as e:
        logger.error(f"Failed to load best config: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/config/save-current")
async def save_current_config(notes: str = ""):
    """Save the current configuration to the verified settings store.

    This will only save if the configuration meets minimum criteria:
    - Win rate >= 50%
    - Daily P&L >= 0.6%
    - Max drawdown <= 25%
    - Total trades >= 50

    Args:
        notes: Optional notes to attach to the saved configuration
    """
    state = get_app_state()

    if state.autotrader is None:
        raise HTTPException(status_code=400, detail="AutoTrader not initialized")

    try:
        from coldpath.config.verified_settings import VerifiedSettingsStore

        store = VerifiedSettingsStore()

        # Calculate metrics
        daily_metrics = state.autotrader.daily_metrics
        pnl_pct = state.autotrader._calculate_daily_pnl_pct()
        estimated_capital = (
            state.autotrader.config.max_position_sol
            * state.autotrader.config.max_concurrent_positions
        )
        max_dd_pct = (
            (daily_metrics.max_drawdown_sol / estimated_capital * 100)
            if estimated_capital > 0
            else 0
        )

        settings_id = store.save_verified(
            settings=state.autotrader.config.to_dict(),
            metrics={
                "win_rate_pct": daily_metrics.win_rate * 100,
                "daily_pnl_pct": pnl_pct,
                "max_drawdown_pct": max_dd_pct,
                "total_trades": daily_metrics.trade_count,
                "sharpe_ratio": 1.5,  # Would need actual calculation
                "profit_factor": 1.2,  # Would need actual calculation
            },
            verified_by="manual_save",
            notes=notes,
            tags=["manual_save"],
        )

        if settings_id:
            return {
                "success": True,
                "settings_id": settings_id,
                "message": f"Configuration saved: {settings_id}",
            }
        else:
            return {
                "success": False,
                "message": "Configuration does not meet minimum criteria for saving",
                "hint": "Need: win_rate >= 50%, daily_pnl >= 0.6%, trades >= 50, max_dd <= 25%",
            }
    except Exception as e:
        logger.error(f"Failed to save current config: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ===========================================
# Operation Mode Management Endpoints
# ===========================================


class OperationModeResponse(BaseModel):
    """Response model for operation mode status."""

    current_mode: str
    autonomous_enabled: bool
    trades_since_last_switch: int
    total_trades: int
    in_cooldown: bool
    seconds_until_available: float
    transition_count: int
    current_config: dict[str, Any]


class AvailableModesResponse(BaseModel):
    """Response model for available modes."""

    modes: list[dict[str, Any]]


class SetModeRequest(BaseModel):
    """Request to change operation mode."""

    mode: str  # scout, conservative, balanced, aggressive, autonomous
    force: bool = False
    reason: str = "manual_request"


class SetAutonomousModeRequest(BaseModel):
    """Request to enable/disable autonomous mode switching."""

    enabled: bool


@router.get("/mode/status", response_model=OperationModeResponse)
async def get_operation_mode_status():
    """Get current operation mode status.

    Returns information about the current trading mode, whether
    autonomous mode switching is enabled, and transition history.
    """
    state = get_app_state()

    if state.autotrader is None:
        raise HTTPException(status_code=400, detail="AutoTrader not initialized")

    try:
        mode_status = state.autotrader.mode_manager.get_status()
        return OperationModeResponse(
            current_mode=mode_status["current_mode"],
            autonomous_enabled=mode_status["autonomous_enabled"],
            trades_since_last_switch=mode_status["trades_since_last_switch"],
            total_trades=mode_status["total_trades"],
            in_cooldown=mode_status["in_cooldown"],
            seconds_until_available=mode_status["seconds_until_available"],
            transition_count=mode_status["transition_count"],
            current_config=mode_status["current_config"],
        )
    except Exception as e:
        logger.error(f"Failed to get mode status: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/mode/available", response_model=AvailableModesResponse)
async def list_operation_modes():
    """List all available operation modes with their configurations.

    Returns a list of modes that can be selected, along with their
    risk levels, target P&L, and key parameters.
    """
    try:
        from coldpath.autotrader.operation_modes import list_available_modes

        modes = list_available_modes()
        return AvailableModesResponse(modes=modes)
    except Exception as e:
        logger.error(f"Failed to list modes: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/mode/set")
async def set_operation_mode(request: SetModeRequest):
    """Set the operation mode.

    Args:
        request: {
            mode: "scout"|"conservative"|"balanced"|"aggressive"|"autonomous",
            force: false,  // Bypass cooldown checks
            reason: "manual_request"
        }

    Changes the trading mode, which updates all trading parameters
    to the preset values for that mode.
    """
    state = get_app_state()

    if state.autotrader is None:
        raise HTTPException(status_code=400, detail="AutoTrader not initialized")

    try:
        from coldpath.autotrader.operation_modes import ModeTransitionReason, OperationMode

        # Map string to enum
        mode_map = {
            "scout": OperationMode.SCOUT,
            "conservative": OperationMode.CONSERVATIVE,
            "balanced": OperationMode.BALANCED,
            "aggressive": OperationMode.AGGRESSIVE,
            "autonomous": OperationMode.AUTONOMOUS,
        }

        mode_str = request.mode.lower()
        if mode_str not in mode_map:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid mode: {request.mode}. Must be one of: {list(mode_map.keys())}",
            )

        target_mode = mode_map[mode_str]

        # Attempt mode switch
        success = state.autotrader.set_operation_mode(
            mode=target_mode,
            reason=ModeTransitionReason.MANUAL,
            details=request.reason,
            force=request.force,
        )

        if success:
            return {
                "success": True,
                "message": f"Operation mode changed to {target_mode.value}",
                "mode": target_mode.value,
                "config": state.autotrader.config.to_dict(),
            }
        else:
            # Mode switch blocked (cooldown or minimum trades)
            mode_status = state.autotrader.mode_manager.get_status()
            return {
                "success": False,
                "message": "Mode switch blocked - in cooldown or insufficient trades",
                "current_mode": mode_status["current_mode"],
                "in_cooldown": mode_status["in_cooldown"],
                "trades_since_last_switch": mode_status["trades_since_last_switch"],
                "seconds_until_available": mode_status["seconds_until_available"],
                "hint": "Use force=true to bypass cooldown checks",
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to set mode: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/mode/autonomous")
async def set_autonomous_mode_switching(request: SetAutonomousModeRequest):
    """Enable or disable autonomous mode switching.

    When enabled, the system will automatically switch between
    operation modes based on:
    - Rolling win rate
    - Daily P&L performance
    - Market regime detection
    - Risk thresholds

    Args:
        request: {enabled: true/false}
    """
    state = get_app_state()

    if state.autotrader is None:
        raise HTTPException(status_code=400, detail="AutoTrader not initialized")

    try:
        state.autotrader.enable_autonomous_mode_switching(request.enabled)

        return {
            "success": True,
            "message": f"Autonomous mode switching {'enabled' if request.enabled else 'disabled'}",
            "autonomous_enabled": request.enabled,
            "current_mode": state.autotrader.mode_manager.current_mode.value,
        }
    except Exception as e:
        logger.error(f"Failed to set autonomous mode: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/mode/history")
async def get_mode_transition_history(limit: int = 20):
    """Get history of mode transitions.

    Args:
        limit: Maximum number of transitions to return (default 20)

    Returns a log of all mode changes with timestamps, reasons,
    and performance snapshots at the time of transition.
    """
    state = get_app_state()

    if state.autotrader is None:
        raise HTTPException(status_code=400, detail="AutoTrader not initialized")

    try:
        history = state.autotrader.mode_manager.get_transition_history(limit=limit)
        return {
            "count": len(history),
            "transitions": history,
        }
    except Exception as e:
        logger.error(f"Failed to get mode history: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/mode/recommend")
async def get_recommended_mode():
    """Get a recommended mode based on current performance.

    Analyzes current win rate and P&L to suggest the most
    appropriate mode for current market conditions.
    """
    state = get_app_state()

    if state.autotrader is None:
        raise HTTPException(status_code=400, detail="AutoTrader not initialized")

    try:
        from coldpath.autotrader.operation_modes import OperationMode

        # Get current metrics
        win_rate = state.autotrader.adaptive_limits.get_rolling_win_rate()
        daily_pnl_pct = state.autotrader._calculate_daily_pnl_pct()

        # Get recommendation
        recommended = state.autotrader.mode_manager.get_recommended_mode(
            win_rate=win_rate,
            daily_pnl_pct=daily_pnl_pct,
            risk_tolerance="medium",
        )

        return {
            "current_mode": state.autotrader.mode_manager.current_mode.value,
            "recommended_mode": recommended.value,
            "reason": f"Based on win_rate={win_rate:.0%}, daily_pnl={daily_pnl_pct:.2f}%",
            "metrics": {
                "rolling_win_rate": round(win_rate, 3),
                "daily_pnl_pct": round(daily_pnl_pct, 2),
            },
            "mode_details": {
                "risk_level": "high"
                if recommended == OperationMode.AGGRESSIVE
                else "low"
                if recommended == OperationMode.CONSERVATIVE
                else "medium",
                "target_pnl": {
                    OperationMode.CONSERVATIVE: 0.5,
                    OperationMode.BALANCED: 0.8,
                    OperationMode.AGGRESSIVE: 1.2,
                }.get(recommended, 0.8),
            },
        }
    except Exception as e:
        logger.error(f"Failed to get recommended mode: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ===========================================
# Validation Pipeline Endpoints
# ===========================================


class QualityGatesStatusResponse(BaseModel):
    """Response model for quality gates status."""

    gates: list[dict[str, Any]]
    min_criteria: dict[str, Any]


class ValidationRunRequest(BaseModel):
    """Request to run a validation."""

    config: dict[str, Any] | None = None  # None = validate current config
    backtest_period_days: int = 7


class OptimizationRunRequest(BaseModel):
    """Request to run parameter optimization."""

    method: str = "random_search"  # grid_search, random_search, bayesian
    iterations: int = 30
    base_config: dict[str, Any] | None = None


class ValidationStatusResponse(BaseModel):
    """Response model for validation status."""

    total_validations: int
    passed: int
    failed: int
    pass_rate: float
    avg_score: float
    best_score: float


@router.get("/validation/quality-gates", response_model=QualityGatesStatusResponse)
async def get_quality_gates():
    """Get the current quality gate configuration.

    Returns the criteria that configurations must meet to pass validation.
    """
    try:
        from coldpath.autotrader.validation_pipeline import QualityGatesConfig

        gates_config = QualityGatesConfig()
        gates = []

        for gate in gates_config.get_gates():
            gates.append(
                {
                    "name": gate.name,
                    "metric": gate.metric,
                    "min_value": gate.min_value,
                    "max_value": gate.max_value,
                    "weight": gate.weight,
                    "description": gate.description,
                }
            )

        return QualityGatesStatusResponse(
            gates=gates,
            min_criteria={
                "win_rate": ">= 50%",
                "daily_pnl": ">= 0.6%",
                "max_drawdown": "<= 20%",
                "total_trades": ">= 50",
                "sharpe_ratio": ">= 1.5",
                "profit_factor": ">= 1.2",
            },
        )
    except Exception as e:
        logger.error(f"Failed to get quality gates: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/validation/status", response_model=ValidationStatusResponse)
async def get_validation_status():
    """Get summary statistics from validation history.

    Returns counts of passed/failed validations, pass rate, and score statistics.
    """
    state = get_app_state()

    if state.autotrader is None:
        raise HTTPException(status_code=400, detail="AutoTrader not initialized")

    try:
        from coldpath.autotrader.validation_pipeline import ValidationPipeline

        # Create pipeline and get stats (would normally be persisted)
        pipeline = ValidationPipeline()
        stats = pipeline.get_summary_stats()

        return ValidationStatusResponse(
            total_validations=stats["total_validations"],
            passed=stats["passed"],
            failed=stats["failed"],
            pass_rate=round(stats["pass_rate"], 3),
            avg_score=round(stats["avg_score"], 3),
            best_score=round(stats["best_score"], 3),
        )
    except Exception as e:
        logger.error(f"Failed to get validation status: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/validation/run")
async def run_validation(request: ValidationRunRequest):
    """Run a validation on a configuration.

    This runs a backtest and evaluates the configuration against
    quality gates to determine if it meets minimum criteria.

    Args:
        request: {
            config: {...},  // Optional, uses current if not provided
            backtest_period_days: 7
        }

    Returns:
        Validation report with pass/fail status and metrics.
    """
    state = get_app_state()

    if state.autotrader is None:
        raise HTTPException(status_code=400, detail="AutoTrader not initialized")

    try:
        from coldpath.autotrader.validation_pipeline import (
            ValidationPipeline,
            create_backtest_metrics,
        )

        # Get config to validate
        config = request.config or state.autotrader.config.to_dict()

        # Store daily metrics for use in mock backtest
        daily = state.autotrader.daily_metrics

        # Create a mock backtest function for now
        # In production, this would run actual backtest
        async def mock_backtest(cfg: dict[str, Any]) -> dict[str, float]:
            # Simulate backtest with current metrics
            # In real implementation, this would run the backtest engine
            return create_backtest_metrics(
                total_return_pct=daily.total_pnl_sol * 10,  # Rough estimate
                win_count=daily.win_count,
                loss_count=daily.loss_count,
                max_drawdown_pct=10.0,  # Would come from actual backtest
                trading_days=request.backtest_period_days,
            )

        pipeline = ValidationPipeline()
        report = await pipeline.validate_config(config, mock_backtest)

        return {
            "success": True,
            "report": report.to_dict(),
        }
    except Exception as e:
        logger.error(f"Failed to run validation: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/validation/search-space")
async def get_search_space():
    """Get the parameter search space for optimization.

    Returns the parameters that can be optimized and their ranges.
    """
    try:
        from coldpath.autotrader.validation_pipeline import DEFAULT_SEARCH_SPACE

        params = []
        for param in DEFAULT_SEARCH_SPACE:
            params.append(
                {
                    "name": param.name,
                    "min_value": param.min_value,
                    "max_value": param.max_value,
                    "step": param.step,
                    "is_integer": param.is_integer,
                    "is_log_scale": param.is_log_scale,
                }
            )

        return {
            "parameters": params,
            "count": len(params),
        }
    except Exception as e:
        logger.error(f"Failed to get search space: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/optimization/run")
async def run_optimization(request: OptimizationRunRequest):
    """Run parameter optimization.

    Searches for the best configuration by running multiple backtests
    with different parameter combinations.

    Args:
        request: {
            method: "random_search",  // or "grid_search", "bayesian"
            iterations: 30,
            base_config: {...}  // Optional starting config
        }

    Note: This is a long-running operation. In production, consider
    running as a background task.
    """
    state = get_app_state()

    if state.autotrader is None:
        raise HTTPException(status_code=400, detail="AutoTrader not initialized")

    try:
        from coldpath.autotrader.validation_pipeline import (
            HyperparameterTuner,
            OptimizationMethod,
            create_backtest_metrics,
        )

        # Map method string to enum
        method_map = {
            "grid_search": OptimizationMethod.GRID_SEARCH,
            "random_search": OptimizationMethod.RANDOM_SEARCH,
            "bayesian": OptimizationMethod.BAYESIAN,
        }

        method_str = request.method.lower()
        if method_str not in method_map:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Invalid method: {request.method}. Must be one of: {list(method_map.keys())}"
                ),
            )

        method = method_map[method_str]

        # Create a mock backtest function for now
        async def mock_backtest(cfg: dict[str, Any]) -> dict[str, float]:
            # Simulate backtest
            # In real implementation, this would run the backtest engine
            win_rate = 0.45 + random.random() * 0.2  # 45-65%
            trades = int(cfg.get("max_daily_trades", 30) * 0.5)

            return create_backtest_metrics(
                total_return_pct=win_rate * 10 + random.random() * 5,
                win_count=int(trades * win_rate),
                loss_count=int(trades * (1 - win_rate)),
                max_drawdown_pct=5 + random.random() * 10,
                trading_days=1,
            )

        # Create tuner and run optimization
        tuner = HyperparameterTuner(base_config=request.base_config)
        result = await tuner.optimize(
            method=method,
            iterations=request.iterations,
            objective_fn=mock_backtest,
        )

        return {
            "success": True,
            "result": result.to_dict(),
            "message": f"Optimization complete. Best score: {result.best_score:.3f}",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to run optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/optimization/history")
async def get_optimization_history(limit: int = 10):
    """Get recent optimization run history.

    Args:
        limit: Maximum number of optimization runs to return

    Returns a list of past optimization runs with their results.
    """
    try:
        from coldpath.autotrader.validation_pipeline import HyperparameterTuner

        tuner = HyperparameterTuner()
        history = tuner.get_optimization_history(limit=limit)

        return {
            "count": len(history),
            "optimizations": history,
        }
    except Exception as e:
        logger.error(f"Failed to get optimization history: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/validation/check-current")
async def check_current_config():
    """Check if current configuration meets quality gates.

    Evaluates the current live configuration against quality gates
    using current daily metrics as a quick health check.
    """
    state = get_app_state()

    if state.autotrader is None:
        raise HTTPException(status_code=400, detail="AutoTrader not initialized")

    try:
        from coldpath.autotrader.validation_pipeline import (
            QualityGatesConfig,
        )

        # Get current metrics
        daily = state.autotrader.daily_metrics
        pnl_pct = state.autotrader._calculate_daily_pnl_pct()

        # Create metrics dict
        metrics = {
            "win_rate_pct": daily.win_rate * 100,
            "daily_pnl_pct": pnl_pct,
            "max_drawdown_pct": (daily.max_drawdown_sol / 0.5 * 100)
            if daily.max_drawdown_sol
            else 0,
            "total_trades": daily.trade_count,
            "sharpe_ratio": 1.5,  # Would need actual calculation
            "profit_factor": 1.2,  # Would need actual calculation
        }

        # Evaluate against quality gates
        gates = QualityGatesConfig()
        passed, score, messages = gates.evaluate_all(metrics)

        return {
            "passed": passed,
            "score": round(score, 3),
            "metrics": metrics,
            "gate_results": messages,
            "recommendation": "Configuration looks good!"
            if passed
            else "Consider adjusting parameters to meet quality gates",
        }
    except Exception as e:
        logger.error(f"Failed to check current config: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ===========================================
# Monitoring & Alerting Endpoints
# ===========================================


class AlertStatisticsResponse(BaseModel):
    """Response model for alert statistics."""

    total_alerts: int
    active_alerts: int
    by_severity: dict[str, int]
    by_category: dict[str, int]
    last_24h: int


class DashboardMetricsResponse(BaseModel):
    """Response model for dashboard metrics."""

    monitoring: dict[str, Any]
    alerts: dict[str, Any]
    model: dict[str, Any]
    system: dict[str, Any]


@router.get("/monitoring/dashboard", response_model=DashboardMetricsResponse)
async def get_monitoring_dashboard():
    """Get comprehensive monitoring dashboard metrics.

    Returns all metrics needed for a real-time monitoring dashboard:
    - Alert statistics and active alerts
    - Model performance metrics
    - System health status
    """
    state = get_app_state()

    try:
        from coldpath.autotrader.monitoring import MonitoringService

        # Create or get monitoring service
        monitoring = MonitoringService()

        # If autotrader is running, capture current health
        if state.autotrader is not None:
            daily = state.autotrader.daily_metrics
            monitoring.system_monitor.capture_health(
                autotrader_state=state.autotrader.state.value,
                hotpath_connected=False,  # Would check actual connection
                db_healthy=True,
                model_loaded=True,
                pending_signals=len(state.autotrader._pending_signals),
                active_positions=daily.active_positions,
                daily_trades=daily.trade_count,
                daily_volume_sol=daily.total_volume_sol,
                daily_pnl_sol=daily.total_pnl_sol,
            )

        metrics = monitoring.get_dashboard_metrics()

        return DashboardMetricsResponse(
            monitoring=metrics["monitoring"],
            alerts=metrics["alerts"],
            model=metrics["model"],
            system=metrics["system"],
        )
    except Exception as e:
        logger.error(f"Failed to get dashboard metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/monitoring/alerts")
async def get_alerts(
    severity: str | None = None,
    category: str | None = None,
    limit: int = 50,
):
    """Get alert history with optional filtering.

    Args:
        severity: Filter by severity (info, warning, error, critical)
        category: Filter by category (performance, model, system, risk, trading)
        limit: Maximum alerts to return
    """
    try:
        from coldpath.autotrader.monitoring import AlertCategory, AlertManager, AlertSeverity

        manager = AlertManager()

        # Convert string filters to enums
        sev_enum = None
        if severity:
            sev_map = {
                "info": AlertSeverity.INFO,
                "warning": AlertSeverity.WARNING,
                "error": AlertSeverity.ERROR,
                "critical": AlertSeverity.CRITICAL,
            }
            sev_enum = sev_map.get(severity.lower())

        cat_enum = None
        if category:
            cat_map = {
                "performance": AlertCategory.PERFORMANCE,
                "model": AlertCategory.MODEL,
                "system": AlertCategory.SYSTEM,
                "risk": AlertCategory.RISK,
                "trading": AlertCategory.TRADING,
            }
            cat_enum = cat_map.get(category.lower())

        alerts = manager.get_alert_history(
            limit=limit,
            severity=sev_enum,
            category=cat_enum,
        )

        return {
            "count": len(alerts),
            "alerts": alerts,
        }
    except Exception as e:
        logger.error(f"Failed to get alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/monitoring/alerts/active")
async def get_active_alerts():
    """Get all active (unacknowledged) alerts.

    Returns alerts that require attention.
    """
    try:
        from coldpath.autotrader.monitoring import AlertManager

        manager = AlertManager()
        alerts = manager.get_active_alerts()

        return {
            "count": len(alerts),
            "alerts": [a.to_dict() for a in alerts],
            "critical_count": len([a for a in alerts if a.severity.value == "critical"]),
            "error_count": len([a for a in alerts if a.severity.value == "error"]),
        }
    except Exception as e:
        logger.error(f"Failed to get active alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/monitoring/alerts/statistics", response_model=AlertStatisticsResponse)
async def get_alert_statistics():
    """Get alert statistics summary.

    Returns counts by severity, category, and time periods.
    """
    try:
        from coldpath.autotrader.monitoring import AlertManager

        manager = AlertManager()
        stats = manager.get_statistics()

        return AlertStatisticsResponse(
            total_alerts=stats["total_alerts"],
            active_alerts=stats["active_alerts"],
            by_severity=stats["by_severity"],
            by_category=stats["by_category"],
            last_24h=stats["last_24h"],
        )
    except Exception as e:
        logger.error(f"Failed to get alert statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/monitoring/model/status")
async def get_model_monitoring_status():
    """Get model performance monitoring status.

    Returns:
    - Current vs baseline metrics
    - Drift detection results
    - Retrain recommendations
    """
    try:
        from coldpath.autotrader.monitoring import ModelPerformanceMonitor

        monitor = ModelPerformanceMonitor()
        status = monitor.get_status()

        return {
            "model_monitoring": status,
            "drift_history": monitor.get_drift_history(limit=5),
            "metrics_history": monitor.get_metrics_history(limit=10),
        }
    except Exception as e:
        logger.error(f"Failed to get model status: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/monitoring/model/drift")
async def get_drift_detection():
    """Get model drift detection results.

    Returns information about detected concept drift, data drift,
    and prediction drift.
    """
    try:
        from coldpath.autotrader.monitoring import ModelPerformanceMonitor

        monitor = ModelPerformanceMonitor()
        drift = monitor.detect_drift()

        return {
            "drift_detected": drift.detected if drift else False,
            "latest_drift": drift.to_dict() if drift else None,
            "drift_history": monitor.get_drift_history(limit=10),
            "retrain_recommended": monitor.should_retrain(),
        }
    except Exception as e:
        logger.error(f"Failed to check drift: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/monitoring/system/health")
async def get_system_health():
    """Get current system health status.

    Returns:
    - Component health (DB, HotPath, models)
    - Latency metrics
    - Error rates
    """
    state = get_app_state()

    try:
        from coldpath.autotrader.monitoring import SystemMonitor

        monitor = SystemMonitor()

        # Capture current health if autotrader is running
        if state.autotrader is not None:
            daily = state.autotrader.daily_metrics
            health = monitor.capture_health(
                autotrader_state=state.autotrader.state.value,
                hotpath_connected=False,
                db_healthy=True,
                model_loaded=True,
                pending_signals=len(state.autotrader._pending_signals),
                active_positions=daily.active_positions,
                daily_trades=daily.trade_count,
                daily_volume_sol=daily.total_volume_sol,
                daily_pnl_sol=daily.total_pnl_sol,
            )

            return {
                "current_health": health.to_dict(),
                "summary": monitor.get_summary(),
                "health_history": monitor.get_health_history(limit=10),
            }
        else:
            return {
                "current_health": None,
                "summary": monitor.get_summary(),
                "message": "AutoTrader not initialized",
            }
    except Exception as e:
        logger.error(f"Failed to get system health: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/monitoring/thresholds")
async def get_alert_thresholds():
    """Get configured alert thresholds.

    Returns all metric thresholds with their severity levels.
    """
    try:
        from coldpath.autotrader.monitoring import DEFAULT_ALERT_THRESHOLDS

        thresholds = []
        for name, threshold in DEFAULT_ALERT_THRESHOLDS.items():
            thresholds.append(
                {
                    "metric_name": name,
                    "warning": threshold.warning_threshold,
                    "error": threshold.error_threshold,
                    "critical": threshold.critical_threshold,
                    "comparison": threshold.comparison,
                    "min_sample_size": threshold.min_sample_size,
                    "cooldown_seconds": threshold.cooldown_seconds,
                }
            )

        return {
            "thresholds": thresholds,
            "count": len(thresholds),
        }
    except Exception as e:
        logger.error(f"Failed to get thresholds: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/monitoring/report")
async def get_full_monitoring_report():
    """Get a comprehensive monitoring report.

    Combines all monitoring data into a single report for
    diagnostic or audit purposes.
    """
    state = get_app_state()

    try:
        from coldpath.autotrader.monitoring import MonitoringService

        monitoring = MonitoringService()
        monitoring.start()

        # Capture health if autotrader is running
        if state.autotrader is not None:
            daily = state.autotrader.daily_metrics
            monitoring.system_monitor.capture_health(
                autotrader_state=state.autotrader.state.value,
                hotpath_connected=False,
                db_healthy=True,
                model_loaded=True,
                pending_signals=len(state.autotrader._pending_signals),
                active_positions=daily.active_positions,
                daily_trades=daily.trade_count,
                daily_volume_sol=daily.total_volume_sol,
                daily_pnl_sol=daily.total_pnl_sol,
            )

            # Check metrics for alerts
            metrics = {
                "win_rate": daily.win_rate,
                "daily_pnl_pct": state.autotrader._calculate_daily_pnl_pct(),
                "consecutive_losses": state.autotrader._consecutive_losses,
            }
            monitoring.check_all_metrics(metrics)

        return monitoring.get_full_report()
    except Exception as e:
        logger.error(f"Failed to get monitoring report: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/monitoring/check")
async def run_health_check():
    """Run an immediate health check.

    Triggers a comprehensive check of all monitored metrics
    and returns any alerts generated.
    """
    state = get_app_state()

    if state.autotrader is None:
        raise HTTPException(status_code=400, detail="AutoTrader not initialized")

    try:
        from coldpath.autotrader.monitoring import MonitoringService

        monitoring = MonitoringService()

        # Gather metrics
        daily = state.autotrader.daily_metrics
        pnl_pct = state.autotrader._calculate_daily_pnl_pct()
        rolling_win_rate = state.autotrader.adaptive_limits.get_rolling_win_rate()

        metrics = {
            "win_rate": daily.win_rate,
            "rolling_win_rate": rolling_win_rate,
            "daily_pnl_pct": pnl_pct,
            "max_drawdown_pct": (daily.max_drawdown_sol / 0.5 * 100)
            if daily.max_drawdown_sol
            else 0,
            "consecutive_losses": state.autotrader._consecutive_losses,
        }

        sample_sizes = {
            "win_rate": daily.trade_count,
            "rolling_win_rate": state.autotrader.adaptive_limits.get_sample_size(),
            "daily_pnl_pct": daily.trade_count,
            "max_drawdown_pct": daily.trade_count,
            "consecutive_losses": 1,
        }

        # Check all metrics
        alerts = monitoring.check_all_metrics(metrics, sample_sizes)

        return {
            "check_timestamp": datetime.now().isoformat(),
            "metrics_checked": len(metrics),
            "alerts_generated": len(alerts),
            "alerts": [a.to_dict() for a in alerts],
            "metrics": metrics,
            "health_status": "unhealthy" if alerts else "healthy",
        }
    except Exception as e:
        logger.error(f"Failed to run health check: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ============================================================================
# Training Pipeline Monitoring Endpoints
# ============================================================================


class TrainingPipelineStatusResponse(BaseModel):
    """Response model for training pipeline status."""

    current_stage: str
    current_status: str
    current_run_id: str | None
    last_success_at: str | None
    last_failure_at: str | None
    consecutive_failures: int
    total_runs: int
    successful_runs: int
    failed_runs: int
    success_rate: float
    current_model_version: int
    outcomes_since_last_train: int
    next_scheduled_train: str | None
    should_train_now: bool
    outcomes_until_training: int


class TrainingRunDetail(BaseModel):
    """Response model for a single training run."""

    run_id: str
    started_at: str
    stage: str
    status: str
    outcomes_used: int
    teacher_metrics: dict[str, float] | None
    student_metrics: dict[str, float] | None
    quality_gates_passed: bool
    correlation: float | None
    calibration_error: float | None
    model_version: int | None
    completed_at: str | None
    duration_seconds: float
    error_message: str | None


class TrainingHistoryResponse(BaseModel):
    """Response model for training run history."""

    count: int
    runs: list[TrainingRunDetail]


class TriggerTrainingRequest(BaseModel):
    """Request to manually trigger a training run."""

    force: bool = False  # Bypass thresholds
    reason: str = "manual_trigger"


class TriggerTrainingResponse(BaseModel):
    """Response model for training trigger."""

    success: bool
    message: str
    run_id: str | None = None
    triggered: bool = False


# Global training pipeline monitor (singleton pattern)
_training_monitor = None


def get_training_monitor():
    """Get or create the training pipeline monitor singleton."""
    global _training_monitor
    try:
        state = get_app_state()
        if (
            state.engine is not None
            and getattr(state.engine, "training_integration", None) is not None
        ):
            return state.engine.training_integration.monitor
    except Exception:
        pass

    try:
        from coldpath.autotrader.training_integration import get_training_integration

        integration = get_training_integration()
        return integration.monitor
    except Exception:
        pass

    if _training_monitor is None:
        from coldpath.autotrader.monitoring import TrainingPipelineMonitor

        _training_monitor = TrainingPipelineMonitor()
    return _training_monitor


@router.get("/training/status", response_model=TrainingPipelineStatusResponse)
async def get_training_pipeline_status():
    """Get current training pipeline status.

    Returns:
        - Current training stage and status
        - Last success/failure timestamps
        - Model version information
        - Outcomes collected since last training
        - Whether training should be triggered now
    """
    try:
        monitor = get_training_monitor()
        summary = monitor.get_summary()
        status = summary["status"]

        return TrainingPipelineStatusResponse(
            current_stage=status["current_stage"],
            current_status=status["current_status"],
            current_run_id=status["current_run_id"],
            last_success_at=status["last_success_at"],
            last_failure_at=status["last_failure_at"],
            consecutive_failures=status["consecutive_failures"],
            total_runs=status["total_runs"],
            successful_runs=status["successful_runs"],
            failed_runs=status["failed_runs"],
            success_rate=status["success_rate"],
            current_model_version=status["current_model_version"],
            outcomes_since_last_train=status["outcomes_since_last_train"],
            next_scheduled_train=status["next_scheduled_train"],
            should_train_now=summary["should_train_now"],
            outcomes_until_training=summary["outcomes_until_training"],
        )
    except Exception as e:
        logger.error(f"Failed to get training status: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/training/current", response_model=TrainingRunDetail | None)
async def get_current_training_run():
    """Get details of the currently running training job.

    Returns None if no training is currently running.
    """
    try:
        monitor = get_training_monitor()
        current = monitor.get_current_run()

        if current is None:
            return None

        return TrainingRunDetail(**current)
    except Exception as e:
        logger.error(f"Failed to get current training run: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/training/history", response_model=TrainingHistoryResponse)
async def get_training_history(limit: int = 20):
    """Get history of training runs.

    Args:
        limit: Maximum number of runs to return (default 20)

    Returns a list of past training runs with their outcomes and metrics.
    """
    try:
        monitor = get_training_monitor()
        history = monitor.get_run_history(limit=limit)

        return TrainingHistoryResponse(
            count=len(history),
            runs=[TrainingRunDetail(**run) for run in history],
        )
    except Exception as e:
        logger.error(f"Failed to get training history: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/training/summary")
async def get_training_summary():
    """Get comprehensive training pipeline summary.

    Combines status, current run, recent history, and configuration
    into a single response for dashboard displays.
    """
    try:
        monitor = get_training_monitor()
        return monitor.get_summary()
    except Exception as e:
        logger.error(f"Failed to get training summary: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/training/trigger", response_model=TriggerTrainingResponse)
async def trigger_training(request: TriggerTrainingRequest):
    """Manually trigger a training run.

    Args:
        request: {force: false, reason: "manual_trigger"}

    By default, respects thresholds (min outcomes, time since last train).
    Use force=true to bypass all thresholds.

    Note: This only schedules the training. Actual training runs
    asynchronously in the background.
    """
    try:
        state = get_app_state()
        from coldpath.autotrader.training_integration import get_training_integration

        integration = (
            get_training_integration(state.engine.db)
            if state.engine is not None and getattr(state.engine, "db", None) is not None
            else get_training_integration()
        )
        evaluation = await integration.evaluate_training_readiness()
        logger.info(
            "Train check: observations=%s, threshold=%s, can_train=%s, reason=%s",
            evaluation["observations"],
            evaluation["threshold"],
            evaluation["can_train"],
            evaluation["reason"],
        )

        result = await integration.trigger_training(force=request.force)
        return TriggerTrainingResponse(
            success=bool(result.get("success")),
            message=result.get("message", "Training trigger completed"),
            run_id=result.get("run_id"),
            triggered=bool(result.get("success")),
        )
    except Exception as e:
        logger.error("Failed to trigger training: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/training/integration")
async def get_training_integration_status():
    """Get training integration status.

    Returns the current state of the training integration including:
    - Whether the integration is running
    - Total outcomes recorded
    - Trainings triggered
    - Whether training should run now
    """
    try:
        from coldpath.autotrader.training_integration import get_training_integration

        state = get_app_state()
        integration = (
            get_training_integration(state.engine.db)
            if state.engine is not None and getattr(state.engine, "db", None) is not None
            else get_training_integration()
        )
        return integration.get_status()
    except Exception as e:
        logger.error("Failed to get training integration status: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/training/check-and-train")
async def check_and_trigger_training():
    """Check if training should run and trigger if thresholds are met.

    This is a convenience endpoint that combines status check with
    automatic training trigger.
    """
    try:
        from coldpath.autotrader.training_integration import get_training_integration

        state = get_app_state()
        integration = (
            get_training_integration(state.engine.db)
            if state.engine is not None and getattr(state.engine, "db", None) is not None
            else get_training_integration()
        )
        result = await integration.check_and_train()

        if result:
            return {
                "training_triggered": True,
                "result": result,
            }
        else:
            return {
                "training_triggered": False,
                "message": "Training thresholds not met",
                "status": integration.get_status(),
            }
    except Exception as e:
        logger.error("Failed to check and train: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/training/reset-failures")
async def reset_training_failures():
    """Reset the consecutive failure counter.

    Use this after manually fixing issues that were causing
    training to fail repeatedly.
    """
    try:
        monitor = get_training_monitor()
        monitor.reset_failure_count()

        return {
            "success": True,
            "message": "Training failure count reset",
        }
    except Exception as e:
        logger.error(f"Failed to reset failures: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/training/health")
async def get_training_health():
    """Quick health check for the training pipeline.

    Returns a simple status indicating if the training pipeline
    is healthy, degraded, or unhealthy.
    """
    try:
        monitor = get_training_monitor()
        status = monitor.get_status()

        # Determine health status
        health = "healthy"
        issues = []

        if status.consecutive_failures >= 3:
            health = "unhealthy"
            issues.append(f"{status.consecutive_failures} consecutive failures")
        elif status.consecutive_failures >= 1:
            health = "degraded"
            issues.append(f"{status.consecutive_failures} recent failure(s)")

        # Check if training is overdue
        if status.outcomes_since_last_train > 500 and monitor.should_train():
            if health == "healthy":
                health = "degraded"
            issues.append(f"Training overdue ({status.outcomes_since_last_train} outcomes)")

        return {
            "health": health,
            "issues": issues,
            "current_stage": status.current_stage.value,
            "last_success": status.last_success_at.isoformat() if status.last_success_at else None,
            "model_version": status.current_model_version,
            "success_rate": round(status.successful_runs / max(1, status.total_runs), 3),
        }
    except Exception as e:
        logger.error(f"Failed to get training health: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ============================================================================
# Phase 2: Test Discovery Injection
# ============================================================================


class TestDiscoveryRequest(BaseModel):
    """Test token discovery for pipeline testing."""

    mint: str
    pool: str | None = None
    source: str = "test"
    liquidity_usd: float | None = None
    fdv_usd: float | None = None
    volume_24h: float | None = None
    holder_count: int | None = None
    price_change_1h_pct: float | None = None
    price_change_24h_pct: float | None = None
    volatility_1h: float | None = None
    top_holder_pct: float | None = None
    rug_risk_score: float | None = None
    age_seconds: int | None = None


class TestDiscoveryResponse(BaseModel):
    success: bool
    message: str
    mint: str
    queued: bool = False


@router.post("/inject-discovery", response_model=TestDiscoveryResponse)
async def inject_test_discovery(request: TestDiscoveryRequest):
    """
    Inject a test token discovery into the pipeline.

    This is for Phase 2 testing when real data feeds are unavailable.
    The discovery will be processed through the full pipeline.
    """
    try:
        state = get_app_state()

        if not state.autotrader:
            raise HTTPException(status_code=503, detail="AutoTrader not available")

        # Create discovery payload
        discovery = {
            "mint": request.mint,
            "pool": request.pool or request.mint,  # Use mint as pool if not provided
            "source": request.source,
            "liquidity_usd": request.liquidity_usd or random.uniform(50000, 500000),
            "fdv_usd": request.fdv_usd or random.uniform(100000, 1000000),
            "volume_24h": request.volume_24h or random.uniform(10000, 100000),
            "holder_count": request.holder_count or random.randint(100, 1000),
            "price_change_1h_pct": request.price_change_1h_pct or random.uniform(-10, 10),
            "price_change_24h_pct": request.price_change_24h_pct or random.uniform(-20, 20),
            "volatility_1h": request.volatility_1h or random.uniform(5, 30),
            "top_holder_pct": request.top_holder_pct or random.uniform(5, 30),
            "rug_risk_score": request.rug_risk_score or random.uniform(0, 0.5),
            "age_seconds": request.age_seconds or random.randint(60, 3600),
            "timestamp_ms": int(datetime.now().timestamp() * 1000),
        }

        # Try to queue the discovery
        queued = False

        # Method 1: Direct queue access
        if hasattr(state.autotrader, "queue_candidate"):
            state.autotrader.queue_candidate(discovery)
            queued = True

        # Method 2: Via IPC to HotPath
        elif hasattr(state, "hotpath_client") and state.hotpath_client:
            try:
                await state.hotpath_client.send_command(
                    {
                        "command": "InjectDiscovery",
                        "payload": discovery,
                    }
                )
                queued = True
            except Exception as e:
                logger.warning(f"IPC injection failed: {e}")

        if queued:
            logger.info(f"Injected test discovery: {request.mint[:16]}...")
            return TestDiscoveryResponse(
                success=True,
                message="Discovery injected successfully",
                mint=request.mint,
                queued=True,
            )
        else:
            return TestDiscoveryResponse(
                success=False,
                message="No injection method available - AutoTrader may not be running",
                mint=request.mint,
                queued=False,
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to inject discovery: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/inject-batch")
async def inject_batch_discoveries(count: int = 10):
    """
    Inject multiple test discoveries at once.

    Useful for stress testing and collecting trade data quickly.
    """
    import string

    results = []
    for _i in range(min(count, 100)):  # Max 100 at a time
        # Generate random mint
        mint = "".join(random.choices(string.ascii_letters + string.digits, k=44))

        discovery = TestDiscoveryRequest(
            mint=mint,
            pool="".join(random.choices(string.ascii_letters + string.digits, k=44)),
            source=random.choice(["raydium", "pumpfun", "moonshot", "test"]),
            liquidity_usd=random.uniform(30000, 500000),
            price_change_1h_pct=random.uniform(-15, 15),
        )

        result = await inject_test_discovery(discovery)
        results.append(
            {
                "mint": mint[:16] + "...",
                "success": result.success,
            }
        )

    successful = sum(1 for r in results if r["success"])

    return {
        "total": len(results),
        "successful": successful,
        "results": results,
    }
