"""
Settings API Routes - Manage verified configurations.

Endpoints:
- POST /settings/verified/save - Save a verified configuration
- GET /settings/verified/best - Get the best performing configuration
- GET /settings/verified/active - Get the currently active configuration
- POST /settings/verified/{id}/activate - Activate a configuration
- GET /settings/verified/list - List all verified configurations
- DELETE /settings/verified/{id} - Delete a configuration
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from coldpath.config.verified_settings import VerifiedSettings, VerifiedSettingsStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/settings", tags=["settings"])


# ===========================================
# Request/Response Models
# ===========================================


class SaveSettingsRequest(BaseModel):
    """Request to save a verified configuration."""

    settings: dict[str, Any]
    metrics: dict[str, float] = Field(
        ..., description="Performance metrics from backtest or live trading"
    )
    verified_by: str = Field(
        default="backtest", description="Source of verification: backtest, live, paper"
    )
    backtest_period: str | None = Field(
        default=None, description="Time period of backtest if applicable"
    )
    notes: str = Field(default="", description="Optional notes")
    tags: list[str] = Field(default_factory=list, description="Optional tags")


class SettingsSummary(BaseModel):
    """Summary of a verified configuration."""

    settings_id: str
    win_rate_pct: float
    daily_pnl_pct: float
    max_drawdown_pct: float
    total_trades: int
    sharpe_ratio: float
    profit_factor: float
    is_active: bool
    verified_at: str
    verified_by: str
    tags: list[str] = []


class SettingsDetail(BaseModel):
    """Detailed view of a verified configuration."""

    settings_id: str
    settings: dict[str, Any]
    win_rate_pct: float
    daily_pnl_pct: float
    max_drawdown_pct: float
    total_trades: int
    sharpe_ratio: float
    profit_factor: float
    avg_win_pct: float
    avg_loss_pct: float
    fill_rate_pct: float
    is_active: bool
    verified_at: str
    verified_by: str
    backtest_period: str | None
    deployed_at: str | None
    deployment_count: int
    notes: str
    tags: list[str]


class SettingsListResponse(BaseModel):
    """Response for listing configurations."""

    count: int
    settings: list[SettingsSummary]


class ActivateResponse(BaseModel):
    """Response for activating a configuration."""

    success: bool
    active_id: str
    message: str


class StatisticsResponse(BaseModel):
    """Response for settings statistics."""

    total_count: int
    active_id: str | None
    best_daily_pnl_pct: float | None
    avg_daily_pnl_pct: float | None
    min_criteria: dict[str, Any]


# ===========================================
# Helper Functions
# ===========================================


def _get_store() -> VerifiedSettingsStore:
    """Get the settings store instance."""
    return VerifiedSettingsStore()


def _settings_to_summary(s: VerifiedSettings) -> SettingsSummary:
    """Convert VerifiedSettings to summary model."""
    return SettingsSummary(
        settings_id=s.settings_id,
        win_rate_pct=s.win_rate_pct,
        daily_pnl_pct=s.daily_pnl_pct,
        max_drawdown_pct=s.max_drawdown_pct,
        total_trades=s.total_trades,
        sharpe_ratio=s.sharpe_ratio,
        profit_factor=s.profit_factor,
        is_active=s.is_active,
        verified_at=s.verified_at,
        verified_by=s.verified_by,
        tags=s.tags,
    )


def _settings_to_detail(s: VerifiedSettings) -> SettingsDetail:
    """Convert VerifiedSettings to detail model."""
    return SettingsDetail(
        settings_id=s.settings_id,
        settings=s.settings,
        win_rate_pct=s.win_rate_pct,
        daily_pnl_pct=s.daily_pnl_pct,
        max_drawdown_pct=s.max_drawdown_pct,
        total_trades=s.total_trades,
        sharpe_ratio=s.sharpe_ratio,
        profit_factor=s.profit_factor,
        avg_win_pct=s.avg_win_pct,
        avg_loss_pct=s.avg_loss_pct,
        fill_rate_pct=s.fill_rate_pct,
        is_active=s.is_active,
        verified_at=s.verified_at,
        verified_by=s.verified_by,
        backtest_period=s.backtest_period,
        deployed_at=s.deployed_at,
        deployment_count=s.deployment_count,
        notes=s.notes,
        tags=s.tags,
    )


# ===========================================
# API Endpoints
# ===========================================


@router.post("/verified/save")
async def save_verified_settings(request: SaveSettingsRequest):
    """Save a verified configuration that meets performance criteria.

    The configuration will only be saved if it meets minimum criteria:
    - Win rate >= 50%
    - Daily P&L >= 0.6%
    - Max drawdown <= 25%
    - Total trades >= 50
    - Sharpe ratio >= 1.5

    Returns the settings_id if saved, or error if criteria not met.
    """
    store = _get_store()

    # Calculate daily_pnl_pct if not provided
    metrics = request.metrics.copy()
    if "daily_pnl_pct" not in metrics and "total_return_pct" in metrics:
        # Estimate daily P&L from total return
        # This is a rough approximation - ideally caller provides exact value
        total_return = metrics["total_return_pct"]
        # Assume backtest period affects this
        metrics["daily_pnl_pct"] = total_return / 10  # Rough daily estimate

    settings_id = store.save_verified(
        settings=request.settings,
        metrics=metrics,
        verified_by=request.verified_by,
        backtest_period=request.backtest_period,
        notes=request.notes,
        tags=request.tags,
    )

    if settings_id is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "Settings do not meet minimum criteria: "
                "win_rate >= 50%, daily_pnl >= 0.6%, "
                "max_drawdown <= 25%, trades >= 50, sharpe >= 1.5"
            ),
        )

    return {
        "success": True,
        "settings_id": settings_id,
        "message": f"Configuration saved successfully: {settings_id}",
    }


@router.get("/verified/best", response_model=SettingsDetail)
async def get_best_settings():
    """Get the best performing verified configuration.

    Returns the configuration with the highest daily P&L percentage
    that meets all minimum criteria.
    """
    store = _get_store()
    best = store.get_best()

    if best is None:
        raise HTTPException(
            status_code=404, detail="No verified settings found. Run a backtest first."
        )

    return _settings_to_detail(best)


@router.get("/verified/active", response_model=SettingsDetail)
async def get_active_settings():
    """Get the currently active configuration for live trading.

    Returns the configuration that has been activated for production use.
    """
    store = _get_store()
    active = store.get_active()

    if active is None:
        raise HTTPException(
            status_code=404, detail="No active settings configured. Activate a configuration first."
        )

    return _settings_to_detail(active)


@router.get("/verified/{settings_id}", response_model=SettingsDetail)
async def get_settings_by_id(settings_id: str):
    """Get a specific configuration by ID."""
    store = _get_store()
    settings = store.get_by_id(settings_id)

    if settings is None:
        raise HTTPException(status_code=404, detail=f"Settings not found: {settings_id}")

    return _settings_to_detail(settings)


@router.post("/verified/{settings_id}/activate", response_model=ActivateResponse)
async def activate_settings(settings_id: str):
    """Set a verified configuration as active for live trading.

    This will:
    1. Deactivate any currently active configuration
    2. Mark the specified configuration as active
    3. Update deployment timestamp and count
    """
    store = _get_store()
    success = store.set_active(settings_id)

    if not success:
        raise HTTPException(status_code=404, detail=f"Settings not found: {settings_id}")

    return ActivateResponse(
        success=True,
        active_id=settings_id,
        message=f"Configuration {settings_id} activated for live trading",
    )


@router.post("/verified/deactivate")
async def deactivate_settings():
    """Deactivate the currently active configuration.

    Returns to using default or no configuration for live trading.
    """
    store = _get_store()
    store.deactivate()

    return {
        "success": True,
        "message": "Active configuration deactivated",
    }


@router.get("/verified/list", response_model=SettingsListResponse)
async def list_verified_settings(
    limit: int = 20,
    tag: str | None = None,
    min_pnl: float | None = None,
):
    """List all verified configurations.

    Args:
        limit: Maximum number of configurations to return (default 20)
        tag: Filter by tag
        min_pnl: Filter by minimum daily P&L percentage
    """
    store = _get_store()
    settings = store.list_all(limit=limit, tag=tag, min_pnl=min_pnl)

    return SettingsListResponse(
        count=len(settings),
        settings=[_settings_to_summary(s) for s in settings],
    )


@router.delete("/verified/{settings_id}")
async def delete_settings(settings_id: str):
    """Delete a configuration.

    Note: Cannot delete the currently active configuration.
    """
    store = _get_store()
    success = store.delete(settings_id)

    if not success:
        raise HTTPException(
            status_code=400,
            detail=(f"Cannot delete configuration: {settings_id}. It may be active or not found."),
        )

    return {
        "success": True,
        "message": f"Configuration {settings_id} deleted",
    }


@router.get("/verified/statistics", response_model=StatisticsResponse)
async def get_settings_statistics():
    """Get statistics about stored configurations."""
    store = _get_store()
    stats = store.get_statistics()

    return StatisticsResponse(**stats)


@router.post("/verified/from-backtest")
async def save_from_backtest(
    backtest_result: dict[str, Any],
    settings: dict[str, Any],
    notes: str = "",
):
    """Save settings directly from a backtest result.

    Convenience endpoint that extracts metrics from a standard
    backtest result and saves the settings.

    Args:
        backtest_result: Full backtest result dictionary
        settings: The configuration parameters used
        notes: Optional notes about this configuration
    """
    # Extract metrics from backtest result
    metrics = backtest_result.get("metrics", {})

    # Also check summary for total_return_pct
    summary = backtest_result.get("summary", {})

    # Build metrics dict
    full_metrics = {
        "win_rate_pct": metrics.get("win_rate_pct", 0),
        "total_trades": metrics.get("total_trades", 0),
        "sharpe_ratio": metrics.get("sharpe_ratio", 0),
        "max_drawdown_pct": metrics.get("max_drawdown_pct", 100),
        "profit_factor": metrics.get("profit_factor", 0),
        "total_return_pct": summary.get("total_return_pct", metrics.get("total_return_pct", 0)),
        "avg_win_pct": metrics.get("avg_win_pct", 0),
        "avg_loss_pct": metrics.get("avg_loss_pct", 0),
        "fill_rate_pct": metrics.get("fill_rate_pct", 100),
    }

    # Calculate daily P&L if not present
    if "daily_pnl_pct" not in full_metrics:
        # Estimate from total return and trade count
        total_return = full_metrics.get("total_return_pct", 0)
        trades = full_metrics.get("total_trades", 1)
        # Rough estimate: assume backtest covers multiple days
        estimated_days = max(1, trades / 20)  # Assume ~20 trades/day
        full_metrics["daily_pnl_pct"] = total_return / estimated_days

    store = _get_store()
    settings_id = store.save_verified(
        settings=settings,
        metrics=full_metrics,
        verified_by="backtest",
        notes=notes,
    )

    if settings_id is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "Settings do not meet minimum criteria for saving. "
                f"Metrics: win_rate={full_metrics.get('win_rate_pct', 0):.1f}%, "
                f"trades={full_metrics.get('total_trades', 0)}, "
                f"sharpe={full_metrics.get('sharpe_ratio', 0):.2f}, "
                f"dd={full_metrics.get('max_drawdown_pct', 100):.1f}%"
            ),
        )

    return {
        "success": True,
        "settings_id": settings_id,
        "metrics_saved": full_metrics,
    }
