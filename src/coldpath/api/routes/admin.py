"""
Admin API routes for server configuration and tuning.

Endpoints:
- GET  /admin/config          - Get current server configuration
- PUT  /admin/config          - Update server configuration
- POST /admin/config/reset    - Reset configuration to defaults
- GET  /admin/tuner/status    - Get tuner status and statistics
- PUT  /admin/tuner/mode      - Set tuning mode
- GET  /admin/metrics         - Get current metrics summary
- GET  /admin/parameters/{name} - Get parameter details
- PUT  /admin/parameters/{name} - Update a specific parameter
- POST /admin/parameters/{name}/rollback - Rollback parameter
- GET  /admin/rules           - List all tuning rules
- PUT  /admin/rules/{name}    - Enable/disable a rule
- POST /admin/health/check    - Trigger health check
"""

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


# Request/Response Models


class ConfigResponse(BaseModel):
    """Current server configuration."""

    parameters: dict[str, float]
    mode: str
    last_tune: float
    stats: dict[str, Any]


class ConfigUpdateRequest(BaseModel):
    """Request to update configuration."""

    parameters: dict[str, float] = Field(..., description="Parameter names and values to update")
    reason: str = Field(default="manual update", description="Reason for the update")


class TunerModeRequest(BaseModel):
    """Request to change tuner mode."""

    mode: str = Field(..., description="Tuning mode: disabled, conservative, moderate, aggressive")


class ParameterUpdateRequest(BaseModel):
    """Request to update a single parameter."""

    value: float = Field(..., description="New parameter value")
    reason: str = Field(default="manual", description="Reason for change")


class RuleUpdateRequest(BaseModel):
    """Request to update a tuning rule."""

    enabled: bool = Field(..., description="Whether the rule is enabled")


class ParameterInfo(BaseModel):
    """Detailed parameter information."""

    name: str
    current_value: float
    default_value: float
    min_value: float
    max_value: float
    step_size: float
    cooldown_seconds: float
    can_adjust: bool
    adjustment_count: int
    recent_adjustments: list[Any]


class RuleInfo(BaseModel):
    """Tuning rule information."""

    name: str
    metric_type: str
    parameter_name: str
    threshold_low: float
    threshold_high: float
    adjustment_low: float
    adjustment_high: float
    priority: int
    enabled: bool


class MetricsResponse(BaseModel):
    """Metrics summary response."""

    metrics: dict[str, Any]
    collection_window_seconds: float


class HealthCheckResponse(BaseModel):
    """Health check response."""

    status: str
    components: dict[str, bool]
    tuner: dict[str, Any]
    recommendations: list[str]


# Dependency injection


def get_tuner() -> Any:
    """Get server tuner instance."""
    from coldpath.ai.server_tuner import get_server_tuner

    return get_server_tuner()


def get_app_state() -> Any:
    """Get application state."""
    from coldpath.api.server import get_app_state

    return get_app_state()


# Endpoints


TunerDep = Annotated[Any, Depends(get_tuner)]
AppStateDep = Annotated[Any, Depends(get_app_state)]


@router.get("/config", response_model=ConfigResponse)
async def get_config(tuner: TunerDep) -> ConfigResponse:
    """Get current server configuration.

    Returns all tunable parameters with their current values,
    tuning mode, and statistics.
    """
    return ConfigResponse(
        parameters=tuner.get_config_dict(),
        mode=tuner.mode.value,
        last_tune=tuner._last_tune,
        stats=tuner.get_stats(),
    )


@router.put("/config")
async def update_config(
    request: ConfigUpdateRequest,
    tuner: TunerDep,
) -> dict[str, Any]:
    """Update server configuration.

    Allows updating multiple parameters at once.
    Parameters are validated against their min/max bounds.
    """
    updated = []
    errors = []

    for name, value in request.parameters.items():
        try:
            tuner.set_parameter(name, value, request.reason)
            updated.append(name)
        except ValueError as e:
            errors.append(f"{name}: {str(e)}")
        except Exception as e:
            errors.append(f"{name}: unexpected error - {str(e)}")

    return {
        "updated": updated,
        "errors": errors,
        "current_config": tuner.get_config_dict(),
    }


@router.post("/config/reset")
async def reset_config(tuner: TunerDep) -> dict[str, Any]:
    """Reset all configuration to defaults.

    Resets all tunable parameters to their default values.
    """
    tuner.reset_all()
    return {
        "status": "reset",
        "current_config": tuner.get_config_dict(),
    }


@router.get("/tuner/status")
async def get_tuner_status(tuner: TunerDep) -> dict[str, Any]:
    """Get detailed tuner status.

    Returns tuner statistics, mode, and recent activity.
    """
    return {
        "running": tuner._running,
        "mode": tuner.mode.value,
        "tune_interval_seconds": tuner._tune_interval,
        "last_tune": tuner._last_tune,
        "stats": tuner.get_stats(),
        "metrics_summary": tuner.get_metrics_summary(),
    }


@router.put("/tuner/mode")
async def set_tuner_mode(
    request: TunerModeRequest,
    tuner: TunerDep,
) -> dict[str, Any]:
    """Set the tuning mode.

    Modes:
    - disabled: No automatic tuning
    - conservative: Small, infrequent adjustments (5 min interval)
    - moderate: Balanced tuning (1 min interval)
    - aggressive: Fast adaptation (15s interval)
    """
    from coldpath.ai.server_tuner import TuningMode

    try:
        mode = TuningMode(request.mode)
        tuner.set_mode(mode)
        return {
            "mode": mode.value,
            "tune_interval_seconds": tuner._tune_interval,
        }
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid mode: {request.mode}. "
                "Valid modes: disabled, conservative, moderate, aggressive"
            ),
        ) from None


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics(tuner: TunerDep) -> MetricsResponse:
    """Get current metrics summary.

    Returns aggregated metrics from the sliding window.
    """
    return MetricsResponse(
        metrics=tuner.get_metrics_summary(),
        collection_window_seconds=tuner._tune_interval * 2,
    )


@router.get("/parameters/{name}", response_model=ParameterInfo)
async def get_parameter(name: str, tuner: TunerDep) -> ParameterInfo:
    """Get detailed information about a parameter.

    Returns current value, bounds, adjustment history, and more.
    """
    info = tuner.get_parameter_info(name)
    if not info:
        raise HTTPException(status_code=404, detail=f"Parameter not found: {name}")

    return ParameterInfo(**info)


@router.put("/parameters/{name}")
async def update_parameter(
    name: str,
    request: ParameterUpdateRequest,
    tuner: TunerDep,
) -> dict[str, Any]:
    """Update a specific parameter.

    Value is validated against parameter bounds.
    """
    try:
        tuner.set_parameter(name, request.value, request.reason)
        return {
            "status": "updated",
            "parameter": tuner.get_parameter_info(name),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/parameters/{name}/rollback")
async def rollback_parameter(name: str, tuner: TunerDep) -> dict[str, Any]:
    """Rollback a parameter to its previous value.

    Reverts the last adjustment made to this parameter.
    """
    if tuner.rollback_parameter(name):
        return {
            "status": "rolled back",
            "parameter": tuner.get_parameter_info(name),
        }
    else:
        raise HTTPException(
            status_code=400, detail=f"Cannot rollback {name}: no adjustment history"
        )


@router.get("/rules")
async def list_rules(tuner: TunerDep) -> dict[str, Any]:
    """List all tuning rules.

    Returns all configured tuning rules with their current state.
    """
    rules = []
    for rule in tuner._rules:
        rules.append(
            RuleInfo(
                name=rule.name,
                metric_type=rule.metric_type.value,
                parameter_name=rule.parameter_name,
                threshold_low=rule.threshold_low,
                threshold_high=rule.threshold_high,
                adjustment_low=rule.adjustment_low,
                adjustment_high=rule.adjustment_high,
                priority=rule.priority,
                enabled=rule.enabled,
            )
        )

    return {
        "rules": [r.model_dump() for r in rules],
        "count": len(rules),
        "enabled_count": sum(1 for r in rules if r.enabled),
    }


@router.put("/rules/{name}")
async def update_rule(
    name: str,
    request: RuleUpdateRequest,
    tuner: TunerDep,
) -> dict[str, Any]:
    """Enable or disable a tuning rule.

    Find rule by name and set its enabled state.
    """
    for rule in tuner._rules:
        if rule.name == name:
            rule.enabled = request.enabled
            return {
                "status": "updated",
                "rule": name,
                "enabled": rule.enabled,
            }

    raise HTTPException(status_code=404, detail=f"Rule not found: {name}")


@router.post("/health/check", response_model=HealthCheckResponse)
async def health_check(
    tuner: TunerDep,
    app_state: AppStateDep,
) -> HealthCheckResponse:
    """Run comprehensive health check.

    Checks all components and provides recommendations
    based on current metrics and configuration.
    """
    recommendations = []

    # Check components
    components = {
        "claude_client": app_state.claude_client is not None,
        "claude_router": app_state.claude_router is not None,
        "bandit_trainer": app_state.bandit_trainer is not None,
        "metrics_engine": app_state.metrics_engine is not None,
        "server_tuner": tuner is not None,
    }

    # Check metrics and generate recommendations
    metrics = tuner.get_metrics_summary()

    # Latency check
    latency_metrics = metrics.get("latency_p95", {})
    if latency_metrics.get("p95", 0) > 1000:
        recommendations.append(
            "High P95 latency detected. Consider increasing latency budgets or "
            "switching to more aggressive caching."
        )

    # Error rate check
    error_metrics = metrics.get("error_rate", {})
    if error_metrics.get("mean", 0) > 0.05:
        recommendations.append(
            "Error rate above 5%. Consider increasing timeouts and retries, "
            "or check API connectivity."
        )

    # Cache efficiency check
    cache_metrics = metrics.get("cache_hit_rate", {})
    if cache_metrics.get("mean", 0) < 0.2:
        recommendations.append(
            "Low cache hit rate. Consider extending cache TTLs or improving cache key strategies."
        )

    # Opus usage check
    opus_metrics = metrics.get("opus_usage", {})
    if opus_metrics.get("mean", 0) > 0.5:
        recommendations.append(
            "High Opus model usage (>50%). Consider raising complexity threshold to reduce costs."
        )

    # Fallback check
    fallback_metrics = metrics.get("fallback_rate", {})
    if fallback_metrics.get("mean", 0) > 0.1:
        recommendations.append(
            "High fallback rate (>10%). Consider increasing analysis budgets "
            "or checking for API issues."
        )

    # Tuner check
    if tuner.mode.value == "disabled":
        recommendations.append(
            "Auto-tuning is disabled. Enable moderate mode for automatic optimization."
        )

    # Determine overall status
    all_healthy = all(components.values())
    status = "healthy" if all_healthy and len(recommendations) < 2 else "degraded"
    if not any(components.values()):
        status = "unhealthy"

    return HealthCheckResponse(
        status=status,
        components=components,
        tuner=tuner.get_stats(),
        recommendations=recommendations,
    )


@router.post("/save")
async def save_config(tuner: TunerDep) -> dict[str, str]:
    """Save current configuration to disk.

    Persists configuration for recovery after restart.
    """
    await tuner.save_config()
    return {"status": "saved"}


@router.post("/load")
async def load_config(tuner: TunerDep) -> dict[str, Any]:
    """Load configuration from disk.

    Restores previously saved configuration.
    """
    await tuner.load_config()
    return {
        "status": "loaded",
        "config": tuner.get_config_dict(),
    }


@router.post("/tuner/force-tune")
async def force_tune(tuner: TunerDep):
    """Force an immediate tuning cycle.

    Runs the tuning logic immediately instead of waiting
    for the next scheduled cycle.
    """
    await tuner._run_tuning_cycle()
    return {
        "status": "tuning cycle completed",
        "adjustments": tuner._stats["total_adjustments"],
        "config": tuner.get_config_dict(),
    }
