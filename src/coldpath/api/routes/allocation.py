"""
Automated Allocation Agent API Routes.

Exposes the Automated Allocation Agent via REST API.
"""

import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from coldpath.ai.automated_allocation_agent import (
    AllocationMode,
    get_allocation_agent,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/allocation", tags=["allocation"])


# =============================================================================
# Request/Response Models
# =============================================================================


class AllocationConfigRequest(BaseModel):
    """Request to update allocation config."""

    mode: str | None = Field(None, description="Operating mode: advisor/semi_auto/full_auto/shadow")
    max_positions: int | None = Field(None, description="Maximum positions")
    max_allocation_per_token_pct: float | None = Field(None, description="Max % per token")
    max_total_exposure_pct: float | None = Field(None, description="Max total exposure %")
    min_ml_score_to_open: float | None = Field(None, description="Min ML score to open")
    min_ml_confidence_to_open: float | None = Field(None, description="Min confidence to open")
    allocation_cycle_seconds: float | None = Field(None, description="Cycle interval")


class OpportunityScanRequest(BaseModel):
    """Request to scan opportunities."""

    min_ml_score: float = Field(0.5, description="Minimum ML score")
    min_liquidity_usd: float = Field(50000, description="Minimum liquidity")
    max_fraud_score: float = Field(0.35, description="Maximum fraud score")
    limit: int = Field(20, description="Max opportunities to return")


class PositionInfo(BaseModel):
    """Position information."""

    symbol: str
    mint: str
    amount: float
    entry_price: float
    current_price: float
    pnl_pct: float
    allocation_pct: float


class PortfolioUpdateRequest(BaseModel):
    """Request to update portfolio state."""

    total_value_usd: float
    cash_usd: float
    positions: list[PositionInfo]


class RunCycleRequest(BaseModel):
    """Request to run an allocation cycle."""

    force: bool = Field(False, description="Force cycle even if recently run")


class ApprovePlanRequest(BaseModel):
    """Request to approve a plan."""

    plan_id: str
    approved: bool
    approver: str | None = None


class MarketDataRequest(BaseModel):
    """Market data for assessment."""

    sol_price: float | None = None
    btc_price: float | None = None
    eth_price: float | None = None
    fear_greed_index: int | None = None
    additional_data: dict[str, Any] | None = None


# =============================================================================
# Routes
# =============================================================================


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "automated-allocation-agent"}


@router.get("/status")
async def get_status():
    """Get allocation agent status."""
    agent = get_allocation_agent()
    stats = agent.get_stats()
    assessment = agent.get_market_assessment()
    last_plan = agent.get_last_plan()

    return {
        "stats": stats,
        "market_assessment": assessment,
        "last_plan_summary": {
            "plan_id": last_plan.get("plan_id"),
            "total_trades": last_plan.get("total_trades"),
            "risk_score": last_plan.get("risk_score"),
        }
        if last_plan
        else None,
    }


@router.get("/config")
async def get_config():
    """Get current allocation configuration."""
    agent = get_allocation_agent()
    return agent.config.to_dict()


@router.post("/config")
async def update_config(request: AllocationConfigRequest):
    """Update allocation configuration."""
    agent = get_allocation_agent()

    # Update fields
    if request.mode:
        mode_map = {
            "advisor": AllocationMode.ADVISOR,
            "semi_auto": AllocationMode.SEMI_AUTO,
            "full_auto": AllocationMode.FULL_AUTO,
            "shadow": AllocationMode.SHADOW,
        }
        agent.config.mode = mode_map.get(request.mode, agent.config.mode)

    if request.max_positions:
        agent.config.max_positions = request.max_positions
    if request.max_allocation_per_token_pct:
        agent.config.max_allocation_per_token_pct = request.max_allocation_per_token_pct
    if request.max_total_exposure_pct:
        agent.config.max_total_exposure_pct = request.max_total_exposure_pct
    if request.min_ml_score_to_open:
        agent.config.min_ml_score_to_open = request.min_ml_score_to_open
    if request.min_ml_confidence_to_open:
        agent.config.min_ml_confidence_to_open = request.min_ml_confidence_to_open
    if request.allocation_cycle_seconds:
        agent.config.allocation_cycle_seconds = request.allocation_cycle_seconds

    return {"status": "updated", "config": agent.config.to_dict()}


@router.post("/start")
async def start_agent(background_tasks: BackgroundTasks):
    """Start the allocation agent loop."""
    agent = get_allocation_agent()

    if agent._running:
        return {"status": "already_running"}

    background_tasks.add_task(agent.start)
    return {"status": "started", "mode": agent.config.mode.value}


@router.post("/stop")
async def stop_agent():
    """Stop the allocation agent loop."""
    agent = get_allocation_agent()
    await agent.stop()
    return {"status": "stopped"}


@router.post("/cycle")
async def run_cycle(request: RunCycleRequest):
    """Run a single allocation cycle."""
    agent = get_allocation_agent()

    # Check if recently run
    if not request.force and agent._last_cycle_time:
        elapsed = (datetime.utcnow() - agent._last_cycle_time).total_seconds()
        if elapsed < agent.config.allocation_cycle_seconds / 2:
            return {
                "status": "skipped",
                "reason": f"Cycle run {elapsed:.0f}s ago",
                "last_plan": agent.get_last_plan(),
            }

    plan = await agent.run_allocation_cycle()
    return plan.to_dict()


@router.post("/assess-market")
async def assess_market(request: MarketDataRequest):
    """Assess current market conditions."""
    agent = get_allocation_agent()

    market_data = {}
    if request.sol_price:
        market_data["sol_price"] = request.sol_price
    if request.btc_price:
        market_data["btc_price"] = request.btc_price
    if request.eth_price:
        market_data["eth_price"] = request.eth_price
    if request.fear_greed_index:
        market_data["fear_greed_index"] = request.fear_greed_index
    if request.additional_data:
        market_data.update(request.additional_data)

    assessment = await agent.assess_market(market_data if market_data else None)
    return assessment.to_dict()


@router.get("/plan/current")
async def get_current_plan():
    """Get the current allocation plan."""
    agent = get_allocation_agent()
    plan = agent.get_last_plan()

    if not plan:
        raise HTTPException(status_code=404, detail="No allocation plan available")

    return plan


@router.get("/plan/history")
async def get_plan_history(limit: int = 10):
    """Get allocation plan history."""
    agent = get_allocation_agent()

    history = [p.to_dict() for p in agent._allocation_history[-limit:]]
    return {
        "total": len(agent._allocation_history),
        "plans": history,
    }


@router.post("/approve")
async def approve_plan(request: ApprovePlanRequest):
    """Approve or reject a plan."""
    agent = get_allocation_agent()

    # Find plan
    plan = None
    for p in agent._allocation_history:
        if p.plan_id == request.plan_id:
            plan = p
            break

    if not plan:
        raise HTTPException(status_code=404, detail="Plan not found")

    plan.approved = request.approved
    plan.approved_by = request.approver or "api"
    plan.approved_at = datetime.utcnow()

    # Execute if approved
    if request.approved:
        for decision in plan.decisions:
            await agent._execute_decision(decision)
            agent._total_executed += 1
            agent._total_decisions += 1

    return {
        "status": "approved" if request.approved else "rejected",
        "plan_id": request.plan_id,
        "executed": request.approved,
    }


@router.get("/decisions/pending")
async def get_pending_decisions():
    """Get pending allocation decisions."""
    agent = get_allocation_agent()
    plan = agent.get_last_plan()

    if not plan:
        return {"decisions": []}

    pending = [
        d
        for d in plan.get("decisions", [])
        if d.get("requires_approval") and not plan.get("approved")
    ]

    return {
        "plan_id": plan.get("plan_id"),
        "pending_count": len(pending),
        "decisions": pending,
    }


@router.get("/stats")
async def get_stats():
    """Get allocation agent statistics."""
    agent = get_allocation_agent()
    return agent.get_stats()


# =============================================================================
# Register with Main App
# =============================================================================


def register_allocation_routes(app):
    """Register allocation routes with the FastAPI app."""
    app.include_router(router)
