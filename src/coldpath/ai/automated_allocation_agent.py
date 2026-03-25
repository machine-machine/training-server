"""
Automated Allocation Agent - FinGPT-powered autonomous portfolio allocation.

This agent provides fully automated portfolio allocation using local LLM analysis:
- Continuous market regime monitoring
- Dynamic position sizing based on ML confidence
- Risk-adjusted allocation recommendations
- Automated rebalancing suggestions
- Portfolio optimization with FinGPT reasoning

Architecture:
┌─────────────────────────────────────────────────────────────────────┐
│                  Automated Allocation Agent                          │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐  │
│  │  FinGPT     │  │  Market     │  │  Portfolio  │  │  Risk     │  │
│  │  (Ollama)   │  │  Data       │  │  State      │  │  Manager  │  │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └─────┬─────┘  │
│         │                │                │                │        │
│         └────────────────┴────────────────┴────────────────┘        │
│                              │                                       │
│  ┌───────────────────────────┴───────────────────────────────────┐  │
│  │                    Allocation Loop (1-5 min)                  │  │
│  ├───────────────────────────────────────────────────────────────┤  │
│  │ 1. assess_market()      → Market regime + sentiment           │  │
│  │ 2. scan_opportunities() → Find high-conviction tokens         │  │
│  │ 3. compute_allocations() → Optimal position sizes             │  │
│  │ 4. validate_risk()      → Ensure risk limits                  │  │
│  │ 5. execute_rebalance()  → Apply allocation changes            │  │
│  │ 6. learn_outcomes()     → Continuous improvement              │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable
from uuid import uuid4

from .fingpt_client import FinGPTClient, get_fingpt_client

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Data Classes
# =============================================================================


class AllocationMode(Enum):
    """Operating mode for allocation agent."""

    ADVISOR = "advisor"  # Suggest only, human approval required
    SEMI_AUTO = "semi_auto"  # Auto-execute small changes, approval for large
    FULL_AUTO = "full_auto"  # Fully autonomous allocation
    SHADOW = "shadow"  # Log recommendations, no execution


class MarketCondition(Enum):
    """Overall market condition assessment."""

    VERY_BULLISH = "very_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    VERY_BEARISH = "very_bearish"
    HIGH_VOLATILITY = "high_volatility"
    LOW_LIQUIDITY = "low_liquidity"


class AllocationAction(Enum):
    """Types of allocation actions."""

    OPEN_POSITION = "open_position"
    CLOSE_POSITION = "close_position"
    INCREASE_POSITION = "increase_position"
    DECREASE_POSITION = "decrease_position"
    REBALANCE = "rebalance"
    HEDGE = "hedge"
    CASH_OUT = "cash_out"


@dataclass
class TokenOpportunity:
    """A token identified as a trading opportunity."""

    symbol: str
    mint: str
    current_price: float
    ml_score: float  # 0-1
    ml_confidence: float  # 0-1
    fraud_score: float  # 0-1
    liquidity_usd: float
    volume_24h: float
    market_cap: float
    price_change_24h: float
    regime_compatibility: float  # 0-1 how well it fits current regime
    finpt_analysis: str | None = None
    recommended_allocation_pct: float = 0.0
    priority: int = 0  # Higher = more important

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "mint": self.mint,
            "current_price": self.current_price,
            "ml_score": self.ml_score,
            "ml_confidence": self.ml_confidence,
            "fraud_score": self.fraud_score,
            "liquidity_usd": self.liquidity_usd,
            "volume_24h": self.volume_24h,
            "market_cap": self.market_cap,
            "price_change_24h": self.price_change_24h,
            "regime_compatibility": self.regime_compatibility,
            "recommended_allocation_pct": self.recommended_allocation_pct,
            "priority": self.priority,
        }


@dataclass
class Position:
    """Current portfolio position."""

    symbol: str
    mint: str
    amount: float
    entry_price: float
    current_price: float
    pnl_pct: float
    pnl_usd: float
    allocation_pct: float  # % of portfolio
    hold_time_hours: float
    ml_score_at_entry: float
    current_ml_score: float
    should_close: bool = False
    close_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "mint": self.mint,
            "amount": self.amount,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "pnl_pct": self.pnl_pct,
            "pnl_usd": self.pnl_usd,
            "allocation_pct": self.allocation_pct,
            "hold_time_hours": self.hold_time_hours,
            "ml_score_at_entry": self.ml_score_at_entry,
            "current_ml_score": self.current_ml_score,
            "should_close": self.should_close,
            "close_reason": self.close_reason,
        }


@dataclass
class AllocationDecision:
    """A single allocation decision."""

    action: AllocationAction
    symbol: str
    mint: str
    current_allocation_pct: float
    target_allocation_pct: float
    amount_to_trade: float  # Positive = buy, negative = sell
    estimated_value_usd: float
    reasoning: str
    confidence: float
    urgency: str  # "immediate", "soon", "scheduled"
    requires_approval: bool
    risk_level: str
    finpt_rationale: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "symbol": self.symbol,
            "mint": self.mint,
            "current_allocation_pct": self.current_allocation_pct,
            "target_allocation_pct": self.target_allocation_pct,
            "amount_to_trade": self.amount_to_trade,
            "estimated_value_usd": self.estimated_value_usd,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "urgency": self.urgency,
            "requires_approval": self.requires_approval,
            "risk_level": self.risk_level,
            "finpt_rationale": self.finpt_rationale,
        }


@dataclass
class AllocationPlan:
    """Complete allocation plan for a cycle."""

    plan_id: str
    timestamp: datetime
    market_condition: MarketCondition
    regime: str
    regime_confidence: float
    portfolio_value_usd: float
    cash_available_usd: float
    decisions: list[AllocationDecision]
    total_trades: int
    estimated_impact_usd: float
    risk_score: float  # 0-100
    execution_priority: list[str]  # Order of mint addresses to execute
    finpt_summary: str
    approved: bool = False
    approved_by: str | None = None
    approved_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "timestamp": self.timestamp.isoformat(),
            "market_condition": self.market_condition.value,
            "regime": self.regime,
            "regime_confidence": self.regime_confidence,
            "portfolio_value_usd": self.portfolio_value_usd,
            "cash_available_usd": self.cash_available_usd,
            "decisions": [d.to_dict() for d in self.decisions],
            "total_trades": self.total_trades,
            "estimated_impact_usd": self.estimated_impact_usd,
            "risk_score": self.risk_score,
            "execution_priority": self.execution_priority,
            "finpt_summary": self.finpt_summary,
            "approved": self.approved,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
        }


@dataclass
class AllocationConfig:
    """Configuration for allocation agent."""

    # Operating mode
    mode: AllocationMode = AllocationMode.ADVISOR

    # Position limits
    max_positions: int = 10
    max_allocation_per_token_pct: float = 15.0  # Max 15% in single token
    max_total_exposure_pct: float = 80.0  # Keep 20% cash
    min_allocation_pct: float = 1.0  # Minimum position size

    # Risk limits
    max_risk_score: int = 70  # Max risk score to allow (0-100)
    max_daily_turnover_pct: float = 50.0  # Max portfolio turnover per day
    max_single_trade_pct: float = 5.0  # Max single trade as % of portfolio

    # Auto-approval thresholds (for SEMI_AUTO mode)
    auto_approve_max_value_usd: float = 100.0
    auto_approve_max_risk: int = 30
    requires_approval_above_pct: float = 3.0  # Changes > 3% need approval

    # ML thresholds
    min_ml_score_to_open: float = 0.65
    min_ml_confidence_to_open: float = 0.70
    max_fraud_score_to_trade: float = 0.35
    ml_score_to_close: float = 0.35  # Close if score drops below

    # Liquidity requirements
    min_liquidity_usd: float = 50_000
    min_volume_24h_usd: float = 10_000

    # Timing
    allocation_cycle_seconds: float = 300.0  # 5 minutes
    cooldown_after_trade_seconds: float = 60.0

    # FinGPT settings
    fingpt_temperature: float = 0.3
    enable_finpt_reasoning: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "max_positions": self.max_positions,
            "max_allocation_per_token_pct": self.max_allocation_per_token_pct,
            "max_total_exposure_pct": self.max_total_exposure_pct,
            "min_allocation_pct": self.min_allocation_pct,
            "max_risk_score": self.max_risk_score,
            "max_daily_turnover_pct": self.max_daily_turnover_pct,
            "max_single_trade_pct": self.max_single_trade_pct,
            "min_ml_score_to_open": self.min_ml_score_to_open,
            "min_ml_confidence_to_open": self.min_ml_confidence_to_open,
            "max_fraud_score_to_trade": self.max_fraud_score_to_trade,
            "allocation_cycle_seconds": self.allocation_cycle_seconds,
        }


@dataclass
class PortfolioState:
    """Current portfolio state."""

    total_value_usd: float
    cash_usd: float
    positions: list[Position]
    daily_pnl_usd: float
    daily_pnl_pct: float
    total_trades_today: int
    turnover_today_pct: float
    last_updated: datetime

    @property
    def exposure_pct(self) -> float:
        """Current exposure as % of portfolio."""
        if self.total_value_usd == 0:
            return 0.0
        return ((self.total_value_usd - self.cash_usd) / self.total_value_usd) * 100

    @property
    def position_count(self) -> int:
        return len(self.positions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_value_usd": self.total_value_usd,
            "cash_usd": self.cash_usd,
            "positions": [p.to_dict() for p in self.positions],
            "daily_pnl_usd": self.daily_pnl_usd,
            "daily_pnl_pct": self.daily_pnl_pct,
            "total_trades_today": self.total_trades_today,
            "turnover_today_pct": self.turnover_today_pct,
            "exposure_pct": self.exposure_pct,
            "position_count": self.position_count,
            "last_updated": self.last_updated.isoformat(),
        }


@dataclass
class MarketAssessment:
    """FinGPT's assessment of current market conditions."""

    condition: MarketCondition
    regime: str
    regime_confidence: float
    sentiment_score: float  # -1 to 1
    volatility_level: str  # "low", "medium", "high"
    risk_on_off: str  # "risk_on", "risk_off", "neutral"
    key_factors: list[str]
    trading_recommendation: str
    suggested_exposure_pct: float
    finpt_analysis: str
    timestamp: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition.value,
            "regime": self.regime,
            "regime_confidence": self.regime_confidence,
            "sentiment_score": self.sentiment_score,
            "volatility_level": self.volatility_level,
            "risk_on_off": self.risk_on_off,
            "key_factors": self.key_factors,
            "trading_recommendation": self.trading_recommendation,
            "suggested_exposure_pct": self.suggested_exposure_pct,
            "finpt_analysis": self.finpt_analysis,
            "timestamp": self.timestamp.isoformat(),
        }


# =============================================================================
# Automated Allocation Agent
# =============================================================================


class AutomatedAllocationAgent:
    """
    Autonomous portfolio allocation agent powered by FinGPT.

    This agent runs in a continuous loop:
    1. Assess market conditions using FinGPT
    2. Scan for trading opportunities
    3. Compute optimal allocations
    4. Validate against risk limits
    5. Execute or propose rebalancing
    6. Learn from outcomes

    Example:
        config = AllocationConfig(mode=AllocationMode.ADVISOR)
        agent = AutomatedAllocationAgent(config)

        # Start the allocation loop
        await agent.start()

        # Or run a single cycle
        plan = await agent.run_allocation_cycle()
    """

    SYSTEM_PROMPT = """You are an expert crypto portfolio allocation AI with deep expertise in:
- Solana DeFi ecosystem (Raydium, Jupiter, Orca, Meteora)
- Dynamic position sizing and portfolio optimization
- Risk management and capital preservation
- Market regime detection and adaptation
- Kelly criterion and other sizing methods
- Liquidity-aware trading strategies

Your role is to:
1. Assess current market conditions and regime
2. Identify high-conviction trading opportunities
3. Recommend optimal position allocations
4. Manage portfolio risk and exposure
5. Adapt to changing market conditions

Always provide:
- Clear reasoning for each allocation decision
- Risk assessment and mitigation strategies
- Confidence levels for recommendations
- Specific position sizing with rationale

You must operate within the risk limits provided and never exceed position size limits.

Respond in JSON format when structured output is requested."""

    def __init__(
        self,
        config: AllocationConfig | None = None,
        fingpt_client: FinGPTClient | None = None,
        portfolio_provider: Callable[[], PortfolioState] | None = None,
        opportunity_scanner: Callable[[], list[TokenOpportunity]] | None = None,
        execution_handler: Callable[[AllocationDecision], bool] | None = None,
        approval_handler: Callable[[AllocationPlan], bool] | None = None,
    ):
        self.config = config or AllocationConfig()
        self.client = fingpt_client or get_fingpt_client()

        # External providers (inject dependencies)
        self._portfolio_provider = portfolio_provider
        self._opportunity_scanner = opportunity_scanner
        self._execution_handler = execution_handler
        self._approval_handler = approval_handler

        # Internal state
        self._running = False
        self._task: asyncio.Task | None = None
        self._last_cycle_time: datetime | None = None
        self._last_plan: AllocationPlan | None = None
        self._market_assessment: MarketAssessment | None = None
        self._daily_trades: list[datetime] = []
        self._allocation_history: list[AllocationPlan] = []

        # Statistics
        self._total_cycles = 0
        self._total_decisions = 0
        self._total_executed = 0
        self._total_rejected = 0
        self._total_latency_ms = 0.0

    # =========================================================================
    # Main Allocation Loop
    # =========================================================================

    async def start(self):
        """Start the continuous allocation loop."""
        if self._running:
            logger.warning("Allocation agent already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._allocation_loop())
        logger.info(f"Started allocation agent in {self.config.mode.value} mode")

    async def stop(self):
        """Stop the allocation loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped allocation agent")

    async def _allocation_loop(self):
        """Main allocation loop."""
        while self._running:
            try:
                await self.run_allocation_cycle()
            except Exception as e:
                logger.error(f"Allocation cycle error: {e}", exc_info=True)

            await asyncio.sleep(self.config.allocation_cycle_seconds)

    async def run_allocation_cycle(self) -> AllocationPlan:
        """
        Run a complete allocation cycle.

        Steps:
        1. Assess market conditions
        2. Get current portfolio state
        3. Scan for opportunities
        4. Compute allocations
        5. Validate risk
        6. Create plan
        7. Execute or request approval
        """
        cycle_start = time.time()
        self._total_cycles += 1
        self._last_cycle_time = datetime.utcnow()

        logger.info("=" * 60)
        logger.info(f"ALLOCATION CYCLE #{self._total_cycles}")
        logger.info("=" * 60)

        # 1. Assess market
        logger.info("Step 1: Assessing market conditions...")
        self._market_assessment = await self.assess_market()
        logger.info(f"  Market: {self._market_assessment.condition.value}")
        logger.info(f"  Regime: {self._market_assessment.regime}")
        logger.info(f"  Suggested Exposure: {self._market_assessment.suggested_exposure_pct:.1f}%")

        # 2. Get portfolio state
        logger.info("Step 2: Getting portfolio state...")
        portfolio = await self._get_portfolio_state()
        logger.info(f"  Portfolio Value: ${portfolio.total_value_usd:,.2f}")
        logger.info(f"  Cash: ${portfolio.cash_usd:,.2f}")
        logger.info(f"  Exposure: {portfolio.exposure_pct:.1f}%")
        logger.info(f"  Positions: {portfolio.position_count}")

        # 3. Scan opportunities
        logger.info("Step 3: Scanning for opportunities...")
        opportunities = await self._scan_opportunities()
        logger.info(f"  Found {len(opportunities)} opportunities")

        # 4. Compute allocations
        logger.info("Step 4: Computing optimal allocations...")
        decisions = await self.compute_allocations(
            portfolio, opportunities, self._market_assessment
        )
        logger.info(f"  Generated {len(decisions)} allocation decisions")

        # 5. Validate risk
        logger.info("Step 5: Validating risk constraints...")
        decisions = self._validate_risk_limits(decisions, portfolio)
        logger.info(f"  {len(decisions)} decisions passed validation")

        # 6. Create plan
        plan = AllocationPlan(
            plan_id=str(uuid4())[:8],
            timestamp=datetime.utcnow(),
            market_condition=self._market_assessment.condition,
            regime=self._market_assessment.regime,
            regime_confidence=self._market_assessment.regime_confidence,
            portfolio_value_usd=portfolio.total_value_usd,
            cash_available_usd=portfolio.cash_usd,
            decisions=decisions,
            total_trades=len(decisions),
            estimated_impact_usd=sum(d.estimated_value_usd for d in decisions),
            risk_score=self._compute_plan_risk(decisions, portfolio),
            execution_priority=[d.mint for d in sorted(decisions, key=lambda x: -x.confidence)],
            finpt_summary="",
        )

        # 7. Generate FinGPT summary
        if self.config.enable_finpt_reasoning:
            plan.finpt_summary = await self._generate_plan_summary(plan, portfolio)

        # 8. Handle based on mode
        await self._handle_plan(plan)

        # Record
        self._last_plan = plan
        self._allocation_history.append(plan)
        if len(self._allocation_history) > 100:
            self._allocation_history = self._allocation_history[-100:]

        cycle_latency = (time.time() - cycle_start) * 1000
        self._total_latency_ms += cycle_latency

        logger.info(f"Cycle completed in {cycle_latency:.0f}ms")
        logger.info(f"Plan ID: {plan.plan_id}, Risk Score: {plan.risk_score}")
        logger.info(f"Total trades: {plan.total_trades}, Impact: ${plan.estimated_impact_usd:,.2f}")

        return plan

    # =========================================================================
    # Market Assessment
    # =========================================================================

    async def assess_market(
        self,
        market_data: dict[str, Any] | None = None,
    ) -> MarketAssessment:
        """
        Assess current market conditions using FinGPT.

        Args:
            market_data: Additional market data (prices, volumes, etc.)

        Returns:
            MarketAssessment with regime, sentiment, and recommendations
        """
        # Default market data
        if market_data is None:
            market_data = {
                "sol_price": 150.0,
                "btc_price": 70000.0,
                "eth_price": 3500.0,
                "market_sentiment": "neutral",
                "fear_greed_index": 50,
            }

        prompt = f"""Assess current market conditions for Solana DeFi trading.

Market Data:
{json.dumps(market_data, indent=2)}

Provide your assessment in JSON format:
{{
  "condition": "very_bullish/bullish/neutral/bearish/very_bearish/high_volatility/low_liquidity",
  "regime": "bull/bear/chop/mev_heavy",
  "regime_confidence": 0.0-1.0,
  "sentiment_score": -1.0 to 1.0,
  "volatility_level": "low/medium/high",
  "risk_on_off": "risk_on/risk_off/neutral",
  "key_factors": ["factor1", "factor2", ...],
  "trading_recommendation": "Brief recommendation for current market",
  "suggested_exposure_pct": 0-100,
  "analysis": "Detailed analysis of market conditions"
}}"""

        response = await self.client.generate(
            prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=self.config.fingpt_temperature,
            json_mode=True,
        )

        if response.error or not response.parsed_json:
            # Fallback to neutral assessment
            logger.warning(f"Market assessment failed: {response.error}")
            return MarketAssessment(
                condition=MarketCondition.NEUTRAL,
                regime="chop",
                regime_confidence=0.5,
                sentiment_score=0.0,
                volatility_level="medium",
                risk_on_off="neutral",
                key_factors=["Assessment unavailable"],
                trading_recommendation="Wait for better market clarity",
                suggested_exposure_pct=50.0,
                finpt_analysis="Market assessment unavailable, using neutral fallback.",
                timestamp=datetime.utcnow(),
            )

        data = response.parsed_json

        # Map condition string to enum
        condition_map = {
            "very_bullish": MarketCondition.VERY_BULLISH,
            "bullish": MarketCondition.BULLISH,
            "neutral": MarketCondition.NEUTRAL,
            "bearish": MarketCondition.BEARISH,
            "very_bearish": MarketCondition.VERY_BEARISH,
            "high_volatility": MarketCondition.HIGH_VOLATILITY,
            "low_liquidity": MarketCondition.LOW_LIQUIDITY,
        }

        return MarketAssessment(
            condition=condition_map.get(data.get("condition", "neutral"), MarketCondition.NEUTRAL),
            regime=data.get("regime", "chop"),
            regime_confidence=float(data.get("regime_confidence", 0.5)),
            sentiment_score=float(data.get("sentiment_score", 0.0)),
            volatility_level=data.get("volatility_level", "medium"),
            risk_on_off=data.get("risk_on_off", "neutral"),
            key_factors=data.get("key_factors", []),
            trading_recommendation=data.get("trading_recommendation", ""),
            suggested_exposure_pct=float(data.get("suggested_exposure_pct", 50.0)),
            finpt_analysis=data.get("analysis", response.content),
            timestamp=datetime.utcnow(),
        )

    # =========================================================================
    # Opportunity Analysis
    # =========================================================================

    async def analyze_opportunity(
        self,
        opportunity: TokenOpportunity,
        market_assessment: MarketAssessment,
    ) -> TokenOpportunity:
        """
        Enhance opportunity analysis with FinGPT reasoning.

        Args:
            opportunity: Token opportunity to analyze
            market_assessment: Current market assessment

        Returns:
            Enhanced opportunity with FinGPT analysis
        """
        prompt = f"""Analyze this trading opportunity for Solana DeFi portfolio allocation.

Market Context:
- Condition: {market_assessment.condition.value}
- Regime: {market_assessment.regime}
- Sentiment: {market_assessment.sentiment_score:.2f}

Token Opportunity:
- Symbol: {opportunity.symbol}
- Price: ${opportunity.current_price:.6f}
- ML Score: {opportunity.ml_score:.3f}
- ML Confidence: {opportunity.ml_confidence:.3f}
- Fraud Score: {opportunity.fraud_score:.3f}
- Liquidity: ${opportunity.liquidity_usd:,.0f}
- 24h Volume: ${opportunity.volume_24h:,.0f}
- Market Cap: ${opportunity.market_cap:,.0f}
- 24h Change: {opportunity.price_change_24h:.2f}%

Provide analysis in JSON format:
{{
  "trade_recommendation": "strong_buy/buy/hold/sell/strong_sell",
  "recommended_allocation_pct": 0.0-15.0,
  "risk_level": "low/medium/high/very_high",
  "key_strengths": ["strength1", ...],
  "key_risks": ["risk1", ...],
  "optimal_entry_strategy": "market/limit/dca",
  "suggested_stop_loss_pct": 0.0-50.0,
  "suggested_take_profit_pct": 0.0-200.0,
  "conviction_score": 0.0-1.0,
  "analysis": "Detailed reasoning for the recommendation"
}}"""

        response = await self.client.generate(
            prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=self.config.fingpt_temperature,
            json_mode=True,
        )

        if response.parsed_json:
            data = response.parsed_json
            opportunity.recommended_allocation_pct = float(
                data.get("recommended_allocation_pct", 0.0)
            )
            opportunity.finpt_analysis = data.get("analysis", "")
            opportunity.priority = int(float(data.get("conviction_score", 0.5)) * 100)

        return opportunity

    # =========================================================================
    # Allocation Computation
    # =========================================================================

    async def compute_allocations(
        self,
        portfolio: PortfolioState,
        opportunities: list[TokenOpportunity],
        market: MarketAssessment,
    ) -> list[AllocationDecision]:
        """
        Compute optimal allocations based on portfolio, opportunities, and market.

        This is the core allocation logic that:
        1. Decides which positions to close
        2. Decides which opportunities to open
        3. Computes optimal position sizes
        4. Creates allocation decisions
        """
        decisions = []

        # Analyze current positions for exit signals
        for position in portfolio.positions:
            exit_decision = await self._evaluate_position_exit(position, market)
            if exit_decision:
                decisions.append(exit_decision)

        # Analyze opportunities for entry signals
        for opportunity in opportunities[:20]:  # Limit to top 20
            if (
                len([d for d in decisions if d.action == AllocationAction.OPEN_POSITION])
                >= self.config.max_positions - portfolio.position_count
            ):
                break

            entry_decision = await self._evaluate_opportunity_entry(opportunity, portfolio, market)
            if entry_decision:
                decisions.append(entry_decision)

        # Use FinGPT for final allocation optimization
        if decisions and self.config.enable_finpt_reasoning:
            decisions = await self._optimize_allocations_with_fingpt(decisions, portfolio, market)

        return decisions

    async def _evaluate_position_exit(
        self,
        position: Position,
        market: MarketAssessment,
    ) -> AllocationDecision | None:
        """Evaluate if a position should be closed."""
        # Hard exit rules
        if position.current_ml_score < self.config.ml_score_to_close:
            return AllocationDecision(
                action=AllocationAction.CLOSE_POSITION,
                symbol=position.symbol,
                mint=position.mint,
                current_allocation_pct=position.allocation_pct,
                target_allocation_pct=0.0,
                amount_to_trade=-position.amount,
                estimated_value_usd=-(position.amount * position.current_price),
                reasoning=(
                    f"ML score dropped below threshold "
                    f"({position.current_ml_score:.2f} < {self.config.ml_score_to_close})"
                ),
                confidence=0.8,
                urgency="soon",
                requires_approval=False,
                risk_level="low",
            )

        # Use FinGPT for nuanced exit decisions
        if self.config.enable_finpt_reasoning:
            prompt = f"""Should this position be closed or held?

Market Context:
- Condition: {market.condition.value}
- Regime: {market.regime}

Position:
- Symbol: {position.symbol}
- P&L: {position.pnl_pct * 100:.2f}% (${position.pnl_usd:,.2f})
- Hold Time: {position.hold_time_hours:.1f} hours
- Current ML Score: {position.current_ml_score:.3f}
- Entry ML Score: {position.ml_score_at_entry:.3f}
- Allocation: {position.allocation_pct:.1f}%

Provide recommendation in JSON format:
{{
  "action": "hold/close/partial_close",
  "confidence": 0.0-1.0,
  "reasoning": "Why",
  "urgency": "immediate/soon/monitor"
}}"""

            response = await self.client.generate(
                prompt,
                system_prompt=self.SYSTEM_PROMPT,
                temperature=0.2,  # Lower for exit decisions
                json_mode=True,
            )

            if response.parsed_json:
                data = response.parsed_json
                if data.get("action") == "close":
                    return AllocationDecision(
                        action=AllocationAction.CLOSE_POSITION,
                        symbol=position.symbol,
                        mint=position.mint,
                        current_allocation_pct=position.allocation_pct,
                        target_allocation_pct=0.0,
                        amount_to_trade=-position.amount,
                        estimated_value_usd=-(position.amount * position.current_price),
                        reasoning=data.get("reasoning", "FinGPT recommended close"),
                        confidence=float(data.get("confidence", 0.5)),
                        urgency=data.get("urgency", "soon"),
                        requires_approval=False,
                        risk_level="low",
                        finpt_rationale=data.get("reasoning"),
                    )

        return None

    async def _evaluate_opportunity_entry(
        self,
        opportunity: TokenOpportunity,
        portfolio: PortfolioState,
        market: MarketAssessment,
    ) -> AllocationDecision | None:
        """Evaluate if an opportunity should be entered."""
        # Hard filter rules
        if opportunity.ml_score < self.config.min_ml_score_to_open:
            return None
        if opportunity.ml_confidence < self.config.min_ml_confidence_to_open:
            return None
        if opportunity.fraud_score > self.config.max_fraud_score_to_trade:
            return None
        if opportunity.liquidity_usd < self.config.min_liquidity_usd:
            return None
        if opportunity.volume_24h < self.config.min_volume_24h_usd:
            return None

        # Check if we already have this position
        if any(p.mint == opportunity.mint for p in portfolio.positions):
            return None

        # Check position limit
        if portfolio.position_count >= self.config.max_positions:
            return None

        # Analyze with FinGPT
        if self.config.enable_finpt_reasoning:
            opportunity = await self.analyze_opportunity(opportunity, market)

        # Compute allocation
        allocation_pct = opportunity.recommended_allocation_pct
        if allocation_pct <= 0:
            # Use Kelly-inspired sizing based on ML confidence
            kelly_fraction = 0.25  # Conservative Kelly
            win_prob = opportunity.ml_confidence
            allocation_pct = max(0, min(15, (2 * win_prob - 1) * kelly_fraction * 100))

        allocation_pct = min(allocation_pct, self.config.max_allocation_per_token_pct)
        allocation_pct = max(allocation_pct, self.config.min_allocation_pct)

        # Check if we have enough cash
        trade_value = portfolio.total_value_usd * (allocation_pct / 100)
        if trade_value > portfolio.cash_usd:
            allocation_pct = (
                (portfolio.cash_usd / portfolio.total_value_usd) * 100 * 0.95
            )  # Leave 5% buffer
            trade_value = portfolio.cash_usd * 0.95

        if trade_value < 10:  # Minimum trade size
            return None

        return AllocationDecision(
            action=AllocationAction.OPEN_POSITION,
            symbol=opportunity.symbol,
            mint=opportunity.mint,
            current_allocation_pct=0.0,
            target_allocation_pct=allocation_pct,
            amount_to_trade=trade_value / opportunity.current_price,
            estimated_value_usd=trade_value,
            reasoning=(
                f"ML score {opportunity.ml_score:.2f}, confidence {opportunity.ml_confidence:.2f}"
            ),
            confidence=opportunity.ml_confidence,
            urgency="soon",
            requires_approval=allocation_pct > self.config.requires_approval_above_pct,
            risk_level="medium" if opportunity.fraud_score < 0.2 else "high",
            finpt_rationale=opportunity.finpt_analysis,
        )

    async def _optimize_allocations_with_fingpt(
        self,
        decisions: list[AllocationDecision],
        portfolio: PortfolioState,
        market: MarketAssessment,
    ) -> list[AllocationDecision]:
        """Use FinGPT to optimize and prioritize allocations."""
        if not decisions:
            return decisions

        prompt = f"""Optimize this allocation plan for a Solana DeFi portfolio.

Portfolio:
- Total Value: ${portfolio.total_value_usd:,.2f}
- Cash: ${portfolio.cash_usd:,.2f}
- Current Exposure: {portfolio.exposure_pct:.1f}%

Market:
- Condition: {market.condition.value}
- Suggested Exposure: {market.suggested_exposure_pct:.1f}%

Proposed Decisions:
{json.dumps([d.to_dict() for d in decisions], indent=2)}

Optimize in JSON format:
{{
  "keep_decisions": [0, 1, 2, ...],  // Indices to keep
  "modified_allocations": {{
    "0": {{"target_allocation_pct": 5.0, "reasoning": "why"}},
    ...
  }},
  "priority_order": [2, 0, 1, ...],  // Execution order
  "overall_assessment": "Brief summary",
  "total_risk_score": 0-100
}}"""

        response = await self.client.generate(
            prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.2,
            json_mode=True,
        )

        if response.parsed_json:
            data = response.parsed_json

            # Filter to kept decisions
            keep_indices = data.get("keep_decisions", list(range(len(decisions))))
            optimized = [decisions[i] for i in keep_indices if i < len(decisions)]

            # Apply modifications
            mods = data.get("modified_allocations", {})
            for idx_str, mod in mods.items():
                idx = int(idx_str)
                if idx < len(optimized):
                    optimized[idx].target_allocation_pct = mod.get(
                        "target_allocation_pct", optimized[idx].target_allocation_pct
                    )

            return optimized

        return decisions

    # =========================================================================
    # Risk Validation
    # =========================================================================

    def _validate_risk_limits(
        self,
        decisions: list[AllocationDecision],
        portfolio: PortfolioState,
    ) -> list[AllocationDecision]:
        """Validate decisions against risk limits."""
        valid = []
        total_turnover = 0.0

        for decision in decisions:
            # Check max single trade
            trade_pct = abs(decision.estimated_value_usd) / portfolio.total_value_usd * 100
            if trade_pct > self.config.max_single_trade_pct:
                decision.estimated_value_usd *= self.config.max_single_trade_pct / trade_pct
                decision.reasoning += " [Reduced for max trade limit]"

            # Check max allocation per token
            if decision.target_allocation_pct > self.config.max_allocation_per_token_pct:
                decision.target_allocation_pct = self.config.max_allocation_per_token_pct
                decision.reasoning += " [Capped at max allocation]"

            # Check daily turnover
            total_turnover += abs(decision.estimated_value_usd)
            turnover_pct = (total_turnover / portfolio.total_value_usd) * 100
            if turnover_pct > self.config.max_daily_turnover_pct:
                break  # Stop adding decisions

            valid.append(decision)

        return valid

    def _compute_plan_risk(
        self,
        decisions: list[AllocationDecision],
        portfolio: PortfolioState,
    ) -> int:
        """Compute overall risk score for the plan (0-100)."""
        if not decisions:
            return 0

        risk = 0

        # Concentration risk
        max_alloc = max(d.target_allocation_pct for d in decisions) if decisions else 0
        risk += min(30, max_alloc * 2)  # Max 30 points

        # Trade count risk
        risk += min(20, len(decisions) * 5)  # Max 20 points

        # Exposure risk
        new_exposure = portfolio.exposure_pct + sum(
            d.target_allocation_pct - d.current_allocation_pct for d in decisions
        )
        risk += min(30, max(0, new_exposure - 50))  # Max 30 points

        # Risk level contribution
        high_risk_count = sum(1 for d in decisions if d.risk_level == "high")
        risk += min(20, high_risk_count * 10)  # Max 20 points

        return min(100, int(risk))

    # =========================================================================
    # Plan Execution
    # =========================================================================

    async def _handle_plan(self, plan: AllocationPlan):
        """Handle allocation plan based on mode."""
        if not plan.decisions:
            logger.info("No decisions to execute")
            return

        if self.config.mode == AllocationMode.SHADOW:
            logger.info(f"[SHADOW] Would execute {len(plan.decisions)} decisions")
            for d in plan.decisions:
                logger.info(f"  - {d.action.value} {d.symbol}: {d.target_allocation_pct:.1f}%")
            return

        if self.config.mode == AllocationMode.ADVISOR:
            logger.info(f"[ADVISOR] Recommending {len(plan.decisions)} decisions")
            plan.requires_approval = True
            return

        # SEMI_AUTO or FULL_AUTO
        for decision in plan.decisions:
            # Check if approval needed
            needs_approval = self.config.mode == AllocationMode.SEMI_AUTO and (
                decision.requires_approval
                or abs(decision.estimated_value_usd) > self.config.auto_approve_max_value_usd
                or plan.risk_score > self.config.auto_approve_max_risk
            )

            if needs_approval:
                # Request approval
                approved = await self._request_approval(plan)
                if not approved:
                    logger.info(f"Plan {plan.plan_id} rejected")
                    plan.approved = False
                    self._total_rejected += 1
                    return
                plan.approved = True
                plan.approved_by = "human"
                plan.approved_at = datetime.utcnow()

            # Execute
            executed = await self._execute_decision(decision)
            if executed:
                self._total_executed += 1
                self._daily_trades.append(datetime.utcnow())
            self._total_decisions += 1

        plan.approved = True

    async def _execute_decision(self, decision: AllocationDecision) -> bool:
        """Execute a single allocation decision."""
        if self._execution_handler:
            try:
                return await self._execution_handler(decision)
            except Exception as e:
                logger.error(f"Execution failed: {e}")
                return False

        # No handler - log what would happen
        logger.info(f"[WOULD EXECUTE] {decision.action.value} {decision.symbol}")
        logger.info(f"  Amount: {decision.amount_to_trade:.4f}")
        logger.info(f"  Value: ${decision.estimated_value_usd:,.2f}")
        logger.info(f"  Reasoning: {decision.reasoning}")
        return True

    async def _request_approval(self, plan: AllocationPlan) -> bool:
        """Request approval for a plan."""
        if self._approval_handler:
            try:
                return await self._approval_handler(plan)
            except Exception as e:
                logger.error(f"Approval request failed: {e}")
                return False

        # No handler - default to reject in auto modes
        logger.warning(f"No approval handler configured - rejecting plan {plan.plan_id}")
        return False

    async def _generate_plan_summary(
        self,
        plan: AllocationPlan,
        portfolio: PortfolioState,
    ) -> str:
        """Generate FinGPT summary of the allocation plan."""
        prompt = f"""Summarize this portfolio allocation plan in 2-3 sentences.

Current Portfolio: ${portfolio.total_value_usd:,.2f}
Plan Risk Score: {plan.risk_score}/100
Market: {plan.market_condition.value}

Decisions:
{
            json.dumps(
                [
                    {
                        "action": d.action.value,
                        "symbol": d.symbol,
                        "allocation": d.target_allocation_pct,
                    }
                    for d in plan.decisions
                ],
                indent=2,
            )
        }

Provide a concise summary of what this plan does and why."""

        response = await self.client.generate(
            prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.5,
            max_tokens=150,
        )

        return response.content if not response.error else "Summary unavailable"

    # =========================================================================
    # Helper Methods
    # =========================================================================

    async def _get_portfolio_state(self) -> PortfolioState:
        """Get current portfolio state."""
        if self._portfolio_provider:
            try:
                return await self._portfolio_provider()
            except Exception as e:
                logger.error(f"Portfolio provider failed: {e}")

        # Default empty portfolio
        return PortfolioState(
            total_value_usd=1000.0,
            cash_usd=1000.0,
            positions=[],
            daily_pnl_usd=0.0,
            daily_pnl_pct=0.0,
            total_trades_today=0,
            turnover_today_pct=0.0,
            last_updated=datetime.utcnow(),
        )

    async def _scan_opportunities(self) -> list[TokenOpportunity]:
        """Scan for trading opportunities."""
        if self._opportunity_scanner:
            try:
                return await self._opportunity_scanner()
            except Exception as e:
                logger.error(f"Opportunity scanner failed: {e}")

        # Return empty list - should be provided by integration
        return []

    def get_stats(self) -> dict[str, Any]:
        """Get agent statistics."""
        # Clean up old daily trades
        cutoff = datetime.utcnow() - timedelta(hours=24)
        self._daily_trades = [t for t in self._daily_trades if t > cutoff]

        return {
            "running": self._running,
            "mode": self.config.mode.value,
            "total_cycles": self._total_cycles,
            "total_decisions": self._total_decisions,
            "total_executed": self._total_executed,
            "total_rejected": self._total_rejected,
            "avg_latency_ms": self._total_latency_ms / max(1, self._total_cycles),
            "trades_today": len(self._daily_trades),
            "last_cycle": self._last_cycle_time.isoformat() if self._last_cycle_time else None,
        }

    def get_last_plan(self) -> dict[str, Any] | None:
        """Get the last allocation plan."""
        return self._last_plan.to_dict() if self._last_plan else None

    def get_market_assessment(self) -> dict[str, Any] | None:
        """Get current market assessment."""
        return self._market_assessment.to_dict() if self._market_assessment else None


# =============================================================================
# Convenience Functions
# =============================================================================


_allocation_agent: AutomatedAllocationAgent | None = None


def get_allocation_agent() -> AutomatedAllocationAgent:
    """Get or create the singleton allocation agent."""
    global _allocation_agent
    if _allocation_agent is None:
        _allocation_agent = AutomatedAllocationAgent()
    return _allocation_agent
