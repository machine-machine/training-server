"""
FinGPT/Ollama client for local financial trading analysis.

Provides a unified interface for running local LLM inference via Ollama
for financial analysis, risk assessment, and trading recommendations.
"""

import asyncio
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)


class OllamaModel(Enum):
    LLAMA_3_2_3B = "llama3.2:3b"
    LLAMA_3_1_8B = "llama3.1:8b"
    LLAMA_3_2_1B = "llama3.2:1b"
    MISTRAL_7B = "mistral:7b"
    CUSTOM = "custom"


@dataclass
class FinGPTResponse:
    content: str
    model_used: str
    latency_ms: float
    tokens_generated: int = 0
    done: bool = True
    error: str | None = None
    parsed_json: dict | None = None


@dataclass
class TokenAnalysis:
    symbol: str
    price_usd: float
    volume_24h: float
    market_cap: float
    liquidity: float | None = None
    price_change_24h: float | None = None
    holders: int | None = None
    age_hours: float | None = None
    contract_verified: bool = False


@dataclass
class TradingRecommendation:
    action: str
    confidence: float
    reasoning: str
    risk_level: str
    suggested_position_size_pct: float
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    key_factors: list[str] = field(default_factory=list)


@dataclass
class FinGPTStats:
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: float = 0.0
    total_tokens: int = 0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / max(1, self.successful_requests)

    @property
    def success_rate(self) -> float:
        return self.successful_requests / max(1, self.total_requests)


FINANCIAL_SYSTEM_PROMPT = """You are FinGPT, a specialized financial analysis AI for
cryptocurrency trading on Solana DEXs. You analyze tokens and provide trading recommendations.

Your expertise includes:
- Solana ecosystem (Raydium, Jupiter, Orca, Meteora)
- Memecoin dynamics and rug pull detection
- Market microstructure and liquidity analysis
- Risk management and position sizing

When analyzing tokens, consider:
1. Liquidity depth and concentration
2. Holder distribution (whale concentration)
3. Token age and contract verification
4. Volume/market cap ratios
5. Price action and momentum
6. Social signals and community metrics

Always provide:
- Clear BUY/HOLD/SELL recommendation
- Confidence level (0.0-1.0)
- Risk assessment (LOW/MEDIUM/HIGH/EXTREME)
- Key factors driving the recommendation
- Suggested position size as % of portfolio

Be concise and focus on actionable insights. Use JSON format when requested."""


class FinGPTClient:
    """
    Ollama-based local LLM client for financial analysis.

    Supports running inference locally via Ollama API for:
    - Token analysis and recommendations
    - Risk assessment
    - Market sentiment analysis
    - Trading strategy evaluation
    """

    DEFAULT_OLLAMA_HOST = "http://localhost:11434"
    DEFAULT_MODEL = OllamaModel.LLAMA_3_2_3B
    DEFAULT_TIMEOUT = 60.0

    def __init__(
        self,
        model: OllamaModel | str = DEFAULT_MODEL,
        ollama_host: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        num_ctx: int = 4096,
    ):
        self.model = model.value if isinstance(model, OllamaModel) else model
        self.ollama_host = ollama_host or os.environ.get("OLLAMA_HOST", self.DEFAULT_OLLAMA_HOST)
        self.timeout = timeout
        self.system_prompt = system_prompt or FINANCIAL_SYSTEM_PROMPT
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.num_ctx = num_ctx
        self._stats = FinGPTStats()
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def is_ollama_running(self) -> bool:
        try:
            session = await self._get_session()
            async with session.get(f"{self.ollama_host}/api/tags") as resp:
                return resp.status == 200
        except Exception as e:
            logger.debug(f"Ollama not running: {e}")
            return False

    async def list_models(self) -> list[str]:
        try:
            session = await self._get_session()
            async with session.get(f"{self.ollama_host}/api/tags") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.warning(f"Failed to list models: {e}")
        return []

    async def pull_model(self, model_name: str | None = None) -> bool:
        model = model_name or self.model
        try:
            session = await self._get_session()
            async with session.post(
                f"{self.ollama_host}/api/pull",
                json={"name": model, "stream": False},
            ) as resp:
                if resp.status == 200:
                    logger.info(f"Pulled model: {model}")
                    return True
        except Exception as e:
            logger.error(f"Failed to pull model {model}: {e}")
        return False

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> FinGPTResponse:
        """
        Generate a response from the local LLM.

        Args:
            prompt: The user prompt
            system_prompt: Override system prompt
            temperature: Override temperature
            max_tokens: Maximum tokens to generate
            json_mode: Request JSON output

        Returns:
            FinGPTResponse with generated content
        """
        self._stats.total_requests += 1
        start_time = time.time()

        system = system_prompt or self.system_prompt
        temp = temperature if temperature is not None else self.temperature

        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "temperature": temp,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "num_ctx": self.num_ctx,
                "num_predict": max_tokens,
            },
        }

        if json_mode:
            payload["format"] = "json"

        try:
            session = await self._get_session()
            async with session.post(
                f"{self.ollama_host}/api/generate",
                json=payload,
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    self._stats.failed_requests += 1
                    return FinGPTResponse(
                        content="",
                        model_used=self.model,
                        latency_ms=(time.time() - start_time) * 1000,
                        error=f"HTTP {resp.status}: {error_text}",
                    )

                data = await resp.json()
                latency_ms = (time.time() - start_time) * 1000

                self._stats.successful_requests += 1
                self._stats.total_latency_ms += latency_ms

                content = data.get("response", "")
                tokens = data.get("eval_count", 0)
                self._stats.total_tokens += tokens

                parsed = None
                if json_mode and content:
                    try:
                        parsed = json.loads(content)
                    except json.JSONDecodeError:
                        pass

                return FinGPTResponse(
                    content=content,
                    model_used=self.model,
                    latency_ms=latency_ms,
                    tokens_generated=tokens,
                    done=data.get("done", True),
                    parsed_json=parsed,
                )

        except TimeoutError:
            self._stats.failed_requests += 1
            return FinGPTResponse(
                content="",
                model_used=self.model,
                latency_ms=self.timeout * 1000,
                error="Request timed out",
            )
        except Exception as e:
            self._stats.failed_requests += 1
            logger.error(f"FinGPT generation error: {e}")
            return FinGPTResponse(
                content="",
                model_used=self.model,
                latency_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )

    async def analyze_token(
        self,
        token: TokenAnalysis,
        additional_context: str | None = None,
    ) -> TradingRecommendation:
        """
        Analyze a token and provide a trading recommendation.

        Args:
            token: Token data to analyze
            additional_context: Extra context (e.g., social signals)

        Returns:
            TradingRecommendation with action and reasoning
        """
        prompt = f"""Analyze this Solana token and provide a trading recommendation.

Token Data:
- Symbol: {token.symbol}
- Price: ${token.price_usd:.6f}
- 24h Volume: ${token.volume_24h:,.0f}
- Market Cap: ${token.market_cap:,.0f}
- Liquidity: {f"${token.liquidity:,.0f}" if token.liquidity else "Unknown"}
- 24h Price Change: {
            f"{token.price_change_24h:.2f}%" if token.price_change_24h is not None else "Unknown"
        }
- Holders: {token.holders if token.holders else "Unknown"}
- Token Age: {f"{token.age_hours:.1f} hours" if token.age_hours else "Unknown"}
- Contract Verified: {"Yes" if token.contract_verified else "No"}

{f"Additional Context: {additional_context}" if additional_context else ""}

Respond in JSON format with this exact structure:
{{
  "action": "BUY" | "HOLD" | "SELL",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation of the recommendation",
  "risk_level": "LOW" | "MEDIUM" | "HIGH" | "EXTREME",
  "suggested_position_size_pct": 0.0-10.0,
  "stop_loss_pct": 0.0-50.0,
  "take_profit_pct": 0.0-200.0,
  "key_factors": ["factor1", "factor2", ...]
}}"""

        response = await self.generate(prompt, json_mode=True)

        if response.error or not response.parsed_json:
            return TradingRecommendation(
                action="HOLD",
                confidence=0.0,
                reasoning=f"Analysis failed: {response.error or 'Invalid response'}",
                risk_level="HIGH",
                suggested_position_size_pct=0.0,
            )

        data = response.parsed_json
        return TradingRecommendation(
            action=data.get("action", "HOLD"),
            confidence=float(data.get("confidence", 0.0)),
            reasoning=data.get("reasoning", ""),
            risk_level=data.get("risk_level", "HIGH"),
            suggested_position_size_pct=float(data.get("suggested_position_size_pct", 0.0)),
            stop_loss_pct=data.get("stop_loss_pct"),
            take_profit_pct=data.get("take_profit_pct"),
            key_factors=data.get("key_factors", []),
        )

    async def assess_risk(
        self,
        portfolio_value: float,
        positions: list[dict[str, Any]],
        market_conditions: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Assess overall portfolio risk.

        Args:
            portfolio_value: Total portfolio value in USD
            positions: List of current positions
            market_conditions: Market regime and conditions

        Returns:
            Risk assessment with recommendations
        """
        prompt = f"""Assess the risk of this trading portfolio.

Portfolio Value: ${portfolio_value:,.2f}

Current Positions:
{json.dumps(positions, indent=2)}

Market Conditions:
{json.dumps(market_conditions, indent=2)}

Provide a risk assessment in JSON format:
{{
  "overall_risk": "LOW" | "MEDIUM" | "HIGH" | "EXTREME",
  "risk_score": 0-100,
  "concentration_risk": 0-100,
  "liquidity_risk": 0-100,
  "volatility_risk": 0-100,
  "recommendations": ["rec1", "rec2", ...],
  "suggested_adjustments": [
    {{"action": "reduce", "symbol": "TOKEN", "pct": 50}},
    ...
  ]
}}"""

        response = await self.generate(prompt, json_mode=True)

        if response.error or not response.parsed_json:
            return {
                "overall_risk": "HIGH",
                "risk_score": 100,
                "error": response.error,
            }

        return response.parsed_json

    async def analyze_sentiment(
        self,
        messages: list[str],
        token_symbol: str | None = None,
    ) -> dict[str, Any]:
        """
        Analyze sentiment from social messages.

        Args:
            messages: List of social media messages
            token_symbol: Optional token to focus analysis on

        Returns:
            Sentiment analysis results
        """
        messages_text = "\n".join(f"- {m}" for m in messages[:50])

        prompt = f"""Analyze the sentiment of these social media messages.

{f"Token of interest: {token_symbol}" if token_symbol else ""}

Messages:
{messages_text}

Provide sentiment analysis in JSON format:
{{
  "overall_sentiment": "VERY_BULLISH" | "BULLISH" | "NEUTRAL" | "BEARISH" | "VERY_BEARISH",
  "sentiment_score": -1.0 to 1.0,
  "confidence": 0.0-1.0,
  "key_themes": ["theme1", "theme2", ...],
  "notable_concerns": ["concern1", ...],
  "momentum": "ACCELERATING" | "STABLE" | "DECLINING"
}}"""

        response = await self.generate(prompt, json_mode=True)

        if response.error or not response.parsed_json:
            return {
                "overall_sentiment": "NEUTRAL",
                "sentiment_score": 0.0,
                "confidence": 0.0,
                "error": response.error,
            }

        return response.parsed_json

    async def evaluate_strategy(
        self,
        strategy_params: dict[str, Any],
        recent_performance: dict[str, Any],
        market_regime: str,
    ) -> dict[str, Any]:
        """
        Evaluate and suggest improvements to trading strategy.

        Args:
            strategy_params: Current strategy parameters
            recent_performance: Recent trading performance metrics
            market_regime: Current market regime description

        Returns:
            Strategy evaluation with suggestions
        """
        prompt = f"""Evaluate this trading strategy and suggest improvements.

Current Strategy Parameters:
{json.dumps(strategy_params, indent=2)}

Recent Performance (last 7 days):
{json.dumps(recent_performance, indent=2)}

Market Regime: {market_regime}

Provide strategy evaluation in JSON format:
{{
  "strategy_health": "EXCELLENT" | "GOOD" | "NEEDS_TWEAKS" | "UNDERPERFORMING" | "CRITICAL",
  "overall_score": 0-100,
  "strengths": ["strength1", ...],
  "weaknesses": ["weakness1", ...],
  "suggested_changes": [
    {{
      "parameter": "param_name",
      "current_value": "x",
      "suggested_value": "y",
      "rationale": "why"
    }}
  ],
  "regime_compatibility": "OPTIMAL" | "ACCEPTABLE" | "POOR",
  "risk_assessment": "Low/Medium/High risk explanation"
}}"""

        response = await self.generate(prompt, json_mode=True)

        if response.error or not response.parsed_json:
            return {
                "strategy_health": "UNKNOWN",
                "overall_score": 0,
                "error": response.error,
            }

        return response.parsed_json

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_requests": self._stats.total_requests,
            "successful_requests": self._stats.successful_requests,
            "failed_requests": self._stats.failed_requests,
            "success_rate": self._stats.success_rate,
            "avg_latency_ms": self._stats.avg_latency_ms,
            "total_tokens": self._stats.total_tokens,
            "model": self.model,
            "ollama_host": self.ollama_host,
        }

    def reset_stats(self):
        self._stats = FinGPTStats()


_fingpt_client: FinGPTClient | None = None


def get_fingpt_client() -> FinGPTClient:
    global _fingpt_client
    if _fingpt_client is None:
        _fingpt_client = FinGPTClient()
    return _fingpt_client


async def ensure_ollama_running() -> bool:
    client = get_fingpt_client()
    if not await client.is_ollama_running():
        logger.info("Starting Ollama server...")
        try:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            await asyncio.sleep(3)
            return await client.is_ollama_running()
        except Exception as e:
            logger.error(f"Failed to start Ollama: {e}")
            return False
    return True
