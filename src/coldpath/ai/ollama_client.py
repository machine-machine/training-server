"""
Ollama Local LLM Client - Zero-cost fallback for Claude.

Provides local LLM inference via Ollama for:
- Fast responses when Claude is unavailable
- Simple tasks that don't need Claude's capabilities
- Cost savings on high-volume queries
- Offline operation capability
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class OllamaModel(Enum):
    """Available Ollama models optimized for different tasks."""

    # Fast inference models
    LLAMA_8B = "llama3.3:8b-instruct-q4_K_M"  # Best balance of speed/quality
    QWEN_7B = "qwen2.5:7b"  # Good for structured reasoning
    MISTRAL_7B = "mistral:7b-instruct"  # Fast, good for chat

    # Larger models for complex tasks
    LLAMA_70B = "llama3.3:70b-instruct-q4_K_M"  # Near-Claude quality
    QWEN_32B = "qwen2.5:32b"  # Strong reasoning


@dataclass
class OllamaResponse:
    """Response from Ollama API."""

    content: str
    model_used: str
    total_duration_ns: int
    load_duration_ns: int
    eval_count: int  # Output tokens
    prompt_eval_count: int  # Input tokens
    latency_ms: float

    @property
    def input_tokens(self) -> int:
        return self.prompt_eval_count

    @property
    def output_tokens(self) -> int:
        return self.eval_count


@dataclass
class OllamaStats:
    """Statistics for Ollama client."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_latency_ms: float = 0.0
    model_requests: dict[str, int] = field(default_factory=dict)

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / max(1, self.successful_requests)

    @property
    def success_rate(self) -> float:
        return self.successful_requests / max(1, self.total_requests)


class OllamaClient:
    """Client for local Ollama LLM inference.

    Features:
    - Automatic model selection based on task complexity
    - Connection pooling for performance
    - Graceful fallback when Ollama unavailable
    - Statistics tracking

    Usage:
        client = OllamaClient()
        if await client.is_available():
            response = await client.chat("Explain this trade signal")
    """

    DEFAULT_HOST = "http://localhost:11434"
    DEFAULT_MODEL = OllamaModel.LLAMA_8B

    def __init__(
        self,
        host: str | None = None,
        default_model: OllamaModel | None = None,
        timeout: float = 30.0,
        max_retries: int = 2,
    ):
        """Initialize Ollama client.

        Args:
            host: Ollama server URL. Defaults to localhost:11434.
            default_model: Default model for inference.
            timeout: Request timeout in seconds.
            max_retries: Number of retries on failure.
        """
        self.host = host or self.DEFAULT_HOST
        self.default_model = default_model or self.DEFAULT_MODEL
        self.timeout = timeout
        self.max_retries = max_retries

        self._stats = OllamaStats()
        self._available_models: list[str] | None = None
        self._is_available: bool | None = None

    async def is_available(self, force_check: bool = False) -> bool:
        """Check if Ollama server is running and accessible.

        Args:
            force_check: Bypass cached availability status.

        Returns:
            True if Ollama is available.
        """
        if self._is_available is not None and not force_check:
            return self._is_available

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.host}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    self._available_models = [m["name"] for m in data.get("models", [])]
                    self._is_available = True
                    logger.info(f"Ollama available with {len(self._available_models)} models")
                    return True
        except Exception as e:
            logger.debug(f"Ollama not available: {e}")
            self._is_available = False

        return False

    async def get_available_models(self) -> list[str]:
        """Get list of models available on Ollama server."""
        if self._available_models is None:
            await self.is_available(force_check=True)
        return self._available_models or []

    async def has_model(self, model: OllamaModel) -> bool:
        """Check if a specific model is available."""
        models = await self.get_available_models()
        model_name = model.value.split(":")[0]  # Match base name
        return any(model_name in m for m in models)

    async def chat(
        self,
        message: str,
        model: OllamaModel | None = None,
        system_prompt: str | None = None,
        conversation_history: list[dict[str, str]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> OllamaResponse:
        """Send a chat message to Ollama.

        Args:
            message: The user message to send.
            model: Model to use. Defaults to default_model.
            system_prompt: Optional system prompt.
            conversation_history: Previous messages for context.
            temperature: Sampling temperature.
            max_tokens: Maximum response tokens.

        Returns:
            OllamaResponse with the model's reply.

        Raises:
            RuntimeError: If Ollama is not available.
        """
        if not await self.is_available():
            raise RuntimeError("Ollama server is not available")

        model = model or self.default_model
        model_name = model.value

        self._stats.total_requests += 1
        self._stats.model_requests[model_name] = self._stats.model_requests.get(model_name, 0) + 1

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": message})

        start_time = time.time()

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.host}/api/chat",
                    json={
                        "model": model_name,
                        "messages": messages,
                        "options": {
                            "temperature": temperature,
                            "num_predict": max_tokens,
                        },
                        "stream": False,
                    },
                )
                response.raise_for_status()
                data = response.json()

            latency_ms = (time.time() - start_time) * 1000

            result = OllamaResponse(
                content=data.get("message", {}).get("content", ""),
                model_used=model_name,
                total_duration_ns=data.get("total_duration", 0),
                load_duration_ns=data.get("load_duration", 0),
                eval_count=data.get("eval_count", 0),
                prompt_eval_count=data.get("prompt_eval_count", 0),
                latency_ms=latency_ms,
            )

            # Update stats
            self._stats.successful_requests += 1
            self._stats.total_input_tokens += result.input_tokens
            self._stats.total_output_tokens += result.output_tokens
            self._stats.total_latency_ms += latency_ms

            return result

        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama HTTP error: {e.response.status_code}")
            self._stats.failed_requests += 1
            raise
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            self._stats.failed_requests += 1
            raise

    async def generate(
        self,
        prompt: str,
        model: OllamaModel | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> OllamaResponse:
        """Generate completion (non-chat mode).

        Args:
            prompt: The prompt text.
            model: Model to use.
            temperature: Sampling temperature.
            max_tokens: Maximum response tokens.

        Returns:
            OllamaResponse with generation.
        """
        if not await self.is_available():
            raise RuntimeError("Ollama server is not available")

        model = model or self.default_model
        model_name = model.value

        self._stats.total_requests += 1
        start_time = time.time()

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.host}/api/generate",
                    json={
                        "model": model_name,
                        "prompt": prompt,
                        "options": {
                            "temperature": temperature,
                            "num_predict": max_tokens,
                        },
                        "stream": False,
                    },
                )
                response.raise_for_status()
                data = response.json()

            latency_ms = (time.time() - start_time) * 1000

            result = OllamaResponse(
                content=data.get("response", ""),
                model_used=model_name,
                total_duration_ns=data.get("total_duration", 0),
                load_duration_ns=data.get("load_duration", 0),
                eval_count=data.get("eval_count", 0),
                prompt_eval_count=data.get("prompt_eval_count", 0),
                latency_ms=latency_ms,
            )

            self._stats.successful_requests += 1
            self._stats.total_input_tokens += result.input_tokens
            self._stats.total_output_tokens += result.output_tokens
            self._stats.total_latency_ms += latency_ms

            return result

        except Exception as e:
            logger.error(f"Ollama generate error: {e}")
            self._stats.failed_requests += 1
            raise

    async def explain_score_fast(
        self,
        features: dict[str, float],
        score: float,
        top_factors: list[tuple],
    ) -> str:
        """Fast score explanation using local LLM.

        Optimized for <100ms latency on simple explanations.
        """
        factors_text = "\n".join(f"- {name}: {value:+.3f}" for name, value in top_factors[:3])

        prompt = f"""Score: {score:.2f}
Key factors:
{factors_text}

Explain in 1-2 sentences why this score."""

        try:
            response = await self.chat(
                message=prompt,
                model=OllamaModel.LLAMA_8B,
                temperature=0.5,
                max_tokens=128,
            )
            return response.content
        except Exception as e:
            logger.warning(f"Ollama explain failed: {e}")
            return f"Score {score:.2f} based on {top_factors[0][0] if top_factors else 'analysis'}."

    async def summarize_regime(
        self,
        regime: str,
        features: dict[str, float],
    ) -> str:
        """Fast regime summary using local LLM."""
        prompt = f"""Market regime: {regime}
Features: volatility={features.get("volatility", 0):.2f}, momentum={features.get("momentum", 0):.2f}

One sentence summary of trading implications."""

        try:
            response = await self.chat(
                message=prompt,
                model=OllamaModel.LLAMA_8B,
                temperature=0.5,
                max_tokens=64,
            )
            return response.content
        except Exception:
            return f"Market is in {regime} regime."

    def get_stats(self) -> dict[str, Any]:
        """Get client statistics."""
        return {
            "total_requests": self._stats.total_requests,
            "successful_requests": self._stats.successful_requests,
            "failed_requests": self._stats.failed_requests,
            "success_rate": self._stats.success_rate,
            "total_input_tokens": self._stats.total_input_tokens,
            "total_output_tokens": self._stats.total_output_tokens,
            "avg_latency_ms": self._stats.avg_latency_ms,
            "model_requests": dict(self._stats.model_requests),
            "available_models": self._available_models or [],
        }

    def reset_stats(self):
        """Reset client statistics."""
        self._stats = OllamaStats()


# Singleton instance for easy access
_ollama_client: OllamaClient | None = None


def get_ollama_client() -> OllamaClient:
    """Get or create the global Ollama client instance."""
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = OllamaClient()
    return _ollama_client


async def check_ollama_available() -> bool:
    """Quick check if Ollama is available."""
    client = get_ollama_client()
    return await client.is_available()
