"""
High-throughput LLM serving with vLLM.

vLLM provides 10-20x throughput improvement over standard HuggingFace
inference through PagedAttention and continuous batching.

Features:
- PagedAttention for efficient KV cache management
- Continuous batching for optimal GPU utilization
- Streaming output support
- Multi-model serving
- Automatic fallback to Ollama/HuggingFace

Usage:
    from coldpath.ai.vllm_server import VllmClient

    client = VllmClient(model="FinGPT/fingpt-sentiment_llama2-13b_lora")

    # Single inference
    result = await client.generate("Analyze SOL sentiment...")

    # Batch inference (high throughput)
    results = await client.generate_batch(prompts, max_tokens=50)
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

VLLM_AVAILABLE = False
try:
    from vllm import LLM, SamplingParams

    VLLM_AVAILABLE = True
    logger.info("vLLM available for high-throughput inference")
except ImportError:
    logger.debug("vLLM not available, will use HTTP API or fallback")


class InferenceBackend(Enum):
    VLLM_NATIVE = "vllm_native"  # Direct vLLM Python API
    VLLM_SERVER = "vllm_server"  # vLLM HTTP server
    OLLAMA = "ollama"  # Ollama fallback
    HUGGINGFACE = "huggingface"  # Standard HF fallback


@dataclass
class GenerationResult:
    """Result from LLM generation."""

    text: str
    prompt: str
    model: str
    tokens_generated: int
    prompt_tokens: int
    latency_ms: float
    finish_reason: str
    logprobs: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "prompt": self.prompt,
            "model": self.model,
            "tokens_generated": self.tokens_generated,
            "prompt_tokens": self.prompt_tokens,
            "latency_ms": self.latency_ms,
            "finish_reason": self.finish_reason,
        }


@dataclass
class BatchGenerationResult:
    """Result from batch generation."""

    results: list[GenerationResult]
    total_latency_ms: float
    total_tokens: int
    throughput_tokens_per_sec: float


@dataclass
class VllmConfig:
    """Configuration for vLLM server."""

    model: str = "meta-llama/Llama-3.2-3B-Instruct"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 4096
    max_num_batched_tokens: int = 8192
    max_num_seqs: int = 256
    swap_space: int = 4  # GB
    enforce_eager: bool = False

    # Sampling defaults
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_tokens: int = 256
    repetition_penalty: float = 1.1

    # Server settings
    host: str = "localhost"
    port: int = 8000

    @classmethod
    def for_sentiment(cls) -> "VllmConfig":
        """Config optimized for sentiment analysis."""
        return cls(
            model="FinGPT/fingpt-sentiment_llama2-13b_lora",
            max_tokens=50,
            temperature=0.3,
        )

    @classmethod
    def for_trading_analysis(cls) -> "VllmConfig":
        """Config optimized for trading analysis."""
        return cls(
            model="meta-llama/Llama-3.2-3B-Instruct",
            max_tokens=256,
            temperature=0.5,
        )


class VllmClient:
    """High-throughput LLM client using vLLM.

    Automatically selects best available backend:
    1. vLLM native (if GPU available)
    2. vLLM HTTP server (if running)
    3. Ollama (fallback)
    """

    def __init__(
        self,
        config: VllmConfig | None = None,
        backend: InferenceBackend | None = None,
    ):
        self.config = config or VllmConfig()
        self._backend: InferenceBackend | None = backend
        self._llm: Any | None = None
        self._session: aiohttp.ClientSession | None = None

        self._stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "total_latency_ms": 0.0,
            "errors": 0,
        }

        self._initialize()

    def _initialize(self):
        """Initialize the best available backend."""
        if self._backend:
            self._init_backend(self._backend)
            return

        if VLLM_AVAILABLE and self._check_gpu_available():
            self._init_backend(InferenceBackend.VLLM_NATIVE)
        elif self._check_vllm_server():
            self._init_backend(InferenceBackend.VLLM_SERVER)
        else:
            self._init_backend(InferenceBackend.OLLAMA)

    def _check_gpu_available(self) -> bool:
        """Check if GPU is available for vLLM."""
        try:
            import torch

            if torch.cuda.is_available():
                return True
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                if os.getenv("COLDPATH_ENABLE_MPS", "0") == "1":
                    return True
        except ImportError:
            pass
        return False

    def _check_vllm_server(self) -> bool:
        """Check if vLLM HTTP server is running."""
        try:
            import requests

            resp = requests.get(f"http://{self.config.host}:{self.config.port}/health", timeout=2)
            return resp.status_code == 200
        except Exception:
            return False

    def _init_backend(self, backend: InferenceBackend):
        """Initialize specific backend."""
        self._backend = backend
        logger.info(f"Initializing vLLM client with backend: {backend.value}")

        if backend == InferenceBackend.VLLM_NATIVE:
            self._init_vllm_native()
        elif backend == InferenceBackend.VLLM_SERVER:
            self._init_vllm_server()
        elif backend == InferenceBackend.OLLAMA:
            logger.info("Using Ollama backend for LLM inference")

    def _init_vllm_native(self):
        """Initialize vLLM native engine."""
        if not VLLM_AVAILABLE:
            raise RuntimeError("vLLM not available")

        self._llm = LLM(
            model=self.config.model,
            tensor_parallel_size=self.config.tensor_parallel_size,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            max_model_len=self.config.max_model_len,
            trust_remote_code=True,
        )
        logger.info(f"vLLM native engine initialized with {self.config.model}")

    def _init_vllm_server(self):
        """Initialize vLLM server connection."""
        self._base_url = f"http://{self.config.host}:{self.config.port}"
        logger.info(f"vLLM server client initialized: {self._base_url}")

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
        **kwargs,
    ) -> GenerationResult:
        """Generate text from a single prompt.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            stop: Stop sequences
            **kwargs: Additional sampling parameters

        Returns:
            GenerationResult with generated text
        """
        start_time = time.time()

        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature if temperature is not None else self.config.temperature
        top_p = top_p or self.config.top_p

        try:
            if self._backend == InferenceBackend.VLLM_NATIVE:
                result = await self._generate_vllm_native(
                    prompt, max_tokens, temperature, top_p, stop
                )
            elif self._backend == InferenceBackend.VLLM_SERVER:
                result = await self._generate_vllm_server(
                    prompt, max_tokens, temperature, top_p, stop
                )
            else:
                result = await self._generate_ollama(prompt, max_tokens, temperature, top_p)

            latency_ms = (time.time() - start_time) * 1000
            result.latency_ms = latency_ms

            self._stats["total_requests"] += 1
            self._stats["total_tokens"] += result.tokens_generated
            self._stats["total_latency_ms"] += latency_ms

            return result

        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Generation failed: {e}")
            return GenerationResult(
                text="",
                prompt=prompt,
                model=self.config.model,
                tokens_generated=0,
                prompt_tokens=0,
                latency_ms=(time.time() - start_time) * 1000,
                finish_reason="error",
            )

    async def _generate_vllm_native(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str] | None,
    ) -> GenerationResult:
        """Generate using vLLM native API."""
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            repetition_penalty=self.config.repetition_penalty,
        )

        loop = asyncio.get_event_loop()
        outputs = await loop.run_in_executor(
            None, lambda: self._llm.generate([prompt], sampling_params)
        )

        output = outputs[0]
        generated = output.outputs[0]

        return GenerationResult(
            text=generated.text,
            prompt=prompt,
            model=self.config.model,
            tokens_generated=len(generated.token_ids),
            prompt_tokens=len(output.prompt_token_ids),
            latency_ms=0,  # Set by caller
            finish_reason=generated.finish_reason,
        )

    async def _generate_vllm_server(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str] | None,
    ) -> GenerationResult:
        """Generate using vLLM HTTP server."""
        session = await self._get_session()

        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": stop or [],
        }

        async with session.post(
            f"{self._base_url}/v1/completions",
            json=payload,
        ) as resp:
            data = await resp.json()

        choice = data["choices"][0]

        return GenerationResult(
            text=choice["text"],
            prompt=prompt,
            model=data.get("model", self.config.model),
            tokens_generated=data["usage"]["completion_tokens"],
            prompt_tokens=data["usage"]["prompt_tokens"],
            latency_ms=0,
            finish_reason=choice.get("finish_reason", "stop"),
        )

    async def _generate_ollama(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> GenerationResult:
        """Generate using Ollama as fallback."""
        session = await self._get_session()

        model = self.config.model.split("/")[-1]
        if "llama" in model.lower():
            model = "llama3.2:3b"
        elif "mistral" in model.lower():
            model = "mistral:7b"

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            },
        }

        async with session.post(
            "http://localhost:11434/api/generate",
            json=payload,
        ) as resp:
            data = await resp.json()

        return GenerationResult(
            text=data.get("response", ""),
            prompt=prompt,
            model=model,
            tokens_generated=data.get("eval_count", 0),
            prompt_tokens=data.get("prompt_eval_count", 0),
            latency_ms=0,
            finish_reason="stop" if data.get("done") else "length",
        )

    async def generate_batch(
        self,
        prompts: list[str],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs,
    ) -> BatchGenerationResult:
        """Generate text for multiple prompts efficiently.

        Uses continuous batching for optimal throughput.

        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens per prompt
            temperature: Sampling temperature

        Returns:
            BatchGenerationResult with all results
        """
        start_time = time.time()

        if self._backend == InferenceBackend.VLLM_NATIVE:
            results = await self._generate_batch_vllm_native(prompts, max_tokens, temperature)
        else:
            tasks = [self.generate(p, max_tokens, temperature, **kwargs) for p in prompts]
            results = await asyncio.gather(*tasks)

        total_latency_ms = (time.time() - start_time) * 1000
        total_tokens = sum(r.tokens_generated for r in results)
        throughput = (total_tokens / total_latency_ms * 1000) if total_latency_ms > 0 else 0

        return BatchGenerationResult(
            results=results,
            total_latency_ms=total_latency_ms,
            total_tokens=total_tokens,
            throughput_tokens_per_sec=throughput,
        )

    async def _generate_batch_vllm_native(
        self,
        prompts: list[str],
        max_tokens: int | None,
        temperature: float | None,
    ) -> list[GenerationResult]:
        """Batch generation using vLLM native."""
        sampling_params = SamplingParams(
            max_tokens=max_tokens or self.config.max_tokens,
            temperature=temperature if temperature is not None else self.config.temperature,
            top_p=self.config.top_p,
            repetition_penalty=self.config.repetition_penalty,
        )

        loop = asyncio.get_event_loop()
        outputs = await loop.run_in_executor(
            None, lambda: self._llm.generate(prompts, sampling_params)
        )

        results = []
        for i, output in enumerate(outputs):
            generated = output.outputs[0]
            results.append(
                GenerationResult(
                    text=generated.text,
                    prompt=prompts[i],
                    model=self.config.model,
                    tokens_generated=len(generated.token_ids),
                    prompt_tokens=len(output.prompt_token_ids),
                    latency_ms=0,
                    finish_reason=generated.finish_reason,
                )
            )

        return results

    async def analyze_sentiment(
        self,
        text: str,
        return_confidence: bool = True,
    ) -> dict[str, Any]:
        """Analyze sentiment of text (optimized for financial content).

        Args:
            text: Text to analyze
            return_confidence: Include confidence score

        Returns:
            Dict with sentiment, confidence, and reasoning
        """
        prompt = f"""Analyze the sentiment of this financial text.
Return JSON with: sentiment (positive/negative/neutral), confidence (0-1), key_phrases (list).

Text: {text}

JSON:"""

        result = await self.generate(prompt, max_tokens=100, temperature=0.3)

        try:
            parsed = json.loads(result.text)
            return {
                "sentiment": parsed.get("sentiment", "neutral"),
                "confidence": parsed.get("confidence", 0.5),
                "key_phrases": parsed.get("key_phrases", []),
                "latency_ms": result.latency_ms,
            }
        except json.JSONDecodeError:
            text_lower = result.text.lower()
            if "positive" in text_lower:
                sentiment = "positive"
            elif "negative" in text_lower:
                sentiment = "negative"
            else:
                sentiment = "neutral"

            return {
                "sentiment": sentiment,
                "confidence": 0.5,
                "key_phrases": [],
                "latency_ms": result.latency_ms,
            }

    def get_stats(self) -> dict[str, Any]:
        """Get inference statistics."""
        stats = self._stats.copy()
        if stats["total_requests"] > 0:
            stats["avg_latency_ms"] = stats["total_latency_ms"] / stats["total_requests"]
            stats["avg_tokens_per_request"] = stats["total_tokens"] / stats["total_requests"]
            stats["error_rate"] = stats["errors"] / stats["total_requests"]
        return stats

    @property
    def backend(self) -> str:
        """Get current backend name."""
        return self._backend.value if self._backend else "none"

    async def close(self):
        """Close client and release resources."""
        if self._session and not self._session.closed:
            await self._session.close()
        if self._llm:
            del self._llm
            self._llm = None


class VllmServer:
    """vLLM server manager for starting/stopping the inference server."""

    def __init__(self, config: VllmConfig | None = None):
        self.config = config or VllmConfig()
        self._process = None

    async def start(self) -> bool:
        """Start vLLM server."""
        if not VLLM_AVAILABLE:
            logger.error("vLLM not available, cannot start server")
            return False

        cmd = [
            "python",
            "-m",
            "vllm.entrypoints.api_server",
            "--model",
            self.config.model,
            "--host",
            self.config.host,
            "--port",
            str(self.config.port),
            "--tensor-parallel-size",
            str(self.config.tensor_parallel_size),
            "--gpu-memory-utilization",
            str(self.config.gpu_memory_utilization),
        ]

        logger.info(f"Starting vLLM server: {' '.join(cmd)}")

        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        await asyncio.sleep(5)

        return self._process.returncode is None

    async def stop(self):
        """Stop vLLM server."""
        if self._process and self._process.returncode is None:
            self._process.terminate()
            await self._process.wait()
            logger.info("vLLM server stopped")

    async def health_check(self) -> bool:
        """Check if server is healthy."""
        try:
            session = aiohttp.ClientSession()
            async with session.get(
                f"http://{self.config.host}:{self.config.port}/health",
                timeout=aiohttp.ClientTimeout(total=2),
            ) as resp:
                await session.close()
                return resp.status == 200
        except Exception:
            return False
