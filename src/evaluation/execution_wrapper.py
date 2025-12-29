"""Execution Wrapper - Routes prompts to multiple LLMs for consistency evaluation."""

import os
import logging
import time
import json
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for an LLM model."""
    name: str
    provider: str  # openai, anthropic, ollama, google
    model_id: str
    temperature: float = 0.0
    max_tokens: int = 2000
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "provider": self.provider,
            "model_id": self.model_id,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "enabled": self.enabled,
        }


@dataclass
class ModelResponse:
    """Response from a single model."""
    model_name: str
    model_id: str
    output: str
    reasoning: Optional[str] = None
    tokens_used: int = 0
    response_time_ms: int = 0
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_id": self.model_id,
            "output": self.output,
            "reasoning": self.reasoning,
            "tokens_used": self.tokens_used,
            "response_time_ms": self.response_time_ms,
            "error": self.error,
            "timestamp": self.timestamp,
        }


@dataclass
class EvaluationResult:
    """Result from running a prompt through multiple models."""
    prompt_id: str
    prompt: str
    context: Optional[str] = None
    golden_answer: Optional[str] = None
    responses: List[ModelResponse] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "prompt": self.prompt,
            "context": self.context,
            "golden_answer": self.golden_answer,
            "responses": [r.to_dict() for r in self.responses],
            "timestamp": self.timestamp,
        }

    def get_outputs(self) -> List[str]:
        """Get list of outputs from all models."""
        return [r.output for r in self.responses if r.output and not r.error]


class ExecutionWrapper:
    """
    Routes prompts to multiple LLMs for consistency evaluation.

    Supports:
    - OpenAI (GPT-4o-mini, GPT-4, etc.)
    - Anthropic (Claude 3 Haiku, etc.)
    - Ollama (local models)
    - Google (Gemini)
    """

    DEFAULT_MODELS = [
        # FREE MODELS (Groq - fast inference, free tier)
        ModelConfig(
            name="groq-llama3-70b",
            provider="groq",
            model_id="llama-3.3-70b-versatile",
            temperature=0.0,
            enabled=True,
        ),
        ModelConfig(
            name="groq-gemma2",
            provider="groq",
            model_id="gemma2-9b-it",
            temperature=0.0,
            enabled=True,
        ),
        ModelConfig(
            name="groq-llama3-8b",
            provider="groq",
            model_id="llama-3.1-8b-instant",
            temperature=0.0,
            enabled=True,
        ),
        # PAID MODELS (optional - disabled by default)
        ModelConfig(
            name="gpt-4o-mini",
            provider="openai",
            model_id="gpt-4o-mini",
            temperature=0.0,
            enabled=False,
        ),
        ModelConfig(
            name="claude-3-haiku",
            provider="anthropic",
            model_id="claude-3-haiku-20240307",
            temperature=0.0,
            enabled=False,
        ),
        # LOCAL MODELS (Ollama - requires local install)
        ModelConfig(
            name="llama3.2",
            provider="ollama",
            model_id="llama3.2",
            temperature=0.0,
            enabled=False,
        ),
    ]

    def __init__(
        self,
        models: Optional[List[ModelConfig]] = None,
        parallel: bool = True,
        timeout_seconds: int = 60,
    ):
        self.models = models or self.DEFAULT_MODELS
        self.parallel = parallel
        self.timeout_seconds = timeout_seconds
        self._init_clients()

    def _init_clients(self):
        """Initialize API clients for each provider."""
        self.clients = {}

        # Groq (FREE - high priority)
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            try:
                from groq import Groq
                self.clients["groq"] = Groq(api_key=groq_key)
                logger.info("Groq client initialized (FREE)")
            except ImportError:
                logger.warning("Groq package not installed. Install with: pip install groq")

        # OpenAI (paid)
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            try:
                from openai import OpenAI
                self.clients["openai"] = OpenAI(api_key=openai_key)
                logger.info("OpenAI client initialized")
            except ImportError:
                logger.warning("OpenAI package not installed")

        # Anthropic (paid)
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key:
            try:
                from anthropic import Anthropic
                self.clients["anthropic"] = Anthropic(api_key=anthropic_key)
                logger.info("Anthropic client initialized")
            except ImportError:
                logger.warning("Anthropic package not installed")

        # Ollama (local)
        try:
            import requests
            # Check if Ollama is running
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                self.clients["ollama"] = True
                logger.info("Ollama client initialized")
        except Exception:
            logger.info("Ollama not available")

        # Google
        google_key = os.getenv("GOOGLE_API_KEY")
        if google_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=google_key)
                self.clients["google"] = genai
                logger.info("Google Gemini client initialized")
            except ImportError:
                logger.warning("Google AI package not installed")

    def execute(
        self,
        prompt: str,
        context: Optional[str] = None,
        golden_answer: Optional[str] = None,
        prompt_id: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Execute a prompt across all enabled models.

        Args:
            prompt: The prompt to send to models
            context: Optional context/background information
            golden_answer: Optional ground truth answer
            prompt_id: Optional ID for this prompt

        Returns:
            EvaluationResult with responses from all models
        """
        # Generate prompt ID if not provided
        if not prompt_id:
            prompt_id = hashlib.md5(prompt.encode()).hexdigest()[:12]

        result = EvaluationResult(
            prompt_id=prompt_id,
            prompt=prompt,
            context=context,
            golden_answer=golden_answer,
        )

        # Get enabled models
        enabled_models = [m for m in self.models if m.enabled]

        if not enabled_models:
            logger.warning("No models enabled")
            return result

        # Execute on all models
        if self.parallel:
            result.responses = self._execute_parallel(prompt, context, enabled_models)
        else:
            result.responses = self._execute_sequential(prompt, context, enabled_models)

        return result

    def _execute_parallel(
        self,
        prompt: str,
        context: Optional[str],
        models: List[ModelConfig],
    ) -> List[ModelResponse]:
        """Execute prompt on models in parallel."""
        responses = []

        with ThreadPoolExecutor(max_workers=len(models)) as executor:
            futures = {
                executor.submit(self._call_model, prompt, context, model): model
                for model in models
            }

            for future in as_completed(futures, timeout=self.timeout_seconds):
                model = futures[future]
                try:
                    response = future.result()
                    responses.append(response)
                except Exception as e:
                    logger.error(f"Error calling {model.name}: {e}")
                    responses.append(ModelResponse(
                        model_name=model.name,
                        model_id=model.model_id,
                        output="",
                        error=str(e),
                    ))

        return responses

    def _execute_sequential(
        self,
        prompt: str,
        context: Optional[str],
        models: List[ModelConfig],
    ) -> List[ModelResponse]:
        """Execute prompt on models sequentially."""
        responses = []

        for model in models:
            try:
                response = self._call_model(prompt, context, model)
                responses.append(response)
            except Exception as e:
                logger.error(f"Error calling {model.name}: {e}")
                responses.append(ModelResponse(
                    model_name=model.name,
                    model_id=model.model_id,
                    output="",
                    error=str(e),
                ))

        return responses

    def _call_model(
        self,
        prompt: str,
        context: Optional[str],
        model: ModelConfig,
    ) -> ModelResponse:
        """Call a single model."""
        start_time = time.time()

        # Build full prompt with context
        full_prompt = prompt
        if context:
            full_prompt = f"Context:\n{context}\n\nQuestion:\n{prompt}"

        try:
            if model.provider == "groq":
                output, tokens = self._call_groq(full_prompt, model)
            elif model.provider == "openai":
                output, tokens = self._call_openai(full_prompt, model)
            elif model.provider == "anthropic":
                output, tokens = self._call_anthropic(full_prompt, model)
            elif model.provider == "ollama":
                output, tokens = self._call_ollama(full_prompt, model)
            elif model.provider == "google":
                output, tokens = self._call_google(full_prompt, model)
            else:
                raise ValueError(f"Unknown provider: {model.provider}")

            response_time = int((time.time() - start_time) * 1000)

            return ModelResponse(
                model_name=model.name,
                model_id=model.model_id,
                output=output,
                tokens_used=tokens,
                response_time_ms=response_time,
            )

        except Exception as e:
            response_time = int((time.time() - start_time) * 1000)
            return ModelResponse(
                model_name=model.name,
                model_id=model.model_id,
                output="",
                error=str(e),
                response_time_ms=response_time,
            )

    def _call_groq(self, prompt: str, model: ModelConfig) -> tuple:
        """Call Groq API (FREE)."""
        if "groq" not in self.clients:
            raise RuntimeError("Groq client not available. Get free API key at console.groq.com")

        client = self.clients["groq"]
        response = client.chat.completions.create(
            model=model.model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=model.temperature,
            max_tokens=model.max_tokens,
        )

        output = response.choices[0].message.content
        tokens = response.usage.total_tokens if response.usage else 0

        return output, tokens

    def _call_openai(self, prompt: str, model: ModelConfig) -> tuple:
        """Call OpenAI API."""
        if "openai" not in self.clients:
            raise RuntimeError("OpenAI client not available")

        client = self.clients["openai"]
        response = client.chat.completions.create(
            model=model.model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=model.temperature,
            max_tokens=model.max_tokens,
        )

        output = response.choices[0].message.content
        tokens = response.usage.total_tokens if response.usage else 0

        return output, tokens

    def _call_anthropic(self, prompt: str, model: ModelConfig) -> tuple:
        """Call Anthropic API."""
        if "anthropic" not in self.clients:
            raise RuntimeError("Anthropic client not available")

        client = self.clients["anthropic"]
        response = client.messages.create(
            model=model.model_id,
            max_tokens=model.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )

        output = response.content[0].text
        tokens = response.usage.input_tokens + response.usage.output_tokens

        return output, tokens

    def _call_ollama(self, prompt: str, model: ModelConfig) -> tuple:
        """Call Ollama local API."""
        if "ollama" not in self.clients:
            raise RuntimeError("Ollama not available")

        import requests

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model.model_id,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": model.temperature,
                },
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()

        output = data.get("response", "")
        tokens = data.get("eval_count", 0) + data.get("prompt_eval_count", 0)

        return output, tokens

    def _call_google(self, prompt: str, model: ModelConfig) -> tuple:
        """Call Google Gemini API."""
        if "google" not in self.clients:
            raise RuntimeError("Google client not available")

        genai = self.clients["google"]
        gen_model = genai.GenerativeModel(model.model_id)

        response = gen_model.generate_content(
            prompt,
            generation_config={
                "temperature": model.temperature,
                "max_output_tokens": model.max_tokens,
            },
        )

        output = response.text
        # Google doesn't easily expose token counts in the same way
        tokens = len(prompt.split()) + len(output.split())  # Rough estimate

        return output, tokens

    def get_available_models(self) -> List[str]:
        """Get list of available (configured and accessible) models."""
        available = []

        for model in self.models:
            if not model.enabled:
                continue

            if model.provider == "groq" and "groq" in self.clients:
                available.append(model.name)
            elif model.provider == "openai" and "openai" in self.clients:
                available.append(model.name)
            elif model.provider == "anthropic" and "anthropic" in self.clients:
                available.append(model.name)
            elif model.provider == "ollama" and "ollama" in self.clients:
                available.append(model.name)
            elif model.provider == "google" and "google" in self.clients:
                available.append(model.name)

        return available

    def health_check(self) -> Dict[str, bool]:
        """Check which model providers are available."""
        return {
            "groq": "groq" in self.clients,
            "openai": "openai" in self.clients,
            "anthropic": "anthropic" in self.clients,
            "ollama": "ollama" in self.clients,
            "google": "google" in self.clients,
        }
