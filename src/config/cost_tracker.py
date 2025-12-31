"""Cost Tracker - Track token usage and costs across LLM calls."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from decimal import Decimal

logger = logging.getLogger(__name__)


# Groq pricing (as of Dec 2024) - per 1M tokens
GROQ_PRICING = {
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.1-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "llama3-70b-8192": {"input": 0.59, "output": 0.79},
    "llama3-8b-8192": {"input": 0.05, "output": 0.08},
    "mixtral-8x7b-32768": {"input": 0.24, "output": 0.24},
    "gemma2-9b-it": {"input": 0.20, "output": 0.20},
    # Default fallback
    "default": {"input": 0.59, "output": 0.79},
}

# OpenAI pricing (for reference)
OPENAI_PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
}


@dataclass
class TokenUsage:
    """Token usage for a single LLM call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __post_init__(self):
        if self.total_tokens == 0:
            self.total_tokens = self.prompt_tokens + self.completion_tokens


@dataclass
class LLMCallCost:
    """Cost breakdown for a single LLM call."""
    model: str
    provider: str  # "groq", "openai", etc.
    tokens: TokenUsage
    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    call_type: str = ""  # "parse_input", "synthesize", "consistency_check", etc.

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "provider": self.provider,
            "prompt_tokens": self.tokens.prompt_tokens,
            "completion_tokens": self.tokens.completion_tokens,
            "total_tokens": self.tokens.total_tokens,
            "input_cost": round(self.input_cost, 6),
            "output_cost": round(self.output_cost, 6),
            "total_cost": round(self.total_cost, 6),
            "timestamp": self.timestamp,
            "call_type": self.call_type,
        }


class CostTracker:
    """
    Track token usage and costs across all LLM calls in a workflow run.

    Usage:
        tracker = CostTracker(run_id="abc123")
        tracker.add_call(model="llama-3.3-70b", tokens=TokenUsage(100, 50), call_type="synthesize")
        summary = tracker.get_summary()
    """

    def __init__(self, run_id: str = "", company_name: str = ""):
        self.run_id = run_id
        self.company_name = company_name
        self.calls: List[LLMCallCost] = []
        self.started_at = datetime.utcnow()

    def calculate_cost(
        self,
        model: str,
        tokens: TokenUsage,
        provider: str = "groq",
    ) -> tuple[float, float, float]:
        """
        Calculate cost for a given model and token usage.

        Returns:
            Tuple of (input_cost, output_cost, total_cost)
        """
        # Get pricing for the model
        if provider == "groq":
            pricing = GROQ_PRICING.get(model, GROQ_PRICING["default"])
        elif provider == "openai":
            pricing = OPENAI_PRICING.get(model, {"input": 0.0, "output": 0.0})
        else:
            pricing = {"input": 0.0, "output": 0.0}

        # Calculate costs (pricing is per 1M tokens)
        input_cost = (tokens.prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (tokens.completion_tokens / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost

        return input_cost, output_cost, total_cost

    def add_call(
        self,
        model: str,
        tokens: TokenUsage,
        provider: str = "groq",
        call_type: str = "",
    ) -> LLMCallCost:
        """Add an LLM call to the tracker."""
        input_cost, output_cost, total_cost = self.calculate_cost(model, tokens, provider)

        call = LLMCallCost(
            model=model,
            provider=provider,
            tokens=tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            call_type=call_type,
        )

        self.calls.append(call)
        logger.debug(f"Added LLM call: {model} - {tokens.total_tokens} tokens - ${total_cost:.6f}")

        return call

    def add_from_groq_response(
        self,
        response: Any,
        model: str,
        call_type: str = "",
    ) -> Optional[LLMCallCost]:
        """
        Add call from a Groq API response object.

        Args:
            response: Groq ChatCompletion response
            model: Model ID used
            call_type: Type of call (e.g., "synthesize", "parse_input")
        """
        try:
            usage = getattr(response, 'usage', None)
            if usage:
                tokens = TokenUsage(
                    prompt_tokens=getattr(usage, 'prompt_tokens', 0) or 0,
                    completion_tokens=getattr(usage, 'completion_tokens', 0) or 0,
                    total_tokens=getattr(usage, 'total_tokens', 0) or 0,
                )
                return self.add_call(model=model, tokens=tokens, provider="groq", call_type=call_type)
            else:
                logger.warning(f"No usage data in Groq response for {call_type}")
                return None
        except Exception as e:
            logger.error(f"Failed to extract usage from Groq response: {e}")
            return None

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all tracked calls."""
        if not self.calls:
            return {
                "run_id": self.run_id,
                "company_name": self.company_name,
                "total_calls": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
                "calls_by_type": {},
                "calls_by_model": {},
            }

        total_tokens = sum(c.tokens.total_tokens for c in self.calls)
        total_prompt_tokens = sum(c.tokens.prompt_tokens for c in self.calls)
        total_completion_tokens = sum(c.tokens.completion_tokens for c in self.calls)
        total_cost = sum(c.total_cost for c in self.calls)

        # Group by call type
        calls_by_type: Dict[str, Dict] = {}
        for call in self.calls:
            ct = call.call_type or "unknown"
            if ct not in calls_by_type:
                calls_by_type[ct] = {"count": 0, "tokens": 0, "cost": 0.0}
            calls_by_type[ct]["count"] += 1
            calls_by_type[ct]["tokens"] += call.tokens.total_tokens
            calls_by_type[ct]["cost"] += call.total_cost

        # Group by model
        calls_by_model: Dict[str, Dict] = {}
        for call in self.calls:
            model = call.model
            if model not in calls_by_model:
                calls_by_model[model] = {"count": 0, "tokens": 0, "cost": 0.0}
            calls_by_model[model]["count"] += 1
            calls_by_model[model]["tokens"] += call.tokens.total_tokens
            calls_by_model[model]["cost"] += call.total_cost

        return {
            "run_id": self.run_id,
            "company_name": self.company_name,
            "total_calls": len(self.calls),
            "total_tokens": total_tokens,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_cost": round(total_cost, 6),
            "total_cost_formatted": f"${total_cost:.4f}",
            "calls_by_type": {k: {**v, "cost": round(v["cost"], 6)} for k, v in calls_by_type.items()},
            "calls_by_model": {k: {**v, "cost": round(v["cost"], 6)} for k, v in calls_by_model.items()},
            "calls": [c.to_dict() for c in self.calls],
        }

    def get_total_cost(self) -> float:
        """Get total cost of all calls."""
        return sum(c.total_cost for c in self.calls)

    def get_total_tokens(self) -> int:
        """Get total tokens used."""
        return sum(c.tokens.total_tokens for c in self.calls)


# Global tracker for the current run
_current_tracker: Optional[CostTracker] = None


def get_cost_tracker(run_id: str = "", company_name: str = "") -> CostTracker:
    """Get or create a cost tracker for the current run."""
    global _current_tracker
    if _current_tracker is None or (run_id and _current_tracker.run_id != run_id):
        _current_tracker = CostTracker(run_id=run_id, company_name=company_name)
    return _current_tracker


def reset_cost_tracker():
    """Reset the global cost tracker."""
    global _current_tracker
    _current_tracker = None


def calculate_cost_for_tokens(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    provider: str = "groq",
) -> Dict[str, float]:
    """
    Standalone function to calculate cost for given tokens.

    Returns:
        Dict with input_cost, output_cost, total_cost
    """
    tracker = CostTracker()
    tokens = TokenUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    input_cost, output_cost, total_cost = tracker.calculate_cost(model, tokens, provider)

    return {
        "input_cost": round(input_cost, 6),
        "output_cost": round(output_cost, 6),
        "total_cost": round(total_cost, 6),
        "total_cost_formatted": f"${total_cost:.6f}",
    }
