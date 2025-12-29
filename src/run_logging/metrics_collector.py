"""Metrics Collector - Collects and aggregates metrics across workflow execution."""

import time
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class StepMetrics:
    """Metrics for a single step."""
    step_name: str
    start_time: float = 0.0
    end_time: float = 0.0
    execution_time_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunMetrics:
    """Aggregated metrics for a complete run."""
    run_id: str
    company_name: str
    total_execution_time_ms: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    successful_steps: int = 0
    failed_steps: int = 0
    steps: List[StepMetrics] = field(default_factory=list)


class MetricsCollector:
    """
    Collects metrics during workflow execution.

    Provides:
    - Step-level timing and token tracking
    - Run-level aggregation
    - Cost estimation
    """

    # Approximate costs per 1M tokens (for estimation)
    TOKEN_COSTS = {
        "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
        "gemma2-9b-it": {"input": 0.20, "output": 0.20},
        "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    }

    def __init__(self, run_id: str, company_name: str):
        self.run_id = run_id
        self.company_name = company_name
        self.run_start_time = time.time()
        self.steps: List[StepMetrics] = []
        self._current_step: Optional[StepMetrics] = None

    def start_step(self, step_name: str) -> None:
        """Start tracking a new step."""
        self._current_step = StepMetrics(
            step_name=step_name,
            start_time=time.time(),
        )
        logger.debug(f"Started step: {step_name}")

    def end_step(
        self,
        success: bool = True,
        error: Optional[str] = None,
        tokens: Dict[str, int] = None,
        metadata: Dict[str, Any] = None,
    ) -> StepMetrics:
        """End the current step and record metrics."""
        if not self._current_step:
            raise RuntimeError("No step in progress")

        step = self._current_step
        step.end_time = time.time()
        step.execution_time_ms = (step.end_time - step.start_time) * 1000
        step.success = success
        step.error = error
        step.metadata = metadata or {}

        if tokens:
            step.prompt_tokens = tokens.get("prompt_tokens", 0)
            step.completion_tokens = tokens.get("completion_tokens", 0)
            step.total_tokens = tokens.get("total_tokens", 0)

        self.steps.append(step)
        self._current_step = None

        logger.debug(f"Completed step: {step.step_name} in {step.execution_time_ms:.2f}ms")
        return step

    def record_step(
        self,
        step_name: str,
        execution_time_ms: float,
        success: bool = True,
        error: Optional[str] = None,
        tokens: Dict[str, int] = None,
        metadata: Dict[str, Any] = None,
    ) -> StepMetrics:
        """Record a step directly without start/end."""
        step = StepMetrics(
            step_name=step_name,
            execution_time_ms=execution_time_ms,
            success=success,
            error=error,
            metadata=metadata or {},
        )

        if tokens:
            step.prompt_tokens = tokens.get("prompt_tokens", 0)
            step.completion_tokens = tokens.get("completion_tokens", 0)
            step.total_tokens = tokens.get("total_tokens", 0)

        self.steps.append(step)
        return step

    def get_run_metrics(self) -> RunMetrics:
        """Get aggregated metrics for the run."""
        total_time = (time.time() - self.run_start_time) * 1000

        metrics = RunMetrics(
            run_id=self.run_id,
            company_name=self.company_name,
            total_execution_time_ms=round(total_time, 2),
            steps=self.steps.copy(),
        )

        for step in self.steps:
            metrics.total_prompt_tokens += step.prompt_tokens
            metrics.total_completion_tokens += step.completion_tokens
            metrics.total_tokens += step.total_tokens

            if step.success:
                metrics.successful_steps += 1
            else:
                metrics.failed_steps += 1

            # Count LLM and tool calls
            if "llm" in step.step_name.lower() or "selection" in step.step_name.lower() or "synthesis" in step.step_name.lower():
                metrics.llm_calls += 1
            if "tool" in step.step_name.lower() or "fetch" in step.step_name.lower():
                metrics.tool_calls += 1

        return metrics

    def estimate_cost(self, model: str = "llama-3.3-70b-versatile") -> Dict[str, float]:
        """Estimate the cost of the run."""
        costs = self.TOKEN_COSTS.get(model, self.TOKEN_COSTS["llama-3.3-70b-versatile"])

        metrics = self.get_run_metrics()

        input_cost = (metrics.total_prompt_tokens / 1_000_000) * costs["input"]
        output_cost = (metrics.total_completion_tokens / 1_000_000) * costs["output"]

        return {
            "model": model,
            "input_tokens": metrics.total_prompt_tokens,
            "output_tokens": metrics.total_completion_tokens,
            "input_cost_usd": round(input_cost, 6),
            "output_cost_usd": round(output_cost, 6),
            "total_cost_usd": round(input_cost + output_cost, 6),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        metrics = self.get_run_metrics()

        return {
            "run_id": metrics.run_id,
            "company_name": metrics.company_name,
            "total_execution_time_ms": metrics.total_execution_time_ms,
            "total_prompt_tokens": metrics.total_prompt_tokens,
            "total_completion_tokens": metrics.total_completion_tokens,
            "total_tokens": metrics.total_tokens,
            "llm_calls": metrics.llm_calls,
            "tool_calls": metrics.tool_calls,
            "successful_steps": metrics.successful_steps,
            "failed_steps": metrics.failed_steps,
            "steps": [
                {
                    "step_name": s.step_name,
                    "execution_time_ms": s.execution_time_ms,
                    "prompt_tokens": s.prompt_tokens,
                    "completion_tokens": s.completion_tokens,
                    "total_tokens": s.total_tokens,
                    "success": s.success,
                    "error": s.error,
                    "metadata": s.metadata,
                }
                for s in metrics.steps
            ],
            "timestamp": datetime.utcnow().isoformat(),
        }
