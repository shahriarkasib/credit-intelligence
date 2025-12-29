"""Base Tool class with logging and metrics."""

import time
import uuid
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result from a tool execution with metrics."""

    # Execution info
    tool_name: str
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    # Metrics
    execution_time_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # Logging
    run_id: Optional[str] = None
    input_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "timestamp": self.timestamp,
            "run_id": self.run_id,
            "input_params": self.input_params,
        }

    def to_log_entry(self) -> Dict[str, Any]:
        """Create a log entry for MongoDB."""
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "execution_time_ms": self.execution_time_ms,
            "timestamp": self.timestamp,
            "run_id": self.run_id,
            "input_params": self.input_params,
            "error": self.error,
            "data_keys": list(self.data.keys()) if self.data else [],
            "data_size": len(str(self.data)) if self.data else 0,
        }


class BaseTool(ABC):
    """
    Base class for all tools.

    Each tool represents a capability that the LLM can choose to use.
    Tools automatically track execution time and log their usage.
    """

    def __init__(self):
        self.name = self._get_name()
        self.description = self._get_description()
        self.when_to_use = self._get_when_to_use()
        self._execution_count = 0
        self._total_execution_time = 0.0
        self._error_count = 0

    @abstractmethod
    def _get_name(self) -> str:
        """Return the tool name."""
        pass

    @abstractmethod
    def _get_description(self) -> str:
        """Return a description of what this tool does."""
        pass

    @abstractmethod
    def _get_when_to_use(self) -> str:
        """Return guidance on when the LLM should use this tool."""
        pass

    @abstractmethod
    def _execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool logic. Override in subclasses."""
        pass

    def execute(self, run_id: Optional[str] = None, **kwargs) -> ToolResult:
        """
        Execute the tool with logging and metrics.

        Args:
            run_id: Optional run ID for tracking
            **kwargs: Tool-specific parameters

        Returns:
            ToolResult with data and metrics
        """
        start_time = time.time()
        self._execution_count += 1

        logger.info(f"Executing tool: {self.name} with params: {kwargs}")

        try:
            data = self._execute(**kwargs)
            execution_time = (time.time() - start_time) * 1000  # ms
            self._total_execution_time += execution_time

            result = ToolResult(
                tool_name=self.name,
                success=True,
                data=data,
                execution_time_ms=round(execution_time, 2),
                run_id=run_id,
                input_params=kwargs,
            )

            logger.info(f"Tool {self.name} completed in {execution_time:.2f}ms")
            return result

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self._error_count += 1

            logger.error(f"Tool {self.name} failed: {e}")

            return ToolResult(
                tool_name=self.name,
                success=False,
                error=str(e),
                execution_time_ms=round(execution_time, 2),
                run_id=run_id,
                input_params=kwargs,
            )

    def get_tool_spec(self) -> Dict[str, Any]:
        """Get tool specification for LLM."""
        return {
            "name": self.name,
            "description": self.description,
            "when_to_use": self.when_to_use,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get tool execution statistics."""
        return {
            "name": self.name,
            "execution_count": self._execution_count,
            "total_execution_time_ms": round(self._total_execution_time, 2),
            "avg_execution_time_ms": round(
                self._total_execution_time / max(self._execution_count, 1), 2
            ),
            "error_count": self._error_count,
            "error_rate": round(
                self._error_count / max(self._execution_count, 1), 4
            ),
        }
