"""Tool Executor - Manages and executes tools with logging."""

import uuid
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from .base_tool import BaseTool, ToolResult
from .sec_tool import SECTool
from .finnhub_tool import FinnhubTool
from .court_tool import CourtListenerTool
from .web_search_tool import WebSearchTool

logger = logging.getLogger(__name__)


class ToolExecutor:
    """
    Manages tool execution with comprehensive logging.

    Provides:
    - Tool registry and discovery
    - Execution with metrics
    - Logging for evaluation
    """

    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self.execution_log: List[Dict[str, Any]] = []
        self._init_tools()

    def _init_tools(self):
        """Initialize all available tools."""
        tools = [
            SECTool(),
            FinnhubTool(),
            CourtListenerTool(),
            WebSearchTool(),
        ]

        for tool in tools:
            self.tools[tool.name] = tool
            logger.info(f"Registered tool: {tool.name}")

    def get_tool_specs(self) -> List[Dict[str, Any]]:
        """Get specifications for all tools (for LLM)."""
        return [tool.get_tool_spec() for tool in self.tools.values()]

    def get_tool_specs_text(self) -> str:
        """Get tool specifications as formatted text for LLM prompt."""
        specs = []
        for i, tool in enumerate(self.tools.values(), 1):
            spec = tool.get_tool_spec()
            specs.append(
                f"{i}. **{spec['name']}**\n"
                f"   Description: {spec['description']}\n"
                f"   When to use:\n{spec['when_to_use']}\n"
            )
        return "\n".join(specs)

    def execute_tool(
        self,
        tool_name: str,
        run_id: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """
        Execute a specific tool.

        Args:
            tool_name: Name of the tool to execute
            run_id: Run ID for tracking
            **kwargs: Tool-specific parameters

        Returns:
            ToolResult with data and metrics
        """
        if tool_name not in self.tools:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"Unknown tool: {tool_name}",
                run_id=run_id,
                input_params=kwargs,
            )

        tool = self.tools[tool_name]
        result = tool.execute(run_id=run_id, **kwargs)

        # Log execution
        self.execution_log.append(result.to_log_entry())

        return result

    def execute_tools(
        self,
        tool_calls: List[Dict[str, Any]],
        run_id: Optional[str] = None,
    ) -> Dict[str, ToolResult]:
        """
        Execute multiple tools.

        Args:
            tool_calls: List of tool calls with name and params
                       [{"name": "fetch_sec_data", "params": {...}}, ...]
            run_id: Run ID for tracking

        Returns:
            Dict mapping tool names to results
        """
        run_id = run_id or str(uuid.uuid4())
        results = {}

        for call in tool_calls:
            tool_name = call.get("name")
            params = call.get("params", {})

            logger.info(f"Executing tool: {tool_name}")
            result = self.execute_tool(tool_name, run_id=run_id, **params)
            results[tool_name] = result

        return results

    def get_execution_summary(self, run_id: Optional[str] = None) -> Dict[str, Any]:
        """Get summary of tool executions."""
        logs = self.execution_log
        if run_id:
            logs = [l for l in logs if l.get("run_id") == run_id]

        if not logs:
            return {"total_executions": 0}

        total_time = sum(l.get("execution_time_ms", 0) for l in logs)
        success_count = sum(1 for l in logs if l.get("success"))
        error_count = len(logs) - success_count

        return {
            "total_executions": len(logs),
            "successful": success_count,
            "failed": error_count,
            "total_execution_time_ms": round(total_time, 2),
            "avg_execution_time_ms": round(total_time / len(logs), 2),
            "tools_used": list(set(l.get("tool_name") for l in logs)),
        }

    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all tools."""
        return {
            tool_name: tool.get_stats()
            for tool_name, tool in self.tools.items()
        }

    def clear_log(self):
        """Clear execution log."""
        self.execution_log = []


# Singleton instance
_executor: Optional[ToolExecutor] = None


def get_tool_executor() -> ToolExecutor:
    """Get the global ToolExecutor instance."""
    global _executor
    if _executor is None:
        _executor = ToolExecutor()
    return _executor
