"""LangChain Tool Executor - Bridges LangChain tools with existing infrastructure.

Provides:
- Unified interface for both LangChain and legacy tool execution
- Tool call parsing from LLM responses
- Execution logging and metrics
- Compatibility layer for gradual migration
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Import LangChain components
try:
    from langchain_core.tools import StructuredTool
    from langchain_core.messages import AIMessage, ToolCall
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    StructuredTool = None
    AIMessage = None
    ToolCall = None
    logger.warning("langchain-core not available")

# Import existing tools infrastructure
from .base_tool import ToolResult
from .tool_executor import ToolExecutor, get_tool_executor
from .langchain_tools import (
    get_langchain_tool,
    get_all_langchain_tools,
    is_langchain_tools_available,
)


class LangChainToolExecutor:
    """
    Executor that bridges LangChain tools with existing infrastructure.

    Features:
    - Execute tools from LangChain StructuredTool or legacy BaseTool
    - Parse tool calls from LLM responses (AIMessage with tool_calls)
    - Maintain execution logs for evaluation
    - Support both synchronous execution patterns
    """

    def __init__(self, use_langchain: bool = True):
        """
        Initialize the LangChain tool executor.

        Args:
            use_langchain: If True, prefer LangChain tools. If False, use legacy.
        """
        self.use_langchain = use_langchain and is_langchain_tools_available()
        self._legacy_executor = get_tool_executor()
        self._langchain_tools: Dict[str, StructuredTool] = {}
        self.execution_log: List[Dict[str, Any]] = []

        # Load LangChain tools if available
        if self.use_langchain:
            for tool in get_all_langchain_tools():
                self._langchain_tools[tool.name] = tool
            logger.info(f"LangChainToolExecutor initialized with {len(self._langchain_tools)} tools")
        else:
            logger.info("LangChainToolExecutor using legacy tools only")

    def get_tools(self) -> List[StructuredTool]:
        """Get all LangChain tools for binding to LLM."""
        return list(self._langchain_tools.values())

    def get_tool_names(self) -> List[str]:
        """Get names of all available tools."""
        if self.use_langchain:
            return list(self._langchain_tools.keys())
        return list(self._legacy_executor.tools.keys())

    def execute_tool(
        self,
        tool_name: str,
        tool_input: Union[Dict[str, Any], BaseModel],
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute a single tool by name.

        Args:
            tool_name: Name of the tool to execute
            tool_input: Input parameters (dict or Pydantic model)
            run_id: Optional run ID for tracking

        Returns:
            Dict with tool result
        """
        run_id = run_id or str(uuid.uuid4())
        start_time = datetime.utcnow()

        # Convert Pydantic model to dict if needed
        if isinstance(tool_input, BaseModel):
            tool_input = tool_input.model_dump()

        logger.info(f"Executing tool: {tool_name} with input: {tool_input}")

        try:
            if self.use_langchain and tool_name in self._langchain_tools:
                # Use LangChain tool
                tool = self._langchain_tools[tool_name]
                result = tool.invoke(tool_input)

                # Result is already a dict from our wrapper
                if isinstance(result, dict):
                    output = result
                else:
                    output = {"result": result}

                success = output.get("success", True)
                error = output.get("error")

            else:
                # Fall back to legacy executor
                legacy_result = self._legacy_executor.execute_tool(
                    tool_name, run_id=run_id, **tool_input
                )
                output = legacy_result.to_dict()
                success = legacy_result.success
                error = legacy_result.error

            # Log execution
            execution_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            log_entry = {
                "run_id": run_id,
                "tool_name": tool_name,
                "input": tool_input,
                "success": success,
                "error": error,
                "execution_time_ms": round(execution_time_ms, 2),
                "timestamp": start_time.isoformat(),
                "executor": "langchain" if self.use_langchain else "legacy",
            }
            self.execution_log.append(log_entry)

            logger.info(f"Tool {tool_name} completed: success={success}, time={execution_time_ms:.0f}ms")

            return output

        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")
            execution_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            log_entry = {
                "run_id": run_id,
                "tool_name": tool_name,
                "input": tool_input,
                "success": False,
                "error": str(e),
                "execution_time_ms": round(execution_time_ms, 2),
                "timestamp": start_time.isoformat(),
                "executor": "langchain" if self.use_langchain else "legacy",
            }
            self.execution_log.append(log_entry)

            return {
                "tool_name": tool_name,
                "success": False,
                "error": str(e),
                "data": {},
            }

    def execute_tool_calls(
        self,
        tool_calls: List[Union[Dict[str, Any], "ToolCall"]],
        run_id: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Execute multiple tool calls (e.g., from an LLM response).

        Args:
            tool_calls: List of tool calls (dicts or LangChain ToolCall objects)
            run_id: Optional run ID for tracking

        Returns:
            Dict mapping tool call IDs to results
        """
        run_id = run_id or str(uuid.uuid4())
        results = {}

        for i, call in enumerate(tool_calls):
            # Parse tool call - handle both dict-like objects and typed dicts
            try:
                # Try dict-style access first (works for TypedDict and regular dicts)
                if hasattr(call, 'get') or isinstance(call, dict):
                    tool_name = call.get("name") or call.get("tool_name")
                    tool_input = call.get("args") or call.get("params") or call.get("input", {})
                    call_id = call.get("id") or f"call_{i}"
                elif hasattr(call, 'name'):
                    # Object with attributes
                    tool_name = call.name
                    tool_input = getattr(call, 'args', {})
                    call_id = getattr(call, 'id', f"call_{i}")
                else:
                    logger.warning(f"Unknown tool call format: {type(call)}")
                    continue
            except Exception as e:
                logger.warning(f"Failed to parse tool call: {e}")
                continue

            # Execute
            result = self.execute_tool(tool_name, tool_input, run_id=run_id)
            results[call_id] = result

        return results

    def execute_from_ai_message(
        self,
        message: "AIMessage",
        run_id: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Execute tool calls from an LLM's AIMessage response.

        Args:
            message: AIMessage from LLM with tool_calls
            run_id: Optional run ID for tracking

        Returns:
            Dict mapping tool call IDs to results

        Example:
            llm_with_tools = bind_tools_to_llm(llm)
            response = llm_with_tools.invoke("What is Apple's financials?")

            if response.tool_calls:
                results = executor.execute_from_ai_message(response)
        """
        if not LANGCHAIN_AVAILABLE:
            logger.error("LangChain not available")
            return {}

        if not hasattr(message, 'tool_calls') or not message.tool_calls:
            logger.info("No tool calls in message")
            return {}

        return self.execute_tool_calls(message.tool_calls, run_id=run_id)

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
            "avg_execution_time_ms": round(total_time / len(logs), 2) if logs else 0,
            "tools_used": list(set(l.get("tool_name") for l in logs)),
            "executor_type": "langchain" if self.use_langchain else "legacy",
        }

    def clear_log(self):
        """Clear execution log."""
        self.execution_log = []


# =============================================================================
# Singleton and Factory Functions
# =============================================================================

_langchain_executor: Optional[LangChainToolExecutor] = None


def get_langchain_tool_executor(use_langchain: bool = True) -> LangChainToolExecutor:
    """
    Get the global LangChainToolExecutor instance.

    Args:
        use_langchain: Whether to prefer LangChain tools

    Returns:
        LangChainToolExecutor instance
    """
    global _langchain_executor
    if _langchain_executor is None:
        _langchain_executor = LangChainToolExecutor(use_langchain=use_langchain)
    return _langchain_executor


def reset_langchain_executor():
    """Reset the global executor (useful for testing)."""
    global _langchain_executor
    _langchain_executor = None
