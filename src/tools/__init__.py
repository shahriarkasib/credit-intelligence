"""Tool-based agents for Credit Intelligence."""

from .base_tool import BaseTool, ToolResult
from .tool_executor import ToolExecutor, get_tool_executor
from .sec_tool import SECTool
from .finnhub_tool import FinnhubTool
from .court_tool import CourtListenerTool
from .web_search_tool import WebSearchTool

# LangChain tool wrappers (Step 2)
from .langchain_tools import (
    get_langchain_tool,
    get_all_langchain_tools,
    bind_tools_to_llm,
    is_langchain_tools_available,
    # Pydantic schemas
    SECToolInput,
    FinnhubToolInput,
    CourtToolInput,
    WebSearchToolInput,
)
from .langchain_executor import (
    LangChainToolExecutor,
    get_langchain_tool_executor,
)

# LangChain Tool Adapters (for AgentExecutor integration)
from .langchain_adapters import (
    wrap_tool_as_langchain,
    get_langchain_tools,
    get_structured_langchain_tools,
    is_langchain_tools_available as is_adapters_available,
)

__all__ = [
    # Legacy tools
    "BaseTool",
    "ToolResult",
    "ToolExecutor",
    "get_tool_executor",
    "SECTool",
    "FinnhubTool",
    "CourtListenerTool",
    "WebSearchTool",
    # LangChain tools
    "get_langchain_tool",
    "get_all_langchain_tools",
    "bind_tools_to_llm",
    "is_langchain_tools_available",
    "LangChainToolExecutor",
    "get_langchain_tool_executor",
    # Input schemas
    "SECToolInput",
    "FinnhubToolInput",
    "CourtToolInput",
    "WebSearchToolInput",
    # Adapters
    "wrap_tool_as_langchain",
    "get_langchain_tools",
    "get_structured_langchain_tools",
    "is_adapters_available",
]
