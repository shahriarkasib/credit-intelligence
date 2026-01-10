"""LangChain Tool Adapters - Wrap custom tools as LangChain Tools.

Provides adapter wrappers that allow our custom BaseTool implementations
to be used with LangChain Agents and chains, enabling:
- Structured tool calling via LangChain Agents
- Automatic callback integration (LangSmith traces, tool events)
- Easy integration with create_tool_calling_agent() and AgentExecutor

Usage:
    from tools.langchain_adapters import get_langchain_tools, wrap_tool_as_langchain

    # Get all tools as LangChain Tools
    lc_tools = get_langchain_tools()

    # Use with LangChain Agent
    agent = create_tool_calling_agent(llm, lc_tools, prompt)

    # Or wrap a single tool
    sec_tool = SECTool()
    lc_sec_tool = wrap_tool_as_langchain(sec_tool)
"""

import logging
from typing import Any, Dict, List, Optional, Type
from functools import wraps

logger = logging.getLogger(__name__)

# Try to import LangChain Tool base classes
try:
    from langchain_core.tools import BaseTool as LangChainBaseTool, tool
    from langchain_core.pydantic_v1 import BaseModel, Field
    LANGCHAIN_TOOLS_AVAILABLE = True
except ImportError:
    try:
        from langchain.tools import BaseTool as LangChainBaseTool, tool
        from pydantic import BaseModel, Field
        LANGCHAIN_TOOLS_AVAILABLE = True
    except ImportError:
        LANGCHAIN_TOOLS_AVAILABLE = False
        LangChainBaseTool = object
        logger.warning("LangChain tools not available. Run: pip install langchain-core")

# Import our custom tools
from .base_tool import BaseTool, ToolResult
from .sec_tool import SECTool
from .finnhub_tool import FinnhubTool
from .court_tool import CourtListenerTool
from .web_search_tool import WebSearchTool


class ToolAdapter(LangChainBaseTool):
    """
    LangChain Tool adapter that wraps our custom BaseTool implementations.

    This adapter provides:
    - LangChain Tool interface compatibility
    - Automatic callback handling via LangChain
    - Structured input/output for Agent tool calling
    """

    # LangChain tool attributes
    name: str = ""
    description: str = ""

    # Our wrapped tool
    _wrapped_tool: Optional[BaseTool] = None
    _run_id: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, tool: BaseTool, run_id: Optional[str] = None, **kwargs):
        """
        Initialize adapter.

        Args:
            tool: Our custom BaseTool instance to wrap
            run_id: Optional run ID for tracking
        """
        super().__init__(
            name=tool.name,
            description=f"{tool.description}\n\nWhen to use: {tool.when_to_use}",
            **kwargs
        )
        self._wrapped_tool = tool
        self._run_id = run_id

    def _run(self, **kwargs) -> str:
        """
        Execute the wrapped tool.

        LangChain Agent will call this with parsed arguments.
        """
        if not self._wrapped_tool:
            return '{"error": "No wrapped tool configured"}'

        result = self._wrapped_tool.execute(run_id=self._run_id, **kwargs)

        # Return JSON string for LLM to parse
        import json
        return json.dumps(result.to_dict(), default=str)

    async def _arun(self, **kwargs) -> str:
        """Async execution (falls back to sync for now)."""
        return self._run(**kwargs)


def wrap_tool_as_langchain(
    tool: BaseTool,
    run_id: Optional[str] = None,
) -> Optional[LangChainBaseTool]:
    """
    Wrap a custom BaseTool as a LangChain Tool.

    Args:
        tool: Our custom BaseTool instance
        run_id: Optional run ID for tracking

    Returns:
        LangChain Tool instance, or None if LangChain not available
    """
    if not LANGCHAIN_TOOLS_AVAILABLE:
        logger.warning("LangChain tools not available")
        return None

    return ToolAdapter(tool=tool, run_id=run_id)


def get_langchain_tools(run_id: Optional[str] = None) -> List[LangChainBaseTool]:
    """
    Get all available tools as LangChain Tools.

    Args:
        run_id: Optional run ID for tracking

    Returns:
        List of LangChain Tool instances
    """
    if not LANGCHAIN_TOOLS_AVAILABLE:
        logger.warning("LangChain tools not available")
        return []

    tools = [
        SECTool(),
        FinnhubTool(),
        CourtListenerTool(),
        WebSearchTool(),
    ]

    lc_tools = []
    for tool in tools:
        wrapped = wrap_tool_as_langchain(tool, run_id=run_id)
        if wrapped:
            lc_tools.append(wrapped)

    logger.info(f"Created {len(lc_tools)} LangChain tool adapters")
    return lc_tools


def is_langchain_tools_available() -> bool:
    """Check if LangChain tool adapters are available."""
    return LANGCHAIN_TOOLS_AVAILABLE


# =============================================================================
# Structured Input Schemas for Tools (for advanced Agent usage)
# =============================================================================

if LANGCHAIN_TOOLS_AVAILABLE:

    class SECToolInput(BaseModel):
        """Input schema for SEC EDGAR tool."""
        company_identifier: str = Field(
            description="Company name, ticker symbol, or CIK number"
        )

    class FinnhubToolInput(BaseModel):
        """Input schema for Finnhub market data tool."""
        ticker: str = Field(description="Stock ticker symbol (e.g., AAPL)")
        company_name: str = Field(description="Company name for reference")

    class CourtToolInput(BaseModel):
        """Input schema for CourtListener legal data tool."""
        company_name: str = Field(description="Company name to search for legal cases")

    class WebSearchToolInput(BaseModel):
        """Input schema for web search tool."""
        company_name: str = Field(description="Company name to search for")


class StructuredToolAdapter(LangChainBaseTool):
    """
    LangChain Tool adapter with structured input schema.

    Provides more reliable argument parsing by defining
    explicit input schemas for each tool.
    """

    name: str = ""
    description: str = ""
    args_schema: Optional[Type[BaseModel]] = None

    _wrapped_tool: Optional[BaseTool] = None
    _run_id: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        tool: BaseTool,
        args_schema: Optional[Type[BaseModel]] = None,
        run_id: Optional[str] = None,
        **kwargs
    ):
        """Initialize with explicit args schema."""
        super().__init__(
            name=tool.name,
            description=f"{tool.description}\n\nWhen to use: {tool.when_to_use}",
            args_schema=args_schema,
            **kwargs
        )
        self._wrapped_tool = tool
        self._run_id = run_id

    def _run(self, **kwargs) -> str:
        """Execute with validated arguments."""
        if not self._wrapped_tool:
            return '{"error": "No wrapped tool configured"}'

        result = self._wrapped_tool.execute(run_id=self._run_id, **kwargs)

        import json
        return json.dumps(result.to_dict(), default=str)

    async def _arun(self, **kwargs) -> str:
        """Async execution."""
        return self._run(**kwargs)


def get_structured_langchain_tools(run_id: Optional[str] = None) -> List[LangChainBaseTool]:
    """
    Get all tools as LangChain Tools with structured input schemas.

    Use this for more reliable argument parsing with LangChain Agents.

    Args:
        run_id: Optional run ID for tracking

    Returns:
        List of LangChain Tools with explicit schemas
    """
    if not LANGCHAIN_TOOLS_AVAILABLE:
        return []

    tools_with_schemas = [
        (SECTool(), SECToolInput),
        (FinnhubTool(), FinnhubToolInput),
        (CourtListenerTool(), CourtToolInput),
        (WebSearchTool(), WebSearchToolInput),
    ]

    lc_tools = []
    for tool, schema in tools_with_schemas:
        wrapped = StructuredToolAdapter(
            tool=tool,
            args_schema=schema,
            run_id=run_id,
        )
        lc_tools.append(wrapped)

    logger.info(f"Created {len(lc_tools)} structured LangChain tool adapters")
    return lc_tools
