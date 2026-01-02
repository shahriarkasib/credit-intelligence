"""LangChain Tool Wrappers for Credit Intelligence.

Provides StructuredTool wrappers around existing tools with:
- Pydantic input schemas for validation
- LangChain-compatible interface for agent patterns
- Automatic bridging to existing BaseTool implementations
"""

import logging
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Try to import LangChain tools
try:
    from langchain_core.tools import StructuredTool, BaseTool as LCBaseTool
    LANGCHAIN_TOOLS_AVAILABLE = True
except ImportError:
    LANGCHAIN_TOOLS_AVAILABLE = False
    StructuredTool = None
    LCBaseTool = None
    logger.warning("langchain-core not installed. LangChain tools not available.")

# Import existing tools
from .base_tool import BaseTool, ToolResult
from .sec_tool import SECTool
from .finnhub_tool import FinnhubTool
from .court_tool import CourtListenerTool
from .web_search_tool import WebSearchTool


# =============================================================================
# Pydantic Input Schemas
# =============================================================================

class SECToolInput(BaseModel):
    """Input schema for SEC EDGAR tool."""
    company_identifier: str = Field(
        description="Stock ticker symbol (e.g., AAPL, GOOGL) or company name"
    )


class FinnhubToolInput(BaseModel):
    """Input schema for Finnhub market data tool."""
    ticker: str = Field(
        description="Stock ticker symbol (e.g., AAPL, MSFT, TSLA)"
    )
    company_name: Optional[str] = Field(
        default="",
        description="Company name for verification (optional)"
    )


class CourtToolInput(BaseModel):
    """Input schema for CourtListener legal data tool."""
    company_name: str = Field(
        description="Name of the company to search for legal records"
    )


class WebSearchToolInput(BaseModel):
    """Input schema for web search tool."""
    company_name: str = Field(
        description="Name of the company to search for"
    )
    search_type: Optional[str] = Field(
        default="general",
        description="Type of search: 'general', 'news', or 'both'"
    )


# =============================================================================
# LangChain Tool Wrappers
# =============================================================================

def _create_sec_tool() -> Optional["StructuredTool"]:
    """Create LangChain StructuredTool for SEC EDGAR data."""
    if not LANGCHAIN_TOOLS_AVAILABLE:
        return None

    sec_tool = SECTool()

    def fetch_sec_data(company_identifier: str) -> Dict[str, Any]:
        """Fetch SEC EDGAR financial filings for US public companies."""
        result = sec_tool.execute(company_identifier=company_identifier)
        return result.to_dict()

    return StructuredTool.from_function(
        func=fetch_sec_data,
        name="fetch_sec_data",
        description=(
            "Fetches SEC EDGAR financial filings for US public companies. "
            "Returns official financial data including revenue, net income, "
            "total assets, liabilities, and recent filings. "
            "Use for US public companies with stock tickers."
        ),
        args_schema=SECToolInput,
        return_direct=False,
    )


def _create_finnhub_tool() -> Optional["StructuredTool"]:
    """Create LangChain StructuredTool for Finnhub market data."""
    if not LANGCHAIN_TOOLS_AVAILABLE:
        return None

    finnhub_tool = FinnhubTool()

    def fetch_market_data(ticker: str, company_name: str = "") -> Dict[str, Any]:
        """Fetch stock market data from Finnhub API."""
        result = finnhub_tool.execute(ticker=ticker, company_name=company_name)
        return result.to_dict()

    return StructuredTool.from_function(
        func=fetch_market_data,
        name="fetch_market_data",
        description=(
            "Fetches stock market data from Finnhub API. "
            "Returns current stock price, market cap, company profile, "
            "and key financial metrics for publicly traded companies. "
            "Requires a stock ticker symbol."
        ),
        args_schema=FinnhubToolInput,
        return_direct=False,
    )


def _create_court_tool() -> Optional["StructuredTool"]:
    """Create LangChain StructuredTool for CourtListener legal data."""
    if not LANGCHAIN_TOOLS_AVAILABLE:
        return None

    court_tool = CourtListenerTool()

    def fetch_legal_data(company_name: str) -> Dict[str, Any]:
        """Search CourtListener for legal records and court cases."""
        result = court_tool.execute(company_name=company_name)
        return result.to_dict()

    return StructuredTool.from_function(
        func=fetch_legal_data,
        name="fetch_legal_data",
        description=(
            "Searches CourtListener for legal records, lawsuits, and court cases "
            "involving a company. Returns information about legal proceedings, "
            "bankruptcies, or regulatory actions. Works for both public and private companies."
        ),
        args_schema=CourtToolInput,
        return_direct=False,
    )


def _create_web_search_tool() -> Optional["StructuredTool"]:
    """Create LangChain StructuredTool for web search."""
    if not LANGCHAIN_TOOLS_AVAILABLE:
        return None

    web_tool = WebSearchTool()

    def web_search(company_name: str, search_type: str = "general") -> Dict[str, Any]:
        """Search the web for company information and news."""
        result = web_tool.execute(company_name=company_name, search_type=search_type)
        return result.to_dict()

    return StructuredTool.from_function(
        func=web_search,
        name="web_search",
        description=(
            "Searches the web using DuckDuckGo for company information and news. "
            "Useful for private companies or when you need recent news and public sentiment. "
            "Good fallback when SEC/Finnhub don't find the company."
        ),
        args_schema=WebSearchToolInput,
        return_direct=False,
    )


# =============================================================================
# Tool Registry and Factory Functions
# =============================================================================

_langchain_tools_cache: Dict[str, "StructuredTool"] = {}


def get_langchain_tool(tool_name: str) -> Optional["StructuredTool"]:
    """
    Get a specific LangChain tool by name.

    Args:
        tool_name: Name of the tool (fetch_sec_data, fetch_market_data, etc.)

    Returns:
        StructuredTool instance or None if not available
    """
    if not LANGCHAIN_TOOLS_AVAILABLE:
        logger.warning("LangChain tools not available")
        return None

    # Check cache
    if tool_name in _langchain_tools_cache:
        return _langchain_tools_cache[tool_name]

    # Create tool
    tool_creators = {
        "fetch_sec_data": _create_sec_tool,
        "fetch_market_data": _create_finnhub_tool,
        "fetch_legal_data": _create_court_tool,
        "web_search": _create_web_search_tool,
    }

    creator = tool_creators.get(tool_name)
    if creator:
        tool = creator()
        if tool:
            _langchain_tools_cache[tool_name] = tool
            return tool

    logger.warning(f"Unknown tool: {tool_name}")
    return None


def get_all_langchain_tools() -> List["StructuredTool"]:
    """
    Get all available LangChain tools.

    Returns:
        List of StructuredTool instances
    """
    if not LANGCHAIN_TOOLS_AVAILABLE:
        logger.warning("LangChain tools not available")
        return []

    tool_names = ["fetch_sec_data", "fetch_market_data", "fetch_legal_data", "web_search"]
    tools = []

    for name in tool_names:
        tool = get_langchain_tool(name)
        if tool:
            tools.append(tool)

    logger.info(f"Loaded {len(tools)} LangChain tools: {[t.name for t in tools]}")
    return tools


def get_tool_input_schema(tool_name: str) -> Optional[Type[BaseModel]]:
    """
    Get the Pydantic input schema for a tool.

    Args:
        tool_name: Name of the tool

    Returns:
        Pydantic BaseModel class or None
    """
    schemas = {
        "fetch_sec_data": SECToolInput,
        "fetch_market_data": FinnhubToolInput,
        "fetch_legal_data": CourtToolInput,
        "web_search": WebSearchToolInput,
    }
    return schemas.get(tool_name)


def is_langchain_tools_available() -> bool:
    """Check if LangChain tools are available."""
    return LANGCHAIN_TOOLS_AVAILABLE


# =============================================================================
# Tool Binding Helper (for LLMs with tool calling)
# =============================================================================

def bind_tools_to_llm(llm: Any, tool_names: Optional[List[str]] = None) -> Any:
    """
    Bind tools to an LLM for tool calling.

    Args:
        llm: LangChain LLM instance (e.g., ChatGroq)
        tool_names: Optional list of tool names to bind (defaults to all)

    Returns:
        LLM with tools bound

    Example:
        from config.langchain_llm import get_chat_groq
        from tools.langchain_tools import bind_tools_to_llm

        llm = get_chat_groq("primary")
        llm_with_tools = bind_tools_to_llm(llm)

        # Now LLM can call tools
        response = llm_with_tools.invoke("What is Apple's stock price?")
    """
    if not LANGCHAIN_TOOLS_AVAILABLE:
        logger.error("LangChain tools not available")
        return llm

    if tool_names:
        tools = [get_langchain_tool(name) for name in tool_names]
        tools = [t for t in tools if t is not None]
    else:
        tools = get_all_langchain_tools()

    if not tools:
        logger.warning("No tools to bind")
        return llm

    try:
        return llm.bind_tools(tools)
    except Exception as e:
        logger.error(f"Failed to bind tools to LLM: {e}")
        return llm
