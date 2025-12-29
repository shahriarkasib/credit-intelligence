"""Web Search Tool - Search the web for company information."""

from typing import Any, Dict
from .base_tool import BaseTool
from data_sources import WebSearchDataSource


class WebSearchTool(BaseTool):
    """
    Tool for searching the web for company information.

    Use this tool when:
    - Company is private (not in SEC/Finnhub)
    - Need recent news articles
    - Need general company information
    - Need public sentiment/reputation
    """

    def __init__(self):
        self._data_source = WebSearchDataSource()
        super().__init__()

    def _get_name(self) -> str:
        return "web_search"

    def _get_description(self) -> str:
        return (
            "Searches the web using DuckDuckGo for company information, "
            "news articles, and general information. Useful for private companies "
            "or when you need recent news and public sentiment."
        )

    def _get_when_to_use(self) -> str:
        return (
            "Use this tool when:\n"
            "- Company is PRIVATE (not publicly traded)\n"
            "- You need recent news articles about the company\n"
            "- You need general company information not in official filings\n"
            "- Other tools (SEC, Finnhub) didn't find the company\n"
            "- You need to understand public sentiment/reputation\n"
            "\n"
            "This tool is a good FALLBACK when other tools fail.\n"
            "\n"
            "Do NOT use when:\n"
            "- You need official financial statements (use SEC)\n"
            "- You need real-time stock prices (use Finnhub)\n"
            "- You need legal records (use CourtListener)"
        )

    def _execute(self, company_name: str, search_type: str = "general", **kwargs) -> Dict[str, Any]:
        """
        Search the web for company information.

        Args:
            company_name: Name of the company to search
            search_type: Type of search - "general", "news", or "both"

        Returns:
            Dict with web search results
        """
        result = self._data_source.get_company_data(company_name)

        if result.success:
            data = result.data

            # Extract key information
            web_results = data.get("web_results", [])
            news = data.get("news", [])

            return {
                "source": "Web Search",
                "found": True,
                "web_results_count": len(web_results),
                "news_count": len(news),
                "web_results": web_results[:5],  # Top 5 results
                "news": news[:5],  # Top 5 news
                "sentiment": data.get("sentiment", {}),
            }
        else:
            return {
                "source": "Web Search",
                "found": False,
                "error": result.error,
            }
