"""Finnhub Tool - Fetch stock market data."""

from typing import Any, Dict
from .base_tool import BaseTool
from data_sources import FinnhubDataSource


class FinnhubTool(BaseTool):
    """
    Tool for fetching stock market data from Finnhub.

    Use this tool when:
    - Need current stock price
    - Need market capitalization
    - Need company profile from market data
    """

    def __init__(self):
        self._data_source = FinnhubDataSource()
        super().__init__()

    def _get_name(self) -> str:
        return "fetch_market_data"

    def _get_description(self) -> str:
        return (
            "Fetches stock market data from Finnhub API. "
            "Returns current stock price, market cap, company profile, "
            "and key financial metrics for publicly traded companies."
        )

    def _get_when_to_use(self) -> str:
        return (
            "Use this tool when:\n"
            "- You need current stock price or market cap\n"
            "- The company is publicly traded\n"
            "- You need real-time or recent market data\n"
            "- You need company industry/sector classification\n"
            "\n"
            "Do NOT use when:\n"
            "- Company is private (not traded on stock exchange)\n"
            "- You only need historical financial filings (use SEC tool)\n"
            "- You need legal/court information"
        )

    def _execute(self, ticker: str, company_name: str = "", **kwargs) -> Dict[str, Any]:
        """
        Fetch market data for a company.

        Args:
            ticker: Stock ticker symbol
            company_name: Company name for verification

        Returns:
            Dict with market data
        """
        # Try ticker first
        identifiers = [ticker]
        if company_name:
            identifiers.extend([
                company_name.upper()[:4],
                company_name.upper()
            ])

        for identifier in identifiers:
            if not identifier:
                continue

            result = self._data_source.get_company_data(identifier)

            if result.success and result.data.get("profile"):
                # Verify company name if provided
                if company_name:
                    profile = result.data.get("profile", {})
                    found_name = profile.get("name", "").lower()
                    search_name = company_name.lower().replace(".com", "").replace(" inc", "")

                    if search_name in found_name or found_name in search_name:
                        return {
                            "source": "Finnhub",
                            "found": True,
                            "ticker": identifier,
                            **result.data,
                        }
                else:
                    return {
                        "source": "Finnhub",
                        "found": True,
                        "ticker": identifier,
                        **result.data,
                    }

        return {
            "source": "Finnhub",
            "found": False,
            "error": "Company not found on stock exchanges",
        }
