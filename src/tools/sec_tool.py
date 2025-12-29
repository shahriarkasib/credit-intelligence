"""SEC EDGAR Tool - Fetch SEC financial filings."""

from typing import Any, Dict
from .base_tool import BaseTool
from data_sources import SECEdgarDataSource


class SECTool(BaseTool):
    """
    Tool for fetching SEC EDGAR financial data.

    Use this tool when:
    - Company is a US public company
    - Need official financial filings (10-K, 10-Q)
    - Need revenue, profit, debt information
    """

    def __init__(self):
        self._data_source = SECEdgarDataSource()
        super().__init__()

    def _get_name(self) -> str:
        return "fetch_sec_data"

    def _get_description(self) -> str:
        return (
            "Fetches SEC EDGAR financial filings for US public companies. "
            "Returns official financial data including revenue, net income, "
            "total assets, liabilities, and recent filings."
        )

    def _get_when_to_use(self) -> str:
        return (
            "Use this tool when:\n"
            "- The company is a US public company (traded on NYSE, NASDAQ, etc.)\n"
            "- You need official financial statements\n"
            "- You have a stock ticker symbol (like AAPL, GOOGL, MSFT)\n"
            "- You need historical financial data\n"
            "\n"
            "Do NOT use when:\n"
            "- Company is private\n"
            "- Company is not US-based\n"
            "- You only need news or general information"
        )

    def _execute(self, company_identifier: str, **kwargs) -> Dict[str, Any]:
        """
        Fetch SEC data for a company.

        Args:
            company_identifier: Ticker symbol or company name

        Returns:
            Dict with SEC filing data
        """
        result = self._data_source.get_company_data(company_identifier)

        if result.success:
            return {
                "source": "SEC EDGAR",
                "found": True,
                **result.data,
            }
        else:
            return {
                "source": "SEC EDGAR",
                "found": False,
                "error": result.error,
            }
