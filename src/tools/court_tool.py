"""CourtListener Tool - Fetch legal/court records."""

from typing import Any, Dict
from .base_tool import BaseTool
from data_sources import CourtListenerDataSource


class CourtListenerTool(BaseTool):
    """
    Tool for fetching court records and legal information.

    Use this tool when:
    - Need to check for lawsuits
    - Need bankruptcy information
    - Need legal risk assessment
    """

    def __init__(self):
        self._data_source = CourtListenerDataSource()
        super().__init__()

    def _get_name(self) -> str:
        return "fetch_legal_data"

    def _get_description(self) -> str:
        return (
            "Searches CourtListener for legal records, lawsuits, and court cases "
            "involving a company. Returns information about any legal proceedings, "
            "bankruptcies, or regulatory actions."
        )

    def _get_when_to_use(self) -> str:
        return (
            "Use this tool when:\n"
            "- You need to check for any lawsuits against the company\n"
            "- You need bankruptcy or legal risk information\n"
            "- Assessing 'willingness to pay' or legal reliability\n"
            "- The company might have legal issues\n"
            "\n"
            "This tool works for BOTH public and private companies.\n"
            "\n"
            "Do NOT use when:\n"
            "- You only need financial data (use SEC or Finnhub)\n"
            "- You only need general company information"
        )

    def _execute(self, company_name: str, **kwargs) -> Dict[str, Any]:
        """
        Fetch legal records for a company.

        Args:
            company_name: Name of the company to search

        Returns:
            Dict with legal/court data
        """
        result = self._data_source.get_company_data(company_name)

        if result.success:
            return {
                "source": "CourtListener",
                "found": True,
                **result.data,
            }
        else:
            return {
                "source": "CourtListener",
                "found": False,
                "error": result.error,
            }
