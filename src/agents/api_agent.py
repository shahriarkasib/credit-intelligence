"""API Agent - Fetches structured data from external APIs."""

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

from data_sources import (
    SECEdgarDataSource,
    FinnhubDataSource,
    CourtListenerDataSource,
)

logger = logging.getLogger(__name__)


@dataclass
class APIAgentResult:
    """Result from API Agent data collection."""
    company: str
    sec_edgar: Dict[str, Any] = field(default_factory=dict)
    finnhub: Dict[str, Any] = field(default_factory=dict)
    court_listener: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "company": self.company,
            "sec_edgar": self.sec_edgar,
            "finnhub": self.finnhub,
            "court_listener": self.court_listener,
            "errors": self.errors,
        }

    def has_financial_data(self) -> bool:
        """Check if we have SEC EDGAR or Finnhub financial data."""
        return bool(self.sec_edgar.get("financials")) or bool(self.finnhub.get("profile"))

    def is_public_company(self) -> bool:
        """Check if company appears to be public."""
        return bool(self.sec_edgar.get("cik")) or bool(self.finnhub.get("profile"))

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of collected data."""
        return {
            "company": self.company,
            "has_sec_data": bool(self.sec_edgar) and not self.sec_edgar.get("error"),
            "has_market_data": bool(self.finnhub) and not self.finnhub.get("error"),
            "has_court_data": bool(self.court_listener) and not self.court_listener.get("error"),
            "is_public_company": self.is_public_company(),
            "error_count": len(self.errors),
        }


class APIAgent:
    """
    API Agent for fetching structured data from external APIs.

    Responsible for:
    - SEC EDGAR (US public company financials)
    - Finnhub (stock/market data)
    - CourtListener (court records)
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._init_data_sources()

    def _init_data_sources(self):
        """Initialize data source connectors."""
        self.sec_edgar = SECEdgarDataSource()
        self.finnhub = FinnhubDataSource()
        self.court_listener = CourtListenerDataSource()

    def fetch_all_data(
        self,
        company_name: str,
        ticker: Optional[str] = None,
        jurisdiction: Optional[str] = None,
        parallel: bool = True,
    ) -> APIAgentResult:
        """
        Fetch data from all sources for a company.

        Args:
            company_name: Name of the company
            ticker: Stock ticker symbol (optional, for US public companies)
            jurisdiction: Jurisdiction code (optional)
            parallel: Whether to fetch data in parallel

        Returns:
            APIAgentResult with data from all sources
        """
        if parallel:
            result = self._fetch_parallel(company_name, ticker)
        else:
            result = self._fetch_sequential(company_name, ticker)

        return result

    def _fetch_parallel(
        self,
        company_name: str,
        ticker: Optional[str],
    ) -> APIAgentResult:
        """Fetch data from all sources in parallel."""
        result = APIAgentResult(company=company_name)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}

            # Submit all fetch tasks
            if ticker:
                futures[executor.submit(self._fetch_sec_edgar, ticker)] = "sec_edgar"
                futures[executor.submit(self._fetch_finnhub, ticker, company_name)] = "finnhub"
            else:
                # Try to find ticker from company name
                futures[executor.submit(self._fetch_sec_edgar, company_name)] = "sec_edgar"
                futures[executor.submit(self._fetch_finnhub, None, company_name)] = "finnhub"

            futures[executor.submit(self._fetch_court_listener, company_name)] = "court_listener"

            # Collect results
            for future in as_completed(futures):
                source = futures[future]
                try:
                    data = future.result()
                    setattr(result, source, data)
                except Exception as e:
                    logger.error(f"Error fetching {source}: {e}")
                    result.errors.append(f"{source}: {str(e)}")

        return result

    def _fetch_sequential(
        self,
        company_name: str,
        ticker: Optional[str],
    ) -> APIAgentResult:
        """Fetch data from all sources sequentially."""
        result = APIAgentResult(company=company_name)

        # SEC EDGAR
        try:
            result.sec_edgar = self._fetch_sec_edgar(ticker or company_name)
        except Exception as e:
            logger.error(f"SEC EDGAR error: {e}")
            result.errors.append(f"sec_edgar: {str(e)}")

        # Finnhub
        try:
            result.finnhub = self._fetch_finnhub(ticker, company_name)
        except Exception as e:
            logger.error(f"Finnhub error: {e}")
            result.errors.append(f"finnhub: {str(e)}")

        # CourtListener
        try:
            result.court_listener = self._fetch_court_listener(company_name)
        except Exception as e:
            logger.error(f"CourtListener error: {e}")
            result.errors.append(f"court_listener: {str(e)}")

        return result

    def _fetch_sec_edgar(self, identifier: str) -> Dict[str, Any]:
        """Fetch SEC EDGAR data for a ticker or company name."""
        logger.info(f"Fetching SEC EDGAR data for {identifier}")
        result = self.sec_edgar.get_company_data(identifier)
        if result.success:
            return result.data
        return {"error": result.error}

    def _fetch_finnhub(self, ticker: Optional[str], company_name: str) -> Dict[str, Any]:
        """Fetch Finnhub data for a ticker."""
        # Try ticker first, then company name variations
        identifiers = []
        if ticker:
            identifiers.append(ticker)
        identifiers.extend([company_name.upper()[:4], company_name.upper()])

        for identifier in identifiers:
            if not identifier:
                continue
            logger.info(f"Fetching Finnhub data for {identifier}")
            result = self.finnhub.get_company_data(identifier)
            if result.success and result.data.get("profile"):
                # Verify company name matches
                profile = result.data.get("profile", {})
                found_name = profile.get("name", "").lower()
                search_name = company_name.lower().replace(".com", "").replace(" inc", "")

                if search_name in found_name or found_name in search_name:
                    return result.data
                else:
                    logger.info(f"Finnhub name mismatch: {found_name} vs {search_name}")

        return {"error": "Company not found on stock exchanges"}

    def _fetch_court_listener(self, company_name: str) -> Dict[str, Any]:
        """Fetch CourtListener data for a company."""
        logger.info(f"Fetching CourtListener data for {company_name}")
        result = self.court_listener.get_company_data(company_name)
        if result.success:
            return result.data
        return {"error": result.error}

    # Individual source methods for granular control
    def fetch_sec_edgar(self, ticker: str) -> Dict[str, Any]:
        """Public method to fetch only SEC EDGAR data."""
        return self._fetch_sec_edgar(ticker)

    def fetch_finnhub(self, ticker: str) -> Dict[str, Any]:
        """Public method to fetch only Finnhub data."""
        return self._fetch_finnhub(ticker, ticker)

    def fetch_court_records(self, company_name: str) -> Dict[str, Any]:
        """Public method to fetch only CourtListener data."""
        return self._fetch_court_listener(company_name)

    def health_check(self) -> Dict[str, bool]:
        """Check health of all data sources."""
        return {
            "sec_edgar": self.sec_edgar.health_check(),
            "finnhub": self.finnhub.health_check(),
            "court_listener": self.court_listener.health_check(),
        }
