"""SEC EDGAR Data Source - Free access to US public company filings."""

import os
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from .base import BaseDataSource, DataSourceResult

logger = logging.getLogger(__name__)


class SECEdgarDataSource(BaseDataSource):
    """
    SEC EDGAR API connector for public company financial data.

    Provides access to:
    - Company filings (10-K, 10-Q, 8-K)
    - Financial facts (revenue, assets, liabilities)
    - Company information

    Rate Limit: 10 requests/second
    Authentication: None (User-Agent required)
    """

    def __init__(self, user_agent: Optional[str] = None):
        user_agent = user_agent or os.getenv(
            "SEC_EDGAR_USER_AGENT",
            "CreditIntelligence Demo contact@example.com"
        )
        super().__init__(
            name="SEC EDGAR",
            base_url="https://data.sec.gov",
            rate_limit=10.0,  # 10 requests per second
        )
        self.session.headers.update({"User-Agent": user_agent})

    def search(self, query: str, **kwargs) -> DataSourceResult:
        """Search for companies by name."""
        # SEC EDGAR full-text search endpoint
        params = {
            "q": query,
            "dateRange": "custom",
            "startdt": "2020-01-01",
            "enddt": datetime.now().strftime("%Y-%m-%d"),
        }

        response = self._make_request(
            "cgi-bin/srch-ia",
            params=params,
        )

        if not response:
            return DataSourceResult(
                source=self.name,
                query=query,
                data={},
                success=False,
                error="Search failed",
            )

        return DataSourceResult(
            source=self.name,
            query=query,
            data={"results": response},
            raw_response=response,
        )

    def get_company_data(self, identifier: str, **kwargs) -> DataSourceResult:
        """
        Get company data by CIK or ticker symbol.

        Args:
            identifier: CIK number (with or without leading zeros) or ticker symbol
        """
        cik = self._resolve_cik(identifier)
        if not cik:
            return DataSourceResult(
                source=self.name,
                query=identifier,
                data={},
                success=False,
                error=f"Could not resolve CIK for {identifier}",
            )

        # Get company submissions (filings list)
        submissions = self._get_submissions(cik)
        if not submissions:
            return DataSourceResult(
                source=self.name,
                query=identifier,
                data={},
                success=False,
                error="Failed to fetch company submissions",
            )

        # Get company facts (financial data)
        facts = self._get_company_facts(cik)

        # Extract key financial metrics
        financial_data = self._extract_financial_metrics(facts) if facts else {}

        return DataSourceResult(
            source=self.name,
            query=identifier,
            data={
                "cik": cik,
                "company_name": submissions.get("name", ""),
                "sic": submissions.get("sic", ""),
                "sic_description": submissions.get("sicDescription", ""),
                "fiscal_year_end": submissions.get("fiscalYearEnd", ""),
                "state_of_incorporation": submissions.get("stateOfIncorporation", ""),
                "recent_filings": self._get_recent_filings(submissions),
                "financials": financial_data,
            },
            raw_response={"submissions": submissions, "facts": facts},
        )

    def _resolve_cik(self, identifier: str) -> Optional[str]:
        """Resolve ticker symbol or CIK to padded CIK."""
        # If it's already a CIK (numeric), just pad it
        if identifier.isdigit():
            return identifier.zfill(10)

        # Try to look up ticker in company tickers file
        # Note: company_tickers.json is on www.sec.gov, not data.sec.gov
        try:
            import requests
            headers = {"User-Agent": self.session.headers.get("User-Agent", "CreditIntelligence/1.0")}
            resp = requests.get("https://www.sec.gov/files/company_tickers.json", headers=headers, timeout=30)
            resp.raise_for_status()
            response = resp.json()
        except Exception:
            response = None

        if response:
            for company in response.values():
                if company.get("ticker", "").upper() == identifier.upper():
                    return str(company.get("cik_str", "")).zfill(10)

        return None

    def _get_submissions(self, cik: str) -> Optional[Dict]:
        """Get company submissions (filing history)."""
        return self._make_request(f"submissions/CIK{cik}.json")

    def _get_company_facts(self, cik: str) -> Optional[Dict]:
        """Get company financial facts from XBRL data."""
        return self._make_request(f"api/xbrl/companyfacts/CIK{cik}.json")

    def _get_recent_filings(self, submissions: Dict, limit: int = 5) -> List[Dict]:
        """Extract recent filings from submissions."""
        recent = submissions.get("filings", {}).get("recent", {})
        filings = []

        forms = recent.get("form", [])[:limit]
        dates = recent.get("filingDate", [])[:limit]
        accessions = recent.get("accessionNumber", [])[:limit]

        for i in range(len(forms)):
            filings.append({
                "form": forms[i] if i < len(forms) else "",
                "filing_date": dates[i] if i < len(dates) else "",
                "accession_number": accessions[i] if i < len(accessions) else "",
            })

        return filings

    def _extract_financial_metrics(self, facts: Dict) -> Dict[str, Any]:
        """Extract key financial metrics from company facts."""
        metrics = {}

        if not facts:
            return metrics

        us_gaap = facts.get("facts", {}).get("us-gaap", {})

        # Define metrics to extract (XBRL tag -> friendly name)
        metric_mappings = {
            "Revenues": "revenue",
            "RevenueFromContractWithCustomerExcludingAssessedTax": "revenue",
            "NetIncomeLoss": "net_income",
            "Assets": "total_assets",
            "Liabilities": "total_liabilities",
            "CashAndCashEquivalentsAtCarryingValue": "cash_and_equivalents",
            "NetCashProvidedByUsedInOperatingActivities": "operating_cash_flow",
            "StockholdersEquity": "stockholders_equity",
        }

        for xbrl_tag, friendly_name in metric_mappings.items():
            if friendly_name in metrics:
                continue  # Skip if already set by a previous tag

            if xbrl_tag in us_gaap:
                units = us_gaap[xbrl_tag].get("units", {})
                usd_values = units.get("USD", [])

                if usd_values:
                    # Get most recent annual value (10-K filing)
                    annual_values = [
                        v for v in usd_values
                        if v.get("form") == "10-K" and "end" in v
                    ]
                    if annual_values:
                        # Sort by end date and get most recent
                        annual_values.sort(key=lambda x: x.get("end", ""), reverse=True)
                        metrics[friendly_name] = {
                            "value": annual_values[0].get("val"),
                            "period_end": annual_values[0].get("end"),
                            "form": annual_values[0].get("form"),
                        }

        # Calculate derived metrics
        if "total_assets" in metrics and "total_liabilities" in metrics:
            assets = metrics["total_assets"]["value"]
            liabilities = metrics["total_liabilities"]["value"]
            if assets and liabilities and assets > 0:
                metrics["debt_to_assets_ratio"] = {
                    "value": round(liabilities / assets, 4),
                    "calculated": True,
                }

        return metrics

    def get_supported_fields(self) -> List[str]:
        return [
            "company_name",
            "cik",
            "sic",
            "sic_description",
            "fiscal_year_end",
            "state_of_incorporation",
            "recent_filings",
            "revenue",
            "net_income",
            "total_assets",
            "total_liabilities",
            "cash_and_equivalents",
            "operating_cash_flow",
            "stockholders_equity",
            "debt_to_assets_ratio",
        ]

    def health_check(self) -> bool:
        """Check if SEC EDGAR is accessible."""
        try:
            import requests
            headers = {"User-Agent": self.session.headers.get("User-Agent", "CreditIntelligence/1.0")}
            resp = requests.get("https://www.sec.gov/files/company_tickers.json", headers=headers, timeout=10)
            return resp.status_code == 200
        except Exception:
            return False
