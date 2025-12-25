"""CourtListener Data Source - Federal and state court records."""

import logging
from typing import Any, Dict, List, Optional

from .base import BaseDataSource, DataSourceResult

logger = logging.getLogger(__name__)


class CourtListenerDataSource(BaseDataSource):
    """
    CourtListener API connector for court records and legal data.

    Provides access to:
    - Federal court opinions
    - PACER/RECAP docket data
    - Bankruptcy filings
    - Judgments and liens

    Rate Limit: 5000 requests/hour
    Authentication: API key required
    """

    def __init__(self, api_key: Optional[str] = None):
        import os
        api_key = api_key or os.getenv("COURTLISTENER_API_KEY")

        super().__init__(
            name="CourtListener",
            base_url="https://www.courtlistener.com/api/rest/v4",
            api_key=api_key,
            rate_limit=1.0,  # 1 request per second
        )

        # Add authorization header if API key provided
        if self.api_key:
            self.session.headers.update({
                "Authorization": f"Token {self.api_key}"
            })

    def search(self, query: str, **kwargs) -> DataSourceResult:
        """
        Search for court cases by party name or keyword.

        Args:
            query: Search query (company or person name)
            case_type: Type of search - 'opinions', 'dockets', 'recap'
        """
        case_type = kwargs.get("case_type", "dockets")

        if case_type == "opinions":
            return self._search_opinions(query, **kwargs)
        elif case_type == "recap":
            return self._search_recap(query, **kwargs)
        else:
            return self._search_dockets(query, **kwargs)

    def _search_dockets(self, query: str, **kwargs) -> DataSourceResult:
        """Search dockets/cases."""
        params = {
            "q": query,
            "order_by": "-date_filed",
            "page_size": kwargs.get("limit", 20),
        }

        response = self._make_request("dockets/", params=params)

        if not response:
            return DataSourceResult(
                source=self.name,
                query=query,
                data={},
                success=False,
                error="Docket search failed",
            )

        results = response.get("results", [])
        processed = [self._process_docket(d) for d in results]

        return DataSourceResult(
            source=self.name,
            query=query,
            data={
                "count": response.get("count", 0),
                "dockets": processed,
            },
            raw_response=response,
        )

    def _search_opinions(self, query: str, **kwargs) -> DataSourceResult:
        """Search court opinions."""
        params = {
            "q": query,
            "order_by": "-date_filed",
            "page_size": kwargs.get("limit", 20),
        }

        response = self._make_request("opinions/", params=params)

        if not response:
            return DataSourceResult(
                source=self.name,
                query=query,
                data={},
                success=False,
                error="Opinion search failed",
            )

        results = response.get("results", [])
        processed = [self._process_opinion(o) for o in results]

        return DataSourceResult(
            source=self.name,
            query=query,
            data={
                "count": response.get("count", 0),
                "opinions": processed,
            },
            raw_response=response,
        )

    def _search_recap(self, query: str, **kwargs) -> DataSourceResult:
        """Search RECAP archive (PACER data)."""
        params = {
            "q": query,
            "order_by": "-date_filed",
            "page_size": kwargs.get("limit", 20),
        }

        response = self._make_request("recap/", params=params)

        if not response:
            return DataSourceResult(
                source=self.name,
                query=query,
                data={},
                success=False,
                error="RECAP search failed",
            )

        results = response.get("results", [])

        return DataSourceResult(
            source=self.name,
            query=query,
            data={
                "count": response.get("count", 0),
                "results": results,
            },
            raw_response=response,
        )

    def get_company_data(self, identifier: str, **kwargs) -> DataSourceResult:
        """
        Get court records for a company.

        Args:
            identifier: Company name to search for
        """
        # Search dockets for the company
        docket_result = self._search_dockets(identifier, limit=10)

        # Search opinions
        opinion_result = self._search_opinions(identifier, limit=10)

        # Categorize cases
        cases = {
            "bankruptcy": [],
            "civil": [],
            "other": [],
        }

        for docket in docket_result.data.get("dockets", []):
            case_type = self._categorize_case(docket)
            cases[case_type].append(docket)

        # Summary statistics
        total_cases = docket_result.data.get("count", 0)
        total_opinions = opinion_result.data.get("count", 0)

        return DataSourceResult(
            source=self.name,
            query=identifier,
            data={
                "company": identifier,
                "total_dockets": total_cases,
                "total_opinions": total_opinions,
                "bankruptcy_cases": cases["bankruptcy"],
                "civil_cases": cases["civil"],
                "other_cases": cases["other"],
                "recent_opinions": opinion_result.data.get("opinions", [])[:5],
                "risk_indicators": self._calculate_risk_indicators(cases, total_cases),
            },
        )

    def _process_docket(self, docket: Dict) -> Dict[str, Any]:
        """Process docket data."""
        return {
            "case_name": docket.get("case_name", ""),
            "docket_number": docket.get("docket_number", ""),
            "court": docket.get("court", ""),
            "court_id": docket.get("court_id", ""),
            "date_filed": docket.get("date_filed"),
            "date_terminated": docket.get("date_terminated"),
            "nature_of_suit": docket.get("nature_of_suit", ""),
            "cause": docket.get("cause", ""),
            "jurisdiction_type": docket.get("jurisdiction_type", ""),
            "pacer_case_id": docket.get("pacer_case_id"),
        }

    def _process_opinion(self, opinion: Dict) -> Dict[str, Any]:
        """Process opinion data."""
        return {
            "case_name": opinion.get("case_name", ""),
            "court": opinion.get("court", ""),
            "date_filed": opinion.get("date_filed"),
            "status": opinion.get("status", ""),
            "citation": opinion.get("citation", []),
            "summary": opinion.get("summary", "")[:500] if opinion.get("summary") else "",
            "url": opinion.get("absolute_url", ""),
        }

    def _categorize_case(self, docket: Dict) -> str:
        """Categorize a case based on nature of suit or court."""
        nature = (docket.get("nature_of_suit") or "").lower()
        court = (docket.get("court") or "").lower()

        if "bankruptcy" in nature or "bankr" in court:
            return "bankruptcy"
        elif any(x in nature for x in ["contract", "civil", "tort", "property"]):
            return "civil"
        else:
            return "other"

    def _calculate_risk_indicators(self, cases: Dict, total: int) -> Dict[str, Any]:
        """Calculate risk indicators from court data."""
        bankruptcy_count = len(cases["bankruptcy"])
        civil_count = len(cases["civil"])

        risk_level = "low"
        if bankruptcy_count > 0:
            risk_level = "high"
        elif civil_count > 5:
            risk_level = "medium"
        elif total > 10:
            risk_level = "medium"

        return {
            "risk_level": risk_level,
            "has_bankruptcy": bankruptcy_count > 0,
            "civil_case_count": civil_count,
            "total_case_count": total,
            "notes": self._generate_risk_notes(cases, total),
        }

    def _generate_risk_notes(self, cases: Dict, total: int) -> List[str]:
        """Generate human-readable risk notes."""
        notes = []

        if len(cases["bankruptcy"]) > 0:
            notes.append(f"Company has {len(cases['bankruptcy'])} bankruptcy filing(s) on record")

        if len(cases["civil"]) > 5:
            notes.append(f"Company has significant civil litigation history ({len(cases['civil'])} cases)")

        if total == 0:
            notes.append("No court records found (may indicate limited US operations or clean record)")

        return notes

    def get_supported_fields(self) -> List[str]:
        return [
            "case_name",
            "docket_number",
            "court",
            "date_filed",
            "date_terminated",
            "nature_of_suit",
            "bankruptcy_cases",
            "civil_cases",
            "judgments",
            "risk_level",
        ]

    def health_check(self) -> bool:
        """Check if CourtListener is accessible."""
        response = self._make_request("courts/", params={"page_size": 1})
        return response is not None
