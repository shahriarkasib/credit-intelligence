"""OpenCorporates Data Source - Global company registry data."""

import os
import logging
from typing import Any, Dict, List, Optional

from .base import BaseDataSource, DataSourceResult

logger = logging.getLogger(__name__)


class OpenCorporatesDataSource(BaseDataSource):
    """
    OpenCorporates API connector for global company registry data.

    Provides access to:
    - Company search across 140+ jurisdictions
    - Company details (name, status, registration, directors)
    - Officer/director information

    Rate Limit: 500 requests/month (free tier), more with API key
    Authentication: API key optional
    """

    # Common jurisdiction codes
    JURISDICTIONS = {
        "US": "us",
        "US-CA": "us_ca",
        "US-DE": "us_de",
        "US-NY": "us_ny",
        "GB": "gb",
        "DE": "de",
        "FR": "fr",
        "CA": "ca",
    }

    def __init__(self, api_key: Optional[str] = None):
        api_key = api_key or os.getenv("OPENCORPORATES_API_KEY")
        super().__init__(
            name="OpenCorporates",
            base_url="https://api.opencorporates.com/v0.4",
            api_key=api_key,
            rate_limit=0.5,  # Conservative for free tier
        )

    def _add_auth(self, params: Dict) -> Dict:
        """Add API key to params if available."""
        if self.api_key:
            params["api_token"] = self.api_key
        return params

    def search(self, query: str, jurisdiction: Optional[str] = None, **kwargs) -> DataSourceResult:
        """
        Search for companies by name.

        Args:
            query: Company name to search
            jurisdiction: Optional jurisdiction code (e.g., 'us_ca', 'gb')
        """
        params = self._add_auth({
            "q": query,
            "per_page": kwargs.get("per_page", 10),
        })

        if jurisdiction:
            params["jurisdiction_code"] = jurisdiction

        response = self._make_request("companies/search", params=params)

        if not response:
            return DataSourceResult(
                source=self.name,
                query=query,
                data={},
                success=False,
                error="Search failed",
            )

        companies = response.get("results", {}).get("companies", [])
        processed_companies = [
            self._process_company(c.get("company", {}))
            for c in companies
        ]

        return DataSourceResult(
            source=self.name,
            query=query,
            data={
                "total_count": response.get("results", {}).get("total_count", 0),
                "companies": processed_companies,
            },
            raw_response=response,
        )

    def get_company_data(
        self,
        identifier: str,
        jurisdiction: Optional[str] = None,
        **kwargs
    ) -> DataSourceResult:
        """
        Get detailed company data.

        Args:
            identifier: Company number or search query
            jurisdiction: Jurisdiction code (required for direct lookup)
        """
        # If jurisdiction provided, try direct lookup
        if jurisdiction:
            response = self._make_request(
                f"companies/{jurisdiction}/{identifier}",
                params=self._add_auth({}),
            )
            if response and "company" in response.get("results", {}):
                company = response["results"]["company"]
                return DataSourceResult(
                    source=self.name,
                    query=identifier,
                    data=self._process_company_detailed(company),
                    raw_response=response,
                )

        # Otherwise, search and get first result
        search_result = self.search(identifier, jurisdiction)
        if not search_result.success or not search_result.data.get("companies"):
            return DataSourceResult(
                source=self.name,
                query=identifier,
                data={},
                success=False,
                error="Company not found",
            )

        # Get detailed data for first match
        first_company = search_result.data["companies"][0]
        if first_company.get("jurisdiction_code") and first_company.get("company_number"):
            return self.get_company_data(
                first_company["company_number"],
                first_company["jurisdiction_code"],
            )

        return DataSourceResult(
            source=self.name,
            query=identifier,
            data=first_company,
        )

    def search_officers(self, name: str, **kwargs) -> DataSourceResult:
        """Search for officers/directors by name."""
        params = self._add_auth({
            "q": name,
            "per_page": kwargs.get("per_page", 10),
        })

        response = self._make_request("officers/search", params=params)

        if not response:
            return DataSourceResult(
                source=self.name,
                query=name,
                data={},
                success=False,
                error="Officer search failed",
            )

        officers = response.get("results", {}).get("officers", [])
        processed_officers = [
            self._process_officer(o.get("officer", {}))
            for o in officers
        ]

        return DataSourceResult(
            source=self.name,
            query=name,
            data={
                "total_count": response.get("results", {}).get("total_count", 0),
                "officers": processed_officers,
            },
            raw_response=response,
        )

    def _process_company(self, company: Dict) -> Dict[str, Any]:
        """Process basic company data."""
        return {
            "company_name": company.get("name", ""),
            "company_number": company.get("company_number", ""),
            "jurisdiction_code": company.get("jurisdiction_code", ""),
            "incorporation_date": company.get("incorporation_date"),
            "dissolution_date": company.get("dissolution_date"),
            "company_type": company.get("company_type", ""),
            "current_status": company.get("current_status", ""),
            "registered_address": self._format_address(
                company.get("registered_address_in_full")
            ),
        }

    def _process_company_detailed(self, company: Dict) -> Dict[str, Any]:
        """Process detailed company data including officers."""
        basic = self._process_company(company)

        # Add additional details
        basic.update({
            "previous_names": [
                {
                    "name": pn.get("company_name", ""),
                    "start_date": pn.get("start_date"),
                    "end_date": pn.get("end_date"),
                }
                for pn in company.get("previous_names", [])
            ],
            "officers": [
                self._process_officer(o.get("officer", {}))
                for o in company.get("officers", [])
            ],
            "industry_codes": company.get("industry_codes", []),
            "registry_url": company.get("registry_url", ""),
            "opencorporates_url": company.get("opencorporates_url", ""),
        })

        return basic

    def _process_officer(self, officer: Dict) -> Dict[str, Any]:
        """Process officer data."""
        return {
            "name": officer.get("name", ""),
            "position": officer.get("position", ""),
            "start_date": officer.get("start_date"),
            "end_date": officer.get("end_date"),
            "nationality": officer.get("nationality"),
            "occupation": officer.get("occupation"),
        }

    def _format_address(self, address: Optional[str]) -> str:
        """Clean up address string."""
        if not address:
            return ""
        # Remove excess whitespace and newlines
        return " ".join(address.split())

    def get_supported_fields(self) -> List[str]:
        return [
            "company_name",
            "company_number",
            "jurisdiction_code",
            "incorporation_date",
            "dissolution_date",
            "company_type",
            "current_status",
            "registered_address",
            "previous_names",
            "officers",
            "industry_codes",
        ]

    def health_check(self) -> bool:
        """Check if OpenCorporates is accessible."""
        # Search for a known company
        response = self._make_request(
            "companies/search",
            params=self._add_auth({"q": "apple", "per_page": 1}),
        )
        return response is not None and "results" in response
