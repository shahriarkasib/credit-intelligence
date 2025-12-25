"""OpenSanctions Data Source - Sanctions, PEPs, and watchlists."""

import logging
from typing import Any, Dict, List, Optional

from .base import BaseDataSource, DataSourceResult

logger = logging.getLogger(__name__)


class OpenSanctionsDataSource(BaseDataSource):
    """
    OpenSanctions API connector for sanctions and compliance data.

    Provides access to:
    - Global sanctions lists
    - Politically Exposed Persons (PEPs)
    - Watchlists and debarment lists
    - Adverse media flags

    Rate Limit: Reasonable use
    Authentication: None required
    """

    # Dataset types
    DATASETS = {
        "default": "All sanctions and enforcement data",
        "sanctions": "International sanctions lists",
        "peps": "Politically exposed persons",
        "crime": "Criminal records and wanted lists",
        "debarment": "Debarment and exclusion lists",
    }

    def __init__(self):
        super().__init__(
            name="OpenSanctions",
            base_url="https://api.opensanctions.org",
            rate_limit=1.0,  # Conservative rate limit
        )

    def search(self, query: str, **kwargs) -> DataSourceResult:
        """
        Search for entities across all datasets.

        Args:
            query: Entity name to search
            schema: Entity type - 'Company', 'Person', 'LegalEntity'
            dataset: Specific dataset to search
        """
        schema = kwargs.get("schema", "LegalEntity")
        dataset = kwargs.get("dataset", "default")

        params = {
            "q": query,
            "schema": schema,
            "limit": kwargs.get("limit", 20),
        }

        response = self._make_request(f"search/{dataset}", params=params)

        if not response:
            return DataSourceResult(
                source=self.name,
                query=query,
                data={},
                success=False,
                error="Search failed",
            )

        results = response.get("results", [])
        processed = [self._process_entity(e) for e in results]

        return DataSourceResult(
            source=self.name,
            query=query,
            data={
                "total": response.get("total", 0),
                "query_text": response.get("query_text", query),
                "results": processed,
            },
            raw_response=response,
        )

    def match(self, name: str, **kwargs) -> DataSourceResult:
        """
        Match an entity against sanctions lists.

        Args:
            name: Entity name to match
            schema: 'Company' or 'Person'
            dataset: Dataset to match against
        """
        schema = kwargs.get("schema", "Company")
        dataset = kwargs.get("dataset", "default")

        # Build properties for matching
        properties = {"name": [name]}

        if kwargs.get("country"):
            properties["country"] = [kwargs["country"]]

        if kwargs.get("address"):
            properties["address"] = [kwargs["address"]]

        params = {
            "schema": schema,
        }

        # Add properties as query params
        for key, values in properties.items():
            params[f"properties.{key}"] = values[0]

        response = self._make_request(f"match/{dataset}", params=params)

        if not response:
            return DataSourceResult(
                source=self.name,
                query=name,
                data={},
                success=False,
                error="Match failed",
            )

        # Process responses
        matches = []
        for dataset_name, data in response.get("responses", {}).items():
            for result in data.get("results", []):
                result["matched_dataset"] = dataset_name
                matches.append(self._process_entity(result))

        # Determine overall match status
        has_match = len(matches) > 0
        highest_score = max([m.get("score", 0) for m in matches], default=0)

        return DataSourceResult(
            source=self.name,
            query=name,
            data={
                "entity_name": name,
                "has_sanctions_match": has_match,
                "highest_match_score": highest_score,
                "match_count": len(matches),
                "matches": matches[:10],  # Limit to top 10
                "risk_assessment": self._assess_risk(matches),
            },
            raw_response=response,
        )

    def get_company_data(self, identifier: str, **kwargs) -> DataSourceResult:
        """
        Check a company against all sanctions datasets.

        Args:
            identifier: Company name to check
        """
        # Perform comprehensive check across datasets
        results = {
            "sanctions": self.match(identifier, schema="Company", dataset="default"),
            "peps": self._check_peps(identifier),
        }

        # Aggregate results
        all_matches = []
        for check_type, result in results.items():
            if result.success and result.data.get("matches"):
                for match in result.data["matches"]:
                    match["check_type"] = check_type
                    all_matches.append(match)

        # Overall risk assessment
        risk = self._aggregate_risk(results)

        return DataSourceResult(
            source=self.name,
            query=identifier,
            data={
                "company": identifier,
                "sanctions_check": results["sanctions"].data if results["sanctions"].success else {},
                "pep_check": results["peps"].data if results["peps"].success else {},
                "all_matches": all_matches,
                "overall_risk": risk,
                "is_sanctioned": any(
                    r.data.get("has_sanctions_match", False)
                    for r in results.values()
                    if r.success
                ),
            },
        )

    def _check_peps(self, name: str) -> DataSourceResult:
        """Check for PEP associations."""
        # Search for PEPs related to this entity
        return self.search(name, schema="Person", dataset="peps")

    def _process_entity(self, entity: Dict) -> Dict[str, Any]:
        """Process entity data."""
        properties = entity.get("properties", {})

        return {
            "id": entity.get("id", ""),
            "schema": entity.get("schema", ""),
            "name": self._get_first(properties.get("name", [])),
            "aliases": properties.get("alias", []),
            "country": self._get_first(properties.get("country", [])),
            "birth_date": self._get_first(properties.get("birthDate", [])),
            "address": self._get_first(properties.get("address", [])),
            "datasets": entity.get("datasets", []),
            "score": entity.get("score", 0),
            "first_seen": entity.get("first_seen"),
            "last_seen": entity.get("last_seen"),
            "topics": properties.get("topics", []),
            "notes": properties.get("notes", []),
        }

    def _get_first(self, lst: List) -> Optional[str]:
        """Get first element of list or None."""
        return lst[0] if lst else None

    def _assess_risk(self, matches: List[Dict]) -> Dict[str, Any]:
        """Assess risk based on matches."""
        if not matches:
            return {
                "level": "low",
                "score": 0,
                "reasons": ["No sanctions matches found"],
            }

        highest_score = max([m.get("score", 0) for m in matches])
        datasets_hit = set()
        for m in matches:
            datasets_hit.update(m.get("datasets", []))

        # Determine risk level
        if highest_score >= 0.9 or "us_ofac_sdn" in datasets_hit:
            level = "critical"
        elif highest_score >= 0.7:
            level = "high"
        elif highest_score >= 0.5:
            level = "medium"
        else:
            level = "low"

        reasons = []
        if "us_ofac_sdn" in datasets_hit:
            reasons.append("Match on US OFAC SDN list")
        if "eu_fsf" in datasets_hit:
            reasons.append("Match on EU sanctions list")
        if "un_sc_sanctions" in datasets_hit:
            reasons.append("Match on UN Security Council sanctions")
        if any("pep" in d.lower() for d in datasets_hit):
            reasons.append("Politically Exposed Person (PEP) match")

        if not reasons:
            reasons.append(f"Potential match with score {highest_score:.2f}")

        return {
            "level": level,
            "score": highest_score,
            "datasets_matched": list(datasets_hit),
            "reasons": reasons,
        }

    def _aggregate_risk(self, results: Dict[str, DataSourceResult]) -> Dict[str, Any]:
        """Aggregate risk from multiple checks."""
        all_risks = []

        for check_type, result in results.items():
            if result.success and result.data.get("risk_assessment"):
                all_risks.append(result.data["risk_assessment"])

        if not all_risks:
            return {
                "level": "low",
                "score": 0,
                "summary": "No risk indicators found",
            }

        # Take highest risk level
        risk_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        highest = max(all_risks, key=lambda x: risk_order.get(x.get("level", "low"), 0))

        all_reasons = []
        for r in all_risks:
            all_reasons.extend(r.get("reasons", []))

        return {
            "level": highest.get("level", "low"),
            "score": max(r.get("score", 0) for r in all_risks),
            "reasons": list(set(all_reasons)),
            "summary": f"Risk level: {highest.get('level', 'low').upper()}",
        }

    def get_supported_fields(self) -> List[str]:
        return [
            "sanctions_status",
            "pep_status",
            "watchlist_status",
            "debarment_status",
            "risk_level",
            "datasets_matched",
            "sanctions_details",
        ]

    def health_check(self) -> bool:
        """Check if OpenSanctions is accessible."""
        # Try a simple search
        result = self.search("test", limit=1)
        return result.success
