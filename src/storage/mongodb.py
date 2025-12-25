"""MongoDB Storage for Credit Intelligence Data."""

import os
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)

try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, OperationFailure
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    logger.warning("pymongo not installed. Run: pip install pymongo")


class CreditIntelligenceDB:
    """
    MongoDB storage for credit intelligence data.

    Collections:
    - companies: Company profiles and metadata
    - assessments: Credit assessments (LLM decisions)
    - raw_data: Raw API responses for auditing
    - evaluations: Consistency evaluation results
    """

    def __init__(self, connection_string: Optional[str] = None):
        """
        Initialize MongoDB connection.

        Args:
            connection_string: MongoDB connection string (or use MONGODB_URI env var)
        """
        self.connection_string = connection_string or os.getenv("MONGODB_URI")
        self.client = None
        self.db = None

        if not MONGODB_AVAILABLE:
            logger.error("pymongo not installed")
            return

        if not self.connection_string:
            logger.warning("No MongoDB connection string provided")
            return

        try:
            self.client = MongoClient(self.connection_string)
            # Test connection
            self.client.admin.command('ping')
            self.db = self.client.credit_intelligence
            logger.info("Connected to MongoDB successfully")
        except ConnectionFailure as e:
            logger.error(f"MongoDB connection failed: {e}")
        except Exception as e:
            logger.error(f"MongoDB error: {e}")

    def is_connected(self) -> bool:
        """Check if connected to MongoDB."""
        return self.db is not None

    # ==================== COMPANY OPERATIONS ====================

    def save_company(self, company_data: Dict[str, Any]) -> Optional[str]:
        """
        Save or update company profile.

        Args:
            company_data: Company information dict

        Returns:
            Inserted/updated document ID
        """
        if not self.is_connected():
            return None

        company_name = company_data.get("company_name") or company_data.get("company")
        if not company_name:
            logger.error("Company name required")
            return None

        # Prepare document
        doc = {
            **company_data,
            "company_name": company_name,
            "updated_at": datetime.utcnow(),
        }

        # Upsert by company name
        result = self.db.companies.update_one(
            {"company_name": company_name},
            {"$set": doc, "$setOnInsert": {"created_at": datetime.utcnow()}},
            upsert=True
        )

        logger.info(f"Saved company: {company_name}")
        return str(result.upserted_id) if result.upserted_id else company_name

    def get_company(self, company_name: str) -> Optional[Dict[str, Any]]:
        """Get company by name."""
        if not self.is_connected():
            return None
        return self.db.companies.find_one({"company_name": company_name})

    def list_companies(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List all companies."""
        if not self.is_connected():
            return []
        return list(self.db.companies.find().limit(limit))

    # ==================== ASSESSMENT OPERATIONS ====================

    def save_assessment(self, assessment: Dict[str, Any]) -> Optional[str]:
        """
        Save a credit assessment (LLM decision).

        Args:
            assessment: Credit assessment from supervisor

        Returns:
            Inserted document ID
        """
        if not self.is_connected():
            return None

        company_name = assessment.get("company") or assessment.get("company_name")
        if not company_name:
            logger.error("Company name required in assessment")
            return None

        # Prepare document
        doc = {
            **assessment,
            "company_name": company_name,
            "saved_at": datetime.utcnow(),
        }

        result = self.db.assessments.insert_one(doc)
        logger.info(f"Saved assessment for: {company_name} (ID: {result.inserted_id})")
        return str(result.inserted_id)

    def get_latest_assessment(self, company_name: str) -> Optional[Dict[str, Any]]:
        """Get the most recent assessment for a company."""
        if not self.is_connected():
            return None
        return self.db.assessments.find_one(
            {"company_name": company_name},
            sort=[("saved_at", -1)]
        )

    def get_assessment_history(self, company_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get assessment history for a company."""
        if not self.is_connected():
            return []
        return list(self.db.assessments.find(
            {"company_name": company_name}
        ).sort("saved_at", -1).limit(limit))

    def get_all_assessments(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all assessments."""
        if not self.is_connected():
            return []
        return list(self.db.assessments.find().sort("saved_at", -1).limit(limit))

    # ==================== RAW DATA OPERATIONS ====================

    def save_raw_data(self, company_name: str, source: str, data: Dict[str, Any]) -> Optional[str]:
        """
        Save raw API response data for auditing.

        Args:
            company_name: Company name
            source: Data source (e.g., "sec_edgar", "finnhub")
            data: Raw API response

        Returns:
            Inserted document ID
        """
        if not self.is_connected():
            return None

        doc = {
            "company_name": company_name,
            "source": source,
            "data": data,
            "fetched_at": datetime.utcnow(),
        }

        result = self.db.raw_data.insert_one(doc)
        logger.debug(f"Saved raw data from {source} for {company_name}")
        return str(result.inserted_id)

    def save_all_raw_data(self, company_name: str, api_data: Dict[str, Any], search_data: Dict[str, Any]) -> None:
        """Save all raw data from a workflow run."""
        if not self.is_connected():
            return

        # Save each API source
        for source, data in api_data.items():
            if data and not isinstance(data, str):  # Skip error strings
                self.save_raw_data(company_name, source, data)

        # Save search data
        if search_data:
            self.save_raw_data(company_name, "web_search", search_data)

    def get_raw_data(self, company_name: str, source: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get raw data for a company."""
        if not self.is_connected():
            return []

        query = {"company_name": company_name}
        if source:
            query["source"] = source

        return list(self.db.raw_data.find(query).sort("fetched_at", -1))

    # ==================== EVALUATION OPERATIONS ====================

    def save_evaluation(self, evaluation: Dict[str, Any]) -> Optional[str]:
        """Save evaluation results."""
        if not self.is_connected():
            return None

        doc = {
            **evaluation,
            "evaluated_at": datetime.utcnow(),
        }

        result = self.db.evaluations.insert_one(doc)
        logger.info(f"Saved evaluation (ID: {result.inserted_id})")
        return str(result.inserted_id)

    def get_evaluations(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get all evaluations."""
        if not self.is_connected():
            return []
        return list(self.db.evaluations.find().sort("evaluated_at", -1).limit(limit))

    # ==================== STATS & UTILITIES ====================

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        if not self.is_connected():
            return {"connected": False}

        return {
            "connected": True,
            "companies_count": self.db.companies.count_documents({}),
            "assessments_count": self.db.assessments.count_documents({}),
            "raw_data_count": self.db.raw_data.count_documents({}),
            "evaluations_count": self.db.evaluations.count_documents({}),
        }

    def get_risk_distribution(self) -> Dict[str, int]:
        """Get distribution of risk levels across assessments."""
        if not self.is_connected():
            return {}

        pipeline = [
            {"$group": {"_id": "$overall_risk_level", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]

        results = list(self.db.assessments.aggregate(pipeline))
        return {r["_id"]: r["count"] for r in results if r["_id"]}

    def close(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")


# Convenience function
def get_db() -> CreditIntelligenceDB:
    """Get a MongoDB connection instance."""
    return CreditIntelligenceDB()


# Test connection
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    db = get_db()
    if db.is_connected():
        print("Connected to MongoDB!")
        print(f"Stats: {db.get_stats()}")
    else:
        print("Failed to connect to MongoDB")
