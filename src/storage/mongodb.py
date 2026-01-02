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

    # ==================== LLM CALL OPERATIONS ====================

    def save_llm_call(self, llm_call: Dict[str, Any]) -> Optional[str]:
        """
        Save an LLM API call log.

        Required fields:
        - run_id: Run identifier
        - company_name: Company being analyzed
        - llm_provider: Provider name (groq, openai, etc.)
        - agent_name: Agent that made the call
        - model: Model used
        - prompt: Input prompt
        - response: LLM response
        - tokens: Token counts

        Args:
            llm_call: LLM call data dict

        Returns:
            Inserted document ID
        """
        if not self.is_connected():
            return None

        doc = {
            **llm_call,
            "logged_at": datetime.utcnow(),
        }

        # Ensure required fields
        if "run_id" not in doc:
            logger.error("run_id required for LLM call log")
            return None

        result = self.db.llm_calls.insert_one(doc)
        logger.debug(f"Saved LLM call for run: {doc.get('run_id')}")
        return str(result.inserted_id)

    def save_llm_call_detailed(
        self,
        run_id: str,
        company_name: str,
        llm_provider: str,
        agent_name: str,
        model: str,
        prompt: str,
        context: str = "",
        response: str = "",
        reasoning: str = "",
        error: str = "",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        response_time_ms: float = 0,
        input_cost: float = 0,
        output_cost: float = 0,
        total_cost: float = 0,
    ) -> Optional[str]:
        """
        Save a detailed LLM call log with all fields.

        This matches Task 17 requirements:
        - llm provider
        - run id
        - agent name
        - input (prompt)
        - Context
        - output
        - Reasoning
        - Errors
        - Tokens Number, response time

        Returns:
            Inserted document ID
        """
        return self.save_llm_call({
            "run_id": run_id,
            "company_name": company_name,
            "llm_provider": llm_provider,
            "agent_name": agent_name,
            "model": model,
            "prompt": prompt,
            "context": context,
            "response": response,
            "reasoning": reasoning,
            "error": error,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "response_time_ms": response_time_ms,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
        })

    def get_llm_calls(
        self,
        run_id: Optional[str] = None,
        company_name: Optional[str] = None,
        agent_name: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get LLM calls with optional filters."""
        if not self.is_connected():
            return []

        query = {}
        if run_id:
            query["run_id"] = run_id
        if company_name:
            query["company_name"] = company_name
        if agent_name:
            query["agent_name"] = agent_name

        return list(
            self.db.llm_calls.find(query)
            .sort("logged_at", -1)
            .limit(limit)
        )

    def get_llm_calls_summary(self, run_id: str) -> Dict[str, Any]:
        """Get summary of LLM calls for a run."""
        if not self.is_connected():
            return {}

        pipeline = [
            {"$match": {"run_id": run_id}},
            {"$group": {
                "_id": "$agent_name",
                "call_count": {"$sum": 1},
                "total_tokens": {"$sum": "$total_tokens"},
                "total_cost": {"$sum": "$total_cost"},
                "avg_response_time": {"$avg": "$response_time_ms"},
            }},
        ]

        results = list(self.db.llm_calls.aggregate(pipeline))

        # Also get totals
        totals = list(self.db.llm_calls.aggregate([
            {"$match": {"run_id": run_id}},
            {"$group": {
                "_id": None,
                "total_calls": {"$sum": 1},
                "total_tokens": {"$sum": "$total_tokens"},
                "total_cost": {"$sum": "$total_cost"},
                "total_time_ms": {"$sum": "$response_time_ms"},
            }},
        ]))

        return {
            "run_id": run_id,
            "by_agent": {r["_id"]: r for r in results},
            "totals": totals[0] if totals else {},
        }

    # ==================== RUN SUMMARY OPERATIONS ====================

    def save_run_summary(self, run_summary: Dict[str, Any]) -> Optional[str]:
        """
        Save a comprehensive run summary.

        Required fields:
        - run_id: Run identifier
        - company_name: Company analyzed

        Recommended fields (per Task 17):
        - status: success/failed
        - risk_level: Final risk assessment
        - credit_score: Final credit score
        - confidence: Confidence score
        - eval_metrics: Dict of all evaluation metrics
        - final_decision: Good/Not Good
        - errors: List of errors
        - started_at, completed_at, duration_ms
        - tools_used, agents_used
        - total_tokens, total_cost

        Returns:
            Inserted document ID
        """
        if not self.is_connected():
            return None

        doc = {
            **run_summary,
            "saved_at": datetime.utcnow(),
        }

        if "run_id" not in doc:
            logger.error("run_id required for run summary")
            return None

        result = self.db.run_summaries.insert_one(doc)
        logger.info(f"Saved run summary for: {doc.get('company_name')} (run: {doc.get('run_id')})")
        return str(result.inserted_id)

    def save_run_summary_detailed(
        self,
        run_id: str,
        company_name: str,
        status: str = "completed",
        # Final Assessment
        risk_level: str = "",
        credit_score: int = 0,
        confidence: float = 0.0,
        reasoning: str = "",
        recommendations: List[str] = None,
        # Evaluation Metrics
        eval_metrics: Dict[str, Any] = None,
        tool_selection_score: float = 0.0,
        data_quality_score: float = 0.0,
        synthesis_score: float = 0.0,
        overall_score: float = 0.0,
        # Final Decision
        final_decision: str = "",  # Good/Not Good
        decision_reasoning: str = "",
        # Execution Details
        errors: List[str] = None,
        warnings: List[str] = None,
        tools_used: List[str] = None,
        agents_used: List[str] = None,
        # Timing
        started_at: str = "",
        completed_at: str = "",
        duration_ms: float = 0.0,
        # Costs
        total_tokens: int = 0,
        total_cost: float = 0.0,
        llm_calls_count: int = 0,
    ) -> Optional[str]:
        """
        Save a detailed run summary with all fields.

        This matches Task 17 requirements for summary run logs.

        Returns:
            Inserted document ID
        """
        return self.save_run_summary({
            "run_id": run_id,
            "company_name": company_name,
            "status": status,
            # Assessment
            "risk_level": risk_level,
            "credit_score": credit_score,
            "confidence": confidence,
            "reasoning": reasoning,
            "recommendations": recommendations or [],
            # Eval metrics
            "eval_metrics": eval_metrics or {},
            "tool_selection_score": tool_selection_score,
            "data_quality_score": data_quality_score,
            "synthesis_score": synthesis_score,
            "overall_score": overall_score,
            # Decision
            "final_decision": final_decision,
            "decision_reasoning": decision_reasoning,
            # Execution
            "errors": errors or [],
            "warnings": warnings or [],
            "tools_used": tools_used or [],
            "agents_used": agents_used or [],
            # Timing
            "started_at": started_at,
            "completed_at": completed_at,
            "duration_ms": duration_ms,
            # Costs
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "llm_calls_count": llm_calls_count,
        })

    def get_run_summary(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get run summary by run_id."""
        if not self.is_connected():
            return None
        return self.db.run_summaries.find_one({"run_id": run_id})

    def get_run_summaries(
        self,
        company_name: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get run summaries with optional filters."""
        if not self.is_connected():
            return []

        query = {}
        if company_name:
            query["company_name"] = company_name
        if status:
            query["status"] = status

        return list(
            self.db.run_summaries.find(query)
            .sort("saved_at", -1)
            .limit(limit)
        )

    def get_run_statistics(self) -> Dict[str, Any]:
        """Get aggregate statistics across all runs."""
        if not self.is_connected():
            return {}

        pipeline = [
            {"$group": {
                "_id": None,
                "total_runs": {"$sum": 1},
                "successful_runs": {
                    "$sum": {"$cond": [{"$eq": ["$status", "completed"]}, 1, 0]}
                },
                "failed_runs": {
                    "$sum": {"$cond": [{"$eq": ["$status", "failed"]}, 1, 0]}
                },
                "avg_credit_score": {"$avg": "$credit_score"},
                "avg_confidence": {"$avg": "$confidence"},
                "avg_overall_score": {"$avg": "$overall_score"},
                "total_tokens": {"$sum": "$total_tokens"},
                "total_cost": {"$sum": "$total_cost"},
                "avg_duration_ms": {"$avg": "$duration_ms"},
            }},
        ]

        results = list(self.db.run_summaries.aggregate(pipeline))

        # Risk level distribution
        risk_pipeline = [
            {"$group": {"_id": "$risk_level", "count": {"$sum": 1}}},
        ]
        risk_dist = list(self.db.run_summaries.aggregate(risk_pipeline))

        # Decision distribution
        decision_pipeline = [
            {"$group": {"_id": "$final_decision", "count": {"$sum": 1}}},
        ]
        decision_dist = list(self.db.run_summaries.aggregate(decision_pipeline))

        return {
            "summary": results[0] if results else {},
            "risk_distribution": {r["_id"]: r["count"] for r in risk_dist if r["_id"]},
            "decision_distribution": {d["_id"]: d["count"] for d in decision_dist if d["_id"]},
        }

    # ==================== LANGGRAPH EVENT OPERATIONS ====================

    def save_langgraph_event(self, event: Dict[str, Any]) -> Optional[str]:
        """
        Save a LangGraph event.

        Args:
            event: LangGraph event dict

        Returns:
            Inserted document ID
        """
        if not self.is_connected():
            return None

        doc = {
            **event,
            "logged_at": datetime.utcnow(),
        }

        result = self.db.langgraph_events.insert_one(doc)
        return str(result.inserted_id)

    def save_langgraph_events_batch(self, events: List[Dict[str, Any]]) -> int:
        """
        Save multiple LangGraph events in batch.

        Args:
            events: List of event dicts

        Returns:
            Number of inserted events
        """
        if not self.is_connected() or not events:
            return 0

        for event in events:
            event["logged_at"] = datetime.utcnow()

        result = self.db.langgraph_events.insert_many(events)
        return len(result.inserted_ids)

    def get_langgraph_events(
        self,
        run_id: Optional[str] = None,
        company_name: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Query LangGraph events.

        Args:
            run_id: Filter by run ID
            company_name: Filter by company name
            event_type: Filter by event type (e.g., "on_chain_start")
            limit: Maximum number of events to return

        Returns:
            List of event documents
        """
        if not self.is_connected():
            return []

        query = {}
        if run_id:
            query["run_id"] = run_id
        if company_name:
            query["company_name"] = company_name
        if event_type:
            query["event_type"] = event_type

        return list(
            self.db.langgraph_events.find(query)
            .sort("timestamp", -1)
            .limit(limit)
        )

    def get_langgraph_run_summary(self, run_id: str) -> Dict[str, Any]:
        """
        Get summary of a LangGraph run.

        Args:
            run_id: Run ID to summarize

        Returns:
            Summary dict with event counts and timing
        """
        if not self.is_connected():
            return {}

        pipeline = [
            {"$match": {"run_id": run_id}},
            {"$group": {
                "_id": "$event_type",
                "count": {"$sum": 1},
                "avg_duration_ms": {"$avg": "$duration_ms"},
                "total_tokens": {"$sum": "$tokens"},
            }},
        ]

        results = list(self.db.langgraph_events.aggregate(pipeline))

        # Get graph start/end for total duration
        graph_events = list(self.db.langgraph_events.find({
            "run_id": run_id,
            "event_type": {"$in": ["graph_start", "graph_end"]}
        }).sort("timestamp", 1))

        return {
            "run_id": run_id,
            "event_summary": {r["_id"]: r for r in results},
            "graph_events": graph_events,
        }

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
            "langgraph_events_count": self.db.langgraph_events.count_documents({}),
            "llm_calls_count": self.db.llm_calls.count_documents({}),
            "run_summaries_count": self.db.run_summaries.count_documents({}),
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
