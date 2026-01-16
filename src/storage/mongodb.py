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


# Collection schemas - matches Google Sheets columns exactly
# MongoDB is schema-less but this documents expected fields for consistency
COLLECTION_SCHEMAS = {
    "runs": [
        "run_id", "company_name", "node", "agent_name", "master_agent", "model", "temperature",
        "status", "started_at", "completed_at", "risk_level", "credit_score", "confidence",
        "total_time_ms", "total_steps", "total_llm_calls", "tools_used", "evaluation_score",
        "workflow_correct", "output_correct", "timestamp"
    ],
    "langgraph_events": [
        "run_id", "company_name", "node", "node_type", "agent_name", "master_agent", "step_number",
        "event_type", "event_name", "model", "temperature", "tokens", "input_preview", "output_preview",
        "duration_ms", "status", "error", "timestamp"
    ],
    "llm_calls": [
        "run_id", "company_name", "node", "node_type", "agent_name", "master_agent", "step_number",
        "call_type", "model", "temperature", "prompt", "response", "reasoning", "context", "current_task",
        "prompt_tokens", "completion_tokens", "total_tokens", "input_cost", "output_cost", "total_cost",
        "execution_time_ms", "status", "error", "timestamp"
    ],
    "tool_calls": [
        "run_id", "company_name", "node", "node_type", "agent_name", "master_agent", "step_number",
        "tool_name", "tool_input", "tool_output", "parent_node", "workflow_phase", "call_depth",
        "parent_tool_id", "execution_time_ms", "status", "error", "timestamp"
    ],
    "assessments": [
        "run_id", "company_name", "node", "node_type", "agent_name", "master_agent", "step_number",
        "model", "temperature", "prompt", "risk_level", "credit_score", "confidence", "reasoning",
        "recommendations", "duration_ms", "status", "timestamp"
    ],
    "evaluations": [
        "run_id", "company_name", "node", "node_type", "agent_name", "master_agent", "step_number",
        "model", "tool_selection_score", "tool_reasoning", "data_quality_score", "data_reasoning",
        "synthesis_score", "synthesis_reasoning", "overall_score", "eval_status", "duration_ms",
        "status", "timestamp"
    ],
    "coalition": [
        "run_id", "company_name", "node", "node_type", "agent_name", "master_agent", "step_number",
        "is_correct", "correctness_score", "confidence", "correctness_category",
        "efficiency_score", "quality_score", "tool_score", "consistency_score",
        "agreement_score", "num_evaluators", "votes_json", "evaluation_time_ms", "status", "timestamp"
    ],
    "agent_metrics": [
        "run_id", "company_name", "node", "node_type", "agent_name", "master_agent", "step_number",
        "model", "intent_correctness", "plan_quality", "tool_choice_correctness", "tool_completeness",
        "trajectory_match", "final_answer_quality", "step_count", "tool_calls", "latency_ms",
        "overall_score", "eval_status", "intent_details", "plan_details", "tool_details",
        "trajectory_details", "answer_details", "status", "timestamp"
    ],
}


class CreditIntelligenceDB:
    """
    MongoDB storage for credit intelligence data.

    Collections (matching Google Sheets):
    - runs: Run summaries
    - langgraph_events: LangGraph framework events
    - llm_calls: LLM API call logs
    - tool_calls: Tool execution logs
    - assessments: Credit assessments
    - evaluations: Evaluation results
    - coalition: Coalition evaluation results
    - agent_metrics: Agent efficiency metrics
    - companies: Company profiles
    - raw_data: Raw API responses for auditing
    - active_runs: Active/running analyses
    - api_keys: API key storage
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

    # ==================== COALITION OPERATIONS ====================

    def save_coalition(
        self,
        run_id: str,
        company_name: str,
        node: str = "",
        node_type: str = "",
        agent_name: str = "",
        master_agent: str = "",
        step_number: int = 0,
        is_correct: bool = None,
        correctness_score: float = 0.0,
        confidence: float = 0.0,
        correctness_category: str = "",
        efficiency_score: float = 0.0,
        quality_score: float = 0.0,
        tool_score: float = 0.0,
        consistency_score: float = 0.0,
        agreement_score: float = 0.0,
        num_evaluators: int = 0,
        votes_json: List[Dict] = None,
        evaluation_time_ms: float = 0.0,
        status: str = "",
    ) -> Optional[str]:
        """
        Save coalition evaluation - matches Google Sheets 'coalition' schema.

        Returns:
            Inserted document ID
        """
        if not self.is_connected():
            return None

        doc = {
            "run_id": run_id,
            "company_name": company_name,
            "node": node,
            "node_type": node_type,
            "agent_name": agent_name,
            "master_agent": master_agent,
            "step_number": step_number,
            "is_correct": is_correct,
            "correctness_score": correctness_score,
            "confidence": confidence,
            "correctness_category": correctness_category,
            "efficiency_score": efficiency_score,
            "quality_score": quality_score,
            "tool_score": tool_score,
            "consistency_score": consistency_score,
            "agreement_score": agreement_score,
            "num_evaluators": num_evaluators,
            "votes_json": votes_json or [],
            "evaluation_time_ms": evaluation_time_ms,
            "status": status,
            "timestamp": datetime.utcnow(),
        }

        result = self.db.coalition.insert_one(doc)
        logger.info(f"Saved coalition for run: {run_id}")
        return str(result.inserted_id)

    def get_coalition(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get coalition evaluation by run_id."""
        if not self.is_connected():
            return None
        return self.db.coalition.find_one({"run_id": run_id})

    # ==================== AGENT METRICS OPERATIONS ====================

    def save_agent_metrics(
        self,
        run_id: str,
        company_name: str,
        node: str = "",
        node_type: str = "",
        agent_name: str = "",
        master_agent: str = "",
        step_number: int = 0,
        model: str = "",
        intent_correctness: float = 0.0,
        plan_quality: float = 0.0,
        tool_choice_correctness: float = 0.0,
        tool_completeness: float = 0.0,
        trajectory_match: float = 0.0,
        final_answer_quality: float = 0.0,
        step_count: int = 0,
        tool_calls: int = 0,
        latency_ms: float = 0.0,
        overall_score: float = 0.0,
        eval_status: str = "",
        intent_details: Dict = None,
        plan_details: Dict = None,
        tool_details: Dict = None,
        trajectory_details: Dict = None,
        answer_details: Dict = None,
        status: str = "",
    ) -> Optional[str]:
        """
        Save agent metrics - matches Google Sheets 'agent_metrics' schema.

        Returns:
            Inserted document ID
        """
        if not self.is_connected():
            return None

        doc = {
            "run_id": run_id,
            "company_name": company_name,
            "node": node,
            "node_type": node_type,
            "agent_name": agent_name,
            "master_agent": master_agent,
            "step_number": step_number,
            "model": model,
            "intent_correctness": intent_correctness,
            "plan_quality": plan_quality,
            "tool_choice_correctness": tool_choice_correctness,
            "tool_completeness": tool_completeness,
            "trajectory_match": trajectory_match,
            "final_answer_quality": final_answer_quality,
            "step_count": step_count,
            "tool_calls": tool_calls,
            "latency_ms": latency_ms,
            "overall_score": overall_score,
            "eval_status": eval_status,
            "intent_details": intent_details or {},
            "plan_details": plan_details or {},
            "tool_details": tool_details or {},
            "trajectory_details": trajectory_details or {},
            "answer_details": answer_details or {},
            "status": status,
            "timestamp": datetime.utcnow(),
        }

        result = self.db.agent_metrics.insert_one(doc)
        logger.info(f"Saved agent metrics for run: {run_id}")
        return str(result.inserted_id)

    def get_agent_metrics(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get agent metrics by run_id."""
        if not self.is_connected():
            return None
        return self.db.agent_metrics.find_one({"run_id": run_id})

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
        node: str = "",
        agent_name: str = "",
        master_agent: str = "",
        model: str = "",
        temperature: float = 0.0,
        status: str = "completed",
        started_at: str = "",
        completed_at: str = "",
        risk_level: str = "",
        credit_score: int = 0,
        confidence: float = 0.0,
        total_time_ms: float = 0.0,
        total_steps: int = 0,
        total_llm_calls: int = 0,
        tools_used: List[str] = None,
        evaluation_score: float = 0.0,
        workflow_correct: bool = None,
        output_correct: bool = None,
    ) -> Optional[str]:
        """
        Save a detailed run summary - matches Google Sheets 'runs' schema.

        Returns:
            Inserted document ID
        """
        return self.save_run_summary({
            "run_id": run_id,
            "company_name": company_name,
            "node": node,
            "agent_name": agent_name,
            "master_agent": master_agent,
            "model": model,
            "temperature": temperature,
            "status": status,
            "started_at": started_at,
            "completed_at": completed_at,
            "risk_level": risk_level,
            "credit_score": credit_score,
            "confidence": confidence,
            "total_time_ms": total_time_ms,
            "total_steps": total_steps,
            "total_llm_calls": total_llm_calls,
            "tools_used": tools_used or [],
            "evaluation_score": evaluation_score,
            "workflow_correct": workflow_correct,
            "output_correct": output_correct,
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
            .sort("timestamp", -1)  # Sort by timestamp (datetime field)
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
            .sort("timestamp", 1)  # Ascending order (oldest first)
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

    # ==================== ACTIVE RUNS OPERATIONS ====================

    def save_active_run(
        self,
        run_id: str,
        company_name: str,
        status: str = "running",
        current_step: int = 0,
        steps: List[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Save or update an active run.

        Args:
            run_id: Unique run identifier
            company_name: Company being analyzed
            status: Run status (running, completed, failed)
            current_step: Current step index
            steps: List of workflow steps

        Returns:
            Document ID
        """
        if not self.is_connected():
            return None

        doc = {
            "run_id": run_id,
            "company_name": company_name,
            "status": status,
            "current_step": current_step,
            "steps": steps or [],
            "updated_at": datetime.utcnow(),
        }

        result = self.db.active_runs.update_one(
            {"run_id": run_id},
            {"$set": doc, "$setOnInsert": {"started_at": datetime.utcnow()}},
            upsert=True
        )

        logger.debug(f"Saved active run: {run_id} for {company_name}")
        return str(result.upserted_id) if result.upserted_id else run_id

    def get_active_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get an active run by run_id."""
        if not self.is_connected():
            return None
        return self.db.active_runs.find_one({"run_id": run_id})

    def get_active_runs(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all active runs.

        Args:
            status: Filter by status (running, completed, failed)

        Returns:
            List of active run documents
        """
        if not self.is_connected():
            return []

        query = {}
        if status:
            query["status"] = status

        return list(
            self.db.active_runs.find(query)
            .sort("started_at", -1)
        )

    def update_active_run(
        self,
        run_id: str,
        status: Optional[str] = None,
        current_step: Optional[int] = None,
        steps: Optional[List[Dict[str, Any]]] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> bool:
        """
        Update an active run's status.

        Args:
            run_id: Run ID to update
            status: New status
            current_step: New step index
            steps: Updated steps list
            result: Final result (for completed runs)
            error: Error message (for failed runs)

        Returns:
            True if updated successfully
        """
        if not self.is_connected():
            return False

        update = {"$set": {"updated_at": datetime.utcnow()}}
        if status:
            update["$set"]["status"] = status
        if current_step is not None:
            update["$set"]["current_step"] = current_step
        if steps is not None:
            update["$set"]["steps"] = steps
        if result is not None:
            update["$set"]["result"] = result
        if error is not None:
            update["$set"]["error"] = error

        result_op = self.db.active_runs.update_one({"run_id": run_id}, update)
        return result_op.modified_count > 0

    def delete_active_run(self, run_id: str) -> bool:
        """
        Delete an active run (after completion).

        Args:
            run_id: Run ID to delete

        Returns:
            True if deleted
        """
        if not self.is_connected():
            return False

        result = self.db.active_runs.delete_one({"run_id": run_id})
        return result.deleted_count > 0

    def cleanup_stale_active_runs(self, max_age_hours: int = 24) -> int:
        """
        Clean up stale active runs that are older than max_age_hours.

        Args:
            max_age_hours: Maximum age in hours

        Returns:
            Number of deleted runs
        """
        if not self.is_connected():
            return 0

        from datetime import timedelta
        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)

        result = self.db.active_runs.delete_many({
            "status": "running",
            "updated_at": {"$lt": cutoff}
        })
        return result.deleted_count

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
            "active_runs_count": self.db.active_runs.count_documents({}),
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

    # ==================== API Key Management ====================

    def get_api_key(self, key_name: str) -> Optional[str]:
        """
        Get an API key from the database.

        Args:
            key_name: The key identifier (e.g., 'GROQ_API_KEY', 'OPENAI_API_KEY')

        Returns:
            The API key value or None if not found
        """
        if not self.is_connected():
            return None

        try:
            doc = self.db.api_keys.find_one({"key_name": key_name})
            if doc:
                return doc.get("key_value")
            return None
        except Exception as e:
            logger.error(f"Failed to get API key {key_name}: {e}")
            return None

    def set_api_key(self, key_name: str, key_value: str) -> bool:
        """
        Set an API key in the database (upsert).

        Args:
            key_name: The key identifier (e.g., 'GROQ_API_KEY', 'OPENAI_API_KEY')
            key_value: The API key value

        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected():
            return False

        try:
            self.db.api_keys.update_one(
                {"key_name": key_name},
                {
                    "$set": {
                        "key_name": key_name,
                        "key_value": key_value,
                        "updated_at": datetime.utcnow(),
                    },
                    "$setOnInsert": {
                        "created_at": datetime.utcnow(),
                    }
                },
                upsert=True
            )
            logger.info(f"API key {key_name} updated in database")
            return True
        except Exception as e:
            logger.error(f"Failed to set API key {key_name}: {e}")
            return False

    def delete_api_key(self, key_name: str) -> bool:
        """
        Delete an API key from the database.

        Args:
            key_name: The key identifier to delete

        Returns:
            True if deleted, False otherwise
        """
        if not self.is_connected():
            return False

        try:
            result = self.db.api_keys.delete_one({"key_name": key_name})
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Failed to delete API key {key_name}: {e}")
            return False

    def get_all_api_keys_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all stored API keys (masked values).

        Returns:
            Dict mapping key_name to status info (is_set, masked, updated_at)
        """
        if not self.is_connected():
            return {}

        try:
            keys = {}
            for doc in self.db.api_keys.find():
                key_name = doc.get("key_name")
                key_value = doc.get("key_value", "")
                if key_name:
                    # Mask the value
                    if len(key_value) > 8:
                        masked = key_value[:4] + "..." + key_value[-4:]
                    elif key_value:
                        masked = "****"
                    else:
                        masked = None

                    keys[key_name] = {
                        "is_set": bool(key_value),
                        "masked": masked,
                        "updated_at": doc.get("updated_at", "").isoformat() if doc.get("updated_at") else None,
                        "source": "database",
                    }
            return keys
        except Exception as e:
            logger.error(f"Failed to get API keys status: {e}")
            return {}

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
