"""Run Logger - Logs complete workflow runs to MongoDB, PostgreSQL, and local JSON fallback."""

import os
import ssl
import uuid
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from pathlib import Path

# Auto-load .env file
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

logger = logging.getLogger(__name__)

# Try to import MongoDB and certifi
try:
    from pymongo import MongoClient
    from pymongo.server_api import ServerApi
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False

try:
    import certifi
    CERTIFI_AVAILABLE = True
except ImportError:
    CERTIFI_AVAILABLE = False

# Try to import PostgreSQL logger
try:
    from run_logging.postgres_logger import PostgresLogger, get_postgres_logger
    POSTGRES_LOGGER_AVAILABLE = True
except ImportError:
    try:
        from src.run_logging.postgres_logger import PostgresLogger, get_postgres_logger
        POSTGRES_LOGGER_AVAILABLE = True
    except ImportError:
        POSTGRES_LOGGER_AVAILABLE = False
        PostgresLogger = None
        get_postgres_logger = None


class RunLogger:
    """
    Comprehensive logger that stores all run data to MongoDB, PostgreSQL, and local JSON fallback.

    Storage Targets:
    - MongoDB: Primary document store (flexible schema)
    - PostgreSQL: SQL store with partitioning for data retention
    - Local JSON: Fallback when databases unavailable

    Collections/Tables:
    - runs: Complete run summaries
    - steps: Individual step logs
    - tool_calls: Tool execution logs
    - evaluations: Evaluation results
    """

    def __init__(self, connection_string: Optional[str] = None, enable_postgres: bool = True):
        self.connection_string = connection_string or os.getenv("MONGODB_URI")
        self.client = None
        self.db = None
        self._postgres_logger: Optional[PostgresLogger] = None

        # Local fallback directory
        self.local_log_dir = Path("data/run_logs")
        self.local_log_dir.mkdir(parents=True, exist_ok=True)
        self._local_runs: Dict[str, Dict[str, Any]] = {}  # In-memory cache for local runs

        # Initialize MongoDB
        if MONGODB_AVAILABLE and self.connection_string:
            try:
                # Connection options for MongoDB Atlas
                options = {
                    "server_api": ServerApi('1'),
                    "serverSelectionTimeoutMS": 10000,
                }
                if CERTIFI_AVAILABLE:
                    options["tlsCAFile"] = certifi.where()

                self.client = MongoClient(self.connection_string, **options)
                self.client.admin.command('ping')
                self.db = self.client.credit_intelligence
                logger.info("RunLogger connected to MongoDB Atlas")
            except Exception as e:
                logger.warning(f"MongoDB connection failed: {e}")

        # Initialize PostgreSQL logger
        if enable_postgres and POSTGRES_LOGGER_AVAILABLE:
            try:
                logger.info("Initializing PostgreSQL logger...")
                self._postgres_logger = get_postgres_logger()
                if self._postgres_logger.is_connected():
                    logger.info("RunLogger connected to PostgreSQL")
                else:
                    logger.info("PostgreSQL not connected, trying to initialize...")
                    if self._postgres_logger.initialize():
                        logger.info("PostgreSQL initialized successfully")
                    else:
                        self._postgres_logger = None
            except Exception as e:
                logger.warning(f"PostgreSQL logger initialization failed: {e}")
                self._postgres_logger = None

    @property
    def postgres(self) -> Optional[PostgresLogger]:
        """Get the PostgreSQL logger instance."""
        return self._postgres_logger

    def is_postgres_connected(self) -> bool:
        """Check if PostgreSQL is connected."""
        if self._postgres_logger is None:
            return False
        if not self._postgres_logger.is_connected():
            # Try to initialize/connect if not connected
            try:
                if self._postgres_logger.initialize():
                    logger.info("PostgreSQL reconnected successfully")
                    return True
            except Exception as e:
                logger.warning(f"Failed to reconnect to PostgreSQL: {e}")
            return False
        return True

    def is_connected(self) -> bool:
        return self.db is not None

    def _save_local_run(self, run_id: str):
        """Save run data to local JSON file."""
        if run_id in self._local_runs:
            filepath = self.local_log_dir / f"run_{run_id}.json"
            with open(filepath, "w") as f:
                json.dump(self._local_runs[run_id], f, indent=2, default=str)

    def _load_local_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Load run data from local JSON file."""
        if run_id in self._local_runs:
            return self._local_runs[run_id]
        filepath = self.local_log_dir / f"run_{run_id}.json"
        if filepath.exists():
            with open(filepath, "r") as f:
                return json.load(f)
        return None

    # ==================== RUN LOGGING ====================

    def start_run(self, company_name: str, context: Dict[str, Any] = None, run_id: str = None) -> str:
        """
        Start a new run and return run_id.

        Args:
            company_name: Company being analyzed
            context: Additional context
            run_id: Optional run_id to use (if not provided, generates a new one)

        Returns:
            run_id: Unique run identifier
        """
        run_id = run_id or str(uuid.uuid4())

        run_doc = {
            "run_id": run_id,
            "company_name": company_name,
            "context": context or {},
            "status": "started",
            "started_at": datetime.utcnow(),
            "steps": [],
            "tool_calls": [],
            "metrics": {},
            "evaluation": {},
        }

        if self.is_connected():
            self.db.runs.insert_one(run_doc)
        else:
            # Local fallback
            self._local_runs[run_id] = run_doc
            self._save_local_run(run_id)
            logger.info(f"Saving run {run_id} locally (MongoDB unavailable)")

        # Also log to PostgreSQL
        if self.is_postgres_connected():
            try:
                self._postgres_logger.log_run(
                    run_id=run_id,
                    company_name=company_name,
                    status="started",
                    started_at=datetime.utcnow().isoformat(),
                )
            except Exception as e:
                logger.warning(f"Failed to log run start to PostgreSQL: {e}")

        logger.info(f"Started run {run_id} for {company_name}")
        return run_id

    def log_step(
        self,
        run_id: str,
        step_name: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        execution_time_ms: float,
        tokens_used: Dict[str, int] = None,
        success: bool = True,
        error: str = None,
    ):
        """Log a workflow step."""
        step_doc = {
            "run_id": run_id,
            "step_name": step_name,
            "input_data": input_data,
            "output_data": output_data,
            "execution_time_ms": execution_time_ms,
            "tokens_used": tokens_used or {},
            "success": success,
            "error": error,
            "timestamp": datetime.utcnow(),
        }

        if self.is_connected():
            # Add to steps collection
            self.db.steps.insert_one(step_doc)
            # Update run document
            self.db.runs.update_one(
                {"run_id": run_id},
                {"$push": {"steps": {
                    "step_name": step_name,
                    "execution_time_ms": execution_time_ms,
                    "success": success,
                }}}
            )
        elif run_id in self._local_runs:
            # Local fallback
            self._local_runs[run_id]["steps"].append({
                "step_name": step_name,
                "execution_time_ms": execution_time_ms,
                "success": success,
            })
            self._save_local_run(run_id)

        logger.debug(f"Logged step {step_name} for run {run_id}")

    def log_tool_call(
        self,
        run_id: str,
        tool_name: str,
        input_params: Dict[str, Any],
        output_data: Dict[str, Any],
        execution_time_ms: float,
        success: bool,
        selection_reason: str = "",
    ):
        """Log a tool execution."""
        tool_doc = {
            "run_id": run_id,
            "tool_name": tool_name,
            "input_params": input_params,
            "output_data": output_data,
            "execution_time_ms": execution_time_ms,
            "success": success,
            "selection_reason": selection_reason,
            "timestamp": datetime.utcnow(),
        }

        if self.is_connected():
            self.db.tool_calls.insert_one(tool_doc)
            self.db.runs.update_one(
                {"run_id": run_id},
                {"$push": {"tool_calls": {
                    "tool_name": tool_name,
                    "success": success,
                    "execution_time_ms": execution_time_ms,
                }}}
            )
        elif run_id in self._local_runs:
            # Local fallback
            self._local_runs[run_id]["tool_calls"].append({
                "tool_name": tool_name,
                "success": success,
                "execution_time_ms": execution_time_ms,
            })
            self._save_local_run(run_id)

        # Also log to PostgreSQL
        pg_connected = self.is_postgres_connected()
        logger.info(f"Tool call {tool_name}: PostgreSQL connected={pg_connected}")
        if pg_connected:
            try:
                result = self._postgres_logger.log_tool_call(
                    run_id=run_id,
                    company_name="",  # Get from run if available
                    tool_name=tool_name,
                    tool_input=input_params,
                    tool_output=output_data,
                    execution_time_ms=execution_time_ms,
                    status="success" if success else "error",
                )
                logger.info(f"Tool call {tool_name} logged to PostgreSQL: {result}")
            except Exception as e:
                logger.warning(f"Failed to log tool call to PostgreSQL: {e}")

        logger.debug(f"Logged tool call {tool_name} for run {run_id}")

    def log_tool_selection(
        self,
        run_id: str,
        company_name: str,
        tools_selected: List[str],
        selection_reasoning: Dict[str, Any],
        llm_metrics: Dict[str, Any],
    ):
        """Log tool selection decision."""
        selection_doc = {
            "run_id": run_id,
            "company_name": company_name,
            "tools_selected": tools_selected,
            "selection_reasoning": selection_reasoning,
            "llm_metrics": llm_metrics,
            "timestamp": datetime.utcnow(),
        }

        if self.is_connected():
            self.db.tool_selections.insert_one(selection_doc)
        elif run_id in self._local_runs:
            # Local fallback
            self._local_runs[run_id]["tool_selection"] = {
                "tools_selected": tools_selected,
                "selection_reasoning": selection_reasoning,
                "llm_metrics": llm_metrics,
            }
            self._save_local_run(run_id)

        logger.info(f"Tool selection for {company_name}: {tools_selected}")

    def log_assessment(
        self,
        run_id: str,
        company_name: str,
        assessment: Dict[str, Any],
        llm_metrics: Dict[str, Any],
    ):
        """Log final credit assessment."""
        assessment_doc = {
            "run_id": run_id,
            "company_name": company_name,
            **assessment,
            "llm_metrics": llm_metrics,
            "timestamp": datetime.utcnow(),
        }

        if self.is_connected():
            self.db.assessments.insert_one(assessment_doc)
        elif run_id in self._local_runs:
            # Local fallback
            self._local_runs[run_id]["assessment"] = {
                **assessment,
                "llm_metrics": llm_metrics,
            }
            self._save_local_run(run_id)

        # Also log to PostgreSQL
        if self.is_postgres_connected():
            try:
                self._postgres_logger.log_assessment(
                    run_id=run_id,
                    company_name=company_name,
                    risk_level=assessment.get("overall_risk_level") or assessment.get("risk_level", ""),
                    credit_score=assessment.get("credit_score_estimate") or assessment.get("credit_score", 0),
                    confidence=assessment.get("confidence_score") or assessment.get("confidence", 0.0),
                    reasoning=assessment.get("llm_reasoning") or assessment.get("reasoning", ""),
                )
            except Exception as e:
                logger.warning(f"Failed to log assessment to PostgreSQL: {e}")

        logger.info(f"Assessment logged for {company_name}: {assessment.get('risk_level')}")

    def complete_run(
        self,
        run_id: str,
        final_result: Dict[str, Any],
        total_metrics: Dict[str, Any],
    ):
        """Mark run as complete with final results."""
        completed_at = datetime.utcnow()

        if self.is_connected():
            self.db.runs.update_one(
                {"run_id": run_id},
                {
                    "$set": {
                        "status": "completed",
                        "completed_at": completed_at,
                        "final_result": final_result,
                        "metrics": total_metrics,
                    }
                }
            )
        elif run_id in self._local_runs:
            # Local fallback
            self._local_runs[run_id]["status"] = "completed"
            self._local_runs[run_id]["completed_at"] = completed_at.isoformat()
            self._local_runs[run_id]["final_result"] = final_result
            self._local_runs[run_id]["metrics"] = total_metrics
            self._save_local_run(run_id)

        # Log completed run to PostgreSQL
        if self.is_postgres_connected():
            try:
                # Get company name and started_at from MongoDB or local cache
                company_name = ""
                started_at = None
                if self.is_connected():
                    run_doc = self.db.runs.find_one({"run_id": run_id})
                    if run_doc:
                        company_name = run_doc.get("company_name", "")
                        started_at = run_doc.get("started_at")
                elif run_id in self._local_runs:
                    company_name = self._local_runs[run_id].get("company_name", "")
                    started_at = self._local_runs[run_id].get("started_at")

                # Convert started_at to ISO format if it's a datetime
                started_at_str = ""
                if started_at:
                    if isinstance(started_at, datetime):
                        started_at_str = started_at.isoformat()
                    elif isinstance(started_at, str) and started_at:
                        started_at_str = started_at

                self._postgres_logger.log_run(
                    run_id=run_id,
                    company_name=company_name,
                    status="completed",
                    risk_level=final_result.get("risk_level", ""),
                    credit_score=final_result.get("credit_score", 0),
                    confidence=final_result.get("confidence", 0.0),
                    reasoning=final_result.get("reasoning", ""),
                    overall_score=final_result.get("evaluation_score", 0.0),
                    started_at=started_at_str if started_at_str else completed_at.isoformat(),
                    completed_at=completed_at.isoformat(),
                    duration_ms=total_metrics.get("total_time_ms", 0),
                    total_tokens=total_metrics.get("total_tokens", 0),
                    total_cost=total_metrics.get("total_cost", 0.0),
                )
                logger.info(f"Logged completed run {run_id} to PostgreSQL")
            except Exception as e:
                logger.warning(f"Failed to log completed run to PostgreSQL: {e}")

        logger.info(f"Completed run {run_id}")

    def fail_run(self, run_id: str, error: str):
        """Mark run as failed."""
        if self.is_connected():
            self.db.runs.update_one(
                {"run_id": run_id},
                {
                    "$set": {
                        "status": "failed",
                        "completed_at": datetime.utcnow(),
                        "error": error,
                    }
                }
            )
        elif run_id in self._local_runs:
            # Local fallback
            self._local_runs[run_id]["status"] = "failed"
            self._local_runs[run_id]["completed_at"] = datetime.utcnow().isoformat()
            self._local_runs[run_id]["error"] = error
            self._save_local_run(run_id)

        logger.error(f"Failed run {run_id}: {error}")

    # ==================== STEP LOGGING ====================

    def log_workflow_step(
        self,
        run_id: str,
        step_name: str,
        step_number: int,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        execution_time_ms: float,
        success: bool = True,
        error: str = None,
    ):
        """Log a detailed workflow step."""
        step_doc = {
            "run_id": run_id,
            "step_name": step_name,
            "step_number": step_number,
            "input_data": input_data,
            "output_data": output_data,
            "execution_time_ms": execution_time_ms,
            "success": success,
            "error": error,
            "timestamp": datetime.utcnow(),
        }

        if self.is_connected():
            self.db.workflow_steps.insert_one(step_doc)
        elif run_id in self._local_runs:
            if "workflow_steps" not in self._local_runs[run_id]:
                self._local_runs[run_id]["workflow_steps"] = []
            self._local_runs[run_id]["workflow_steps"].append(step_doc)
            self._save_local_run(run_id)

        logger.debug(f"Logged workflow step {step_name} for run {run_id}")

    def log_llm_call(
        self,
        run_id: str,
        call_type: str,
        model: str,
        prompt: str,
        response: str,
        prompt_tokens: int,
        completion_tokens: int,
        execution_time_ms: float,
    ):
        """Log an LLM API call."""
        llm_doc = {
            "run_id": run_id,
            "call_type": call_type,
            "model": model,
            "prompt": prompt or "",  # Full prompt - no truncation
            "response": response or "",  # Full response - no truncation
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "execution_time_ms": execution_time_ms,
            "timestamp": datetime.utcnow(),
        }

        if self.is_connected():
            self.db.llm_calls.insert_one(llm_doc)
        elif run_id in self._local_runs:
            if "llm_calls" not in self._local_runs[run_id]:
                self._local_runs[run_id]["llm_calls"] = []
            self._local_runs[run_id]["llm_calls"].append(llm_doc)
            self._save_local_run(run_id)

        # Also log to PostgreSQL
        if self.is_postgres_connected():
            try:
                self._postgres_logger.log_llm_call(
                    run_id=run_id,
                    company_name="",
                    call_type=call_type,
                    model=model,
                    prompt=prompt or "",
                    response=response or "",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                    execution_time_ms=execution_time_ms,
                )
            except Exception as e:
                logger.warning(f"Failed to log LLM call to PostgreSQL: {e}")

        logger.debug(f"Logged LLM call {call_type} for run {run_id}")

    def log_data_source(
        self,
        run_id: str,
        source_name: str,
        success: bool,
        records_found: int,
        data_summary: Dict[str, Any],
        execution_time_ms: float,
    ):
        """Log data source fetch result."""
        source_doc = {
            "run_id": run_id,
            "source_name": source_name,
            "success": success,
            "records_found": records_found,
            "data_summary": data_summary,
            "execution_time_ms": execution_time_ms,
            "timestamp": datetime.utcnow(),
        }

        if self.is_connected():
            self.db.data_sources.insert_one(source_doc)
        elif run_id in self._local_runs:
            if "data_sources" not in self._local_runs[run_id]:
                self._local_runs[run_id]["data_sources"] = []
            self._local_runs[run_id]["data_sources"].append(source_doc)
            self._save_local_run(run_id)

        logger.debug(f"Logged data source {source_name} for run {run_id}")

    def log_consistency_score(
        self,
        run_id: str,
        company_name: str,
        evaluation_type: str,
        num_runs: int,
        risk_level_consistency: float,
        score_consistency: float,
        reasoning_similarity: float,
        overall_consistency: float,
        risk_levels: List[str],
        credit_scores: List[int],
    ):
        """Log consistency evaluation scores."""
        consistency_doc = {
            "run_id": run_id,
            "company_name": company_name,
            "evaluation_type": evaluation_type,
            "num_runs": num_runs,
            "risk_level_consistency": risk_level_consistency,
            "score_consistency": score_consistency,
            "reasoning_similarity": reasoning_similarity,
            "overall_consistency": overall_consistency,
            "risk_levels": risk_levels,
            "credit_scores": credit_scores,
            "timestamp": datetime.utcnow(),
        }

        if self.is_connected():
            self.db.consistency_scores.insert_one(consistency_doc)
        elif run_id in self._local_runs:
            self._local_runs[run_id]["consistency"] = consistency_doc
            self._save_local_run(run_id)

        logger.info(f"Logged {evaluation_type} consistency for {company_name}: {overall_consistency:.2f}")

    # ==================== TASK 17: DETAILED LOGGING ====================

    def log_llm_call_detailed(
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
    ):
        """
        Log a detailed LLM call (Task 17 compliant).

        Stores in llm_calls_detailed collection with all fields:
        - llm_provider, run_id, agent_name
        - prompt, context, response, reasoning
        - error, tokens, response_time_ms, costs
        """
        llm_doc = {
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
            "input_cost": round(input_cost, 6),
            "output_cost": round(output_cost, 6),
            "total_cost": round(total_cost, 6),
            "response_time_ms": response_time_ms,
            "timestamp": datetime.utcnow(),
        }

        if self.is_connected():
            self.db.llm_calls_detailed.insert_one(llm_doc)
        elif run_id in self._local_runs:
            if "llm_calls_detailed" not in self._local_runs[run_id]:
                self._local_runs[run_id]["llm_calls_detailed"] = []
            self._local_runs[run_id]["llm_calls_detailed"].append(llm_doc)
            self._save_local_run(run_id)

        # Also log to PostgreSQL
        if self.is_postgres_connected():
            try:
                self._postgres_logger.log_llm_call(
                    run_id=run_id,
                    company_name=company_name,
                    agent_name=agent_name,
                    model=model,
                    provider=llm_provider,
                    prompt=prompt,
                    context=context,
                    response=response,
                    reasoning=reasoning,
                    error=error,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                    input_cost=input_cost,
                    output_cost=output_cost,
                    total_cost=total_cost,
                    execution_time_ms=response_time_ms,
                )
            except Exception as e:
                logger.warning(f"Failed to log detailed LLM call to PostgreSQL: {e}")

        logger.debug(f"Logged detailed LLM call {agent_name}/{model} for run {run_id}")

    def log_run_summary_detailed(
        self,
        run_id: str,
        company_name: str,
        status: str = "completed",
        # Assessment
        risk_level: str = "",
        credit_score: int = 0,
        confidence: float = 0.0,
        reasoning: str = "",
        # Eval metrics
        tool_selection_score: float = 0.0,
        data_quality_score: float = 0.0,
        synthesis_score: float = 0.0,
        overall_score: float = 0.0,
        # Decision
        final_decision: str = "",
        decision_reasoning: str = "",
        # Execution details
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
    ):
        """
        Log a comprehensive run summary (Task 17 compliant).

        Stores in run_summaries collection with all fields:
        - company_name, run_id, status
        - risk_level, credit_score, confidence, reasoning
        - ALL eval metrics
        - final_decision, decision_reasoning
        - errors, warnings, tools_used, agents_used
        - timing and cost information
        """
        # Use current time for completed_at if not provided
        now = datetime.utcnow()
        actual_completed_at = completed_at if completed_at else now.isoformat()

        summary_doc = {
            "run_id": run_id,
            "company_name": company_name,
            "status": status,
            "risk_level": risk_level,
            "credit_score": credit_score,
            "confidence": round(confidence, 4),
            "reasoning": reasoning,
            "tool_selection_score": round(tool_selection_score, 4),
            "data_quality_score": round(data_quality_score, 4),
            "synthesis_score": round(synthesis_score, 4),
            "overall_score": round(overall_score, 4),
            "final_decision": final_decision,
            "decision_reasoning": decision_reasoning,
            "errors": errors or [],
            "warnings": warnings or [],
            "tools_used": tools_used or [],
            "agents_used": agents_used or [],
            "started_at": started_at,
            "completed_at": actual_completed_at,
            "duration_ms": duration_ms,
            "total_tokens": total_tokens,
            "total_cost": round(total_cost, 6),
            "llm_calls_count": llm_calls_count,
            "timestamp": now,
        }

        if self.is_connected():
            self.db.run_summaries.insert_one(summary_doc)
        elif run_id in self._local_runs:
            self._local_runs[run_id]["run_summary"] = summary_doc
            self._save_local_run(run_id)

        # Also log to PostgreSQL
        if self.is_postgres_connected():
            try:
                # Ensure timestamps are not empty strings
                pg_started_at = started_at if started_at else now.isoformat()
                pg_completed_at = actual_completed_at if actual_completed_at else now.isoformat()

                self._postgres_logger.log_run(
                    run_id=run_id,
                    company_name=company_name,
                    status=status,
                    risk_level=risk_level,
                    credit_score=credit_score,
                    confidence=confidence,
                    reasoning=reasoning,
                    overall_score=overall_score,
                    final_decision=final_decision,
                    decision_reasoning=decision_reasoning,
                    errors=errors,
                    warnings=warnings,
                    tools_used=tools_used,
                    agents_used=agents_used,
                    started_at=pg_started_at,
                    completed_at=pg_completed_at,
                    duration_ms=duration_ms,
                    total_tokens=total_tokens,
                    total_cost=total_cost,
                    llm_calls_count=llm_calls_count,
                )
                logger.info(f"Logged run summary to PostgreSQL for {company_name}")
            except Exception as e:
                logger.warning(f"Failed to log run summary to PostgreSQL: {e}")

        logger.info(f"Logged detailed run summary for {company_name} (run: {run_id})")

    def log_agent_metrics(
        self,
        run_id: str,
        company_name: str,
        # Core agent metrics (0-1 scores)
        intent_correctness: float = 0.0,
        plan_quality: float = 0.0,
        tool_choice_correctness: float = 0.0,
        tool_completeness: float = 0.0,
        trajectory_match: float = 0.0,
        final_answer_quality: float = 0.0,
        # Execution metrics
        step_count: int = 0,
        tool_calls: int = 0,
        latency_ms: float = 0.0,
        # Overall score
        overall_score: float = 0.0,
        # Details (dicts)
        intent_details: Dict[str, Any] = None,
        plan_details: Dict[str, Any] = None,
        tool_details: Dict[str, Any] = None,
        trajectory_details: Dict[str, Any] = None,
        answer_details: Dict[str, Any] = None,
    ):
        """
        Log agent efficiency metrics (Task 4 compliant).

        Stores in agent_metrics collection with all fields:
        - intent_correctness: Did the agent understand the task?
        - plan_quality: How good was the execution plan?
        - tool_choice_correctness: Did agent choose correct tools? (precision)
        - tool_completeness: Did agent use all needed tools? (recall)
        - trajectory_match: Did agent follow expected execution path?
        - final_answer_quality: Is the final output correct and complete?
        - step_count, tool_calls, latency_ms: Execution metrics
        """
        metrics_doc = {
            "run_id": run_id,
            "company_name": company_name,
            # Core metrics
            "intent_correctness": round(intent_correctness, 4),
            "plan_quality": round(plan_quality, 4),
            "tool_choice_correctness": round(tool_choice_correctness, 4),
            "tool_completeness": round(tool_completeness, 4),
            "trajectory_match": round(trajectory_match, 4),
            "final_answer_quality": round(final_answer_quality, 4),
            # Execution metrics
            "step_count": step_count,
            "tool_calls": tool_calls,
            "latency_ms": round(latency_ms, 2),
            # Overall
            "overall_score": round(overall_score, 4),
            # Details
            "intent_details": intent_details or {},
            "plan_details": plan_details or {},
            "tool_details": tool_details or {},
            "trajectory_details": trajectory_details or {},
            "answer_details": answer_details or {},
            "timestamp": datetime.utcnow(),
        }

        if self.is_connected():
            self.db.agent_metrics.insert_one(metrics_doc)
        elif run_id in self._local_runs:
            self._local_runs[run_id]["agent_metrics"] = metrics_doc
            self._save_local_run(run_id)

        logger.info(f"Logged agent metrics for {company_name} (run: {run_id}, overall: {overall_score:.4f})")

    def get_llm_calls_detailed(self, run_id: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get detailed LLM calls, optionally filtered by run_id."""
        if not self.is_connected():
            return []

        query = {"run_id": run_id} if run_id else {}
        return list(self.db.llm_calls_detailed.find(query).sort("timestamp", -1).limit(limit))

    def get_run_summaries(self, company_name: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get run summaries, optionally filtered by company name."""
        if not self.is_connected():
            return []

        query = {"company_name": company_name} if company_name else {}
        return list(self.db.run_summaries.find(query).sort("timestamp", -1).limit(limit))

    # ==================== EVALUATION LOGGING ====================

    def log_evaluation(
        self,
        run_id: str,
        evaluation_type: str,
        metrics: Dict[str, Any],
        scores: Dict[str, float],
    ):
        """Log evaluation results."""
        eval_doc = {
            "run_id": run_id,
            "evaluation_type": evaluation_type,
            "metrics": metrics,
            "scores": scores,
            "timestamp": datetime.utcnow(),
        }

        if self.is_connected():
            self.db.evaluations.insert_one(eval_doc)
            self.db.runs.update_one(
                {"run_id": run_id},
                {"$set": {f"evaluation.{evaluation_type}": scores}}
            )
        elif run_id in self._local_runs:
            # Local fallback
            if "evaluation" not in self._local_runs[run_id]:
                self._local_runs[run_id]["evaluation"] = {}
            self._local_runs[run_id]["evaluation"][evaluation_type] = scores
            self._save_local_run(run_id)

        logger.info(f"Logged {evaluation_type} evaluation for run {run_id}")

    # ==================== RETRIEVAL ====================

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get a complete run by ID."""
        if self.is_connected():
            return self.db.runs.find_one({"run_id": run_id})
        # Try local fallback
        return self._load_local_run(run_id)

    def get_run_steps(self, run_id: str) -> List[Dict[str, Any]]:
        """Get all steps for a run."""
        if self.is_connected():
            return list(self.db.steps.find({"run_id": run_id}).sort("timestamp", 1))
        # Try local fallback
        run = self._load_local_run(run_id)
        return run.get("steps", []) if run else []

    def get_run_tool_calls(self, run_id: str) -> List[Dict[str, Any]]:
        """Get all tool calls for a run."""
        if self.is_connected():
            return list(self.db.tool_calls.find({"run_id": run_id}).sort("timestamp", 1))
        # Try local fallback
        run = self._load_local_run(run_id)
        return run.get("tool_calls", []) if run else []

    def get_recent_runs(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent runs."""
        if self.is_connected():
            return list(self.db.runs.find().sort("started_at", -1).limit(limit))
        # Try local fallback - read all local run files
        runs = []
        for filepath in self.local_log_dir.glob("run_*.json"):
            try:
                with open(filepath, "r") as f:
                    runs.append(json.load(f))
            except Exception:
                continue
        # Sort by started_at
        runs.sort(key=lambda x: x.get("started_at", ""), reverse=True)
        return runs[:limit]

    def get_runs_for_company(self, company_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get runs for a specific company."""
        if self.is_connected():
            return list(self.db.runs.find(
                {"company_name": company_name}
            ).sort("started_at", -1).limit(limit))
        # Local fallback
        runs = self.get_recent_runs(limit=1000)
        filtered = [r for r in runs if r.get("company_name") == company_name]
        return filtered[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """Get overall statistics."""
        if self.is_connected():
            return {
                "connected": True,
                "storage": "mongodb",
                "total_runs": self.db.runs.count_documents({}),
                "completed_runs": self.db.runs.count_documents({"status": "completed"}),
                "failed_runs": self.db.runs.count_documents({"status": "failed"}),
                "total_steps": self.db.steps.count_documents({}),
                "total_tool_calls": self.db.tool_calls.count_documents({}),
                "total_evaluations": self.db.evaluations.count_documents({}),
            }

        # Local fallback stats
        runs = self.get_recent_runs(limit=1000)
        return {
            "connected": False,
            "storage": "local_json",
            "local_log_dir": str(self.local_log_dir),
            "total_runs": len(runs),
            "completed_runs": len([r for r in runs if r.get("status") == "completed"]),
            "failed_runs": len([r for r in runs if r.get("status") == "failed"]),
            "total_steps": sum(len(r.get("steps", [])) for r in runs),
            "total_tool_calls": sum(len(r.get("tool_calls", [])) for r in runs),
        }


# Singleton instance
_logger: Optional[RunLogger] = None


def get_run_logger(force_new: bool = False) -> RunLogger:
    """Get the global RunLogger instance."""
    global _logger
    if _logger is None or force_new:
        _logger = RunLogger()
    return _logger
