"""
PostgreSQL Logger for Credit Intelligence.

This module provides logging functionality to PostgreSQL, designed to replace
Google Sheets for better scalability, querying, and data retention.

The logger uses the same interface as SheetsLogger for easy migration.
"""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from dataclasses import asdict

logger = logging.getLogger(__name__)

# Import PostgresStorage
try:
    from storage.postgres import PostgresStorage, get_postgres_storage, POSTGRES_AVAILABLE
except ImportError:
    try:
        from src.storage.postgres import PostgresStorage, get_postgres_storage, POSTGRES_AVAILABLE
    except ImportError:
        POSTGRES_AVAILABLE = False
        PostgresStorage = None
        get_postgres_storage = None


class PostgresLogger:
    """
    Logger that writes workflow data to PostgreSQL.

    Provides the same interface as SheetsLogger for easy migration.

    Table names match Google Sheets exactly:
    - runs, llm_calls, tool_calls, langgraph_events, plans, prompts,
    - data_sources, assessments, evaluations, tool_selections,
    - consistency_scores, cross_model_eval, llm_judge_results,
    - agent_metrics, log_tests, coalition, api_keys
    """

    # Table names match Google Sheets exactly (identity mapping)
    TABLE_MAPPING = {
        "runs": "runs",
        "llm_calls": "llm_calls",
        "tool_calls": "tool_calls",
        "langgraph_events": "langgraph_events",
        "plans": "plans",
        "prompts": "prompts",
        "data_sources": "data_sources",
        "assessments": "assessments",
        "evaluations": "evaluations",
        "tool_selections": "tool_selections",
        "consistency_scores": "consistency_scores",
        "cross_model_eval": "cross_model_eval",
        "llm_judge_results": "llm_judge_results",
        "agent_metrics": "agent_metrics",
        "log_tests": "log_tests",
        "coalition": "coalition",
        "api_keys": "api_keys",
    }

    # Column mapping for field name normalization
    COLUMN_MAPPING = {
        # Sheets column -> Postgres column
        "variables_json": "variables",
        "tool_input": "tool_input",
        "tool_output": "tool_output",
        "precision": "precision_score",
        "recall": "recall_score",
        "generated_by": None,  # Skip this column
    }

    def __init__(self, storage: Optional[PostgresStorage] = None):
        """
        Initialize the PostgresLogger.

        Args:
            storage: PostgresStorage instance. If not provided, uses global instance.
        """
        self._storage = storage
        self._initialized = False

    @property
    def storage(self) -> Optional[PostgresStorage]:
        """Get the PostgresStorage instance."""
        if self._storage is None and POSTGRES_AVAILABLE:
            self._storage = get_postgres_storage()
        return self._storage

    def is_connected(self) -> bool:
        """Check if connected to PostgreSQL."""
        storage = self.storage
        return storage is not None and storage.is_connected()

    def initialize(self) -> bool:
        """Initialize the database schema."""
        storage = self.storage
        if storage is None:
            logger.warning("PostgresStorage not available")
            return False

        if not storage.is_connected():
            if not storage.connect():
                return False

        result = storage.initialize_schema()
        self._initialized = result
        return result

    def _get_table_name(self, sheet_name: str) -> str:
        """Get PostgreSQL table name from sheet name."""
        return self.TABLE_MAPPING.get(sheet_name, sheet_name)

    def _normalize_columns(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize column names for PostgreSQL."""
        normalized = {}

        for key, value in data.items():
            # Convert to lowercase and replace spaces
            col_name = key.lower().replace(" ", "_")

            # Apply column mapping
            if col_name in self.COLUMN_MAPPING:
                mapped = self.COLUMN_MAPPING[col_name]
                if mapped is None:
                    continue  # Skip this column
                col_name = mapped

            # Handle timestamp
            if col_name == "timestamp":
                if isinstance(value, str):
                    try:
                        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
                    except ValueError:
                        value = datetime.now(timezone.utc)
                elif not isinstance(value, datetime):
                    value = datetime.now(timezone.utc)

            normalized[col_name] = value

        # Ensure timestamp is always present
        if "timestamp" not in normalized:
            normalized["timestamp"] = datetime.now(timezone.utc)

        return normalized

    def log(self, sheet_name: str, data: Dict[str, Any]) -> bool:
        """
        Log a single row to a table.

        Args:
            sheet_name: Name of the sheet/table
            data: Dictionary of column names to values

        Returns:
            True if successful
        """
        if not self.is_connected():
            logger.warning("Not connected to PostgreSQL")
            return False

        table_name = self._get_table_name(sheet_name)
        normalized_data = self._normalize_columns(data)

        return self.storage.insert(table_name, normalized_data)

    def log_many(self, sheet_name: str, data_list: List[Dict[str, Any]]) -> int:
        """
        Log multiple rows to a table.

        Args:
            sheet_name: Name of the sheet/table
            data_list: List of dictionaries

        Returns:
            Number of rows inserted
        """
        if not self.is_connected() or not data_list:
            return 0

        table_name = self._get_table_name(sheet_name)
        normalized_list = [self._normalize_columns(d) for d in data_list]

        return self.storage.insert_many(table_name, normalized_list)

    # Convenience methods matching SheetsLogger interface

    def log_run(
        self,
        run_id: str,
        company_name: str,
        status: str = "completed",
        risk_level: str = "",
        credit_score: int = 0,
        confidence: float = 0.0,
        reasoning: str = "",
        overall_score: float = 0.0,
        final_decision: str = "",
        decision_reasoning: str = "",
        errors: List[str] = None,
        warnings: List[str] = None,
        tools_used: List[str] = None,
        agents_used: List[str] = None,
        started_at: str = "",
        completed_at: str = "",
        duration_ms: float = 0.0,
        total_tokens: int = 0,
        total_cost: float = 0.0,
        llm_calls_count: int = 0,
        **kwargs,
    ) -> bool:
        """Log a run summary."""
        data = {
            "run_id": run_id,
            "company_name": company_name,
            "status": status,
            "risk_level": risk_level,
            "credit_score": credit_score,
            "confidence": confidence,
            "reasoning": reasoning,
            "overall_score": overall_score,
            "final_decision": final_decision,
            "decision_reasoning": decision_reasoning,
            "errors": errors or [],
            "warnings": warnings or [],
            "tools_used": tools_used or [],
            "agents_used": agents_used or [],
            "started_at": started_at,
            "completed_at": completed_at,
            "duration_ms": duration_ms,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "llm_calls_count": llm_calls_count,
        }
        data.update(kwargs)
        return self.log("runs", data)

    def log_llm_call(
        self,
        run_id: str,
        company_name: str,
        node: str = "",
        node_type: str = "",
        agent_name: str = "",
        step_number: int = 0,
        call_type: str = "",
        model: str = "",
        provider: str = "",
        temperature: float = 0.0,
        prompt: str = "",
        response: str = "",
        reasoning: str = "",
        context: str = "",
        current_task: str = "",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        input_cost: float = 0.0,
        output_cost: float = 0.0,
        total_cost: float = 0.0,
        execution_time_ms: float = 0.0,
        status: str = "success",
        error: str = "",
        **kwargs,
    ) -> bool:
        """Log an LLM call."""
        data = {
            "run_id": run_id,
            "company_name": company_name,
            "node": node,
            "node_type": node_type,
            "agent_name": agent_name,
            "step_number": step_number,
            "call_type": call_type,
            "model": model,
            "provider": provider,
            "temperature": temperature,
            "prompt": prompt[:50000] if prompt else "",  # Truncate long prompts
            "response": response[:50000] if response else "",
            "reasoning": reasoning,
            "context": context,
            "current_task": current_task,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "execution_time_ms": execution_time_ms,
            "status": status,
            "error": error,
        }
        data.update(kwargs)
        return self.log("llm_calls", data)

    def log_tool_call(
        self,
        run_id: str,
        company_name: str,
        tool_name: str,
        tool_input: Any = None,
        tool_output: Any = None,
        execution_time_ms: float = 0.0,
        status: str = "success",
        error: str = "",
        **kwargs,
    ) -> bool:
        """Log a tool call."""
        data = {
            "run_id": run_id,
            "company_name": company_name,
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_output": tool_output,
            "execution_time_ms": execution_time_ms,
            "status": status,
            "error": error,
        }
        data.update(kwargs)
        return self.log("tool_calls", data)

    def log_langgraph_event(
        self,
        run_id: str,
        company_name: str,
        event_type: str,
        node: str = "",
        agent_name: str = "",
        duration_ms: float = 0.0,
        **kwargs,
    ) -> bool:
        """Log a LangGraph event."""
        data = {
            "run_id": run_id,
            "company_name": company_name,
            "event_type": event_type,
            "node": node,
            "agent_name": agent_name,
            "duration_ms": duration_ms,
        }
        data.update(kwargs)
        return self.log("langgraph_events", data)

    def log_assessment(
        self,
        run_id: str,
        company_name: str,
        risk_level: str = "",
        credit_score: int = 0,
        confidence: float = 0.0,
        reasoning: str = "",
        **kwargs,
    ) -> bool:
        """Log an assessment."""
        data = {
            "run_id": run_id,
            "company_name": company_name,
            "risk_level": risk_level,
            "credit_score": credit_score,
            "confidence": confidence,
            "reasoning": reasoning,
        }
        data.update(kwargs)
        return self.log("assessments", data)

    def log_evaluation(
        self,
        run_id: str,
        company_name: str,
        evaluation_type: str = "",
        overall_score: float = 0.0,
        tool_selection_score: float = 0.0,
        data_quality_score: float = 0.0,
        synthesis_score: float = 0.0,
        **kwargs,
    ) -> bool:
        """Log an evaluation."""
        data = {
            "run_id": run_id,
            "company_name": company_name,
            "evaluation_type": evaluation_type,
            "overall_score": overall_score,
            "tool_selection_score": tool_selection_score,
            "data_quality_score": data_quality_score,
            "synthesis_score": synthesis_score,
        }
        data.update(kwargs)
        return self.log("evaluations", data)

    def log_tool_selection(
        self,
        run_id: str,
        company_name: str,
        selected_tools: List[str] = None,
        expected_tools: List[str] = None,
        precision: float = 0.0,
        recall: float = 0.0,
        f1_score: float = 0.0,
        **kwargs,
    ) -> bool:
        """Log tool selection evaluation."""
        data = {
            "run_id": run_id,
            "company_name": company_name,
            "selected_tools": selected_tools or [],
            "expected_tools": expected_tools or [],
            "precision_score": precision,
            "recall_score": recall,
            "f1_score": f1_score,
        }
        data.update(kwargs)
        return self.log("tool_selections", data)

    def log_agent_metrics(
        self,
        run_id: str,
        company_name: str,
        agent_name: str = "",
        overall_score: float = 0.0,
        intent_correctness: float = 0.0,
        plan_quality: float = 0.0,
        tool_choice_correctness: float = 0.0,
        tool_completeness: float = 0.0,
        trajectory_match: float = 0.0,
        final_answer_quality: float = 0.0,
        **kwargs,
    ) -> bool:
        """Log agent metrics."""
        data = {
            "run_id": run_id,
            "company_name": company_name,
            "agent_name": agent_name,
            "overall_score": overall_score,
            "intent_correctness": intent_correctness,
            "plan_quality": plan_quality,
            "tool_choice_correctness": tool_choice_correctness,
            "tool_completeness": tool_completeness,
            "trajectory_match": trajectory_match,
            "final_answer_quality": final_answer_quality,
        }
        data.update(kwargs)
        return self.log("agent_metrics", data)

    def log_coalition(
        self,
        run_id: str,
        company_name: str,
        correctness_score: float = 0.0,
        confidence: float = 0.0,
        correctness_category: str = "",
        votes: List[Dict] = None,
        **kwargs,
    ) -> bool:
        """Log coalition evaluation."""
        data = {
            "run_id": run_id,
            "company_name": company_name,
            "correctness_score": correctness_score,
            "confidence": confidence,
            "correctness_category": correctness_category,
            "votes": votes or [],
        }
        data.update(kwargs)
        return self.log("coalition", data)

    def log_log_test(
        self,
        run_id: str,
        company_name: str,
        verification_status: str = "",
        total_tables_logged: int = 0,
        **kwargs,
    ) -> bool:
        """Log a log test verification result."""
        data = {
            "run_id": run_id,
            "company_name": company_name,
            "verification_status": verification_status,
            "total_tables_logged": total_tables_logged,
        }
        data.update(kwargs)
        return self.log("log_tests", data)

    # Query methods

    def get_runs(
        self,
        limit: int = 50,
        company_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get recent runs."""
        if not self.is_connected():
            return []
        return self.storage.get_runs(limit=limit, company_name=company_name)

    def get_run_details(self, run_id: str) -> Dict[str, Any]:
        """Get comprehensive details for a run."""
        if not self.is_connected():
            return {"run_id": run_id, "found": False}
        return self.storage.get_run_details(run_id)

    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregate statistics."""
        if not self.is_connected():
            return {}
        return self.storage.get_statistics()

    def apply_retention(self, months_to_keep: int = 3) -> List[str]:
        """
        Apply data retention policy by dropping old partitions.

        Args:
            months_to_keep: Number of months of data to keep

        Returns:
            List of dropped partition names
        """
        if not self.is_connected():
            return []
        return self.storage.drop_old_partitions(months_to_keep)


# Global instance
_postgres_logger: Optional[PostgresLogger] = None


def get_postgres_logger() -> PostgresLogger:
    """Get or create the global PostgresLogger instance."""
    global _postgres_logger
    if _postgres_logger is None:
        _postgres_logger = PostgresLogger()
    return _postgres_logger


def init_postgres_logger() -> bool:
    """Initialize the PostgresLogger and schema."""
    pg_logger = get_postgres_logger()
    return pg_logger.initialize()
