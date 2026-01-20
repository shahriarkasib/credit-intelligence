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
    # Now PostgreSQL matches Google Sheets exactly, so minimal mapping needed
    COLUMN_MAPPING = {
        "generated_by": None,  # Skip this column (not in PostgreSQL)
        # These now match directly - no transformation needed
        "variables_json": "variables_json",
        "precision": "precision",
        "recall": "recall",
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
        if storage is None:
            logger.debug("PostgresStorage is None")
            return False
        connected = storage.is_connected()
        logger.debug(f"PostgresStorage.is_connected() = {connected}")
        return connected

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
            logger.warning(f"Not connected to PostgreSQL, cannot log to {sheet_name}")
            return False

        table_name = self._get_table_name(sheet_name)
        normalized_data = self._normalize_columns(data)

        result = self.storage.insert(table_name, normalized_data)
        if result:
            logger.info(f"Successfully logged to PostgreSQL table {table_name}")
        else:
            logger.warning(f"Failed to log to PostgreSQL table {table_name}")
        return result

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
        # Performance scores (3 key metrics)
        tool_overall_score: float = None,
        agent_overall_score: float = None,
        workflow_overall_score: float = None,
        **kwargs,
    ) -> bool:
        """Log a run summary - matches Google Sheets 'runs' schema."""
        # Remove fields that aren't in the schema
        for field in ["reasoning", "duration_ms", "total_cost", "total_tokens",
                      "errors", "warnings", "final_decision", "decision_reasoning",
                      "overall_score", "tool_selection_score", "synthesis_score",
                      "data_quality_score", "llm_calls_count", "agents_used"]:
            kwargs.pop(field, None)

        # Handle empty string timestamps - PostgreSQL TIMESTAMPTZ needs valid values or NULL
        now_iso = datetime.now(timezone.utc).isoformat()
        actual_started_at = started_at if started_at else now_iso
        actual_completed_at = completed_at if completed_at else now_iso

        data = {
            "run_id": run_id,
            "company_name": company_name,
            "node": node,
            "agent_name": agent_name,
            "master_agent": master_agent or "supervisor",
            "model": model,
            "temperature": temperature,
            "status": status,
            "started_at": actual_started_at,
            "completed_at": actual_completed_at,
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
            # Performance scores (3 key metrics)
            "tool_overall_score": tool_overall_score,
            "agent_overall_score": agent_overall_score,
            "workflow_overall_score": workflow_overall_score,
        }
        data.update(kwargs)
        return self.log("runs", data)

    def update_run_correctness(
        self,
        run_id: str,
        workflow_correct: bool = None,
        output_correct: bool = None,
    ) -> bool:
        """
        Update the workflow_correct and output_correct columns for an existing run.

        This is called after coalition evaluation completes to update the run
        with correctness values that weren't available when the run was initially logged.

        Args:
            run_id: The run ID to update
            workflow_correct: Whether the workflow executed correctly
            output_correct: Whether the output quality is acceptable

        Returns:
            True if update was successful
        """
        if not self.is_connected():
            logger.warning("Not connected to PostgreSQL, cannot update run correctness")
            return False

        try:
            # Build UPDATE query
            query = """
                UPDATE wf_runs
                SET workflow_correct = %s,
                    output_correct = %s
                WHERE run_id = %s
            """

            # Execute the update
            with self.storage._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, (workflow_correct, output_correct, run_id))
                    conn.commit()
                    rows_updated = cursor.rowcount

            if rows_updated > 0:
                logger.info(f"Updated run {run_id} correctness: workflow={workflow_correct}, output={output_correct}")
                return True
            else:
                logger.warning(f"No run found with run_id {run_id} to update")
                return False

        except Exception as e:
            logger.error(f"Failed to update run correctness in PostgreSQL: {e}")
            return False

    def log_llm_call(
        self,
        run_id: str,
        company_name: str,
        node: str = "",
        node_type: str = "",
        agent_name: str = "",
        master_agent: str = "supervisor",
        step_number: int = 0,
        call_type: str = "",
        model: str = "",
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
        """Log an LLM call - matches Google Sheets schema."""
        # Remove provider from kwargs if present (not in schema)
        kwargs.pop("provider", None)
        data = {
            "run_id": run_id,
            "company_name": company_name,
            "node": node,
            "node_type": node_type,
            "agent_name": agent_name,
            "master_agent": master_agent or "supervisor",
            "step_number": step_number,
            "call_type": call_type,
            "model": model,
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
        # Hierarchy fields for joinability
        node: str = "",
        node_type: str = "tool",
        agent_name: str = "",
        master_agent: str = "supervisor",
        step_number: int = 0,
        # Additional hierarchy fields
        parent_node: str = "",
        workflow_phase: str = "",
        call_depth: int = 0,
        parent_tool_id: str = "",
        **kwargs,
    ) -> bool:
        """Log a tool call with hierarchy tracking."""
        # Ensure tool_input and tool_output are JSON-compatible (dict/list)
        # Strings need to be wrapped in a dict for JSONB columns
        if tool_input is not None and not isinstance(tool_input, (dict, list)):
            tool_input = {"value": str(tool_input)}
        if tool_output is not None and not isinstance(tool_output, (dict, list)):
            tool_output = {"value": str(tool_output)}

        data = {
            "run_id": run_id,
            "company_name": company_name,
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_output": tool_output,
            "execution_time_ms": execution_time_ms,
            "status": status,
            "error": error,
            # Hierarchy fields for joinability with lg_events and other tables
            "node": node or tool_name,  # Use tool_name as fallback
            "node_type": node_type or "tool",
            "agent_name": agent_name,
            "master_agent": master_agent or "supervisor",
            "step_number": step_number,
            # Additional hierarchy
            "parent_node": parent_node or node or tool_name,
            "workflow_phase": workflow_phase,
            "call_depth": call_depth,
            "parent_tool_id": parent_tool_id,
        }
        data.update(kwargs)
        return self.log("tool_calls", data)

    def log_plan(
        self,
        run_id: str,
        company_name: str,
        full_plan: List[Dict[str, Any]] = None,
        num_tasks: int = 0,
        plan_summary: str = "",
        # Hierarchy fields
        node: str = "create_plan",
        node_type: str = "agent",
        agent_name: str = "",
        master_agent: str = "supervisor",
        status: str = "completed",
        **kwargs,
    ) -> bool:
        """Log a task plan - matches Google Sheets schema."""
        # Remove step_number from kwargs if present (not in wf_plans schema)
        kwargs.pop("step_number", None)

        plan_list = full_plan or []
        data = {
            "run_id": run_id,
            "company_name": company_name,
            "full_plan": plan_list,
            "num_tasks": num_tasks or len(plan_list),
            "plan_summary": plan_summary or f"Plan with {len(plan_list)} tasks",
            # Hierarchy fields - wf_plans doesn't have node_type or step_number
            "node": node or "create_plan",
            "agent_name": agent_name,
            "master_agent": master_agent or "supervisor",
            "status": status,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        # Add individual tasks (up to 10)
        for i in range(1, 11):
            if i <= len(plan_list):
                data[f"task_{i}"] = str(plan_list[i-1]) if plan_list[i-1] else ""
            else:
                data[f"task_{i}"] = ""
        data.update(kwargs)
        return self.log("plans", data)

    def log_data_source(
        self,
        run_id: str,
        company_name: str,
        source_name: str,
        success: bool = True,
        records_found: int = 0,
        data_summary: str = "",
        execution_time_ms: float = 0.0,
        error: str = "",
        # Hierarchy fields for joinability
        node: str = "fetch_api_data",
        node_type: str = "tool",
        agent_name: str = "",
        master_agent: str = "supervisor",
        step_number: int = 0,
        **kwargs,
    ) -> bool:
        """Log a data source fetch with hierarchy tracking."""
        data = {
            "run_id": run_id,
            "company_name": company_name,
            "source_name": source_name,
            "status": "success" if success else "error",
            "records_found": records_found,
            "data_summary": data_summary[:50000] if data_summary else "",
            "execution_time_ms": execution_time_ms,
            "error": error,
            # Hierarchy fields for joinability
            "node": node or "fetch_api_data",
            "node_type": node_type or "tool",
            "agent_name": agent_name,
            "master_agent": master_agent or "supervisor",
            "step_number": step_number,
        }
        data.update(kwargs)
        return self.log("data_sources", data)

    def log_langgraph_event(
        self,
        run_id: str,
        company_name: str,
        event_type: str,
        node: str = "",
        node_type: str = "",
        agent_name: str = "",
        master_agent: str = "supervisor",
        step_number: int = 0,
        duration_ms: float = 0.0,
        **kwargs,
    ) -> bool:
        """Log a LangGraph event with hierarchy tracking."""
        data = {
            "run_id": run_id,
            "company_name": company_name,
            "event_type": event_type,
            "node": node,
            "node_type": node_type,
            "agent_name": agent_name,
            "master_agent": master_agent or "supervisor",
            "step_number": step_number,
            "duration_ms": duration_ms,
        }
        data.update(kwargs)
        return self.log("langgraph_events", data)

    def log_prompt(
        self,
        run_id: str,
        company_name: str,
        prompt_id: str,
        prompt_name: str,
        category: str,
        system_prompt: str = "",
        user_prompt: str = "",
        variables: Dict[str, Any] = None,
        # Hierarchy fields
        node: str = "",
        node_type: str = "llm",
        agent_name: str = "",
        master_agent: str = "supervisor",
        step_number: int = 0,
        model: str = "",
        temperature: float = None,
        **kwargs,
    ) -> bool:
        """Log a prompt with hierarchy tracking - matches Google Sheets schema."""
        data = {
            "run_id": run_id,
            "company_name": company_name,
            "prompt_id": prompt_id,
            "prompt_name": prompt_name,
            "category": category,
            "system_prompt": system_prompt[:10000] if system_prompt else "",
            "user_prompt": user_prompt[:10000] if user_prompt else "",
            "variables_json": variables or {},  # Match Google Sheets column name
            # Hierarchy fields
            "node": node,
            "node_type": node_type or "llm",
            "agent_name": agent_name,
            "master_agent": master_agent or "supervisor",
            "step_number": step_number,
            "model": model,
            "temperature": temperature,
        }
        data.update(kwargs)
        return self.log("prompts", data)

    def log_assessment(
        self,
        run_id: str,
        company_name: str,
        risk_level: str = "",
        credit_score: int = 0,
        confidence: float = 0.0,
        reasoning: str = "",
        # Hierarchy fields
        node: str = "synthesize",
        node_type: str = "agent",
        agent_name: str = "",
        master_agent: str = "supervisor",
        step_number: int = 0,
        **kwargs,
    ) -> bool:
        """Log an assessment with hierarchy tracking."""
        data = {
            "run_id": run_id,
            "company_name": company_name,
            "risk_level": risk_level,
            "credit_score": credit_score,
            "confidence": confidence,
            "reasoning": reasoning,
            # Hierarchy fields
            "node": node or "synthesize",
            "node_type": node_type or "agent",
            "agent_name": agent_name,
            "master_agent": master_agent or "supervisor",
            "step_number": step_number,
        }
        data.update(kwargs)
        return self.log("assessments", data)

    def log_evaluation(
        self,
        run_id: str,
        company_name: str,
        overall_score: float = 0.0,
        tool_selection_score: float = 0.0,
        data_quality_score: float = 0.0,
        synthesis_score: float = 0.0,
        tool_reasoning: str = "",
        data_reasoning: str = "",
        synthesis_reasoning: str = "",
        eval_status: str = "",
        duration_ms: float = 0.0,
        model: str = "",
        # Hierarchy fields
        node: str = "evaluate",
        node_type: str = "agent",
        agent_name: str = "",
        master_agent: str = "supervisor",
        step_number: int = 0,
        **kwargs,
    ) -> bool:
        """Log an evaluation with hierarchy tracking - matches Google Sheets schema."""
        # Remove evaluation_type from kwargs if present (not in schema)
        kwargs.pop("evaluation_type", None)
        data = {
            "run_id": run_id,
            "company_name": company_name,
            "overall_score": overall_score,
            "tool_selection_score": tool_selection_score,
            "tool_reasoning": tool_reasoning,
            "data_quality_score": data_quality_score,
            "data_reasoning": data_reasoning,
            "synthesis_score": synthesis_score,
            "synthesis_reasoning": synthesis_reasoning,
            "eval_status": eval_status,
            "duration_ms": duration_ms,
            "model": model,
            "status": "completed",
            # Hierarchy fields
            "node": node or "evaluate",
            "node_type": node_type or "agent",
            "agent_name": agent_name,
            "master_agent": master_agent or "supervisor",
            "step_number": step_number,
        }
        data.update(kwargs)
        return self.log("evaluations", data)

    def log_tool_selection(
        self,
        run_id: str,
        company_name: str,
        selected_tools: List[str] = None,
        expected_tools: List[str] = None,
        correct_tools: List[str] = None,
        missing_tools: List[str] = None,
        extra_tools: List[str] = None,
        precision: float = 0.0,
        recall: float = 0.0,
        f1_score: float = 0.0,
        reasoning: str = "",
        duration_ms: float = 0.0,
        model: str = "",
        # Hierarchy fields
        node: str = "create_plan",
        node_type: str = "agent",
        agent_name: str = "",
        master_agent: str = "supervisor",
        step_number: int = 0,
        **kwargs,
    ) -> bool:
        """Log tool selection evaluation - matches Google Sheets schema."""
        data = {
            "run_id": run_id,
            "company_name": company_name,
            "selected_tools": selected_tools or [],
            "expected_tools": expected_tools or [],
            "correct_tools": correct_tools or [],
            "missing_tools": missing_tools or [],
            "extra_tools": extra_tools or [],
            "precision": precision,  # Match Google Sheets column name
            "recall": recall,  # Match Google Sheets column name
            "f1_score": f1_score,
            "reasoning": reasoning,
            "duration_ms": duration_ms,
            "model": model,
            "status": "completed",
            # Hierarchy fields
            "node": node or "create_plan",
            "node_type": node_type or "agent",
            "agent_name": agent_name,
            "master_agent": master_agent or "supervisor",
            "step_number": step_number,
        }
        data.update(kwargs)
        return self.log("tool_selections", data)

    def log_consistency_score(
        self,
        run_id: str,
        company_name: str,
        model_name: str = "",
        evaluation_type: str = "",
        num_runs: int = 0,
        risk_level_consistency: float = 0.0,
        score_consistency: float = 0.0,
        overall_consistency: float = 0.0,
        risk_levels: List[str] = None,
        credit_scores: List[int] = None,
        # Hierarchy fields
        node: str = "evaluate",
        node_type: str = "agent",
        agent_name: str = "",
        master_agent: str = "supervisor",
        step_number: int = 0,
        **kwargs,
    ) -> bool:
        """Log consistency score with hierarchy tracking."""
        data = {
            "run_id": run_id,
            "company_name": company_name,
            "model_name": model_name,
            "evaluation_type": evaluation_type,
            "num_runs": num_runs,
            "risk_level_consistency": risk_level_consistency,
            "score_consistency": score_consistency,
            "overall_consistency": overall_consistency,
            "risk_levels": risk_levels or [],
            "credit_scores": credit_scores or [],
            # Hierarchy fields
            "node": node or "evaluate",
            "node_type": node_type or "agent",
            "agent_name": agent_name,
            "master_agent": master_agent or "supervisor",
            "step_number": step_number,
        }
        data.update(kwargs)
        return self.log("consistency_scores", data)

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
        step_count: int = 0,
        tool_calls: int = 0,
        latency_ms: float = 0.0,
        eval_status: str = "",
        model: str = "",
        # Hierarchy fields
        node: str = "evaluate",
        node_type: str = "agent",
        master_agent: str = "supervisor",
        step_number: int = 0,
        **kwargs,
    ) -> bool:
        """Log agent metrics - matches Google Sheets schema."""
        # Handle tool_calls_count -> tool_calls mapping
        if "tool_calls_count" in kwargs:
            tool_calls = kwargs.pop("tool_calls_count")
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
            "step_count": step_count,
            "tool_calls": tool_calls,  # Match Google Sheets column name
            "latency_ms": latency_ms,
            "eval_status": eval_status,
            "model": model,
            "status": "completed",
            # Hierarchy fields
            "node": node or "evaluate",
            "node_type": node_type or "agent",
            "master_agent": master_agent or "supervisor",
            "step_number": step_number,
        }
        data.update(kwargs)
        return self.log("agent_metrics", data)

    def log_llm_judge_result(
        self,
        run_id: str,
        company_name: str,
        model_used: str = "",
        accuracy_score: float = 0.0,
        completeness_score: float = 0.0,
        consistency_score: float = 0.0,
        actionability_score: float = 0.0,
        data_utilization_score: float = 0.0,
        overall_score: float = 0.0,
        overall_reasoning: str = "",
        suggestions: List[str] = None,
        # Hierarchy fields
        node: str = "evaluate",
        node_type: str = "llm",
        agent_name: str = "llm_judge",
        master_agent: str = "supervisor",
        step_number: int = 0,
        **kwargs,
    ) -> bool:
        """Log LLM-as-judge result with hierarchy tracking."""
        data = {
            "run_id": run_id,
            "company_name": company_name,
            "model_used": model_used,
            "accuracy_score": accuracy_score,
            "completeness_score": completeness_score,
            "consistency_score": consistency_score,
            "actionability_score": actionability_score,
            "data_utilization_score": data_utilization_score,
            "overall_score": overall_score,
            "overall_reasoning": overall_reasoning,
            "suggestions": suggestions or [],
            # Hierarchy fields
            "node": node or "evaluate",
            "node_type": node_type or "llm",
            "agent_name": agent_name or "llm_judge",
            "master_agent": master_agent or "supervisor",
            "step_number": step_number,
        }
        data.update(kwargs)
        return self.log("llm_judge_results", data)

    def log_coalition(
        self,
        run_id: str,
        company_name: str,
        node: str = "evaluate",
        node_type: str = "evaluator",
        agent_name: str = "coalition_evaluator",
        master_agent: str = "supervisor",
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
        **kwargs,
    ) -> bool:
        """Log coalition evaluation - matches Google Sheets schema."""
        # Handle 'votes' -> 'votes_json' mapping
        if "votes" in kwargs:
            votes_json = kwargs.pop("votes")
        data = {
            "run_id": run_id,
            "company_name": company_name,
            "node": node or "evaluate",
            "node_type": node_type or "evaluator",
            "agent_name": agent_name or "coalition_evaluator",
            "master_agent": master_agent or "supervisor",
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
        }
        data.update(kwargs)
        return self.log("coalition", data)

    def log_node_scoring(
        self,
        run_id: str,
        company_name: str,
        node: str = "",
        node_type: str = "",
        agent_name: str = "",
        master_agent: str = "supervisor",
        step_number: int = 0,
        task_description: str = "",
        task_completed: bool = True,
        quality_score: float = 0.0,
        quality_reasoning: str = "",
        input_summary: str = "",
        output_summary: str = "",
        judge_model: str = "",
        **kwargs,
    ) -> bool:
        """Log LLM judge quality score for a node - matches Google Sheets schema."""
        data = {
            "run_id": run_id,
            "company_name": company_name,
            "node": node,
            "node_type": node_type,
            "agent_name": agent_name,
            "master_agent": master_agent or "supervisor",
            "step_number": step_number,
            "task_description": task_description,
            "task_completed": task_completed,
            "quality_score": quality_score,
            "quality_reasoning": quality_reasoning,
            "input_summary": input_summary,
            "output_summary": output_summary,
            "judge_model": judge_model,
        }
        data.update(kwargs)
        return self.log("node_scoring", data)

    def log_cross_model_eval(
        self,
        run_id: str,
        company_name: str,
        models_compared: List[str] = None,
        num_models: int = 0,
        risk_level_agreement: float = 0.0,
        credit_score_mean: float = 0.0,
        credit_score_std: float = 0.0,
        credit_score_range: float = 0.0,
        confidence_agreement: float = 0.0,
        best_model: str = "",
        best_model_reasoning: str = "",
        cross_model_agreement: float = 0.0,
        model_results: Dict[str, Any] = None,
        **kwargs,
    ) -> bool:
        """Log cross-model evaluation result."""
        data = {
            "run_id": run_id,
            "company_name": company_name,
            "models_compared": models_compared or [],
            "num_models": num_models,
            "risk_level_agreement": risk_level_agreement,
            "credit_score_mean": credit_score_mean,
            "credit_score_std": credit_score_std,
            "credit_score_range": credit_score_range,
            "confidence_agreement": confidence_agreement,
            "best_model": best_model,
            "best_model_reasoning": best_model_reasoning,
            "cross_model_agreement": cross_model_agreement,
            "model_results": model_results or {},
        }
        data.update(kwargs)
        return self.log("cross_model_eval", data)

    def log_log_test(
        self,
        run_id: str,
        company_name: str,
        verification_status: str = "",
        total_sheets_logged: int = 0,
        # Individual sheet counts
        runs: int = 0,
        langgraph_events: int = 0,
        llm_calls: int = 0,
        tool_calls: int = 0,
        assessments: int = 0,
        evaluations: int = 0,
        tool_selections: int = 0,
        consistency_scores: int = 0,
        data_sources: int = 0,
        plans: int = 0,
        prompts: int = 0,
        cross_model_eval: int = 0,
        llm_judge_results: int = 0,
        agent_metrics: int = 0,
        coalition: int = 0,
        **kwargs,
    ) -> bool:
        """Log a log test verification result - matches Google Sheets schema."""
        # Handle old column name mappings from graph.py
        if "total_tables_logged" in kwargs:
            total_sheets_logged = kwargs.pop("total_tables_logged")

        # Map *_logged parameters to new column names (without _logged suffix)
        column_mappings = {
            "runs_logged": "runs",
            "langgraph_events_logged": "langgraph_events",
            "llm_calls_logged": "llm_calls",
            "tool_calls_logged": "tool_calls",
            "assessments_logged": "assessments",
            "evaluations_logged": "evaluations",
            "tool_selections_logged": "tool_selections",
            "consistency_scores_logged": "consistency_scores",
            "data_sources_logged": "data_sources",
            "plans_logged": "plans",
            "prompts_logged": "prompts",
            "cross_model_eval_logged": "cross_model_eval",
            "llm_judge_results_logged": "llm_judge_results",
            "agent_metrics_logged": "agent_metrics",
            "coalition_logged": "coalition",
        }

        # Extract values from kwargs using old names, then remove them
        for old_name, new_name in column_mappings.items():
            if old_name in kwargs:
                # Use locals() equivalent to set variable
                if new_name == "runs":
                    runs = kwargs.pop(old_name)
                elif new_name == "langgraph_events":
                    langgraph_events = kwargs.pop(old_name)
                elif new_name == "llm_calls":
                    llm_calls = kwargs.pop(old_name)
                elif new_name == "tool_calls":
                    tool_calls = kwargs.pop(old_name)
                elif new_name == "assessments":
                    assessments = kwargs.pop(old_name)
                elif new_name == "evaluations":
                    evaluations = kwargs.pop(old_name)
                elif new_name == "tool_selections":
                    tool_selections = kwargs.pop(old_name)
                elif new_name == "consistency_scores":
                    consistency_scores = kwargs.pop(old_name)
                elif new_name == "data_sources":
                    data_sources = kwargs.pop(old_name)
                elif new_name == "plans":
                    plans = kwargs.pop(old_name)
                elif new_name == "prompts":
                    prompts = kwargs.pop(old_name)
                elif new_name == "cross_model_eval":
                    cross_model_eval = kwargs.pop(old_name)
                elif new_name == "llm_judge_results":
                    llm_judge_results = kwargs.pop(old_name)
                elif new_name == "agent_metrics":
                    agent_metrics = kwargs.pop(old_name)
                elif new_name == "coalition":
                    coalition = kwargs.pop(old_name)

        data = {
            "run_id": run_id,
            "company_name": company_name,
            "runs": runs,
            "langgraph_events": langgraph_events,
            "llm_calls": llm_calls,
            "tool_calls": tool_calls,
            "assessments": assessments,
            "evaluations": evaluations,
            "tool_selections": tool_selections,
            "consistency_scores": consistency_scores,
            "data_sources": data_sources,
            "plans": plans,
            "prompts": prompts,
            "cross_model_eval": cross_model_eval,
            "llm_judge_results": llm_judge_results,
            "agent_metrics": agent_metrics,
            "coalition": coalition,
            "total_sheets_logged": total_sheets_logged,
            "verification_status": verification_status,
        }
        data.update(kwargs)
        return self.log("log_tests", data)

    def log_state_dump(
        self,
        run_id: str,
        company_name: str,
        company_info: Dict[str, Any] = None,
        plan: List[Dict] = None,
        api_data: Dict[str, Any] = None,
        search_data: Dict[str, Any] = None,
        assessment: Dict[str, Any] = None,
        evaluation: Dict[str, Any] = None,
        errors: List[str] = None,
        duration_ms: float = 0.0,
        status: str = "completed",
        node: str = "evaluate",
        master_agent: str = "supervisor",
        step_number: int = 8,
        **kwargs,
    ) -> bool:
        """Log full workflow state dump - matches Google Sheets schema."""
        import json

        # Calculate sizes
        plan_json_str = json.dumps(plan or [])
        api_data_json = json.dumps(api_data or {})
        search_data_json = json.dumps(search_data or {})

        # Extract assessment fields
        risk_level = assessment.get("overall_risk_level", "") if assessment else ""
        credit_score = assessment.get("credit_score_estimate", 0) if assessment else 0
        confidence = assessment.get("confidence_score", 0) if assessment else 0

        # Build evaluation scores summary
        coalition_score = 0.0
        agent_metrics_score = 0.0
        if evaluation:
            if "coalition" in evaluation:
                coalition_score = evaluation["coalition"].get("correctness_score", 0)
            if "agent_metrics" in evaluation:
                agent_metrics_score = evaluation["agent_metrics"].get("overall_agent_score", 0)
            # Also check overall_score as fallback for agent_metrics
            if agent_metrics_score == 0 and "overall_score" in evaluation:
                agent_metrics_score = evaluation.get("overall_score", 0)

        # Allow kwargs to override scores (from graph.py)
        coalition_score = kwargs.pop("coalition_score", coalition_score)
        agent_metrics_score = kwargs.pop("agent_metrics_score", agent_metrics_score)

        # Create summary strings for api_data and search_data
        api_summary = f"{len(api_data)} sources" if api_data else "No API data"
        search_summary = f"{len(search_data.get('results', []))} results" if search_data and isinstance(search_data, dict) else "No search data"

        data = {
            "run_id": run_id,
            "company_name": company_name,
            "node": node or "evaluate",
            "master_agent": master_agent or "supervisor",
            "step_number": step_number,
            # Company info - use _json suffix to match schema
            "company_info_json": company_info or {},
            # Plan - use _json suffix
            "plan_json": plan or [],
            "plan_size_bytes": len(plan_json_str),
            "plan_tasks_count": len(plan) if plan else 0,
            # API data - use _summary suffix
            "api_data_summary": api_summary,
            "api_data_size_bytes": len(api_data_json),
            "api_sources_count": len(api_data) if api_data else 0,
            # Search data - use _summary suffix
            "search_data_summary": search_summary,
            "search_data_size_bytes": len(search_data_json),
            # Assessment fields
            "risk_level": risk_level,
            "credit_score": credit_score,
            "confidence": confidence,
            "assessment_json": assessment or {},
            # Evaluation - use _json suffix
            "coalition_score": coalition_score,
            "agent_metrics_score": agent_metrics_score,
            "evaluation_json": evaluation or {},
            # Errors - use _json suffix
            "errors_json": errors or [],
            "error_count": len(errors) if errors else 0,
            # Metadata
            "total_state_size_bytes": len(plan_json_str) + len(api_data_json) + len(search_data_json),
            "duration_ms": duration_ms,
            "status": status,
        }
        data.update(kwargs)
        return self.log("state_dumps", data)

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
