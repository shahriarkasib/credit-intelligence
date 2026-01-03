"""Workflow Logger - Comprehensive logging for every step of the workflow."""

import time
import functools
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from .run_logger import get_run_logger
from .sheets_logger import get_sheets_logger

logger = logging.getLogger(__name__)


class WorkflowLogger:
    """
    Comprehensive logger that captures every step of the workflow.

    Logs to both MongoDB and Google Sheets:
    - Step-by-step execution with inputs/outputs
    - LLM calls with prompts and responses
    - Data source fetches with results
    - Timing for every operation
    """

    def __init__(self):
        self.run_logger = get_run_logger()
        self.sheets_logger = get_sheets_logger()
        self._step_counter: Dict[str, int] = {}  # run_id -> step count
        self._run_info: Dict[str, Dict[str, Any]] = {}  # run_id -> run info

    def start_run(self, company_name: str, context: Dict[str, Any] = None) -> str:
        """Start a new run and return run_id."""
        run_id = self.run_logger.start_run(company_name, context)
        self._step_counter[run_id] = 0
        self._run_info[run_id] = {
            "company_name": company_name,
            "started_at": time.time(),
        }
        return run_id

    def _get_step_number(self, run_id: str) -> int:
        """Get and increment step counter."""
        if run_id not in self._step_counter:
            self._step_counter[run_id] = 0
        self._step_counter[run_id] += 1
        return self._step_counter[run_id]

    def log_step(
        self,
        run_id: str,
        company_name: str,
        step_name: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        execution_time_ms: float,
        success: bool = True,
        error: str = None,
    ):
        """Log a workflow step to all storage."""
        step_number = self._get_step_number(run_id)

        # Log to MongoDB (full data - no truncation)
        if self.run_logger.is_connected():
            self.run_logger.log_workflow_step(
                run_id=run_id,
                step_name=step_name,
                step_number=step_number,
                input_data=input_data,  # Full data for MongoDB
                output_data=output_data,  # Full data for MongoDB
                execution_time_ms=execution_time_ms,
                success=success,
                error=error,
            )

        # Log to Google Sheets
        if self.sheets_logger.is_connected():
            self.sheets_logger.log_step(
                run_id=run_id,
                company_name=company_name,
                step_name=step_name,
                step_number=step_number,
                input_data=input_data,
                output_data=output_data,
                execution_time_ms=execution_time_ms,
                success=success,
                error=error or "",
            )

        logger.info(f"[{run_id[:8]}] Step {step_number}: {step_name} ({execution_time_ms:.0f}ms) - {'OK' if success else 'FAILED'}")

    def log_llm_call(
        self,
        run_id: str,
        company_name: str,
        call_type: str,
        model: str,
        prompt: str,
        response: str,
        prompt_tokens: int,
        completion_tokens: int,
        execution_time_ms: float,
        input_cost: float = 0.0,
        output_cost: float = 0.0,
        total_cost: float = 0.0,
    ):
        """Log an LLM API call with cost tracking to all storage."""
        # Calculate cost if not provided
        if total_cost == 0 and (prompt_tokens > 0 or completion_tokens > 0):
            try:
                from config.cost_tracker import calculate_cost_for_tokens
                cost = calculate_cost_for_tokens(
                    model=model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    provider="groq",
                )
                input_cost = cost["input_cost"]
                output_cost = cost["output_cost"]
                total_cost = cost["total_cost"]
            except Exception:
                pass

        # Log to MongoDB
        if self.run_logger.is_connected():
            self.run_logger.log_llm_call(
                run_id=run_id,
                call_type=call_type,
                model=model,
                prompt=prompt,
                response=response,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                execution_time_ms=execution_time_ms,
            )

        # Log to Google Sheets with cost
        if self.sheets_logger.is_connected():
            self.sheets_logger.log_llm_call(
                run_id=run_id,
                company_name=company_name,
                call_type=call_type,
                model=model,
                prompt=prompt,
                response=response,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                execution_time_ms=execution_time_ms,
                input_cost=input_cost,
                output_cost=output_cost,
                total_cost=total_cost,
            )

        logger.debug(f"[{run_id[:8]}] LLM {call_type}: {prompt_tokens}+{completion_tokens} tokens, ${total_cost:.6f} ({execution_time_ms:.0f}ms)")

    def log_data_source(
        self,
        run_id: str,
        company_name: str,
        source_name: str,
        success: bool,
        records_found: int,
        data_summary: Any,
        execution_time_ms: float,
    ):
        """Log a data source fetch to all storage."""
        # Log to MongoDB (full data - no truncation)
        if self.run_logger.is_connected():
            self.run_logger.log_data_source(
                run_id=run_id,
                source_name=source_name,
                success=success,
                records_found=records_found,
                data_summary=data_summary if isinstance(data_summary, dict) else {"raw": data_summary},  # Full data
                execution_time_ms=execution_time_ms,
            )

        # Log to Google Sheets (truncated for cell limits)
        if self.sheets_logger.is_connected():
            self.sheets_logger.log_data_source(
                run_id=run_id,
                company_name=company_name,
                source_name=source_name,
                success=success,
                records_found=records_found,
                data_summary=str(data_summary)[:50000] if data_summary else "No data",  # Increased limit
                execution_time_ms=execution_time_ms,
            )

        logger.debug(f"[{run_id[:8]}] Data source {source_name}: {records_found} records ({execution_time_ms:.0f}ms)")

    def log_tool_call(
        self,
        run_id: str,
        company_name: str,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_output: Any,
        execution_time_ms: float,
        success: bool = True,
        error: str = None,
    ):
        """Log a tool call to all storage."""
        # Log to Google Sheets
        if self.sheets_logger.is_connected():
            self.sheets_logger.log_tool_call(
                run_id=run_id,
                company_name=company_name,
                tool_name=tool_name,
                tool_input=tool_input,
                tool_output=tool_output,
                execution_time_ms=execution_time_ms,
                success=success,
                error=error,
            )
        logger.debug(f"[{run_id[:8]}] Tool {tool_name}: {execution_time_ms:.0f}ms - {'OK' if success else 'FAILED'}")

    def log_assessment(
        self,
        run_id: str,
        company_name: str,
        risk_level: str,
        credit_score: int,
        confidence: float,
        reasoning: str,
        recommendations: List[str],
        risk_factors: List[str],
        positive_factors: List[str],
    ):
        """Log assessment to all storage."""
        # Log to Google Sheets
        if self.sheets_logger.is_connected():
            self.sheets_logger.log_assessment(
                run_id=run_id,
                company_name=company_name,
                risk_level=risk_level,
                credit_score=credit_score,
                confidence=confidence,
                reasoning=reasoning,
                recommendations=recommendations,
            )
        logger.info(f"[{run_id[:8]}] Assessment: {risk_level} (score: {credit_score})")

    def log_evaluation(
        self,
        run_id: str,
        company_name: str,
        tool_selection_score: float,
        data_quality_score: float,
        synthesis_score: float,
        overall_score: float,
        tool_reasoning: str = "",
        data_reasoning: str = "",
        synthesis_reasoning: str = "",
        details: Dict[str, Any] = None,
    ):
        """Log evaluation with reasoning to all storage."""
        # Log to Google Sheets
        if self.sheets_logger.is_connected():
            self.sheets_logger.log_evaluation(
                run_id=run_id,
                company_name=company_name,
                tool_selection_score=tool_selection_score,
                data_quality_score=data_quality_score,
                synthesis_score=synthesis_score,
                overall_score=overall_score,
                tool_reasoning=tool_reasoning,
                data_reasoning=data_reasoning,
                synthesis_reasoning=synthesis_reasoning,
            )
        logger.info(f"[{run_id[:8]}] Evaluation: {overall_score:.2f}")

    def log_tool_selection(
        self,
        run_id: str,
        company_name: str,
        selected_tools: List[str],
        expected_tools: List[str],
        precision: float,
        recall: float,
        f1_score: float,
        correct_tools: List[str] = None,
        missing_tools: List[str] = None,
        extra_tools: List[str] = None,
        reasoning: str = "",
    ):
        """Log tool selection with reasoning to all storage."""
        # Log to Google Sheets
        if self.sheets_logger.is_connected():
            self.sheets_logger.log_tool_selection(
                run_id=run_id,
                company_name=company_name,
                selected_tools=selected_tools,
                expected_tools=expected_tools,
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                correct_tools=correct_tools,
                missing_tools=missing_tools,
                extra_tools=extra_tools,
                reasoning=reasoning,
            )
        logger.info(f"[{run_id[:8]}] Tool selection F1: {f1_score:.2f}")

    def log_consistency(
        self,
        run_id: str,
        company_name: str,
        model_name: str,  # e.g., "llama-3.3-70b-versatile" or "overall"
        evaluation_type: str,
        num_runs: int,
        consistency_data: Dict[str, Any],
        risk_levels: List[str],
        credit_scores: List[int],
    ):
        """Log consistency evaluation to all storage."""
        # Log to MongoDB (model_name not yet supported in MongoDB schema)
        if self.run_logger.is_connected():
            self.run_logger.log_consistency_score(
                run_id=run_id,
                company_name=company_name,
                evaluation_type=evaluation_type,
                num_runs=num_runs,
                risk_level_consistency=consistency_data.get("risk_level_consistency", 0),
                score_consistency=consistency_data.get("score_consistency", 0),
                reasoning_similarity=consistency_data.get("reasoning_similarity", 0),
                overall_consistency=consistency_data.get("overall_consistency", 0),
                risk_levels=risk_levels,
                credit_scores=credit_scores,
            )

        # Log to Google Sheets (with model_name)
        if self.sheets_logger.is_connected():
            self.sheets_logger.log_consistency_score(
                run_id=run_id,
                company_name=company_name,
                model_name=model_name,
                evaluation_type=evaluation_type,
                num_runs=num_runs,
                risk_level_consistency=consistency_data.get("risk_level_consistency", 0),
                score_consistency=consistency_data.get("score_consistency", 0),
                score_std=consistency_data.get("score_std", 0),
                overall_consistency=consistency_data.get("overall_consistency", 0),
                risk_levels=risk_levels,
                credit_scores=credit_scores,
            )

        logger.info(f"[{run_id[:8]}] {model_name} {evaluation_type} consistency: {consistency_data.get('overall_consistency', 0):.2f}")

    def complete_run(self, run_id: str, final_result: Dict[str, Any], total_metrics: Dict[str, Any]):
        """Mark run as complete and log to all storage."""
        self.run_logger.complete_run(run_id, final_result, total_metrics)

        # Log to Google Sheets
        if self.sheets_logger.is_connected() and run_id in self._run_info:
            run_info = self._run_info[run_id]
            total_time_ms = (time.time() - run_info["started_at"]) * 1000

            # Extract tools_used from api_data keys or explicit tools_used field
            tools_used = final_result.get("tools_used", [])
            if not tools_used and "api_data" in final_result:
                tools_used = list(final_result["api_data"].keys())

            self.sheets_logger.log_run(
                run_id=run_id,
                company_name=run_info["company_name"],
                status="completed",
                risk_level=final_result.get("risk_level", ""),
                credit_score=final_result.get("credit_score", 0),
                confidence=final_result.get("confidence", 0),
                total_time_ms=total_time_ms,
                tools_used=tools_used,
                evaluation_score=final_result.get("evaluation_score", 0),
            )

        # Cleanup
        if run_id in self._step_counter:
            del self._step_counter[run_id]
        if run_id in self._run_info:
            del self._run_info[run_id]

    def fail_run(self, run_id: str, error: str):
        """Mark run as failed."""
        self.run_logger.fail_run(run_id, error)
        if run_id in self._step_counter:
            del self._step_counter[run_id]

    def _truncate_data(self, data: Dict[str, Any], max_str_len: int = 500) -> Dict[str, Any]:
        """Truncate string values in dict for storage."""
        if not isinstance(data, dict):
            return {"value": str(data)[:max_str_len]}

        result = {}
        for key, value in data.items():
            if isinstance(value, str) and len(value) > max_str_len:
                result[key] = value[:max_str_len] + "..."
            elif isinstance(value, dict):
                result[key] = self._truncate_data(value, max_str_len)
            elif isinstance(value, list) and len(value) > 10:
                result[key] = value[:10] + ["...truncated..."]
            else:
                result[key] = value
        return result

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

        Logs to both MongoDB and Google Sheets with all fields:
        - llm_provider, run_id, agent_name
        - prompt, context, response, reasoning
        - error, tokens, response_time_ms, costs
        """
        # Calculate cost if not provided
        if total_cost == 0 and (prompt_tokens > 0 or completion_tokens > 0):
            try:
                from config.cost_tracker import calculate_cost_for_tokens
                cost = calculate_cost_for_tokens(
                    model=model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    provider=llm_provider,
                )
                input_cost = cost["input_cost"]
                output_cost = cost["output_cost"]
                total_cost = cost["total_cost"]
            except Exception:
                pass

        # Log to MongoDB
        if self.run_logger.is_connected():
            self.run_logger.log_llm_call_detailed(
                run_id=run_id,
                company_name=company_name,
                llm_provider=llm_provider,
                agent_name=agent_name,
                model=model,
                prompt=prompt,
                context=context,
                response=response,
                reasoning=reasoning,
                error=error,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                response_time_ms=response_time_ms,
                input_cost=input_cost,
                output_cost=output_cost,
                total_cost=total_cost,
            )

        # Log to Google Sheets
        if self.sheets_logger.is_connected():
            self.sheets_logger.log_llm_call_detailed(
                run_id=run_id,
                company_name=company_name,
                llm_provider=llm_provider,
                agent_name=agent_name,
                model=model,
                prompt=prompt,
                context=context,
                response=response,
                reasoning=reasoning,
                error=error,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                response_time_ms=response_time_ms,
                input_cost=input_cost,
                output_cost=output_cost,
                total_cost=total_cost,
            )

        logger.debug(f"[{run_id[:8]}] LLM {agent_name}/{model}: {prompt_tokens}+{completion_tokens} tokens, ${total_cost:.6f}")

    def log_run_summary(
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

        Logs to both MongoDB and Google Sheets with all fields:
        - company_name, run_id, status
        - risk_level, credit_score, confidence, reasoning
        - ALL eval metrics
        - final_decision, decision_reasoning
        - errors, warnings, tools_used, agents_used
        - timing and cost information
        """
        # Log to MongoDB
        if self.run_logger.is_connected():
            self.run_logger.log_run_summary_detailed(
                run_id=run_id,
                company_name=company_name,
                status=status,
                risk_level=risk_level,
                credit_score=credit_score,
                confidence=confidence,
                reasoning=reasoning,
                tool_selection_score=tool_selection_score,
                data_quality_score=data_quality_score,
                synthesis_score=synthesis_score,
                overall_score=overall_score,
                final_decision=final_decision,
                decision_reasoning=decision_reasoning,
                errors=errors,
                warnings=warnings,
                tools_used=tools_used,
                agents_used=agents_used,
                started_at=started_at,
                completed_at=completed_at,
                duration_ms=duration_ms,
                total_tokens=total_tokens,
                total_cost=total_cost,
                llm_calls_count=llm_calls_count,
            )

        # Log to Google Sheets
        if self.sheets_logger.is_connected():
            self.sheets_logger.log_run_summary(
                run_id=run_id,
                company_name=company_name,
                status=status,
                risk_level=risk_level,
                credit_score=credit_score,
                confidence=confidence,
                reasoning=reasoning,
                tool_selection_score=tool_selection_score,
                data_quality_score=data_quality_score,
                synthesis_score=synthesis_score,
                overall_score=overall_score,
                final_decision=final_decision,
                decision_reasoning=decision_reasoning,
                errors=errors,
                warnings=warnings,
                tools_used=tools_used,
                agents_used=agents_used,
                started_at=started_at,
                completed_at=completed_at,
                duration_ms=duration_ms,
                total_tokens=total_tokens,
                total_cost=total_cost,
                llm_calls_count=llm_calls_count,
            )

        logger.info(f"[{run_id[:8]}] Run summary logged: {company_name} - {risk_level} (score: {credit_score})")


# Singleton instance
_workflow_logger: Optional[WorkflowLogger] = None


def get_workflow_logger() -> WorkflowLogger:
    """Get the global WorkflowLogger instance."""
    global _workflow_logger
    if _workflow_logger is None:
        _workflow_logger = WorkflowLogger()
    return _workflow_logger


def log_node(node_name: str):
    """
    Decorator to automatically log node execution.

    Usage:
        @log_node("parse_input")
        def parse_input(state):
            ...
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
            wf_logger = get_workflow_logger()
            run_id = state.get("run_id", "unknown")
            company_name = state.get("company_name", "unknown")

            # Capture relevant input
            input_summary = {
                "company_name": company_name,
                "status": state.get("status", ""),
            }

            start_time = time.time()
            error = None
            success = True

            try:
                result = func(state)

                # Capture relevant output
                output_summary = {
                    "status": result.get("status", ""),
                    "has_errors": bool(result.get("errors")),
                }

                # Add specific outputs based on node
                if "company_info" in result:
                    output_summary["company_info"] = {
                        "is_public": result["company_info"].get("is_public_company"),
                        "ticker": result["company_info"].get("ticker"),
                    }
                if "task_plan" in result:
                    output_summary["num_tasks"] = len(result.get("task_plan", []))
                if "api_data" in result:
                    output_summary["api_sources"] = list(result.get("api_data", {}).keys())
                if "search_data" in result:
                    output_summary["search_results"] = bool(result.get("search_data"))
                if "assessment" in result and result["assessment"]:
                    output_summary["risk_level"] = result["assessment"].get("overall_risk_level")
                    output_summary["credit_score"] = result["assessment"].get("credit_score_estimate")
                if "evaluation" in result:
                    output_summary["evaluation"] = result.get("evaluation", {})

                return result

            except Exception as e:
                error = str(e)
                success = False
                raise

            finally:
                execution_time_ms = (time.time() - start_time) * 1000

                wf_logger.log_step(
                    run_id=run_id,
                    company_name=company_name,
                    step_name=node_name,
                    input_data=input_summary,
                    output_data=output_summary if success else {"error": error},
                    execution_time_ms=execution_time_ms,
                    success=success,
                    error=error,
                )

        return wrapper
    return decorator
