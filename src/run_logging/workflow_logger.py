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
        self._llm_call_counter: Dict[str, int] = {}  # run_id -> llm call count
        self._run_info: Dict[str, Dict[str, Any]] = {}  # run_id -> run info

    def start_run(self, company_name: str, context: Dict[str, Any] = None, run_id: str = None) -> str:
        """Start a new run and return run_id.

        Args:
            company_name: Company being analyzed
            context: Additional context
            run_id: Optional run_id to use (if not provided, generates a new one)
        """
        run_id = self.run_logger.start_run(company_name, context, run_id=run_id)
        self._step_counter[run_id] = 0
        self._llm_call_counter[run_id] = 0
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

    def _increment_llm_calls(self, run_id: str) -> int:
        """Increment and return LLM call counter."""
        if run_id not in self._llm_call_counter:
            self._llm_call_counter[run_id] = 0
        self._llm_call_counter[run_id] += 1
        return self._llm_call_counter[run_id]

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
        # Node tracking fields
        node: str = "",
        agent_name: str = "",
        model: str = "",
        temperature: float = None,
    ):
        """Log a workflow step to all storage."""
        step_number = self._get_step_number(run_id)

        # Use step_name as node if not provided
        effective_node = node or step_name

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
                node=effective_node,
                agent_name=agent_name,
                model=model,
                temperature=temperature,
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
        # Node tracking fields
        node: str = "",
        agent_name: str = "",
        temperature: float = None,
        step_number: int = 0,
        # Task tracking
        current_task: str = "",
    ):
        """Log an LLM API call with cost and task tracking to all storage."""
        # Increment LLM call counter
        self._increment_llm_calls(run_id)

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

        # Log to Google Sheets with cost and task tracking
        if self.sheets_logger.is_connected():
            self.sheets_logger.log_llm_call(
                run_id=run_id,
                company_name=company_name,
                node=node or call_type,
                agent_name=agent_name,
                temperature=temperature,
                step_number=step_number,
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
                current_task=current_task,
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
        # Node tracking fields
        node: str = "fetch_api_data",
        agent_name: str = "data_fetcher",
        step_number: int = 0,
        error: str = "",
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
                node=node,
                agent_name=agent_name,
                step_number=step_number,
                success=success,
                records_found=records_found,
                data_summary=str(data_summary)[:50000] if data_summary else "No data",  # Increased limit
                execution_time_ms=execution_time_ms,
                error=error,
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
        # Node tracking fields
        node: str = "",
        agent_name: str = "",
        step_number: int = 0,
        # Hierarchy tracking fields
        parent_node: str = "",
        workflow_phase: str = "",
        call_depth: int = 0,
        parent_tool_id: str = "",
    ):
        """Log a tool call to all storage with hierarchy tracking."""
        # Use tool_name as node if not provided
        effective_node = node or tool_name

        # Log to Google Sheets
        if self.sheets_logger.is_connected():
            self.sheets_logger.log_tool_call(
                run_id=run_id,
                company_name=company_name,
                tool_name=tool_name,
                node=effective_node,
                agent_name=agent_name,
                step_number=step_number,
                tool_input=tool_input,
                tool_output=tool_output,
                execution_time_ms=execution_time_ms,
                success=success,
                error=error,
                # Pass hierarchy fields
                parent_node=parent_node or effective_node,
                workflow_phase=workflow_phase,
                call_depth=call_depth,
                parent_tool_id=parent_tool_id,
            )

        # Log to PostgreSQL via run_logger
        self.run_logger.log_tool_call(
            run_id=run_id,
            tool_name=tool_name,
            input_params=tool_input,
            output_data=tool_output,
            execution_time_ms=execution_time_ms,
            success=success,
            selection_reason=f"node={effective_node}, agent={agent_name}",
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
        # Node tracking fields
        node: str = "synthesize",
        agent_name: str = "llm_analyst",
        model: str = "",
        temperature: float = None,
        step_number: int = 0,
        prompt: str = "",
        duration_ms: float = 0,
        status: str = "ok",
    ):
        """Log assessment to all storage."""
        # Log to Google Sheets
        if self.sheets_logger.is_connected():
            self.sheets_logger.log_assessment(
                run_id=run_id,
                company_name=company_name,
                node=node,
                agent_name=agent_name,
                model=model,
                temperature=temperature,
                step_number=step_number,
                prompt=prompt,
                duration_ms=duration_ms,
                status=status,
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
        # Node tracking fields
        node: str = "evaluate",
        node_type: str = "agent",
        agent_name: str = "workflow_evaluator",
        step_number: int = 0,
        model: str = "",
        duration_ms: float = 0,
        status: str = "ok",
    ):
        """Log evaluation with reasoning to all storage."""
        # Log to Google Sheets
        if self.sheets_logger.is_connected():
            self.sheets_logger.log_evaluation(
                run_id=run_id,
                company_name=company_name,
                node=node,
                node_type=node_type,
                agent_name=agent_name,
                step_number=step_number,
                model=model,
                duration_ms=duration_ms,
                status=status,
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
        # Node tracking fields
        node: str = "create_plan",
        agent_name: str = "planner",
        step_number: int = 0,
        model: str = "",
    ):
        """Log tool selection with reasoning to all storage."""
        # Log to Google Sheets
        if self.sheets_logger.is_connected():
            self.sheets_logger.log_tool_selection(
                run_id=run_id,
                company_name=company_name,
                node=node,
                agent_name=agent_name,
                step_number=step_number,
                model=model,
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
        # Node tracking fields
        node: str = "evaluate",
        agent_name: str = "consistency_evaluator",
        step_number: int = 0,
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
                node=node,
                agent_name=agent_name,
                step_number=step_number,
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

            # Get counters before cleanup
            total_steps = self._step_counter.get(run_id, 0)
            total_llm_calls = self._llm_call_counter.get(run_id, 0)

            self.sheets_logger.log_run(
                run_id=run_id,
                company_name=run_info["company_name"],
                node="complete_run",
                agent_name="credit_intelligence_workflow",
                model=final_result.get("model", ""),
                status="completed",
                risk_level=final_result.get("risk_level", ""),
                credit_score=final_result.get("credit_score", 0),
                confidence=final_result.get("confidence", 0),
                total_time_ms=total_time_ms,
                total_steps=total_steps,
                total_llm_calls=total_llm_calls,
                tools_used=tools_used,
                evaluation_score=final_result.get("evaluation_score", 0),
            )

        # Cleanup
        if run_id in self._step_counter:
            del self._step_counter[run_id]
        if run_id in self._llm_call_counter:
            del self._llm_call_counter[run_id]
        if run_id in self._run_info:
            del self._run_info[run_id]

    def fail_run(self, run_id: str, error: str):
        """Mark run as failed."""
        self.run_logger.fail_run(run_id, error)
        if run_id in self._step_counter:
            del self._step_counter[run_id]
        if run_id in self._llm_call_counter:
            del self._llm_call_counter[run_id]

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
        # Node tracking fields
        node: str = "",
        step_number: int = 0,
        temperature: float = None,
        # Original fields
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
        # Increment LLM call counter
        self._increment_llm_calls(run_id)

        # Use agent_name as node if not provided
        effective_node = node or agent_name

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
                node=effective_node,
                step_number=step_number,
                temperature=temperature,
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
        # Node tracking fields
        node: str = "complete_run",
        agent_name: str = "workflow_orchestrator",
        model: str = "",
        temperature: float = None,
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
        total_steps: int = 0,
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
                node=node,
                agent_name=agent_name,
                model=model,
                temperature=temperature,
                total_steps=total_steps,
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
        # Node tracking fields
        node: str = "evaluate",
        agent_name: str = "agent_efficiency_evaluator",
        model: str = "",
    ):
        """
        Log agent efficiency metrics (Task 4 compliant).

        Logs standard agentic metrics to MongoDB and Google Sheets:
        - intent_correctness: Did the agent understand the task?
        - plan_quality: How good was the execution plan?
        - tool_choice_correctness: Did agent choose correct tools? (precision)
        - tool_completeness: Did agent use all needed tools? (recall)
        - trajectory_match: Did agent follow expected execution path?
        - final_answer_quality: Is the final output correct and complete?
        - step_count, tool_calls, latency_ms: Execution metrics
        """
        # Log to MongoDB
        if self.run_logger.is_connected():
            self.run_logger.log_agent_metrics(
                run_id=run_id,
                company_name=company_name,
                intent_correctness=intent_correctness,
                plan_quality=plan_quality,
                tool_choice_correctness=tool_choice_correctness,
                tool_completeness=tool_completeness,
                trajectory_match=trajectory_match,
                final_answer_quality=final_answer_quality,
                step_count=step_count,
                tool_calls=tool_calls,
                latency_ms=latency_ms,
                overall_score=overall_score,
                intent_details=intent_details,
                plan_details=plan_details,
                tool_details=tool_details,
                trajectory_details=trajectory_details,
                answer_details=answer_details,
            )

        # Log to Google Sheets
        if self.sheets_logger.is_connected():
            self.sheets_logger.log_agent_metrics(
                run_id=run_id,
                company_name=company_name,
                node=node,
                agent_name=agent_name,
                model=model,
                intent_correctness=intent_correctness,
                plan_quality=plan_quality,
                tool_choice_correctness=tool_choice_correctness,
                tool_completeness=tool_completeness,
                trajectory_match=trajectory_match,
                final_answer_quality=final_answer_quality,
                step_count=step_count,
                tool_calls=tool_calls,
                latency_ms=latency_ms,
                overall_score=overall_score,
                intent_details=intent_details,
                plan_details=plan_details,
                tool_details=tool_details,
                trajectory_details=trajectory_details,
                answer_details=answer_details,
            )

            # Also log to openevals_metrics sheet
            self.sheets_logger.log_openevals_metrics(
                run_id=run_id,
                company_name=company_name,
                model_used=model,
                node=node,
                agent_name=agent_name,
                intent_correctness=intent_correctness,
                plan_quality=plan_quality,
                tool_choice_correctness=tool_choice_correctness,
                tool_completeness=tool_completeness,
                trajectory_match=trajectory_match,
                final_answer_quality=final_answer_quality,
                step_count=step_count,
                tool_calls=tool_calls,
                latency_ms=latency_ms,
                overall_score=overall_score,
                intent_details=intent_details,
                plan_details=plan_details,
                tool_details=tool_details,
                trajectory_details=trajectory_details,
                answer_details=answer_details,
            )

        logger.info(f"[{run_id[:8]}] Agent metrics logged: overall_score={overall_score:.4f}")

    def log_unified_metrics(
        self,
        run_id: str,
        company_name: str,
        # Accuracy metrics
        faithfulness: float = 0.0,
        hallucination: float = 0.0,
        answer_relevancy: float = 0.0,
        factual_accuracy: float = 0.0,
        final_answer_quality: float = 0.0,
        accuracy_score: float = 0.0,
        # Consistency metrics
        same_model_consistency: float = 0.0,
        cross_model_consistency: float = 0.0,
        risk_level_agreement: float = 0.0,
        semantic_similarity: float = 0.0,
        consistency_score: float = 0.0,
        # Agent efficiency metrics
        intent_correctness: float = 0.0,
        plan_quality: float = 0.0,
        tool_choice_correctness: float = 0.0,
        tool_completeness: float = 0.0,
        trajectory_match: float = 0.0,
        agent_final_answer: float = 0.0,
        agent_efficiency_score: float = 0.0,
        # Overall
        overall_quality_score: float = 0.0,
        libraries_used: List[str] = None,
        evaluation_time_ms: float = 0.0,
        # Node tracking fields
        node: str = "evaluate",
        agent_name: str = "unified_evaluator",
        model: str = "",
    ):
        """
        Log unified evaluation metrics (DeepEval + OpenEvals + Built-in).

        Logs to Google Sheets (unified_metrics sheet).
        """
        # Log to Google Sheets
        if self.sheets_logger.is_connected():
            self.sheets_logger.log_unified_metrics(
                run_id=run_id,
                company_name=company_name,
                node=node,
                agent_name=agent_name,
                model=model,
                faithfulness=faithfulness,
                hallucination=hallucination,
                answer_relevancy=answer_relevancy,
                factual_accuracy=factual_accuracy,
                final_answer_quality=final_answer_quality,
                accuracy_score=accuracy_score,
                same_model_consistency=same_model_consistency,
                cross_model_consistency=cross_model_consistency,
                risk_level_agreement=risk_level_agreement,
                semantic_similarity=semantic_similarity,
                consistency_score=consistency_score,
                intent_correctness=intent_correctness,
                plan_quality=plan_quality,
                tool_choice_correctness=tool_choice_correctness,
                tool_completeness=tool_completeness,
                trajectory_match=trajectory_match,
                agent_final_answer=agent_final_answer,
                agent_efficiency_score=agent_efficiency_score,
                overall_quality_score=overall_quality_score,
                libraries_used=libraries_used,
                evaluation_time_ms=evaluation_time_ms,
            )

            # Also log DeepEval metrics if available (faithfulness, hallucination are from DeepEval)
            if faithfulness > 0 or hallucination > 0 or answer_relevancy > 0:
                self.sheets_logger.log_deepeval_metrics(
                    run_id=run_id,
                    company_name=company_name,
                    model_used=model,
                    node=node,
                    agent_name=agent_name,
                    answer_relevancy=answer_relevancy,
                    faithfulness=faithfulness,
                    hallucination=hallucination,
                    overall_score=accuracy_score,
                    evaluation_time_ms=evaluation_time_ms,
                )

        logger.info(f"[{run_id[:8]}] Unified metrics logged: accuracy={accuracy_score:.2f}, "
                   f"consistency={consistency_score:.2f}, agent={agent_efficiency_score:.2f}, "
                   f"overall={overall_quality_score:.2f}")

    def log_llm_judge_result(
        self,
        run_id: str,
        company_name: str,
        model_used: str,
        # Node tracking fields
        node: str = "evaluate",
        agent_name: str = "llm_judge",
        step_number: int = 0,
        temperature: float = None,
        # Dimension scores (0-1)
        accuracy_score: float = 0.0,
        completeness_score: float = 0.0,
        consistency_score: float = 0.0,
        actionability_score: float = 0.0,
        data_utilization_score: float = 0.0,
        overall_score: float = 0.0,
        # Reasoning
        accuracy_reasoning: str = "",
        completeness_reasoning: str = "",
        consistency_reasoning: str = "",
        actionability_reasoning: str = "",
        data_utilization_reasoning: str = "",
        overall_reasoning: str = "",
        # Benchmark comparison
        benchmark_alignment: float = 0.0,
        benchmark_comparison: str = "",
        # Suggestions
        suggestions: List[str] = None,
        # Metadata
        tokens_used: int = 0,
        evaluation_cost: float = 0.0,
    ):
        """
        Log LLM-as-a-judge evaluation result (Task 21 compliant).

        Logs to Google Sheets with all evaluation dimensions and reasoning.
        """
        # Log to Google Sheets
        if self.sheets_logger.is_connected():
            self.sheets_logger.log_llm_judge_result(
                run_id=run_id,
                company_name=company_name,
                model_used=model_used,
                node=node,
                agent_name=agent_name,
                step_number=step_number,
                temperature=temperature,
                accuracy_score=accuracy_score,
                completeness_score=completeness_score,
                consistency_score=consistency_score,
                actionability_score=actionability_score,
                data_utilization_score=data_utilization_score,
                overall_score=overall_score,
                accuracy_reasoning=accuracy_reasoning,
                completeness_reasoning=completeness_reasoning,
                consistency_reasoning=consistency_reasoning,
                actionability_reasoning=actionability_reasoning,
                data_utilization_reasoning=data_utilization_reasoning,
                overall_reasoning=overall_reasoning,
                benchmark_alignment=benchmark_alignment,
                benchmark_comparison=benchmark_comparison,
                suggestions=suggestions,
                tokens_used=tokens_used,
                evaluation_cost=evaluation_cost,
            )

        logger.info(f"[{run_id[:8]}] LLM Judge result logged: overall_score={overall_score:.4f}")


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
