"""Run Evaluation - Main entry point for running and evaluating the credit workflow."""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from typing import Any, Dict, List, Optional

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from agents.tool_supervisor import ToolSupervisor
from run_logging import MetricsCollector, get_run_logger, get_sheets_logger, get_workflow_logger
from evaluation import (
    WorkflowEvaluator,
    ToolSelectionEvaluator,
    ConsistencyScorer,
    EvaluationBrain,
    get_evaluation_brain,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class EvaluationRunner:
    """
    Main runner for credit intelligence evaluation.

    Supports:
    1. Single run evaluation
    2. Multi-run consistency evaluation (same model, N runs)
    3. Cross-model evaluation (different models)
    """

    def __init__(self, model: str = "primary", log_to_mongodb: bool = True, log_to_sheets: bool = True, run_llm_judge: bool = True):
        self.supervisor = ToolSupervisor(model=model)
        self.evaluator = WorkflowEvaluator()
        self.evaluation_brain = get_evaluation_brain()
        self.run_logger = get_run_logger() if log_to_mongodb else None
        self.sheets_logger = get_sheets_logger() if log_to_sheets else None
        self.workflow_logger = get_workflow_logger()
        self.run_llm_judge = run_llm_judge
        self.results_dir = "data/evaluation_results"
        os.makedirs(self.results_dir, exist_ok=True)

        if self.sheets_logger and self.sheets_logger.is_connected():
            logger.info(f"Google Sheets logging enabled: {self.sheets_logger.get_spreadsheet_url()}")

    def run_single_assessment(
        self,
        company_name: str,
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Run a single credit assessment with full evaluation.

        Args:
            company_name: Company to assess
            context: Additional context

        Returns:
            Complete assessment with evaluation
        """
        logger.info(f"Running assessment for: {company_name}")

        # Start logging
        run_id = None
        if self.run_logger and self.run_logger.is_connected():
            run_id = self.run_logger.start_run(company_name, context)

        try:
            # Run the assessment
            result = self.supervisor.run_full_assessment(company_name, context)
            run_id = run_id or result.get("run_id")

            # Evaluate the run using basic evaluator
            evaluation = self.evaluator.evaluate_single_run(
                run_id=run_id,
                company_name=company_name,
                tool_selection=result.get("tool_selection", {}),
                tool_results=result.get("tool_results", {}),
                assessment=result.get("assessment", {}),
                execution_metrics={"total_execution_time_ms": result.get("total_execution_time_ms", 0)},
            )

            # Add evaluation to result
            result["evaluation"] = evaluation.to_dict()

            # Run comprehensive evaluation with LLM Judge if enabled
            if self.run_llm_judge:
                try:
                    # Build state for EvaluationBrain
                    state = {
                        "company_info": result.get("tool_selection", {}).get("selection", {}).get("company_analysis", {}),
                        "api_data": result.get("tool_results", {}).get("results", {}),
                        "assessment": result.get("assessment", {}).get("assessment", {}),
                    }

                    # Run comprehensive evaluation
                    comprehensive_eval = self.evaluation_brain.evaluate_run(
                        run_id=run_id,
                        company_name=company_name,
                        state=state,
                        run_llm_judge=True,
                    )

                    # Add comprehensive evaluation to result
                    result["comprehensive_evaluation"] = comprehensive_eval.to_dict()

                    # Log LLM Judge results to Google Sheets (always log, even if score is 0)
                    if self.workflow_logger:
                        self.workflow_logger.log_llm_judge_result(
                            run_id=run_id,
                            company_name=company_name,
                            model_used="llama-3.3-70b-versatile",
                            accuracy_score=comprehensive_eval.llm_accuracy,
                            completeness_score=comprehensive_eval.llm_completeness,
                            consistency_score=comprehensive_eval.llm_consistency,
                            actionability_score=comprehensive_eval.llm_actionability,
                            data_utilization_score=comprehensive_eval.llm_data_utilization,
                            overall_score=comprehensive_eval.llm_judge_overall,
                            accuracy_reasoning=comprehensive_eval.llm_judge_details.get("accuracy_reasoning", ""),
                            completeness_reasoning=comprehensive_eval.llm_judge_details.get("completeness_reasoning", ""),
                            consistency_reasoning=comprehensive_eval.llm_judge_details.get("consistency_reasoning", ""),
                            actionability_reasoning=comprehensive_eval.llm_judge_details.get("actionability_reasoning", ""),
                            data_utilization_reasoning=comprehensive_eval.llm_judge_details.get("data_utilization_reasoning", ""),
                            overall_reasoning=comprehensive_eval.llm_judge_details.get("overall_reasoning", ""),
                            suggestions=comprehensive_eval.llm_suggestions,
                            tokens_used=comprehensive_eval.llm_judge_details.get("tokens_used", 0),
                            evaluation_cost=comprehensive_eval.llm_judge_details.get("cost", 0),
                        )

                    logger.info(f"  LLM Judge Score: {comprehensive_eval.llm_judge_overall:.2f}")
                    logger.info(f"  Overall Grade: {comprehensive_eval.overall_grade}")

                except Exception as e:
                    logger.warning(f"LLM Judge evaluation failed: {e}")

            # Log to all storage (MongoDB + Google Sheets)
            self._log_to_all(run_id, result, evaluation)

            logger.info(f"Assessment complete for {company_name}")
            logger.info(f"  Risk Level: {result.get('assessment', {}).get('assessment', {}).get('risk_level')}")
            logger.info(f"  Tool Selection Score: {evaluation.tool_selection_score:.2f}")
            logger.info(f"  Synthesis Score: {evaluation.synthesis_consistency:.2f}")

            return result

        except Exception as e:
            logger.error(f"Assessment failed: {e}")
            if self.run_logger and run_id:
                self.run_logger.fail_run(run_id, str(e))
            raise

    def run_consistency_evaluation(
        self,
        company_name: str,
        num_runs: int = 3,
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Run multiple assessments to evaluate consistency.

        Args:
            company_name: Company to assess
            num_runs: Number of runs to perform
            context: Additional context

        Returns:
            Consistency evaluation results
        """
        logger.info(f"Running {num_runs} assessments for consistency evaluation: {company_name}")

        assessments = []
        for i in range(num_runs):
            logger.info(f"Run {i+1}/{num_runs}...")
            try:
                result = self.supervisor.run_full_assessment(company_name, context)
                assessments.append(result)
            except Exception as e:
                logger.error(f"Run {i+1} failed: {e}")
                continue

        if len(assessments) < 2:
            return {"error": "Not enough successful runs for consistency evaluation"}

        # Evaluate consistency
        consistency = self.evaluator.evaluate_consistency(company_name, assessments)

        # Extract risk levels and credit scores for logging
        risk_levels = [a.get("assessment", {}).get("assessment", {}).get("risk_level", "unknown") for a in assessments]
        credit_scores = [a.get("assessment", {}).get("assessment", {}).get("credit_score", 0) for a in assessments]

        # Log consistency to MongoDB
        if self.run_logger and self.run_logger.is_connected():
            self.run_logger.log_consistency_score(
                run_id=assessments[0].get("run_id", "") if assessments else "",
                company_name=company_name,
                evaluation_type="same_model",
                num_runs=len(assessments),
                risk_level_consistency=consistency.get("risk_level_consistency", 0),
                score_consistency=consistency.get("score_consistency", 0),
                reasoning_similarity=consistency.get("reasoning_similarity", 0),
                overall_consistency=consistency.get("overall_consistency", 0),
                risk_levels=risk_levels,
                credit_scores=credit_scores,
            )

        # Log consistency to Google Sheets
        if self.sheets_logger and self.sheets_logger.is_connected():
            self.sheets_logger.log_consistency_score(
                run_id=assessments[0].get("run_id", "") if assessments else "",
                company_name=company_name,
                model_name="same_model_evaluation",
                evaluation_type="same_model",
                num_runs=len(assessments),
                risk_level_consistency=consistency.get("risk_level_consistency", 0),
                score_consistency=consistency.get("score_consistency", 0),
                score_std=consistency.get("score_std", 0),
                overall_consistency=consistency.get("overall_consistency", 0),
                risk_levels=risk_levels,
                credit_scores=credit_scores,
            )

        result = {
            "company_name": company_name,
            "num_runs": len(assessments),
            "consistency": consistency,
            "assessments": assessments,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Log summary
        logger.info(f"Consistency Evaluation Complete:")
        logger.info(f"  Overall Consistency: {consistency.get('overall_consistency', 0):.2f}")
        logger.info(f"  Risk Level Consistency: {consistency.get('risk_level_consistency', 0):.2f}")
        logger.info(f"  Credit Score Range: {consistency.get('score_range', 0)}")

        return result

    def run_batch_evaluation(
        self,
        companies: List[str],
        num_runs_per_company: int = 1,
    ) -> Dict[str, Any]:
        """
        Run evaluation for multiple companies.

        Args:
            companies: List of company names
            num_runs_per_company: Runs per company for consistency

        Returns:
            Batch evaluation results
        """
        logger.info(f"Running batch evaluation for {len(companies)} companies")

        results = []
        for company in companies:
            try:
                if num_runs_per_company > 1:
                    result = self.run_consistency_evaluation(
                        company, num_runs_per_company
                    )
                else:
                    result = self.run_single_assessment(company)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed for {company}: {e}")
                results.append({"company_name": company, "error": str(e)})

        # Get summary
        summary = self.evaluator.get_summary()

        batch_result = {
            "total_companies": len(companies),
            "successful": len([r for r in results if "error" not in r]),
            "summary": summary,
            "results": results,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Save results
        self._save_results(batch_result, "batch_evaluation")

        return batch_result

    def _log_to_mongodb(self, run_id: str, result: Dict[str, Any]):
        """Log results to MongoDB - comprehensive logging."""
        try:
            company_name = result.get("company_name", "")
            tool_selection = result.get("tool_selection", {})
            selection = tool_selection.get("selection", {})
            tool_results = result.get("tool_results", {})

            # 1. Log tool selection
            self.run_logger.log_tool_selection(
                run_id=run_id,
                company_name=company_name,
                tools_selected=[t.get("name") for t in selection.get("tools_to_use", [])],
                selection_reasoning=selection,
                llm_metrics=tool_selection.get("llm_metrics", {}),
            )

            # 2. Log tool executions with detailed steps
            step_num = 1
            for metric in tool_results.get("execution_metrics", []):
                tool_name = metric.get("tool_name", "")
                success = metric.get("success", False)
                exec_time = metric.get("execution_time_ms", 0)

                # Log tool call
                self.run_logger.log_tool_call(
                    run_id=run_id,
                    tool_name=tool_name,
                    input_params={},
                    output_data={},
                    execution_time_ms=exec_time,
                    success=success,
                    selection_reason=metric.get("reason", ""),
                )

                # Log workflow step
                self.run_logger.log_workflow_step(
                    run_id=run_id,
                    step_name=f"Execute {tool_name}",
                    step_number=step_num,
                    input_data={"tool": tool_name, "reason": metric.get("reason", "")},
                    output_data={"success": success, "time_ms": exec_time},
                    execution_time_ms=exec_time,
                    success=success,
                )

                # Log data source
                data = tool_results.get("results", {}).get(tool_name, {})
                records = 0
                if isinstance(data, dict):
                    records = len(data.get("filings", [])) or len(data.get("data", [])) or (1 if data else 0)
                self.run_logger.log_data_source(
                    run_id=run_id,
                    source_name=tool_name,
                    success=success,
                    records_found=records,
                    data_summary={"sample": str(data)[:500]} if data else {},
                    execution_time_ms=exec_time,
                )
                step_num += 1

            # 3. Log LLM calls
            ts_metrics = tool_selection.get("llm_metrics", {})
            if ts_metrics:
                self.run_logger.log_llm_call(
                    run_id=run_id,
                    call_type="tool_selection",
                    model=ts_metrics.get("model", "unknown"),
                    prompt=f"Select tools for {company_name}",
                    response=str([t.get("name") for t in selection.get("tools_to_use", [])]),
                    prompt_tokens=ts_metrics.get("prompt_tokens", 0),
                    completion_tokens=ts_metrics.get("completion_tokens", 0),
                    execution_time_ms=ts_metrics.get("execution_time_ms", 0),
                )

            assess_metrics = result.get("assessment", {}).get("llm_metrics", {})
            if assess_metrics:
                assessment_data = result.get("assessment", {}).get("assessment", {})
                self.run_logger.log_llm_call(
                    run_id=run_id,
                    call_type="assessment",
                    model=assess_metrics.get("model", "unknown"),
                    prompt=f"Generate assessment for {company_name}",
                    response=f"Risk: {assessment_data.get('risk_level')}, Score: {assessment_data.get('credit_score')}",
                    prompt_tokens=assess_metrics.get("prompt_tokens", 0),
                    completion_tokens=assess_metrics.get("completion_tokens", 0),
                    execution_time_ms=assess_metrics.get("execution_time_ms", 0),
                )

            # 4. Log assessment
            assessment = result.get("assessment", {})
            self.run_logger.log_assessment(
                run_id=run_id,
                company_name=company_name,
                assessment=assessment.get("assessment", {}),
                llm_metrics=assessment.get("llm_metrics", {}),
            )

            # 5. Complete run
            self.run_logger.complete_run(
                run_id=run_id,
                final_result=result,
                total_metrics={
                    "total_execution_time_ms": result.get("total_execution_time_ms", 0),
                },
            )

        except Exception as e:
            logger.error(f"Failed to log to MongoDB: {e}")

    def _log_to_sheets(self, run_id: str, result: Dict[str, Any], evaluation: Any = None):
        """Log results to Google Sheets - comprehensive logging."""
        if not self.sheets_logger or not self.sheets_logger.is_connected():
            return

        try:
            company_name = result.get("company_name", "")
            tool_selection = result.get("tool_selection", {})
            selection = tool_selection.get("selection", {})
            assessment = result.get("assessment", {}).get("assessment", {})
            tool_results = result.get("tool_results", {})

            # 1. Log tool selection
            tools_selected = [t.get("name") for t in selection.get("tools_to_use", [])]
            self.sheets_logger.log_tool_selection(
                run_id=run_id,
                company_name=company_name,
                selected_tools=tools_selected,
                reasoning=selection.get("company_analysis", {}).get("reasoning", ""),
            )

            # 2. Log each tool call and data source
            step_num = 1
            for metric in tool_results.get("execution_metrics", []):
                tool_name = metric.get("tool_name", "")
                success = metric.get("success", False)
                exec_time = metric.get("execution_time_ms", 0)

                # Log tool call
                self.sheets_logger.log_tool_call(
                    run_id=run_id,
                    company_name=company_name,
                    tool_name=tool_name,
                    tool_input={"reason": metric.get("reason", "")},
                    tool_output={},
                    execution_time_ms=exec_time,
                    success=success,
                )

                # Log as workflow step
                self.sheets_logger.log_step(
                    run_id=run_id,
                    company_name=company_name,
                    step_name=f"Execute {tool_name}",
                    step_number=step_num,
                    input_data={"tool": tool_name, "reason": metric.get("reason", "")},
                    output_data={"success": success, "time_ms": exec_time},
                    execution_time_ms=exec_time,
                    success=success,
                )

                # Log data source result
                data = tool_results.get("results", {}).get(tool_name, {})
                records = 0
                if isinstance(data, dict):
                    records = len(data.get("filings", [])) or len(data.get("data", [])) or (1 if data else 0)
                self.sheets_logger.log_data_source(
                    run_id=run_id,
                    company_name=company_name,
                    source_name=tool_name,
                    success=success,
                    records_found=records,
                    data_summary=str(data)[:200] if data else "No data",
                    execution_time_ms=exec_time,
                )
                step_num += 1

            # 3. Log LLM calls
            # Tool selection LLM call
            ts_metrics = tool_selection.get("llm_metrics", {})
            if ts_metrics:
                self.sheets_logger.log_llm_call(
                    run_id=run_id,
                    company_name=company_name,
                    call_type="tool_selection",
                    model=ts_metrics.get("model", "unknown"),
                    prompt=f"Select tools for {company_name}",
                    response=str(tools_selected),
                    prompt_tokens=ts_metrics.get("prompt_tokens", 0),
                    completion_tokens=ts_metrics.get("completion_tokens", 0),
                    execution_time_ms=ts_metrics.get("execution_time_ms", 0),
                )

            # Assessment LLM call
            assess_metrics = result.get("assessment", {}).get("llm_metrics", {})
            if assess_metrics:
                self.sheets_logger.log_llm_call(
                    run_id=run_id,
                    company_name=company_name,
                    call_type="assessment",
                    model=assess_metrics.get("model", "unknown"),
                    prompt=f"Generate assessment for {company_name}",
                    response=f"Risk: {assessment.get('risk_level')}, Score: {assessment.get('credit_score')}",
                    prompt_tokens=assess_metrics.get("prompt_tokens", 0),
                    completion_tokens=assess_metrics.get("completion_tokens", 0),
                    execution_time_ms=assess_metrics.get("execution_time_ms", 0),
                )

            # 4. Log assessment
            self.sheets_logger.log_assessment(
                run_id=run_id,
                company_name=company_name,
                risk_level=assessment.get("risk_level", ""),
                credit_score=assessment.get("credit_score", 0) or 0,
                confidence=assessment.get("confidence", 0) or 0,
                reasoning=assessment.get("reasoning", ""),
                recommendations=assessment.get("recommendations", []),
            )

            # 5. Log evaluation if available
            if evaluation:
                tool_score = getattr(evaluation, 'tool_selection_score', 0) or 0
                data_score = getattr(evaluation, 'data_completeness', 0) or 0
                synthesis_score = getattr(evaluation, 'synthesis_consistency', 0) or 0
                overall = (tool_score + data_score + synthesis_score) / 3 if any([tool_score, data_score, synthesis_score]) else 0

                self.sheets_logger.log_evaluation(
                    run_id=run_id,
                    company_name=company_name,
                    tool_selection_score=tool_score,
                    data_quality_score=data_score,
                    synthesis_score=synthesis_score,
                    overall_score=overall,
                )

            # 6. Log run summary
            self.sheets_logger.log_run(
                run_id=run_id,
                company_name=company_name,
                status="completed",
                risk_level=assessment.get("risk_level", ""),
                credit_score=assessment.get("credit_score"),
                confidence=assessment.get("confidence"),
                total_time_ms=result.get("total_execution_time_ms", 0),
                tools_used=tools_selected,
                evaluation_score=getattr(evaluation, 'tool_selection_score', None) if evaluation else None,
            )

            logger.info(f"Logged to Google Sheets: {run_id}")

        except Exception as e:
            logger.error(f"Failed to log to Google Sheets: {e}")

    def _log_to_all(self, run_id: str, result: Dict[str, Any], evaluation: Any = None):
        """Log to both MongoDB and Google Sheets."""
        if self.run_logger and self.run_logger.is_connected():
            self._log_to_mongodb(run_id, result)
        self._log_to_sheets(run_id, result, evaluation)

    def run_cross_model_evaluation(
        self,
        company_name: str,
        models: List[str] = None,
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Run assessment with multiple different models to evaluate cross-model consistency.

        Args:
            company_name: Company to assess
            models: List of models to use (default: all 3)
            context: Additional context

        Returns:
            Cross-model consistency evaluation results
        """
        from agents.tool_supervisor import ToolSupervisor

        models = models or ["primary", "fast", "balanced"]
        model_names = {
            "primary": "llama-3.3-70b-versatile",
            "fast": "llama-3.1-8b-instant",
            "balanced": "gemma2-9b-it",
        }

        logger.info(f"Running cross-model evaluation for {company_name} with {len(models)} models")

        model_assessments = {}

        for model_key in models:
            model_name = model_names.get(model_key, model_key)
            logger.info(f"Running with model: {model_name}...")

            try:
                supervisor = ToolSupervisor(model=model_key)
                result = supervisor.run_full_assessment(company_name, context)
                model_assessments[model_name] = result
            except Exception as e:
                logger.error(f"Model {model_name} failed: {e}")
                model_assessments[model_name] = {"error": str(e)}

        # Evaluate cross-model consistency
        valid_assessments = {k: v for k, v in model_assessments.items() if "error" not in v}

        if len(valid_assessments) < 2:
            return {"error": "Not enough successful model runs for cross-model evaluation"}

        consistency = self.evaluator.evaluate_cross_model_consistency(
            company_name,
            valid_assessments,
        )

        # Extract data for logging
        risk_levels = [v.get("assessment", {}).get("assessment", {}).get("risk_level", "unknown") for v in valid_assessments.values()]
        credit_scores = [v.get("assessment", {}).get("assessment", {}).get("credit_score", 0) for v in valid_assessments.values()]

        # Log cross-model consistency to MongoDB
        if self.run_logger and self.run_logger.is_connected():
            self.run_logger.log_consistency_score(
                run_id=list(valid_assessments.values())[0].get("run_id", "") if valid_assessments else "",
                company_name=company_name,
                evaluation_type="cross_model",
                num_runs=len(valid_assessments),
                risk_level_consistency=consistency.get("risk_level_consistency", 0),
                score_consistency=consistency.get("score_consistency", 0),
                reasoning_similarity=consistency.get("reasoning_similarity", 0),
                overall_consistency=consistency.get("overall_consistency", 0),
                risk_levels=risk_levels,
                credit_scores=credit_scores,
            )

        # Log cross-model consistency to Google Sheets
        if self.sheets_logger and self.sheets_logger.is_connected():
            self.sheets_logger.log_consistency_score(
                run_id=list(valid_assessments.values())[0].get("run_id", "") if valid_assessments else "",
                company_name=company_name,
                model_name="cross_model_evaluation",
                evaluation_type="cross_model",
                num_runs=len(valid_assessments),
                risk_level_consistency=consistency.get("risk_level_consistency", 0),
                score_consistency=consistency.get("score_consistency", 0),
                score_std=consistency.get("score_std", 0),
                overall_consistency=consistency.get("overall_consistency", 0),
                risk_levels=risk_levels,
                credit_scores=credit_scores,
            )

        result = {
            "company_name": company_name,
            "evaluation_type": "cross_model",
            "models_used": list(model_assessments.keys()),
            "successful_models": list(valid_assessments.keys()),
            "consistency": consistency,
            "model_results": {
                model: {
                    "risk_level": v.get("assessment", {}).get("assessment", {}).get("risk_level"),
                    "credit_score": v.get("assessment", {}).get("assessment", {}).get("credit_score"),
                    "confidence": v.get("assessment", {}).get("assessment", {}).get("confidence"),
                }
                for model, v in valid_assessments.items()
            },
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Log summary
        logger.info(f"\nCross-Model Evaluation Complete:")
        logger.info(f"  Models: {list(valid_assessments.keys())}")
        logger.info(f"  Overall Consistency: {consistency.get('overall_consistency', 0):.2f}")
        logger.info(f"  Risk Level Consistency: {consistency.get('risk_level_consistency', 0):.2f}")
        logger.info(f"  Credit Score Range: {consistency.get('score_range', 0)}")

        # Print individual model results
        for model, data in result["model_results"].items():
            logger.info(f"  {model}: risk={data['risk_level']}, score={data['credit_score']}")

        # Save results
        self._save_results(result, "cross_model_evaluation")

        return result

    def run_full_evaluation(
        self,
        company_name: str,
        same_model_runs: int = 3,
        cross_model: bool = True,
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Run BOTH same-model consistency AND cross-model validation.

        This is the comprehensive evaluation that combines both approaches.

        Args:
            company_name: Company to assess
            same_model_runs: Number of runs for same-model consistency
            cross_model: Whether to also run cross-model evaluation
            context: Additional context

        Returns:
            Complete evaluation with both consistency scores
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"FULL EVALUATION: {company_name}")
        logger.info(f"{'='*60}")

        results = {
            "company_name": company_name,
            "evaluation_type": "full",
            "timestamp": datetime.utcnow().isoformat(),
        }

        # 1. Same-Model Consistency (run N times with primary model)
        logger.info(f"\n[1/2] Same-Model Consistency ({same_model_runs} runs)...")
        same_model_result = self.run_consistency_evaluation(
            company_name, same_model_runs, context
        )
        results["same_model_consistency"] = same_model_result.get("consistency", {})
        results["same_model_runs"] = same_model_runs

        # 2. Cross-Model Validation (run with 3 different models)
        if cross_model:
            logger.info(f"\n[2/2] Cross-Model Validation...")
            cross_model_result = self.run_cross_model_evaluation(company_name, context=context)
            results["cross_model_consistency"] = cross_model_result.get("consistency", {})
            results["model_results"] = cross_model_result.get("model_results", {})

        # Summary
        logger.info(f"\n{'='*60}")
        logger.info(f"EVALUATION SUMMARY: {company_name}")
        logger.info(f"{'='*60}")
        logger.info(f"Same-Model Consistency: {results.get('same_model_consistency', {}).get('overall_consistency', 'N/A')}")
        if cross_model:
            logger.info(f"Cross-Model Consistency: {results.get('cross_model_consistency', {}).get('overall_consistency', 'N/A')}")

        # Save results
        self._save_results(results, "full_evaluation")

        return results

    def _save_results(self, results: Dict[str, Any], prefix: str):
        """Save results to JSON file."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)

        with open(filepath, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Results saved to: {filepath}")
        return filepath


def main():
    parser = argparse.ArgumentParser(description="Run Credit Intelligence Evaluation")

    parser.add_argument(
        "--company",
        type=str,
        help="Single company to evaluate",
    )
    parser.add_argument(
        "--companies",
        type=str,
        help="Comma-separated list of companies",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of runs per company for same-model consistency",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="primary",
        choices=["primary", "fast", "balanced"],
        help="Model to use for single/consistency runs",
    )
    parser.add_argument(
        "--cross-model",
        action="store_true",
        help="Run cross-model validation (uses all 3 models)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run FULL evaluation (same-model consistency + cross-model validation)",
    )
    parser.add_argument(
        "--no-mongodb",
        action="store_true",
        help="Disable MongoDB logging",
    )
    parser.add_argument(
        "--no-llm-judge",
        action="store_true",
        help="Disable LLM-as-a-Judge evaluation (saves tokens)",
    )

    args = parser.parse_args()

    runner = EvaluationRunner(
        model=args.model,
        log_to_mongodb=not args.no_mongodb,
        run_llm_judge=not args.no_llm_judge,
    )

    if args.company:
        if args.full:
            # Run both same-model and cross-model evaluation
            result = runner.run_full_evaluation(
                args.company,
                same_model_runs=args.runs if args.runs > 1 else 3,
                cross_model=True,
            )
        elif args.cross_model:
            # Run only cross-model validation
            result = runner.run_cross_model_evaluation(args.company)
        elif args.runs > 1:
            # Run same-model consistency
            result = runner.run_consistency_evaluation(args.company, args.runs)
        else:
            # Single run
            result = runner.run_single_assessment(args.company)
        print(json.dumps(result, indent=2, default=str))

    elif args.companies:
        companies = [c.strip() for c in args.companies.split(",")]
        result = runner.run_batch_evaluation(companies, args.runs)
        print(json.dumps(result, indent=2, default=str))

    else:
        # Run sample evaluation
        sample_companies = [
            "Apple Inc",
            "Microsoft Corporation",
            "Acme Private Co",
        ]
        result = runner.run_batch_evaluation(sample_companies)
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
