"""
Batch Evaluation Runner - Runs evaluations with comprehensive logging.

This script:
1. Runs assessments for multiple companies (multiple runs each)
2. Logs EACH individual run to Google Sheets
3. Evaluates consistency across runs
4. Logs LLM Judge results for EACH run
5. Creates detailed per-run logging in langsmith_eval_runs
"""

import os
import sys
import uuid
import time
import logging
import argparse
from datetime import datetime, timezone
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.tool_supervisor import ToolSupervisor
from evaluation import (
    LLMJudgeEvaluator,
    WorkflowEvaluator,
    get_evaluation_brain,
)
from evaluation.langsmith_eval.logger import get_eval_logger, EvaluationRun, ExampleResult
from run_logging import get_sheets_logger, get_run_logger, get_workflow_logger

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Test companies with expected values
TEST_COMPANIES = [
    {
        "name": "Apple Inc",
        "ticker": "AAPL",
        "expected_risk_level": "low",
        "expected_score_range": (70, 95),
        "expected_tools": ["fetch_sec_data", "fetch_market_data", "fetch_legal_data"],
        "company_type": "public_us",
    },
    {
        "name": "Microsoft Corporation",
        "ticker": "MSFT",
        "expected_risk_level": "low",
        "expected_score_range": (75, 95),
        "expected_tools": ["fetch_sec_data", "fetch_market_data", "fetch_legal_data"],
        "company_type": "public_us",
    },
    {
        "name": "Tesla Inc",
        "ticker": "TSLA",
        "expected_risk_level": "low",
        "expected_score_range": (60, 90),
        "expected_tools": ["fetch_sec_data", "fetch_market_data", "fetch_legal_data"],
        "company_type": "public_us",
    },
]


class BatchEvaluationRunner:
    """Runs batch evaluations with detailed per-run logging."""

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        num_runs_per_company: int = 3,
    ):
        self.supervisor = ToolSupervisor(model=model)
        self.model_name = model
        self.num_runs_per_company = num_runs_per_company
        self.llm_judge = LLMJudgeEvaluator()
        self.workflow_evaluator = WorkflowEvaluator()
        self.evaluation_brain = get_evaluation_brain()

        # All loggers
        self.eval_logger = get_eval_logger()
        self.sheets_logger = get_sheets_logger()
        self.run_logger = get_run_logger()
        self.workflow_logger = get_workflow_logger()

        logger.info(f"BatchEvaluationRunner initialized")
        logger.info(f"  Model: {self.model_name}")
        logger.info(f"  Runs per company: {self.num_runs_per_company}")
        logger.info(f"  Sheets: {self.sheets_logger.is_connected() if self.sheets_logger else False}")
        logger.info(f"  MongoDB: {self.run_logger.is_connected() if self.run_logger else False}")

    def run_single_assessment(self, company_name: str, run_number: int) -> Dict[str, Any]:
        """Run a single assessment and return detailed results."""
        run_id = str(uuid.uuid4())
        start_time = time.time()

        try:
            result = self.supervisor.run_full_assessment(company_name)
            duration_ms = (time.time() - start_time) * 1000

            assessment = result.get("assessment", {}).get("assessment", {})
            tool_selection = result.get("tool_selection", {})
            tool_results = result.get("tool_results", {})

            # Get tools that were actually selected and used
            selected_tools = tool_selection.get("tools_selected", [])
            if not selected_tools:
                # Try to get from selection dict
                selection = tool_selection.get("selection", {})
                tools_to_use = selection.get("tools_to_use", [])
                selected_tools = [t.get("name") for t in tools_to_use if isinstance(t, dict)]

            return {
                "run_id": run_id,
                "run_number": run_number,
                "company_name": company_name,
                "model_name": self.model_name,
                "success": True,
                "duration_ms": duration_ms,
                # Assessment results
                "risk_level": assessment.get("risk_level", "unknown"),
                "credit_score": assessment.get("credit_score", 0),
                "confidence": assessment.get("confidence", 0),
                "reasoning": assessment.get("reasoning", ""),
                "risk_factors": assessment.get("risk_factors", []),
                "positive_factors": assessment.get("positive_factors", []),
                "recommendations": assessment.get("recommendations", []),
                # Tool selection
                "tools_selected": selected_tools,
                "tool_selection_reasoning": tool_selection.get("selection", {}).get("execution_order_reasoning", ""),
                # Raw result for further analysis
                "full_result": result,
            }
        except Exception as e:
            logger.error(f"Assessment failed for {company_name} (run {run_number}): {e}")
            return {
                "run_id": run_id,
                "run_number": run_number,
                "company_name": company_name,
                "model_name": self.model_name,
                "success": False,
                "error": str(e),
                "duration_ms": (time.time() - start_time) * 1000,
            }

    def calculate_tool_selection_f1(self, selected: List[str], expected: List[str]) -> Dict[str, float]:
        """Calculate precision, recall, F1 for tool selection."""
        selected_set = set(selected)
        expected_set = set(expected)

        true_positives = len(selected_set & expected_set)
        precision = true_positives / len(selected_set) if selected_set else 0.0
        recall = true_positives / len(expected_set) if expected_set else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "correct_tools": list(selected_set & expected_set),
            "missing_tools": list(expected_set - selected_set),
            "extra_tools": list(selected_set - expected_set),
        }

    def run_company_evaluation(self, company_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run multiple assessments for a company and log EACH run individually.
        """
        company_name = company_config["name"]
        expected_risk = company_config["expected_risk_level"]
        expected_score_range = company_config["expected_score_range"]
        expected_tools = company_config["expected_tools"]

        logger.info(f"\n{'='*50}")
        logger.info(f"Evaluating: {company_name}")
        logger.info(f"Expected: risk={expected_risk}, score={expected_score_range}, tools={expected_tools}")
        logger.info(f"Running {self.num_runs_per_company} assessments...")
        logger.info(f"{'='*50}")

        all_runs = []

        for run_num in range(1, self.num_runs_per_company + 1):
            logger.info(f"\n--- Run {run_num}/{self.num_runs_per_company} ---")

            # Run assessment
            run_result = self.run_single_assessment(company_name, run_num)

            if run_result.get("success"):
                # Calculate tool selection metrics
                tool_metrics = self.calculate_tool_selection_f1(
                    run_result.get("tools_selected", []),
                    expected_tools
                )
                run_result["tool_selection_f1"] = tool_metrics["f1"]
                run_result["tool_selection_precision"] = tool_metrics["precision"]
                run_result["tool_selection_recall"] = tool_metrics["recall"]
                run_result["correct_tools"] = tool_metrics["correct_tools"]
                run_result["missing_tools"] = tool_metrics["missing_tools"]
                run_result["extra_tools"] = tool_metrics["extra_tools"]

                # Check accuracy vs expected
                run_result["risk_level_correct"] = run_result["risk_level"].lower() == expected_risk.lower()
                score = run_result["credit_score"]
                run_result["score_in_range"] = expected_score_range[0] <= score <= expected_score_range[1]

                # Run LLM Judge for THIS specific run
                llm_judge_result = self.llm_judge.evaluate(
                    run_id=run_result["run_id"],
                    company_name=company_name,
                    assessment=run_result.get("full_result", {}).get("assessment", {}).get("assessment", {}),
                    api_data=run_result.get("full_result", {}).get("tool_results", {}).get("results", {}),
                )
                run_result["llm_judge_score"] = llm_judge_result.overall_score
                run_result["llm_judge_accuracy"] = llm_judge_result.accuracy_score
                run_result["llm_judge_completeness"] = llm_judge_result.completeness_score
                run_result["llm_judge_consistency"] = llm_judge_result.consistency_score
                run_result["llm_judge_reasoning"] = llm_judge_result.overall_reasoning

                # Log THIS run to langsmith_eval_runs (one row per run)
                self._log_individual_run(run_result, company_config)

                # Log LLM Judge result for THIS run
                if self.sheets_logger and self.sheets_logger.is_connected():
                    self.sheets_logger.log_llm_judge_result(
                        run_id=run_result["run_id"],
                        company_name=company_name,
                        model_used=self.model_name,
                        accuracy_score=llm_judge_result.accuracy_score,
                        completeness_score=llm_judge_result.completeness_score,
                        consistency_score=llm_judge_result.consistency_score,
                        actionability_score=llm_judge_result.actionability_score,
                        data_utilization_score=llm_judge_result.data_utilization_score,
                        overall_score=llm_judge_result.overall_score,
                        accuracy_reasoning=llm_judge_result.accuracy_reasoning,
                        completeness_reasoning=llm_judge_result.completeness_reasoning,
                        consistency_reasoning=llm_judge_result.consistency_reasoning,
                        actionability_reasoning=llm_judge_result.actionability_reasoning,
                        data_utilization_reasoning=llm_judge_result.data_utilization_reasoning,
                        overall_reasoning=llm_judge_result.overall_reasoning,
                        suggestions=llm_judge_result.suggestions,
                        tokens_used=llm_judge_result.tokens_used,
                        evaluation_cost=llm_judge_result.evaluation_cost,
                    )

                logger.info(f"  Risk: {run_result['risk_level']} (correct: {run_result['risk_level_correct']})")
                logger.info(f"  Score: {run_result['credit_score']} (in range: {run_result['score_in_range']})")
                logger.info(f"  Tools: {run_result['tools_selected']} (F1: {run_result['tool_selection_f1']:.2f})")
                logger.info(f"  LLM Judge: {run_result['llm_judge_score']:.2f}")

            all_runs.append(run_result)

            # Rate limiting between runs
            if run_num < self.num_runs_per_company:
                time.sleep(2)

        # Now evaluate consistency across all runs
        successful_runs = [r for r in all_runs if r.get("success")]

        if len(successful_runs) >= 2:
            # Evaluate consistency using LLM Judge
            consistency_result = self.llm_judge.evaluate_consistency(
                company_name=company_name,
                assessments=[r for r in successful_runs],
                model_name=self.model_name,
            )

            # Log consistency evaluation
            if self.sheets_logger and self.sheets_logger.is_connected():
                self.sheets_logger.log_model_consistency(
                    eval_id=str(uuid.uuid4())[:8],
                    company_name=company_name,
                    model_name=self.model_name,
                    num_runs=len(successful_runs),
                    risk_level_consistency=consistency_result.risk_level_consistency,
                    credit_score_mean=consistency_result.credit_score_mean,
                    credit_score_std=consistency_result.credit_score_std,
                    confidence_variance=consistency_result.confidence_variance,
                    reasoning_similarity=consistency_result.reasoning_similarity,
                    risk_factors_overlap=consistency_result.risk_factors_overlap,
                    recommendations_overlap=consistency_result.recommendations_overlap,
                    overall_consistency=consistency_result.overall_consistency,
                    is_consistent=consistency_result.is_consistent,
                    consistency_grade=consistency_result.consistency_grade,
                    llm_judge_analysis=consistency_result.llm_judge_analysis,
                    llm_judge_concerns=consistency_result.llm_judge_concerns,
                    run_details=consistency_result.run_details,
                )

            logger.info(f"\n--- Consistency Results ---")
            logger.info(f"  Grade: {consistency_result.consistency_grade}")
            logger.info(f"  Overall: {consistency_result.overall_consistency:.2f}")
            logger.info(f"  Risk Level Consistency: {consistency_result.risk_level_consistency:.2f}")

        return {
            "company_name": company_name,
            "model_name": self.model_name,
            "total_runs": len(all_runs),
            "successful_runs": len(successful_runs),
            "runs": all_runs,
        }

    def _log_individual_run(self, run_result: Dict[str, Any], company_config: Dict[str, Any]):
        """Log a single run to langsmith_eval_runs sheet."""
        company_name = company_config["name"]

        # Create dataset name from company
        dataset_name = f"{company_name.replace(' ', '_').replace('.', '')}"

        eval_run = EvaluationRun(
            eval_id=run_result["run_id"],
            dataset_name=dataset_name,
            experiment_name=f"run_{run_result['run_number']}_{self.model_name}",
            total_examples=1,
            passed_examples=1 if run_result.get("risk_level_correct") and run_result.get("score_in_range") else 0,
            failed_examples=0 if run_result.get("risk_level_correct") and run_result.get("score_in_range") else 1,
            avg_risk_accuracy=1.0 if run_result.get("risk_level_correct") else 0.0,
            avg_score_accuracy=1.0 if run_result.get("score_in_range") else 0.0,
            avg_tool_selection_f1=run_result.get("tool_selection_f1", 0),
            avg_synthesis_quality=1.0 if run_result.get("reasoning") else 0.5,
            avg_trajectory_match=run_result.get("llm_judge_score", 0),
            overall_score=(
                (1.0 if run_result.get("risk_level_correct") else 0.0) +
                (1.0 if run_result.get("score_in_range") else 0.0) +
                run_result.get("tool_selection_f1", 0) +
                run_result.get("llm_judge_score", 0)
            ) / 4,
            duration_seconds=run_result.get("duration_ms", 0) / 1000,
            model_config={
                "model": self.model_name,
                "company": company_name,
                "run_number": run_result["run_number"],
                "risk_level": run_result.get("risk_level"),
                "credit_score": run_result.get("credit_score"),
                "tools_selected": run_result.get("tools_selected", []),
            },
        )
        self.eval_logger.log_evaluation_run(eval_run)

        # Also log to langsmith_eval_examples
        example_result = ExampleResult(
            eval_id=run_result["run_id"],
            example_id=f"{run_result['run_id'][:8]}_run{run_result['run_number']}",
            company_name=company_name,
            risk_level_correct=run_result.get("risk_level_correct", False),
            credit_score_accuracy=1.0 if run_result.get("score_in_range") else 0.0,
            tool_selection_f1=run_result.get("tool_selection_f1", 0),
            synthesis_quality=1.0 if run_result.get("reasoning") else 0.5,
            trajectory_match=run_result.get("llm_judge_score", 0),
            llm_judge_score=run_result.get("llm_judge_score", 0),
            actual_risk_level=run_result.get("risk_level", ""),
            expected_risk_level=company_config["expected_risk_level"],
            actual_credit_score=run_result.get("credit_score", 0),
            expected_credit_score_range=str(company_config["expected_score_range"]),
            actual_tools=", ".join(run_result.get("tools_selected", [])),
            expected_tools=", ".join(company_config["expected_tools"]),
            tool_selection_comment=f"P:{run_result.get('tool_selection_precision', 0):.2f} R:{run_result.get('tool_selection_recall', 0):.2f} F1:{run_result.get('tool_selection_f1', 0):.2f}",
            synthesis_comment=run_result.get("llm_judge_reasoning", "")[:500],
            passed=run_result.get("risk_level_correct", False) and run_result.get("score_in_range", False),
            error="",
        )
        self.eval_logger.log_example_result(example_result)

    def run_full_evaluation(
        self,
        companies: List[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run evaluation for all companies."""
        companies = companies or TEST_COMPANIES

        logger.info(f"\n{'='*60}")
        logger.info(f"STARTING BATCH EVALUATION")
        logger.info(f"{'='*60}")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Companies: {len(companies)}")
        logger.info(f"Runs per company: {self.num_runs_per_company}")
        logger.info(f"Total runs: {len(companies) * self.num_runs_per_company}")
        logger.info(f"{'='*60}\n")

        start_time = time.time()
        all_company_results = []

        for company_config in companies:
            result = self.run_company_evaluation(company_config)
            all_company_results.append(result)

        duration = time.time() - start_time

        # Summary
        total_runs = sum(r["total_runs"] for r in all_company_results)
        successful_runs = sum(r["successful_runs"] for r in all_company_results)

        logger.info(f"\n{'='*60}")
        logger.info(f"BATCH EVALUATION COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Total companies: {len(companies)}")
        logger.info(f"Total runs: {total_runs}")
        logger.info(f"Successful runs: {successful_runs}")
        logger.info(f"Duration: {duration:.1f}s")
        logger.info(f"{'='*60}")

        # Wait for async writes
        time.sleep(3)

        return {
            "model": self.model_name,
            "total_companies": len(companies),
            "total_runs": total_runs,
            "successful_runs": successful_runs,
            "duration_seconds": duration,
            "results": all_company_results,
        }


def main():
    parser = argparse.ArgumentParser(description="Run batch evaluation with detailed logging")
    parser.add_argument("--model", default="llama-3.3-70b-versatile", help="Model name")
    parser.add_argument("--runs", type=int, default=3, help="Runs per company")
    parser.add_argument("--company", default=None, help="Single company to test")
    parser.add_argument("--quick", action="store_true", help="Quick test (1 company, 2 runs)")
    args = parser.parse_args()

    # Determine companies to test
    if args.company:
        companies = [c for c in TEST_COMPANIES if args.company.lower() in c["name"].lower()]
        if not companies:
            # Create custom company config
            companies = [{
                "name": args.company,
                "ticker": "UNKNOWN",
                "expected_risk_level": "low",
                "expected_score_range": (50, 95),
                "expected_tools": ["fetch_sec_data", "fetch_market_data", "fetch_legal_data"],
                "company_type": "public_us",
            }]
    elif args.quick:
        companies = TEST_COMPANIES[:1]
        args.runs = 2
    else:
        companies = TEST_COMPANIES

    runner = BatchEvaluationRunner(
        model=args.model,
        num_runs_per_company=args.runs,
    )

    result = runner.run_full_evaluation(companies=companies)

    print(f"\n{'='*60}")
    print(f"Results logged to Google Sheets:")
    print(f"  - langsmith_eval_runs: {result['total_runs']} rows (one per run)")
    print(f"  - langsmith_eval_examples: {result['total_runs']} rows")
    print(f"  - llm_judge_results: {result['successful_runs']} rows")
    print(f"  - model_consistency: {result['total_companies']} rows")
    print(f"\nSpreadsheet: {runner.sheets_logger.get_spreadsheet_url()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
