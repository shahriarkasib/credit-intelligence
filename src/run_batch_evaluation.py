"""
Batch Evaluation Runner - Runs evaluations with LangSmith logging.

This script:
1. Runs assessments for multiple companies
2. Evaluates consistency (same model, multiple runs)
3. Logs results to langsmith_eval_runs sheet
4. Logs to MongoDB and Google Sheets
"""

import os
import sys
import uuid
import time
import logging
import argparse
from datetime import datetime
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
from run_logging import get_sheets_logger, get_run_logger

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Test companies with expected values (ground truth)
TEST_COMPANIES = [
    {
        "name": "Apple Inc",
        "expected_risk_level": "low",
        "expected_score_range": (70, 95),
        "expected_tools": ["fetch_sec_data", "fetch_market_data"],
        "company_type": "public_us",
    },
    {
        "name": "Microsoft Corporation",
        "expected_risk_level": "low",
        "expected_score_range": (75, 95),
        "expected_tools": ["fetch_sec_data", "fetch_market_data"],
        "company_type": "public_us",
    },
    {
        "name": "Tesla Inc",
        "expected_risk_level": "low",
        "expected_score_range": (60, 85),
        "expected_tools": ["fetch_sec_data", "fetch_market_data"],
        "company_type": "public_us",
    },
]


class BatchEvaluationRunner:
    """Runs batch evaluations and logs to LangSmith eval sheets."""

    def __init__(
        self,
        model: str = "primary",
        num_runs_per_company: int = 2,
        log_to_sheets: bool = True,
        log_to_mongodb: bool = True,
    ):
        self.supervisor = ToolSupervisor(model=model)
        self.model = model
        self.num_runs_per_company = num_runs_per_company
        self.llm_judge = LLMJudgeEvaluator()
        self.workflow_evaluator = WorkflowEvaluator()
        self.evaluation_brain = get_evaluation_brain()

        self.eval_logger = get_eval_logger()
        self.sheets_logger = get_sheets_logger() if log_to_sheets else None
        self.run_logger = get_run_logger() if log_to_mongodb else None

        logger.info(f"BatchEvaluationRunner initialized: model={model}, runs_per_company={num_runs_per_company}")
        logger.info(f"Sheets connected: {self.eval_logger.sheets_connected()}")
        logger.info(f"MongoDB connected: {self.eval_logger.mongodb_connected()}")

    def run_single_assessment(self, company_name: str) -> Dict[str, Any]:
        """Run a single assessment and return results."""
        run_id = str(uuid.uuid4())

        try:
            result = self.supervisor.run_full_assessment(company_name)

            assessment = result.get("assessment", {}).get("assessment", {})
            tool_selection = result.get("tool_selection", {})

            return {
                "run_id": run_id,
                "company_name": company_name,
                "success": True,
                "risk_level": assessment.get("risk_level", "unknown"),
                "credit_score": assessment.get("credit_score", 0),
                "confidence": assessment.get("confidence", 0),
                "reasoning": assessment.get("reasoning", ""),
                "risk_factors": assessment.get("risk_factors", []),
                "recommendations": assessment.get("recommendations", []),
                "tools_used": tool_selection.get("tools_selected", []),
                "full_result": result,
            }
        except Exception as e:
            logger.error(f"Assessment failed for {company_name}: {e}")
            return {
                "run_id": run_id,
                "company_name": company_name,
                "success": False,
                "error": str(e),
            }

    def evaluate_company(self, company_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run multiple assessments for a company and evaluate consistency."""
        company_name = company_config["name"]
        expected_risk = company_config["expected_risk_level"]
        expected_score_range = company_config["expected_score_range"]
        expected_tools = company_config["expected_tools"]

        logger.info(f"Evaluating {company_name} ({self.num_runs_per_company} runs)...")

        # Run multiple assessments
        assessments = []
        for i in range(self.num_runs_per_company):
            logger.info(f"  Run {i+1}/{self.num_runs_per_company}...")
            result = self.run_single_assessment(company_name)
            if result.get("success"):
                assessments.append(result)
            time.sleep(1)  # Rate limiting

        if not assessments:
            return {
                "company_name": company_name,
                "success": False,
                "error": "All assessments failed",
            }

        # Evaluate consistency using LLM Judge
        consistency_result = self.llm_judge.evaluate_consistency(
            company_name=company_name,
            assessments=[a for a in assessments],
            model_name=self.model,
        )

        # Calculate accuracy against expected values
        risk_correct = sum(1 for a in assessments if a.get("risk_level", "").lower() == expected_risk.lower())
        risk_accuracy = risk_correct / len(assessments)

        scores = [a.get("credit_score", 0) for a in assessments]
        avg_score = sum(scores) / len(scores) if scores else 0
        score_in_range = expected_score_range[0] <= avg_score <= expected_score_range[1]
        score_accuracy = 1.0 if score_in_range else max(0, 1 - abs(avg_score - sum(expected_score_range)/2) / 50)

        # Tool selection accuracy
        first_tools = assessments[0].get("tools_used", [])
        tool_correct = set(first_tools) & set(expected_tools)
        tool_f1 = len(tool_correct) / max(len(first_tools), len(expected_tools), 1)

        # Overall pass/fail
        passed = risk_accuracy >= 0.5 and score_in_range and consistency_result.overall_consistency >= 0.6

        return {
            "company_name": company_name,
            "success": True,
            "num_runs": len(assessments),
            # Accuracy metrics
            "risk_level_correct": risk_accuracy >= 0.5,
            "risk_accuracy": risk_accuracy,
            "score_accuracy": score_accuracy,
            "actual_risk_levels": [a.get("risk_level") for a in assessments],
            "actual_credit_scores": scores,
            "expected_risk_level": expected_risk,
            "expected_score_range": expected_score_range,
            # Tool selection
            "tool_selection_f1": tool_f1,
            "actual_tools": first_tools,
            "expected_tools": expected_tools,
            # Consistency
            "consistency_grade": consistency_result.consistency_grade,
            "overall_consistency": consistency_result.overall_consistency,
            "risk_level_consistency": consistency_result.risk_level_consistency,
            "credit_score_std": consistency_result.credit_score_std,
            # Synthesis quality
            "synthesis_quality": 1.0 if assessments[0].get("reasoning") else 0.5,
            # Overall
            "passed": passed,
            "llm_judge_analysis": consistency_result.llm_judge_analysis,
        }

    def run_batch_evaluation(
        self,
        companies: List[Dict[str, Any]] = None,
        experiment_name: str = None,
    ) -> Dict[str, Any]:
        """Run evaluation for multiple companies and log to langsmith_eval_runs."""
        companies = companies or TEST_COMPANIES
        experiment_name = experiment_name or f"batch_eval_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        eval_id = str(uuid.uuid4())[:8]

        logger.info(f"Starting batch evaluation: {experiment_name}")
        logger.info(f"Eval ID: {eval_id}")
        logger.info(f"Companies: {len(companies)}")

        start_time = time.time()
        results = []

        for company_config in companies:
            try:
                result = self.evaluate_company(company_config)
                results.append(result)

                # Log individual example result
                example_result = ExampleResult(
                    eval_id=eval_id,
                    example_id=f"{eval_id}_{company_config['name'][:10]}",
                    company_name=company_config["name"],
                    risk_level_correct=result.get("risk_level_correct", False),
                    credit_score_accuracy=result.get("score_accuracy", 0),
                    tool_selection_f1=result.get("tool_selection_f1", 0),
                    synthesis_quality=result.get("synthesis_quality", 0),
                    trajectory_match=result.get("overall_consistency", 0),
                    llm_judge_score=result.get("overall_consistency", 0),
                    actual_risk_level=str(result.get("actual_risk_levels", [])),
                    expected_risk_level=result.get("expected_risk_level", ""),
                    actual_credit_score=int(sum(result.get("actual_credit_scores", [0])) / max(len(result.get("actual_credit_scores", [1])), 1)),
                    expected_credit_score_range=str(result.get("expected_score_range", "")),
                    actual_tools=", ".join(result.get("actual_tools", [])),
                    expected_tools=", ".join(result.get("expected_tools", [])),
                    tool_selection_comment=f"F1: {result.get('tool_selection_f1', 0):.2f}",
                    synthesis_comment=result.get("llm_judge_analysis", "")[:500],
                    passed=result.get("passed", False),
                    error="" if result.get("success") else result.get("error", ""),
                )
                self.eval_logger.log_example_result(example_result)

            except Exception as e:
                logger.error(f"Failed to evaluate {company_config['name']}: {e}")
                results.append({
                    "company_name": company_config["name"],
                    "success": False,
                    "error": str(e),
                    "passed": False,
                })

        duration = time.time() - start_time

        # Calculate aggregate metrics
        successful = [r for r in results if r.get("success")]
        passed = sum(1 for r in successful if r.get("passed"))

        avg_risk_acc = sum(r.get("risk_accuracy", 0) for r in successful) / len(successful) if successful else 0
        avg_score_acc = sum(r.get("score_accuracy", 0) for r in successful) / len(successful) if successful else 0
        avg_tool_f1 = sum(r.get("tool_selection_f1", 0) for r in successful) / len(successful) if successful else 0
        avg_synthesis = sum(r.get("synthesis_quality", 0) for r in successful) / len(successful) if successful else 0
        avg_consistency = sum(r.get("overall_consistency", 0) for r in successful) / len(successful) if successful else 0

        overall_score = (avg_risk_acc + avg_score_acc + avg_tool_f1 + avg_synthesis + avg_consistency) / 5

        # Log evaluation run summary to langsmith_eval_runs
        eval_run = EvaluationRun(
            eval_id=eval_id,
            dataset_name="credit_intelligence_test",
            experiment_name=experiment_name,
            total_examples=len(companies),
            passed_examples=passed,
            failed_examples=len(companies) - passed,
            avg_risk_accuracy=avg_risk_acc,
            avg_score_accuracy=avg_score_acc,
            avg_tool_selection_f1=avg_tool_f1,
            avg_synthesis_quality=avg_synthesis,
            avg_trajectory_match=avg_consistency,
            overall_score=overall_score,
            duration_seconds=duration,
            model_config={"model": self.model, "runs_per_company": self.num_runs_per_company},
        )
        self.eval_logger.log_evaluation_run(eval_run)

        # Log consistency evaluation to sheets
        if self.sheets_logger and self.sheets_logger.is_connected():
            for r in successful:
                self.sheets_logger.log_model_consistency(
                    eval_id=eval_id,
                    company_name=r["company_name"],
                    model_name=self.model,
                    num_runs=r.get("num_runs", 0),
                    risk_level_consistency=r.get("risk_level_consistency", 0),
                    credit_score_mean=sum(r.get("actual_credit_scores", [0])) / max(len(r.get("actual_credit_scores", [1])), 1),
                    credit_score_std=r.get("credit_score_std", 0),
                    overall_consistency=r.get("overall_consistency", 0),
                    is_consistent=r.get("overall_consistency", 0) >= 0.75,
                    consistency_grade=r.get("consistency_grade", ""),
                    llm_judge_analysis=r.get("llm_judge_analysis", ""),
                )

        # Wait for async writes
        time.sleep(3)

        summary = {
            "eval_id": eval_id,
            "experiment_name": experiment_name,
            "total_companies": len(companies),
            "successful_evaluations": len(successful),
            "passed": passed,
            "failed": len(companies) - passed,
            "avg_risk_accuracy": avg_risk_acc,
            "avg_score_accuracy": avg_score_acc,
            "avg_tool_selection_f1": avg_tool_f1,
            "avg_synthesis_quality": avg_synthesis,
            "avg_consistency": avg_consistency,
            "overall_score": overall_score,
            "duration_seconds": duration,
            "results": results,
        }

        logger.info(f"\n{'='*60}")
        logger.info(f"BATCH EVALUATION COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Eval ID: {eval_id}")
        logger.info(f"Passed: {passed}/{len(companies)}")
        logger.info(f"Overall Score: {overall_score:.2f}")
        logger.info(f"Duration: {duration:.1f}s")
        logger.info(f"Logged to: langsmith_eval_runs, langsmith_eval_examples")
        logger.info(f"{'='*60}")

        return summary


def main():
    parser = argparse.ArgumentParser(description="Run batch evaluation")
    parser.add_argument("--model", default="primary", help="Model to use")
    parser.add_argument("--runs", type=int, default=2, help="Runs per company")
    parser.add_argument("--experiment", default=None, help="Experiment name")
    parser.add_argument("--quick", action="store_true", help="Quick test with 1 company")
    args = parser.parse_args()

    companies = TEST_COMPANIES[:1] if args.quick else TEST_COMPANIES

    runner = BatchEvaluationRunner(
        model=args.model,
        num_runs_per_company=args.runs,
    )

    result = runner.run_batch_evaluation(
        companies=companies,
        experiment_name=args.experiment,
    )

    print(f"\nResults logged to langsmith_eval_runs sheet!")
    print(f"Spreadsheet: {runner.eval_logger.get_spreadsheet_url()}")


if __name__ == "__main__":
    main()
