"""
Batch Evaluation Runner - Multi-Model Evaluation with Cross-Model Comparison.

This script:
1. Runs assessments with MULTIPLE models (not just one)
2. Runs multiple iterations per model for consistency testing
3. Compares results ACROSS models using LLM Judge
4. Logs everything to Google Sheets with full details
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


# Available models for evaluation
AVAILABLE_MODELS = [
    "llama-3.3-70b-versatile",      # Primary model
    "llama-3.1-8b-instant",          # Fast, smaller model
    "mixtral-8x7b-32768",            # Mixtral model
    "gemma2-9b-it",                  # Gemma model
]

# Default models to use for evaluation
DEFAULT_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]

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


class MultiModelEvaluationRunner:
    """
    Runs evaluations across MULTIPLE models with cross-model comparison.

    For each company:
    1. Run N iterations with Model A
    2. Run N iterations with Model B
    3. Compare results across models using LLM Judge
    4. Log everything with full details
    """

    def __init__(
        self,
        models: List[str] = None,
        runs_per_model: int = 3,
        judge_model: str = "llama-3.3-70b-versatile",
    ):
        self.models = models or DEFAULT_MODELS
        self.runs_per_model = runs_per_model
        self.judge_model = judge_model

        # Initialize supervisors for each model
        self.supervisors: Dict[str, ToolSupervisor] = {}
        for model in self.models:
            self.supervisors[model] = ToolSupervisor(model=model)

        # LLM Judge uses configurable model
        self.llm_judge = LLMJudgeEvaluator(model=judge_model)
        self.workflow_evaluator = WorkflowEvaluator()
        self.evaluation_brain = get_evaluation_brain()

        # Loggers
        self.eval_logger = get_eval_logger()
        self.sheets_logger = get_sheets_logger()
        self.run_logger = get_run_logger()
        self.workflow_logger = get_workflow_logger()

        logger.info(f"MultiModelEvaluationRunner initialized")
        logger.info(f"  Models to evaluate: {self.models}")
        logger.info(f"  Runs per model: {self.runs_per_model}")
        logger.info(f"  LLM Judge model: {self.judge_model}")
        logger.info(f"  Sheets connected: {self.sheets_logger.is_connected() if self.sheets_logger else False}")

    def run_single_assessment(
        self,
        company_name: str,
        model_name: str,
        run_number: int
    ) -> Dict[str, Any]:
        """Run a single assessment with a specific model."""
        run_id = str(uuid.uuid4())
        start_time = time.time()

        try:
            supervisor = self.supervisors[model_name]
            result = supervisor.run_full_assessment(company_name)
            duration_ms = (time.time() - start_time) * 1000

            assessment = result.get("assessment", {}).get("assessment", {})
            tool_selection = result.get("tool_selection", {})

            # Get tools selected
            selected_tools = tool_selection.get("tools_selected", [])
            if not selected_tools:
                selection = tool_selection.get("selection", {})
                tools_to_use = selection.get("tools_to_use", [])
                selected_tools = [t.get("name") for t in tools_to_use if isinstance(t, dict)]

            return {
                "run_id": run_id,
                "run_number": run_number,
                "company_name": company_name,
                "model_name": model_name,
                "success": True,
                "duration_ms": duration_ms,
                # Assessment
                "risk_level": assessment.get("risk_level", "unknown"),
                "credit_score": assessment.get("credit_score", 0),
                "confidence": assessment.get("confidence", 0),
                "reasoning": assessment.get("reasoning", ""),
                "risk_factors": assessment.get("risk_factors", []),
                "positive_factors": assessment.get("positive_factors", []),
                "recommendations": assessment.get("recommendations", []),
                # Tools
                "tools_selected": selected_tools,
                # Full result
                "full_result": result,
            }
        except Exception as e:
            logger.error(f"Assessment failed for {company_name} with {model_name} (run {run_number}): {e}")
            return {
                "run_id": run_id,
                "run_number": run_number,
                "company_name": company_name,
                "model_name": model_name,
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

        return {"precision": precision, "recall": recall, "f1": f1}

    def evaluate_company_with_all_models(self, company_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run evaluation for a company across ALL models.

        Returns per-model results and cross-model comparison.
        """
        company_name = company_config["name"]
        expected_risk = company_config["expected_risk_level"]
        expected_score_range = company_config["expected_score_range"]
        expected_tools = company_config["expected_tools"]

        logger.info(f"\n{'='*60}")
        logger.info(f"EVALUATING: {company_name}")
        logger.info(f"Expected: risk={expected_risk}, score={expected_score_range}")
        logger.info(f"Models: {self.models}")
        logger.info(f"Runs per model: {self.runs_per_model}")
        logger.info(f"Total runs: {len(self.models) * self.runs_per_model}")
        logger.info(f"{'='*60}")

        # Store results per model
        model_results: Dict[str, List[Dict]] = {model: [] for model in self.models}
        all_runs = []

        # Run each model
        for model_name in self.models:
            logger.info(f"\n--- Model: {model_name} ---")

            for run_num in range(1, self.runs_per_model + 1):
                logger.info(f"  Run {run_num}/{self.runs_per_model}...")

                run_result = self.run_single_assessment(company_name, model_name, run_num)

                if run_result.get("success"):
                    # Calculate metrics
                    tool_metrics = self.calculate_tool_selection_f1(
                        run_result.get("tools_selected", []),
                        expected_tools
                    )
                    run_result["tool_selection_f1"] = tool_metrics["f1"]
                    run_result["tool_selection_precision"] = tool_metrics["precision"]
                    run_result["tool_selection_recall"] = tool_metrics["recall"]

                    # Accuracy checks
                    run_result["risk_level_correct"] = run_result["risk_level"].lower() == expected_risk.lower()
                    score = run_result["credit_score"]
                    run_result["score_in_range"] = expected_score_range[0] <= score <= expected_score_range[1]

                    # Run LLM Judge for this run
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

                    # Log to sheets
                    self._log_individual_run(run_result, company_config)

                    # Log LLM Judge result
                    if self.sheets_logger and self.sheets_logger.is_connected():
                        self.sheets_logger.log_llm_judge_result(
                            run_id=run_result["run_id"],
                            company_name=company_name,
                            model_used=model_name,
                            accuracy_score=llm_judge_result.accuracy_score,
                            completeness_score=llm_judge_result.completeness_score,
                            consistency_score=llm_judge_result.consistency_score,
                            actionability_score=llm_judge_result.actionability_score,
                            data_utilization_score=llm_judge_result.data_utilization_score,
                            overall_score=llm_judge_result.overall_score,
                            accuracy_reasoning=llm_judge_result.accuracy_reasoning,
                            completeness_reasoning=llm_judge_result.completeness_reasoning,
                            consistency_reasoning=llm_judge_result.consistency_reasoning,
                            overall_reasoning=llm_judge_result.overall_reasoning,
                            suggestions=llm_judge_result.suggestions,
                            tokens_used=llm_judge_result.tokens_used,
                        )

                    logger.info(f"    Risk: {run_result['risk_level']} | Score: {run_result['credit_score']} | LLM Judge: {run_result['llm_judge_score']:.2f}")

                model_results[model_name].append(run_result)
                all_runs.append(run_result)

                # Rate limiting
                time.sleep(1)

        # Evaluate consistency WITHIN each model
        logger.info(f"\n--- Consistency Analysis (per model) ---")
        consistency_results = {}

        for model_name in self.models:
            successful_runs = [r for r in model_results[model_name] if r.get("success")]
            if len(successful_runs) >= 2:
                consistency = self.llm_judge.evaluate_consistency(
                    company_name=company_name,
                    assessments=successful_runs,
                    model_name=model_name,
                )
                consistency_results[model_name] = consistency

                # Log to sheets
                if self.sheets_logger and self.sheets_logger.is_connected():
                    self.sheets_logger.log_model_consistency(
                        eval_id=str(uuid.uuid4())[:8],
                        company_name=company_name,
                        model_name=model_name,
                        num_runs=len(successful_runs),
                        risk_level_consistency=consistency.risk_level_consistency,
                        credit_score_mean=consistency.credit_score_mean,
                        credit_score_std=consistency.credit_score_std,
                        overall_consistency=consistency.overall_consistency,
                        is_consistent=consistency.is_consistent,
                        consistency_grade=consistency.consistency_grade,
                        llm_judge_analysis=consistency.llm_judge_analysis,
                    )

                logger.info(f"  {model_name}: Grade={consistency.consistency_grade} ({consistency.overall_consistency:.2f})")

        # Cross-model comparison
        logger.info(f"\n--- Cross-Model Comparison ---")
        cross_model_result = None

        if len(self.models) >= 2:
            # Get best run from each model for comparison
            model_best_runs = {}
            for model_name in self.models:
                successful = [r for r in model_results[model_name] if r.get("success")]
                if successful:
                    # Use the run with highest LLM judge score
                    best = max(successful, key=lambda x: x.get("llm_judge_score", 0))
                    model_best_runs[model_name] = best

            if len(model_best_runs) >= 2:
                cross_model_result = self.llm_judge.evaluate_cross_model(
                    company_name=company_name,
                    model_assessments=model_best_runs,
                )

                # Log cross-model evaluation
                if self.sheets_logger and self.sheets_logger.is_connected():
                    self.sheets_logger.log_cross_model_eval(
                        eval_id=str(uuid.uuid4())[:8],
                        company_name=company_name,
                        models_compared=list(model_best_runs.keys()),
                        num_models=len(model_best_runs),
                        risk_level_agreement=cross_model_result.risk_level_agreement,
                        credit_score_mean=cross_model_result.credit_score_mean,
                        credit_score_std=cross_model_result.credit_score_std,
                        credit_score_range=cross_model_result.credit_score_range,
                        best_model=cross_model_result.best_model,
                        best_model_reasoning=cross_model_result.best_model_reasoning,
                        cross_model_agreement=cross_model_result.cross_model_agreement,
                        llm_judge_analysis=cross_model_result.llm_judge_analysis,
                        model_recommendations=cross_model_result.model_recommendations,
                        model_results=cross_model_result.model_results,
                    )

                logger.info(f"  Best Model: {cross_model_result.best_model}")
                logger.info(f"  Cross-Model Agreement: {cross_model_result.cross_model_agreement:.2f}")
                logger.info(f"  Risk Level Agreement: {cross_model_result.risk_level_agreement:.2f}")

        return {
            "company_name": company_name,
            "models_evaluated": self.models,
            "runs_per_model": self.runs_per_model,
            "total_runs": len(all_runs),
            "successful_runs": sum(1 for r in all_runs if r.get("success")),
            "model_results": model_results,
            "consistency_results": consistency_results,
            "cross_model_result": cross_model_result,
            "all_runs": all_runs,
        }

    def _log_individual_run(self, run_result: Dict[str, Any], company_config: Dict[str, Any]):
        """Log a single run to langsmith_eval_runs sheet."""
        company_name = company_config["name"]
        model_name = run_result["model_name"]

        eval_run = EvaluationRun(
            eval_id=run_result["run_id"],
            dataset_name=company_name.replace(" ", "_").replace(".", ""),
            experiment_name=f"{model_name}_run{run_result['run_number']}",
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
                "model": model_name,
                "company": company_name,
                "run_number": run_result["run_number"],
                "risk_level": run_result.get("risk_level"),
                "credit_score": run_result.get("credit_score"),
                "confidence": run_result.get("confidence"),
                "tools_selected": run_result.get("tools_selected", []),
                "llm_judge_score": run_result.get("llm_judge_score", 0),
            },
        )
        self.eval_logger.log_evaluation_run(eval_run)

        # Also log to examples
        example_result = ExampleResult(
            eval_id=run_result["run_id"],
            example_id=f"{run_result['run_id'][:8]}_{model_name[:10]}_run{run_result['run_number']}",
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
            tool_selection_comment=f"Model: {model_name} | F1: {run_result.get('tool_selection_f1', 0):.2f}",
            synthesis_comment=run_result.get("llm_judge_reasoning", "")[:500],
            passed=run_result.get("risk_level_correct", False) and run_result.get("score_in_range", False),
        )
        self.eval_logger.log_example_result(example_result)

    def run_full_evaluation(self, companies: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run full multi-model evaluation for all companies."""
        companies = companies or TEST_COMPANIES

        total_runs = len(companies) * len(self.models) * self.runs_per_model

        logger.info(f"\n{'='*70}")
        logger.info(f"STARTING MULTI-MODEL EVALUATION")
        logger.info(f"{'='*70}")
        logger.info(f"Companies: {len(companies)}")
        logger.info(f"Models: {self.models}")
        logger.info(f"Runs per model: {self.runs_per_model}")
        logger.info(f"Total runs: {total_runs}")
        logger.info(f"LLM Judge model: {self.judge_model}")
        logger.info(f"{'='*70}\n")

        start_time = time.time()
        all_results = []

        for company_config in companies:
            result = self.evaluate_company_with_all_models(company_config)
            all_results.append(result)

        duration = time.time() - start_time

        # Summary
        total_successful = sum(r["successful_runs"] for r in all_results)

        logger.info(f"\n{'='*70}")
        logger.info(f"MULTI-MODEL EVALUATION COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"Companies: {len(companies)}")
        logger.info(f"Models: {len(self.models)}")
        logger.info(f"Total runs: {total_runs}")
        logger.info(f"Successful: {total_successful}")
        logger.info(f"Duration: {duration:.1f}s")
        logger.info(f"{'='*70}")

        # Wait for async writes
        time.sleep(3)

        return {
            "models": self.models,
            "judge_model": self.judge_model,
            "companies": len(companies),
            "runs_per_model": self.runs_per_model,
            "total_runs": total_runs,
            "successful_runs": total_successful,
            "duration_seconds": duration,
            "results": all_results,
        }


def main():
    parser = argparse.ArgumentParser(description="Multi-model evaluation with cross-model comparison")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help=f"Models to evaluate. Available: {AVAILABLE_MODELS}")
    parser.add_argument("--runs", type=int, default=2, help="Runs per model")
    parser.add_argument("--judge", default="llama-3.3-70b-versatile", help="Model for LLM Judge")
    parser.add_argument("--company", default=None, help="Single company to test")
    parser.add_argument("--quick", action="store_true", help="Quick test (1 company, 2 models, 1 run each)")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    args = parser.parse_args()

    if args.list_models:
        print("Available models:")
        for m in AVAILABLE_MODELS:
            print(f"  - {m}")
        return

    # Determine companies
    if args.company:
        companies = [c for c in TEST_COMPANIES if args.company.lower() in c["name"].lower()]
        if not companies:
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
        args.models = DEFAULT_MODELS[:2]
        args.runs = 1
    else:
        companies = TEST_COMPANIES

    runner = MultiModelEvaluationRunner(
        models=args.models,
        runs_per_model=args.runs,
        judge_model=args.judge,
    )

    result = runner.run_full_evaluation(companies=companies)

    print(f"\n{'='*70}")
    print(f"RESULTS LOGGED TO GOOGLE SHEETS")
    print(f"{'='*70}")
    print(f"Models evaluated: {result['models']}")
    print(f"LLM Judge model: {result['judge_model']}")
    print(f"Total runs: {result['total_runs']}")
    print(f"")
    print(f"Sheets updated:")
    print(f"  - langsmith_eval_runs: {result['successful_runs']} rows")
    print(f"  - langsmith_eval_examples: {result['successful_runs']} rows")
    print(f"  - llm_judge_results: {result['successful_runs']} rows")
    print(f"  - model_consistency: {len(result['models']) * result['companies']} rows")
    print(f"  - cross_model_eval: {result['companies']} rows")
    print(f"")
    print(f"Spreadsheet: {runner.sheets_logger.get_spreadsheet_url()}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
