"""
Evaluation Runner for LangSmith Evaluations.

Provides functions to run evaluations using LangSmith SDK's evaluate() and aevaluate().
"""

import os
import uuid
import time
import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, Union
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import LangSmith
try:
    from langsmith import Client, evaluate, aevaluate
    from langsmith.schemas import Run, Example
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    Client = None
    evaluate = None
    aevaluate = None
    logger.warning("LangSmith SDK not installed. Install with: pip install langsmith")

# Import local modules
from .evaluators import (
    correct_risk_level,
    credit_score_accuracy,
    tool_selection_f1,
    synthesis_quality,
    assessment_completeness,
    trajectory_evaluator,
    CreditEvaluatorSuite,
)
from .datasets import get_dataset_manager, get_dataset
from .logger import get_eval_logger, EvaluationRun, ExampleResult


# =============================================================================
# EVALUATION RUNNER CLASS
# =============================================================================

class EvaluationRunner:
    """
    Runner for LangSmith evaluations.

    Handles:
    - Running evaluations on datasets
    - Collecting and processing results
    - Logging to separate Google Sheets tab
    - Integration with LangSmith experiments
    """

    def __init__(
        self,
        client: Optional[Any] = None,
        log_to_sheets: bool = True,
        log_to_mongodb: bool = True,
    ):
        self.client = client
        self.log_to_sheets = log_to_sheets
        self.log_to_mongodb = log_to_mongodb

        if LANGSMITH_AVAILABLE and client is None:
            api_key = os.getenv("LANGCHAIN_API_KEY")
            if api_key:
                try:
                    self.client = Client(api_key=api_key)
                except Exception as e:
                    logger.warning(f"Failed to create LangSmith client: {e}")

        self.eval_logger = get_eval_logger() if (log_to_sheets or log_to_mongodb) else None

    def run_evaluation(
        self,
        target: Callable,
        dataset_name: str,
        evaluators: List[Callable] = None,
        experiment_prefix: str = "credit_eval",
        max_concurrency: int = 4,
        include_llm_judge: bool = False,
    ) -> Dict[str, Any]:
        """
        Run evaluation synchronously.

        Args:
            target: Function to evaluate (takes inputs dict, returns outputs dict)
            dataset_name: Name of the dataset to use
            evaluators: List of evaluator functions (default: use CreditEvaluatorSuite)
            experiment_prefix: Prefix for experiment name
            max_concurrency: Max concurrent evaluations
            include_llm_judge: Whether to include LLM-as-judge evaluator

        Returns:
            Dict with evaluation results and summary
        """
        if not LANGSMITH_AVAILABLE:
            logger.error("LangSmith not available for evaluation")
            return self._run_local_evaluation(target, dataset_name, evaluators)

        # Get or create evaluators
        if evaluators is None:
            suite = CreditEvaluatorSuite(include_llm_judge=include_llm_judge)
            evaluators = suite.get_row_evaluators()

        # Generate evaluation ID
        eval_id = str(uuid.uuid4())[:8]
        experiment_name = f"{experiment_prefix}_{eval_id}"

        logger.info(f"Starting evaluation: {experiment_name} on dataset: {dataset_name}")
        start_time = time.time()

        try:
            # Run LangSmith evaluation
            results = evaluate(
                target,
                data=dataset_name,
                evaluators=evaluators,
                experiment_prefix=experiment_prefix,
                max_concurrency=max_concurrency,
            )

            # Process results
            processed = self._process_results(results, eval_id, dataset_name, experiment_name)

            duration = time.time() - start_time

            # Log to sheets/mongodb
            if self.eval_logger:
                self.eval_logger.log_batch_results(
                    eval_id=eval_id,
                    dataset_name=dataset_name,
                    experiment_name=experiment_name,
                    results=processed["examples"],
                    duration_seconds=duration,
                    model_config={"max_concurrency": max_concurrency},
                )

            logger.info(f"Evaluation complete: {processed['summary']['passed']}/{processed['summary']['total']} passed")

            return {
                "eval_id": eval_id,
                "experiment_name": experiment_name,
                "dataset_name": dataset_name,
                "duration_seconds": duration,
                "summary": processed["summary"],
                "examples": processed["examples"],
                "langsmith_results": results,
            }

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {
                "eval_id": eval_id,
                "experiment_name": experiment_name,
                "error": str(e),
            }

    async def run_async_evaluation(
        self,
        target: Callable,
        dataset_name: str,
        evaluators: List[Callable] = None,
        experiment_prefix: str = "credit_eval",
        max_concurrency: int = 4,
        include_llm_judge: bool = False,
    ) -> Dict[str, Any]:
        """
        Run evaluation asynchronously.

        Args:
            target: Async function to evaluate
            dataset_name: Name of the dataset to use
            evaluators: List of evaluator functions
            experiment_prefix: Prefix for experiment name
            max_concurrency: Max concurrent evaluations
            include_llm_judge: Whether to include LLM-as-judge

        Returns:
            Dict with evaluation results
        """
        if not LANGSMITH_AVAILABLE:
            logger.error("LangSmith not available")
            return {"error": "LangSmith not available"}

        if evaluators is None:
            suite = CreditEvaluatorSuite(include_llm_judge=include_llm_judge)
            evaluators = suite.get_row_evaluators()

        eval_id = str(uuid.uuid4())[:8]
        experiment_name = f"{experiment_prefix}_{eval_id}"

        logger.info(f"Starting async evaluation: {experiment_name}")
        start_time = time.time()

        try:
            results = await aevaluate(
                target,
                data=dataset_name,
                evaluators=evaluators,
                experiment_prefix=experiment_prefix,
                max_concurrency=max_concurrency,
            )

            processed = self._process_results(results, eval_id, dataset_name, experiment_name)
            duration = time.time() - start_time

            if self.eval_logger:
                self.eval_logger.log_batch_results(
                    eval_id=eval_id,
                    dataset_name=dataset_name,
                    experiment_name=experiment_name,
                    results=processed["examples"],
                    duration_seconds=duration,
                )

            return {
                "eval_id": eval_id,
                "experiment_name": experiment_name,
                "duration_seconds": duration,
                "summary": processed["summary"],
                "examples": processed["examples"],
            }

        except Exception as e:
            logger.error(f"Async evaluation failed: {e}")
            return {"error": str(e)}

    def evaluate_single(
        self,
        target: Callable,
        inputs: Dict[str, Any],
        expected_outputs: Dict[str, Any],
        evaluators: List[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a single example without using a dataset.

        Args:
            target: Function to evaluate
            inputs: Input dict
            expected_outputs: Expected outputs
            evaluators: List of evaluators

        Returns:
            Evaluation results
        """
        if evaluators is None:
            suite = CreditEvaluatorSuite()
            evaluators = suite.get_row_evaluators()

        # Run target
        start_time = time.time()
        try:
            outputs = target(inputs)
            success = True
            error = None
        except Exception as e:
            outputs = {}
            success = False
            error = str(e)

        duration = time.time() - start_time

        # Run evaluators
        results = {}
        for evaluator in evaluators:
            try:
                result = evaluator(outputs, expected_outputs)
                if isinstance(result, bool):
                    results[evaluator.__name__] = {"score": 1.0 if result else 0.0}
                elif isinstance(result, (int, float)):
                    results[evaluator.__name__] = {"score": float(result)}
                elif isinstance(result, dict):
                    key = result.get("key", evaluator.__name__)
                    results[key] = result
            except Exception as e:
                results[evaluator.__name__] = {"score": 0.0, "error": str(e)}

        # Determine pass/fail
        scores = [r.get("score", 0) for r in results.values()]
        avg_score = sum(scores) / len(scores) if scores else 0
        passed = avg_score >= 0.5 and success

        return {
            "inputs": inputs,
            "outputs": outputs,
            "expected_outputs": expected_outputs,
            "results": results,
            "avg_score": avg_score,
            "passed": passed,
            "success": success,
            "error": error,
            "duration_seconds": duration,
        }

    def _process_results(
        self,
        results: Any,
        eval_id: str,
        dataset_name: str,
        experiment_name: str,
    ) -> Dict[str, Any]:
        """Process LangSmith evaluation results into our format."""
        examples = []

        # Results from LangSmith evaluate() is an ExperimentResults object
        # We need to iterate through it
        try:
            result_list = list(results)
        except:
            result_list = []

        for i, result in enumerate(result_list):
            try:
                # Extract from result object
                run = getattr(result, 'run', None)
                example = getattr(result, 'example', None)
                evaluation_results = getattr(result, 'evaluation_results', {})

                outputs = run.outputs if run else {}
                inputs = example.inputs if example else {}
                reference_outputs = example.outputs if example else {}

                # Extract scores from evaluation_results
                risk_correct = self._get_score(evaluation_results, "correct_risk_level", 0)
                score_acc = self._get_score(evaluation_results, "credit_score_accuracy", 0)
                tool_f1 = self._get_score(evaluation_results, "tool_selection_f1", 0)
                synthesis = self._get_score(evaluation_results, "synthesis_quality", 0)
                trajectory = self._get_score(evaluation_results, "trajectory_match", 0)
                llm_judge = self._get_score(evaluation_results, "llm_judge", 0)

                # Extract comments
                tool_comment = self._get_comment(evaluation_results, "tool_selection_f1")
                synthesis_comment = self._get_comment(evaluation_results, "synthesis_quality")

                # Determine pass/fail
                scores = [risk_correct, score_acc, tool_f1, synthesis]
                avg = sum(scores) / len(scores) if scores else 0
                passed = avg >= 0.5

                examples.append({
                    "example_id": str(example.id) if example else f"example_{i}",
                    "company_name": inputs.get("company_name", ""),
                    "risk_level_correct": risk_correct >= 0.5,
                    "credit_score_accuracy": score_acc,
                    "tool_selection_f1": tool_f1,
                    "synthesis_quality": synthesis,
                    "trajectory_match": trajectory,
                    "llm_judge_score": llm_judge,
                    "actual_risk_level": outputs.get("risk_level", outputs.get("assessment", {}).get("overall_risk_level", "")),
                    "expected_risk_level": reference_outputs.get("expected_risk_level", ""),
                    "actual_credit_score": outputs.get("credit_score", outputs.get("assessment", {}).get("credit_score_estimate", 0)),
                    "expected_credit_score_range": reference_outputs.get("expected_credit_score_range", []),
                    "actual_tools": self._extract_tools(outputs),
                    "expected_tools": reference_outputs.get("expected_tools", []),
                    "tool_selection_comment": tool_comment,
                    "synthesis_comment": synthesis_comment,
                    "passed": passed,
                    "error": "",
                })

            except Exception as e:
                logger.warning(f"Failed to process result {i}: {e}")
                examples.append({
                    "example_id": f"example_{i}",
                    "passed": False,
                    "error": str(e),
                })

        # Calculate summary
        total = len(examples)
        passed_count = sum(1 for e in examples if e.get("passed", False))

        summary = {
            "total": total,
            "passed": passed_count,
            "failed": total - passed_count,
            "pass_rate": passed_count / total if total else 0,
            "avg_risk_accuracy": sum(1 for e in examples if e.get("risk_level_correct", False)) / total if total else 0,
            "avg_score_accuracy": sum(e.get("credit_score_accuracy", 0) for e in examples) / total if total else 0,
            "avg_tool_f1": sum(e.get("tool_selection_f1", 0) for e in examples) / total if total else 0,
            "avg_synthesis": sum(e.get("synthesis_quality", 0) for e in examples) / total if total else 0,
        }

        return {"summary": summary, "examples": examples}

    def _get_score(self, results: Dict, key: str, default: float = 0.0) -> float:
        """Extract score from evaluation results."""
        if not results:
            return default
        for result in results.values() if isinstance(results, dict) else []:
            if hasattr(result, 'key') and result.key == key:
                return getattr(result, 'score', default)
            if isinstance(result, dict) and result.get("key") == key:
                return result.get("score", default)
        return default

    def _get_comment(self, results: Dict, key: str) -> str:
        """Extract comment from evaluation results."""
        if not results:
            return ""
        for result in results.values() if isinstance(results, dict) else []:
            if hasattr(result, 'key') and result.key == key:
                return getattr(result, 'comment', "")
            if isinstance(result, dict) and result.get("key") == key:
                return result.get("comment", "")
        return ""

    def _extract_tools(self, outputs: Dict) -> List[str]:
        """Extract tools used from outputs."""
        tools = outputs.get("tools_used", [])
        if not tools:
            api_data = outputs.get("api_data", {})
            for key in api_data:
                if api_data.get(key):
                    tools.append(f"fetch_{key}")
            if outputs.get("search_data"):
                tools.append("web_search")
        return tools

    def _run_local_evaluation(
        self,
        target: Callable,
        dataset_name: str,
        evaluators: List[Callable] = None,
    ) -> Dict[str, Any]:
        """Run evaluation locally without LangSmith."""
        logger.info(f"Running local evaluation on dataset: {dataset_name}")

        dataset = get_dataset(dataset_name)
        if not dataset:
            return {"error": f"Dataset not found: {dataset_name}"}

        examples = dataset.get("examples", [])
        if not examples:
            return {"error": "Dataset has no examples"}

        if evaluators is None:
            suite = CreditEvaluatorSuite()
            evaluators = suite.get_row_evaluators()

        eval_id = str(uuid.uuid4())[:8]
        results = []

        for i, example in enumerate(examples):
            inputs = example.get("inputs", {})
            expected = example.get("outputs", {})

            result = self.evaluate_single(target, inputs, expected, evaluators)
            result["example_id"] = f"example_{i}"
            result["company_name"] = inputs.get("company_name", "")
            results.append(result)

        # Summary
        total = len(results)
        passed = sum(1 for r in results if r.get("passed", False))

        return {
            "eval_id": eval_id,
            "dataset_name": dataset_name,
            "summary": {
                "total": total,
                "passed": passed,
                "failed": total - passed,
                "pass_rate": passed / total if total else 0,
            },
            "examples": results,
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def run_evaluation(
    target: Callable,
    dataset_name: str,
    evaluators: List[Callable] = None,
    experiment_prefix: str = "credit_eval",
    log_to_sheets: bool = True,
    log_to_mongodb: bool = True,
) -> Dict[str, Any]:
    """
    Run evaluation on a dataset.

    Args:
        target: Function to evaluate
        dataset_name: Name of the dataset
        evaluators: Custom evaluators (optional)
        experiment_prefix: Experiment prefix
        log_to_sheets: Log to Google Sheets
        log_to_mongodb: Log to MongoDB

    Returns:
        Evaluation results
    """
    runner = EvaluationRunner(
        log_to_sheets=log_to_sheets,
        log_to_mongodb=log_to_mongodb,
    )
    return runner.run_evaluation(
        target=target,
        dataset_name=dataset_name,
        evaluators=evaluators,
        experiment_prefix=experiment_prefix,
    )


async def run_async_evaluation(
    target: Callable,
    dataset_name: str,
    evaluators: List[Callable] = None,
    experiment_prefix: str = "credit_eval",
    log_to_sheets: bool = True,
    log_to_mongodb: bool = True,
) -> Dict[str, Any]:
    """
    Run async evaluation on a dataset.

    Args:
        target: Async function to evaluate
        dataset_name: Name of the dataset
        evaluators: Custom evaluators (optional)
        experiment_prefix: Experiment prefix
        log_to_sheets: Log to Google Sheets
        log_to_mongodb: Log to MongoDB

    Returns:
        Evaluation results
    """
    runner = EvaluationRunner(
        log_to_sheets=log_to_sheets,
        log_to_mongodb=log_to_mongodb,
    )
    return await runner.run_async_evaluation(
        target=target,
        dataset_name=dataset_name,
        evaluators=evaluators,
        experiment_prefix=experiment_prefix,
    )


def evaluate_single_run(
    target: Callable,
    company_name: str,
    expected_risk_level: str = None,
    expected_score_range: tuple = None,
    expected_tools: List[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate a single company analysis.

    Args:
        target: Function to evaluate
        company_name: Company to analyze
        expected_risk_level: Expected risk level (optional)
        expected_score_range: Expected score range (optional)
        expected_tools: Expected tools (optional)

    Returns:
        Evaluation results
    """
    runner = EvaluationRunner()

    inputs = {"company_name": company_name}
    expected = {}
    if expected_risk_level:
        expected["expected_risk_level"] = expected_risk_level
    if expected_score_range:
        expected["expected_credit_score_range"] = list(expected_score_range)
    if expected_tools:
        expected["expected_tools"] = expected_tools

    return runner.evaluate_single(target, inputs, expected)
