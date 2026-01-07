"""
LangSmith Integration - Fetches traces and logs them to Google Sheets.

This module:
1. Fetches recent traces from LangSmith API
2. Logs them to langsmith_traces sheet
3. Manages evaluation examples and runs

Requires valid LANGCHAIN_API_KEY set in config/settings.yaml
"""

import os
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Try to import langsmith
try:
    from langsmith import Client
    from langsmith.schemas import Run
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    logger.warning("LangSmith not installed. Run: pip install langsmith")

# Try to import sheets logger
try:
    from run_logging.sheets_logger import get_sheets_logger
    SHEETS_AVAILABLE = True
except ImportError:
    SHEETS_AVAILABLE = False


class LangSmithIntegration:
    """
    Integrates LangSmith traces with Google Sheets logging.

    Usage:
        integration = LangSmithIntegration()

        # Fetch and log recent traces
        integration.fetch_and_log_traces(limit=10)

        # Log a specific run
        integration.log_run_trace(run_id="...")
    """

    def __init__(self, project_name: str = None):
        self.project_name = project_name or os.getenv("LANGCHAIN_PROJECT", "credit-intelligence")
        self._client = None
        self._sheets_logger = None

        if not LANGSMITH_AVAILABLE:
            logger.error("LangSmith not available")
            return

        if not os.getenv("LANGCHAIN_API_KEY"):
            logger.warning("LANGCHAIN_API_KEY not set - LangSmith integration disabled")
            return

    @property
    def client(self):
        """Lazy load LangSmith client."""
        if self._client is None and LANGSMITH_AVAILABLE:
            try:
                self._client = Client()
            except Exception as e:
                logger.error(f"Failed to create LangSmith client: {e}")
        return self._client

    @property
    def sheets_logger(self):
        """Lazy load sheets logger."""
        if self._sheets_logger is None and SHEETS_AVAILABLE:
            self._sheets_logger = get_sheets_logger()
        return self._sheets_logger

    def fetch_and_log_traces(
        self,
        limit: int = 20,
        hours_back: int = 24,
    ) -> int:
        """
        Fetch recent traces from LangSmith and log to sheets.

        Args:
            limit: Max number of traces to fetch
            hours_back: How far back to look for traces

        Returns:
            Number of traces logged
        """
        if not self.client:
            logger.error("LangSmith client not available")
            return 0

        if not self.sheets_logger:
            logger.error("Sheets logger not available")
            return 0

        try:
            # Fetch recent runs
            start_time = datetime.utcnow() - timedelta(hours=hours_back)

            runs = list(self.client.list_runs(
                project_name=self.project_name,
                start_time=start_time,
                limit=limit,
                is_root=True,  # Only root runs (not child spans)
            ))

            logger.info(f"Fetched {len(runs)} traces from LangSmith")

            logged = 0
            for run in runs:
                try:
                    self._log_run_to_sheets(run)
                    logged += 1
                except Exception as e:
                    logger.warning(f"Failed to log run {run.id}: {e}")

            return logged

        except Exception as e:
            logger.error(f"Failed to fetch traces: {e}")
            return 0

    def _log_run_to_sheets(self, run: "Run"):
        """Log a single run to the langsmith_traces sheet."""
        # Extract run info
        run_id = str(run.id)

        # Try to get company name from inputs
        company_name = ""
        if run.inputs:
            company_name = (
                run.inputs.get("company_name") or
                run.inputs.get("input", {}).get("company_name") or
                ""
            )

        # Calculate latency
        latency_ms = 0
        if run.start_time and run.end_time:
            latency_ms = (run.end_time - run.start_time).total_seconds() * 1000

        # Get input/output previews
        input_preview = str(run.inputs)[:10000] if run.inputs else ""
        output_preview = str(run.outputs)[:10000] if run.outputs else ""

        # Get error if any
        error = run.error or ""

        # Get model from extra metadata
        model = ""
        temperature = None
        if run.extra:
            metadata = run.extra.get("metadata", {})
            model = metadata.get("model", "") or metadata.get("ls_model_name", "")
            temperature = metadata.get("temperature")

        # Log to sheets (matching existing method signature)
        self.sheets_logger.log_langsmith_trace(
            run_id=run_id,
            company_name=company_name,
            step_name=run.name or "",
            run_type=run.run_type or "",
            status=run.status or "unknown",
            latency_ms=latency_ms,
            node=run.name or "",
            node_type=run.run_type or "",
            model=model,
            temperature=temperature,
            error=error,
            input_preview=input_preview,
            output_preview=output_preview,
        )

    def log_run_trace(self, run_id: str) -> bool:
        """
        Log a specific run by ID.

        Args:
            run_id: The LangSmith run ID

        Returns:
            True if successful
        """
        if not self.client or not self.sheets_logger:
            return False

        try:
            run = self.client.read_run(run_id)
            self._log_run_to_sheets(run)
            return True
        except Exception as e:
            logger.error(f"Failed to log run {run_id}: {e}")
            return False

    # ==================== EVALUATION EXAMPLES ====================

    def create_evaluation_example(
        self,
        dataset_name: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None,
    ) -> Optional[str]:
        """
        Create an evaluation example in LangSmith dataset.

        Args:
            dataset_name: Name of the dataset
            inputs: Input data for the example
            outputs: Expected output data
            metadata: Additional metadata

        Returns:
            Example ID if successful
        """
        if not self.client:
            return None

        try:
            # Get or create dataset
            try:
                dataset = self.client.read_dataset(dataset_name=dataset_name)
            except Exception:
                dataset = self.client.create_dataset(dataset_name=dataset_name)

            # Create example
            example = self.client.create_example(
                inputs=inputs,
                outputs=outputs or {},
                dataset_id=dataset.id,
                metadata=metadata or {},
            )

            # Log to sheets
            if self.sheets_logger:
                self.sheets_logger.log_langsmith_eval_example(
                    example_id=str(example.id),
                    dataset_name=dataset_name,
                    company_name=inputs.get("company_name", ""),
                    input_data=inputs,
                    expected_output=outputs or {},
                    metadata=metadata or {},
                )

            logger.info(f"Created evaluation example: {example.id}")
            return str(example.id)

        except Exception as e:
            logger.error(f"Failed to create evaluation example: {e}")
            return None

    def fetch_and_log_eval_examples(
        self,
        dataset_name: str = "credit-intelligence-eval",
        limit: int = 50,
    ) -> int:
        """
        Fetch evaluation examples from LangSmith and log to sheets.

        Args:
            dataset_name: Name of the dataset
            limit: Max examples to fetch

        Returns:
            Number of examples logged
        """
        if not self.client or not self.sheets_logger:
            return 0

        try:
            dataset = self.client.read_dataset(dataset_name=dataset_name)
            examples = list(self.client.list_examples(dataset_id=dataset.id, limit=limit))

            logged = 0
            for example in examples:
                try:
                    self.sheets_logger.log_langsmith_eval_example(
                        example_id=str(example.id),
                        dataset_name=dataset_name,
                        company_name=example.inputs.get("company_name", "") if example.inputs else "",
                        input_data=example.inputs or {},
                        expected_output=example.outputs or {},
                        metadata=example.metadata or {},
                    )
                    logged += 1
                except Exception as e:
                    logger.warning(f"Failed to log example {example.id}: {e}")

            return logged

        except Exception as e:
            logger.error(f"Failed to fetch evaluation examples: {e}")
            return 0

    # ==================== EVALUATION RUNS ====================

    def run_evaluation(
        self,
        dataset_name: str = "credit-intelligence-eval",
        evaluator_name: str = "credit_assessment_quality",
    ) -> Optional[str]:
        """
        Run an evaluation on a dataset.

        Args:
            dataset_name: Name of the dataset to evaluate
            evaluator_name: Name of the evaluator to use

        Returns:
            Evaluation run ID if successful
        """
        if not self.client:
            return None

        try:
            from langsmith.evaluation import evaluate
            from evaluation.langsmith_eval.evaluators import (
                get_all_evaluators,
            )

            # Get dataset
            dataset = self.client.read_dataset(dataset_name=dataset_name)

            # Get evaluators
            evaluators = get_all_evaluators()

            # Run evaluation
            results = evaluate(
                lambda inputs: self._run_workflow(inputs),
                data=dataset_name,
                evaluators=evaluators,
                experiment_prefix=f"eval-{datetime.now().strftime('%Y%m%d-%H%M')}",
            )

            # Log results to sheets
            if self.sheets_logger and results:
                for result in results:
                    self.sheets_logger.log_langsmith_eval_run(
                        eval_run_id=str(result.get("run_id", "")),
                        dataset_name=dataset_name,
                        evaluator_name=evaluator_name,
                        company_name=result.get("inputs", {}).get("company_name", ""),
                        score=result.get("score", 0),
                        feedback=result.get("feedback", ""),
                        model_output=result.get("output", {}),
                        expected_output=result.get("expected", {}),
                        latency_ms=result.get("latency_ms", 0),
                        status="completed",
                    )

            return str(results) if results else None

        except Exception as e:
            logger.error(f"Failed to run evaluation: {e}")
            return None

    def _run_workflow(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run the credit assessment workflow for evaluation."""
        try:
            from agents.graph import run_sync_with_logging
            company_name = inputs.get("company_name", "Unknown")
            result = run_sync_with_logging(company_name)
            return result
        except Exception as e:
            return {"error": str(e)}

    def fetch_and_log_eval_runs(
        self,
        project_name: str = None,
        limit: int = 20,
    ) -> int:
        """
        Fetch evaluation runs from LangSmith and log to sheets.

        Args:
            project_name: Project name (defaults to current project)
            limit: Max runs to fetch

        Returns:
            Number of runs logged
        """
        if not self.client or not self.sheets_logger:
            return 0

        project = project_name or self.project_name

        try:
            # Get runs with feedback (evaluation runs)
            runs = list(self.client.list_runs(
                project_name=project,
                has_feedback=True,
                limit=limit,
            ))

            logged = 0
            for run in runs:
                try:
                    # Get feedback for this run
                    feedbacks = list(self.client.list_feedback(run_ids=[run.id]))

                    for feedback in feedbacks:
                        self.sheets_logger.log_langsmith_eval_run(
                            eval_run_id=str(run.id),
                            dataset_name="",
                            evaluator_name=feedback.key or "",
                            company_name=run.inputs.get("company_name", "") if run.inputs else "",
                            score=feedback.score or 0,
                            feedback=feedback.comment or "",
                            model_output=run.outputs or {},
                            expected_output={},
                            latency_ms=(run.end_time - run.start_time).total_seconds() * 1000 if run.end_time and run.start_time else 0,
                            status=run.status or "unknown",
                        )
                        logged += 1

                except Exception as e:
                    logger.warning(f"Failed to log eval run {run.id}: {e}")

            return logged

        except Exception as e:
            logger.error(f"Failed to fetch evaluation runs: {e}")
            return 0


# Singleton instance
_langsmith_integration: Optional[LangSmithIntegration] = None


def get_langsmith_integration() -> LangSmithIntegration:
    """Get or create LangSmith integration instance."""
    global _langsmith_integration
    if _langsmith_integration is None:
        _langsmith_integration = LangSmithIntegration()
    return _langsmith_integration


def fetch_and_log_all_langsmith_data(
    traces_limit: int = 20,
    examples_limit: int = 50,
    eval_runs_limit: int = 20,
) -> Dict[str, int]:
    """
    Convenience function to fetch and log all LangSmith data.

    Returns:
        Dict with counts of items logged for each category
    """
    integration = get_langsmith_integration()

    results = {
        "traces": integration.fetch_and_log_traces(limit=traces_limit),
        "eval_examples": integration.fetch_and_log_eval_examples(limit=examples_limit),
        "eval_runs": integration.fetch_and_log_eval_runs(limit=eval_runs_limit),
    }

    logger.info(f"LangSmith data sync complete: {results}")
    return results
