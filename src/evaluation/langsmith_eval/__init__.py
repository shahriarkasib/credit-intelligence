"""
LangSmith Evaluation Module - Separate evaluation framework using LangSmith SDK.

This module provides:
- Custom evaluators for credit intelligence workflow
- Dataset management for test cases
- Trajectory evaluation using agentevals
- Separate logging to Google Sheets (langsmith_evaluations tab)

Usage:
    from evaluation.langsmith_eval import (
        run_evaluation,
        create_dataset,
        LangSmithEvalLogger,
    )

    # Run evaluation on a dataset
    results = await run_evaluation(
        graph=my_graph,
        dataset_name="credit_intel_test",
    )
"""

from .evaluators import (
    correct_risk_level,
    credit_score_accuracy,
    tool_selection_f1,
    synthesis_quality,
    assessment_completeness,
    trajectory_evaluator,
    llm_as_judge_evaluator,
    CreditEvaluatorSuite,
)

from .datasets import (
    DatasetManager,
    create_credit_dataset,
    add_test_case,
    get_dataset,
    list_datasets,
)

from .runner import (
    run_evaluation,
    run_async_evaluation,
    evaluate_single_run,
    EvaluationRunner,
)

from .logger import (
    LangSmithEvalLogger,
    get_eval_logger,
)

__all__ = [
    # Evaluators
    "correct_risk_level",
    "credit_score_accuracy",
    "tool_selection_f1",
    "synthesis_quality",
    "assessment_completeness",
    "trajectory_evaluator",
    "llm_as_judge_evaluator",
    "CreditEvaluatorSuite",
    # Datasets
    "DatasetManager",
    "create_credit_dataset",
    "add_test_case",
    "get_dataset",
    "list_datasets",
    # Runner
    "run_evaluation",
    "run_async_evaluation",
    "evaluate_single_run",
    "EvaluationRunner",
    # Logger
    "LangSmithEvalLogger",
    "get_eval_logger",
]
