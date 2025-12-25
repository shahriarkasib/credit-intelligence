"""Evaluation Runner - Main entry point for consistency evaluation."""

import argparse
import json
import logging
import os
from typing import Any, Dict, List, Optional
from datetime import datetime

from .execution_wrapper import ExecutionWrapper, ModelConfig
from .consistency_scorer import ConsistencyScorer
from .correctness_scorer import CorrectnessScorer
from .analyzer import CorrelationAnalyzer, EvaluationRecord

logger = logging.getLogger(__name__)


def load_test_set(filepath: str) -> List[Dict[str, Any]]:
    """Load golden test set from JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)

    # Support both list and dict formats
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        return data.get("test_cases", data.get("samples", []))
    else:
        raise ValueError(f"Invalid test set format in {filepath}")


def run_evaluation(
    test_set_path: str,
    models: Optional[List[str]] = None,
    output_dir: str = "data/results",
    consistency_method: str = "semantic_similarity",
    correctness_method: str = "semantic_similarity",
    consistency_threshold: float = 0.85,
    correctness_threshold: float = 0.90,
    parallel: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run full evaluation pipeline.

    Args:
        test_set_path: Path to golden test set JSON
        models: List of model names to use (optional)
        output_dir: Directory for output files
        consistency_method: Method for consistency scoring
        correctness_method: Method for correctness scoring
        consistency_threshold: Threshold for consistency
        correctness_threshold: Threshold for correctness
        parallel: Run models in parallel
        verbose: Print progress

    Returns:
        Evaluation results dictionary
    """
    # Setup logging
    if verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    logger.info(f"Loading test set from: {test_set_path}")
    test_cases = load_test_set(test_set_path)
    logger.info(f"Loaded {len(test_cases)} test cases")

    # Initialize components
    model_configs = None
    if models:
        model_configs = [
            ModelConfig(name=m, provider=_get_provider(m), model_id=_get_model_id(m), enabled=True)
            for m in models
        ]

    wrapper = ExecutionWrapper(models=model_configs, parallel=parallel)
    consistency_scorer = ConsistencyScorer(
        method=consistency_method,
        similarity_threshold=consistency_threshold,
    )
    correctness_scorer = CorrectnessScorer(
        method=correctness_method,
        similarity_threshold=correctness_threshold,
    )
    analyzer = CorrelationAnalyzer(results_dir=output_dir)

    # Check available models
    available = wrapper.get_available_models()
    logger.info(f"Available models: {available}")

    if len(available) < 2:
        logger.warning("Less than 2 models available. Consistency scoring may not be meaningful.")

    # Run evaluation
    for i, test_case in enumerate(test_cases):
        prompt_id = test_case.get("id", f"test_{i}")
        prompt = test_case.get("prompt", test_case.get("question", ""))
        context = test_case.get("context")
        golden_answer = test_case.get("golden_answer", test_case.get("answer", ""))

        if not prompt or not golden_answer:
            logger.warning(f"Skipping test case {prompt_id}: missing prompt or golden answer")
            continue

        if verbose:
            logger.info(f"Processing test case {i+1}/{len(test_cases)}: {prompt_id}")

        # Execute on all models
        exec_result = wrapper.execute(
            prompt=prompt,
            context=context,
            golden_answer=golden_answer,
            prompt_id=prompt_id,
        )

        # Get outputs
        outputs = exec_result.get_outputs()
        model_names = [r.model_name for r in exec_result.responses if r.output and not r.error]

        if len(outputs) < 2:
            logger.warning(f"Test case {prompt_id}: Not enough model outputs ({len(outputs)})")
            continue

        # Score consistency
        consistency_result = consistency_scorer.score(outputs)

        # Score correctness
        correctness_result = correctness_scorer.score(outputs, golden_answer, model_names)

        # Add to analyzer
        analyzer.add_result(
            prompt_id=prompt_id,
            prompt=prompt,
            golden_answer=golden_answer,
            model_outputs=[r.to_dict() for r in exec_result.responses],
            consistency_result=consistency_result.to_dict(),
            correctness_result=correctness_result.to_dict(),
        )

    # Generate report
    report = analyzer.analyze()

    if verbose:
        print("\n" + report.get_summary())

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_file = analyzer.save_results()

    logger.info(f"Evaluation complete. Results saved to: {results_file}")

    return report.to_dict()


def _get_provider(model_name: str) -> str:
    """Get provider from model name."""
    model_lower = model_name.lower()
    if "gpt" in model_lower or "openai" in model_lower:
        return "openai"
    elif "claude" in model_lower or "anthropic" in model_lower:
        return "anthropic"
    elif "llama" in model_lower or "mistral" in model_lower or "ollama" in model_lower:
        return "ollama"
    elif "gemini" in model_lower or "google" in model_lower:
        return "google"
    else:
        return "openai"  # Default


def _get_model_id(model_name: str) -> str:
    """Get model ID from model name."""
    # Common mappings
    mappings = {
        "gpt-4o-mini": "gpt-4o-mini",
        "gpt-4": "gpt-4",
        "gpt-4o": "gpt-4o",
        "gpt-3.5-turbo": "gpt-3.5-turbo",
        "claude-3-haiku": "claude-3-haiku-20240307",
        "claude-3-sonnet": "claude-3-sonnet-20240229",
        "claude-3-opus": "claude-3-opus-20240229",
        "llama3.2": "llama3.2",
        "llama3": "llama3",
        "mistral": "mistral",
        "gemini-flash": "gemini-1.5-flash",
        "gemini-pro": "gemini-1.5-pro",
    }

    return mappings.get(model_name.lower(), model_name)


def create_sample_test_set(output_path: str = "data/golden_test_sets/sample_finance.json"):
    """Create a sample golden test set for demonstration."""
    test_cases = [
        {
            "id": "fin_001",
            "prompt": "What is the formula for calculating Return on Equity (ROE)?",
            "golden_answer": "ROE = Net Income / Shareholders' Equity",
            "category": "finance_formula",
        },
        {
            "id": "fin_002",
            "prompt": "What does a current ratio below 1.0 indicate about a company?",
            "golden_answer": "A current ratio below 1.0 indicates that a company may have difficulty paying its short-term obligations, as its current liabilities exceed its current assets.",
            "category": "financial_analysis",
        },
        {
            "id": "fin_003",
            "prompt": "What is the difference between revenue and net income?",
            "golden_answer": "Revenue is the total amount of money generated from sales, while net income is the profit remaining after all expenses, taxes, and costs are deducted from revenue.",
            "category": "finance_basics",
        },
        {
            "id": "fin_004",
            "prompt": "Is Apple Inc a public or private company?",
            "golden_answer": "Public",
            "category": "classification",
        },
        {
            "id": "fin_005",
            "prompt": "What type of financial statement shows a company's assets, liabilities, and equity at a specific point in time?",
            "golden_answer": "Balance Sheet",
            "category": "finance_basics",
        },
        {
            "id": "credit_001",
            "prompt": "What are the main factors that affect a company's creditworthiness?",
            "golden_answer": "The main factors affecting a company's creditworthiness include: financial health (revenue, profitability, cash flow), debt levels, payment history, industry risk, management quality, and company age/stability.",
            "category": "credit_analysis",
        },
        {
            "id": "credit_002",
            "prompt": "What is a debt-to-equity ratio of 2.0 considered?",
            "golden_answer": "A debt-to-equity ratio of 2.0 is generally considered high, indicating that the company has twice as much debt as equity, which may pose higher financial risk.",
            "category": "credit_analysis",
        },
        {
            "id": "risk_001",
            "prompt": "Should a company on the OFAC sanctions list be approved for credit?",
            "golden_answer": "No, a company on the OFAC sanctions list should not be approved for credit as it is legally prohibited to conduct business with sanctioned entities.",
            "category": "compliance",
        },
        {
            "id": "math_001",
            "prompt": "If a company has $500 million in revenue and $50 million in net income, what is the profit margin?",
            "golden_answer": "10%",
            "category": "calculation",
        },
        {
            "id": "math_002",
            "prompt": "A company has current assets of $2 million and current liabilities of $1.5 million. What is the current ratio?",
            "golden_answer": "1.33",
            "category": "calculation",
        },
    ]

    # Create directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump({"test_cases": test_cases, "created": datetime.utcnow().isoformat()}, f, indent=2)

    print(f"Sample test set created: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Consistency-as-Correctness Evaluation")
    parser.add_argument(
        "--test-set",
        required=True,
        help="Path to golden test set JSON file",
    )
    parser.add_argument(
        "--models",
        help="Comma-separated list of model names to use",
    )
    parser.add_argument(
        "--output-dir",
        default="data/results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--consistency-method",
        default="semantic_similarity",
        choices=["exact_match", "semantic_similarity", "keyword_overlap"],
        help="Method for consistency scoring",
    )
    parser.add_argument(
        "--correctness-method",
        default="semantic_similarity",
        choices=["exact_match", "semantic_similarity", "contains", "classification"],
        help="Method for correctness scoring",
    )
    parser.add_argument(
        "--consistency-threshold",
        type=float,
        default=0.85,
        help="Threshold for consistency (0-1)",
    )
    parser.add_argument(
        "--correctness-threshold",
        type=float,
        default=0.90,
        help="Threshold for correctness (0-1)",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run models sequentially instead of parallel",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )
    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Create a sample test set and exit",
    )

    args = parser.parse_args()

    if args.create_sample:
        create_sample_test_set()
        exit(0)

    models = args.models.split(",") if args.models else None

    run_evaluation(
        test_set_path=args.test_set,
        models=models,
        output_dir=args.output_dir,
        consistency_method=args.consistency_method,
        correctness_method=args.correctness_method,
        consistency_threshold=args.consistency_threshold,
        correctness_threshold=args.correctness_threshold,
        parallel=not args.sequential,
        verbose=not args.quiet,
    )
