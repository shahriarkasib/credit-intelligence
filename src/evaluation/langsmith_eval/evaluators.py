"""
Custom Evaluators for Credit Intelligence using LangSmith SDK.

These evaluators are designed to work with LangSmith's evaluate() and aevaluate() functions.
Each evaluator follows the LangSmith signature conventions.
"""

import logging
from typing import Any, Dict, List, Optional, Sequence, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Try to import LangSmith and agentevals
try:
    from langsmith.schemas import Run, Example
    from langsmith import Client
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    Run = None
    Example = None
    Client = None
    logger.warning("LangSmith SDK not installed. Install with: pip install langsmith")

try:
    from agentevals import (
        create_trajectory_match_evaluator,
        create_async_trajectory_match_evaluator,
    )
    AGENTEVALS_AVAILABLE = True
except ImportError:
    AGENTEVALS_AVAILABLE = False
    create_trajectory_match_evaluator = None
    create_async_trajectory_match_evaluator = None
    logger.warning("agentevals not installed. Install with: pip install agentevals")


# =============================================================================
# SIMPLE EVALUATORS (New LangSmith SDK signature)
# =============================================================================

def correct_risk_level(outputs: Dict, reference_outputs: Dict) -> bool:
    """
    Check if the predicted risk level exactly matches the expected.

    Args:
        outputs: Model outputs containing 'risk_level' or 'overall_risk_level'
        reference_outputs: Expected outputs with 'risk_level' or 'expected_risk_level'

    Returns:
        True if risk levels match
    """
    actual = (
        outputs.get("risk_level") or
        outputs.get("overall_risk_level") or
        outputs.get("assessment", {}).get("overall_risk_level", "")
    ).lower()

    expected = (
        reference_outputs.get("expected_risk_level") or
        reference_outputs.get("risk_level", "")
    ).lower()

    return actual == expected


def credit_score_accuracy(outputs: Dict, reference_outputs: Dict) -> float:
    """
    Calculate accuracy based on how close the credit score is to expected.

    Args:
        outputs: Model outputs containing 'credit_score' or 'credit_score_estimate'
        reference_outputs: Expected outputs with 'credit_score' or score range

    Returns:
        Score from 0.0 to 1.0 (1.0 = perfect match)
    """
    actual = (
        outputs.get("credit_score") or
        outputs.get("credit_score_estimate") or
        outputs.get("assessment", {}).get("credit_score_estimate", 0)
    )

    # Handle range-based expectations
    if "expected_credit_score_range" in reference_outputs:
        min_score, max_score = reference_outputs["expected_credit_score_range"]
        if min_score <= actual <= max_score:
            return 1.0
        # Calculate how far outside the range
        if actual < min_score:
            distance = min_score - actual
        else:
            distance = actual - max_score
        return max(0.0, 1.0 - (distance / 50))  # 50 point buffer

    # Exact comparison
    expected = reference_outputs.get("expected_credit_score", reference_outputs.get("credit_score", 50))
    diff = abs(actual - expected)
    return max(0.0, 1.0 - (diff / 100))


def tool_selection_f1(outputs: Dict, reference_outputs: Dict) -> Dict[str, Any]:
    """
    Calculate F1 score for tool selection accuracy.

    Args:
        outputs: Model outputs containing 'tools_used' or similar
        reference_outputs: Expected outputs with 'expected_tools'

    Returns:
        Dict with key, score, and reasoning
    """
    # Extract actual tools used
    actual_tools = set(
        outputs.get("tools_used") or
        outputs.get("selected_tools") or
        _extract_tools_from_state(outputs)
    )

    # Extract expected tools
    expected_tools = set(reference_outputs.get("expected_tools", []))

    if not expected_tools:
        return {"key": "tool_selection_f1", "score": 1.0, "comment": "No expected tools defined"}

    # Calculate metrics
    true_positives = len(actual_tools & expected_tools)
    precision = true_positives / len(actual_tools) if actual_tools else 0.0
    recall = true_positives / len(expected_tools) if expected_tools else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    correct = list(actual_tools & expected_tools)
    missing = list(expected_tools - actual_tools)
    extra = list(actual_tools - expected_tools)

    comment = f"Precision: {precision:.2f}, Recall: {recall:.2f}. "
    if correct:
        comment += f"Correct: {', '.join(correct)}. "
    if missing:
        comment += f"Missing: {', '.join(missing)}. "
    if extra:
        comment += f"Extra: {', '.join(extra)}."

    return {
        "key": "tool_selection_f1",
        "score": f1,
        "comment": comment,
    }


def synthesis_quality(outputs: Dict, reference_outputs: Dict = None) -> Dict[str, Any]:
    """
    Evaluate the quality of the synthesized assessment.

    Checks for presence and quality of required fields.

    Args:
        outputs: Model outputs containing assessment
        reference_outputs: Optional expected outputs (unused)

    Returns:
        Dict with key, score, and reasoning
    """
    assessment = outputs.get("assessment", outputs)

    # Required fields
    checks = {
        "has_risk_level": bool(assessment.get("overall_risk_level") or assessment.get("risk_level")),
        "has_credit_score": bool(assessment.get("credit_score_estimate") or assessment.get("credit_score")),
        "has_reasoning": bool(assessment.get("llm_reasoning") or assessment.get("reasoning")),
        "has_recommendations": bool(assessment.get("recommendations")),
        "has_confidence": bool(assessment.get("confidence_score") or assessment.get("confidence")),
    }

    # Calculate score
    present = sum(checks.values())
    score = present / len(checks)

    # Build comment
    present_fields = [k.replace("has_", "") for k, v in checks.items() if v]
    missing_fields = [k.replace("has_", "") for k, v in checks.items() if not v]

    comment = f"Present: {', '.join(present_fields)}. " if present_fields else ""
    comment += f"Missing: {', '.join(missing_fields)}." if missing_fields else ""

    return {
        "key": "synthesis_quality",
        "score": score,
        "comment": comment.strip(),
    }


def assessment_completeness(outputs: Dict, reference_outputs: Dict = None) -> float:
    """
    Calculate assessment completeness as a simple score.

    Args:
        outputs: Model outputs
        reference_outputs: Optional expected outputs

    Returns:
        Score from 0.0 to 1.0
    """
    assessment = outputs.get("assessment", outputs)

    required_fields = [
        "overall_risk_level", "credit_score_estimate", "confidence_score",
        "recommendations", "risk_factors", "positive_factors"
    ]

    present = sum(1 for f in required_fields if assessment.get(f))
    return present / len(required_fields)


# =============================================================================
# LEGACY SIGNATURE EVALUATORS (Run, Example objects)
# =============================================================================

def tool_selection_evaluator(run: 'Run', example: 'Example') -> Dict[str, Any]:
    """
    Evaluate tool selection using Run and Example objects.

    For use with older LangSmith SDK patterns.
    """
    outputs = run.outputs or {}
    reference_outputs = example.outputs or {}
    return tool_selection_f1(outputs, reference_outputs)


def risk_level_evaluator(run: 'Run', example: 'Example') -> Dict[str, Any]:
    """
    Evaluate risk level correctness using Run and Example objects.
    """
    outputs = run.outputs or {}
    reference_outputs = example.outputs or {}
    is_correct = correct_risk_level(outputs, reference_outputs)
    return {
        "key": "risk_level_correct",
        "score": 1.0 if is_correct else 0.0,
    }


# =============================================================================
# TRAJECTORY EVALUATORS
# =============================================================================

def trajectory_evaluator(
    outputs: Dict,
    reference_outputs: Dict,
    match_mode: str = "unordered",
) -> Dict[str, Any]:
    """
    Evaluate agent trajectory (sequence of tool calls / node visits).

    Args:
        outputs: Model outputs containing trajectory info
        reference_outputs: Expected trajectory
        match_mode: One of 'strict', 'unordered', 'subset', 'superset'

    Returns:
        Dict with key, score, and reasoning
    """
    if not AGENTEVALS_AVAILABLE:
        return {
            "key": "trajectory_match",
            "score": 0.0,
            "comment": "agentevals not installed",
        }

    # Extract trajectory from outputs
    actual_trajectory = _extract_trajectory(outputs)
    expected_trajectory = reference_outputs.get("expected_trajectory", [])

    if not expected_trajectory:
        return {
            "key": "trajectory_match",
            "score": 1.0,
            "comment": "No expected trajectory defined",
        }

    # Compare trajectories based on mode
    if match_mode == "strict":
        is_match = actual_trajectory == expected_trajectory
        score = 1.0 if is_match else 0.0
    elif match_mode == "unordered":
        actual_set = set(actual_trajectory)
        expected_set = set(expected_trajectory)
        overlap = len(actual_set & expected_set)
        score = overlap / len(expected_set) if expected_set else 1.0
    elif match_mode == "subset":
        # Actual should be subset of expected
        actual_set = set(actual_trajectory)
        expected_set = set(expected_trajectory)
        is_subset = actual_set <= expected_set
        score = 1.0 if is_subset else len(actual_set & expected_set) / len(actual_set) if actual_set else 1.0
    elif match_mode == "superset":
        # Actual should be superset of expected
        actual_set = set(actual_trajectory)
        expected_set = set(expected_trajectory)
        is_superset = actual_set >= expected_set
        score = 1.0 if is_superset else len(actual_set & expected_set) / len(expected_set) if expected_set else 1.0
    else:
        score = 0.0

    return {
        "key": "trajectory_match",
        "score": score,
        "comment": f"Mode: {match_mode}. Actual: {actual_trajectory}. Expected: {expected_trajectory}",
    }


def create_graph_trajectory_evaluator(match_mode: str = "unordered"):
    """
    Create a trajectory evaluator for LangGraph node sequences.

    Args:
        match_mode: One of 'strict', 'unordered', 'subset', 'superset'

    Returns:
        Evaluator function
    """
    def evaluator(outputs: Dict, reference_outputs: Dict) -> Dict[str, Any]:
        return trajectory_evaluator(outputs, reference_outputs, match_mode)

    return evaluator


# =============================================================================
# LLM-AS-JUDGE EVALUATORS
# =============================================================================

def llm_as_judge_evaluator(
    outputs: Dict,
    inputs: Dict,
    judge_model: str = "gpt-4o",
    criteria: str = None,
) -> Dict[str, Any]:
    """
    Use an LLM to judge the quality of the assessment.

    Args:
        outputs: Model outputs
        inputs: Original inputs
        judge_model: Model to use as judge
        criteria: Custom criteria prompt

    Returns:
        Dict with key, score, and reasoning
    """
    try:
        from openai import OpenAI
    except ImportError:
        return {
            "key": "llm_judge",
            "score": 0.0,
            "comment": "OpenAI not installed for LLM judge",
        }

    import os
    if not os.getenv("OPENAI_API_KEY"):
        return {
            "key": "llm_judge",
            "score": 0.0,
            "comment": "OPENAI_API_KEY not set",
        }

    default_criteria = """
    Evaluate the credit assessment on the following criteria (score 1-5 each):
    1. Completeness - Does it cover all relevant risk factors?
    2. Accuracy - Is the risk level appropriate given the company info?
    3. Reasoning - Is the reasoning clear, logical, and well-supported?
    4. Actionability - Are the recommendations practical and useful?

    Provide scores for each criterion and an overall score (1-5).
    """

    client = OpenAI()
    company_name = inputs.get("company_name", "Unknown")
    assessment = outputs.get("assessment", outputs)

    prompt = f"""
    {criteria or default_criteria}

    Company: {company_name}

    Assessment:
    - Risk Level: {assessment.get('overall_risk_level', 'N/A')}
    - Credit Score: {assessment.get('credit_score_estimate', 'N/A')}
    - Confidence: {assessment.get('confidence_score', 'N/A')}
    - Reasoning: {assessment.get('llm_reasoning', 'N/A')[:500]}
    - Recommendations: {assessment.get('recommendations', [])}

    Respond with JSON:
    {{"completeness": <1-5>, "accuracy": <1-5>, "reasoning": <1-5>, "actionability": <1-5>, "overall": <1-5>, "feedback": "<brief feedback>"}}
    """

    try:
        response = client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )

        import json
        result = json.loads(response.choices[0].message.content)

        # Normalize to 0-1 scale
        overall_score = result.get("overall", 3) / 5.0

        return {
            "key": "llm_judge",
            "score": overall_score,
            "comment": result.get("feedback", ""),
            "details": result,
        }

    except Exception as e:
        logger.error(f"LLM judge failed: {e}")
        return {
            "key": "llm_judge",
            "score": 0.0,
            "comment": f"Error: {str(e)}",
        }


# =============================================================================
# SUMMARY EVALUATORS (Experiment-level)
# =============================================================================

def overall_accuracy(runs: Sequence['Run'], examples: Sequence['Example']) -> Dict[str, Any]:
    """
    Calculate overall accuracy across all runs in an experiment.

    Args:
        runs: All runs in the experiment
        examples: All examples

    Returns:
        Dict with aggregated metrics
    """
    if not runs or not examples:
        return {"key": "overall_accuracy", "score": 0.0}

    correct = 0
    for run, example in zip(runs, examples):
        outputs = run.outputs or {}
        reference_outputs = example.outputs or {}
        if correct_risk_level(outputs, reference_outputs):
            correct += 1

    accuracy = correct / len(runs)
    return {
        "key": "overall_accuracy",
        "score": accuracy,
        "comment": f"{correct}/{len(runs)} correct risk levels",
    }


def average_f1_score(runs: Sequence['Run'], examples: Sequence['Example']) -> Dict[str, Any]:
    """
    Calculate average F1 score across all runs.
    """
    if not runs or not examples:
        return {"key": "average_f1", "score": 0.0}

    f1_scores = []
    for run, example in zip(runs, examples):
        outputs = run.outputs or {}
        reference_outputs = example.outputs or {}
        result = tool_selection_f1(outputs, reference_outputs)
        f1_scores.append(result.get("score", 0.0))

    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    return {
        "key": "average_f1",
        "score": avg_f1,
        "comment": f"Average F1 across {len(runs)} runs",
    }


# =============================================================================
# EVALUATOR SUITE
# =============================================================================

@dataclass
class CreditEvaluatorSuite:
    """
    Collection of all evaluators for credit intelligence.

    Usage:
        suite = CreditEvaluatorSuite()
        evaluators = suite.get_all_evaluators()

        results = evaluate(
            my_graph,
            data="credit_dataset",
            evaluators=evaluators,
        )
    """
    include_llm_judge: bool = False
    include_trajectory: bool = True
    trajectory_mode: str = "unordered"

    def get_row_evaluators(self) -> List:
        """Get evaluators that run on each row."""
        evaluators = [
            correct_risk_level,
            credit_score_accuracy,
            tool_selection_f1,
            synthesis_quality,
            assessment_completeness,
        ]

        if self.include_trajectory:
            evaluators.append(create_graph_trajectory_evaluator(self.trajectory_mode))

        if self.include_llm_judge:
            evaluators.append(llm_as_judge_evaluator)

        return evaluators

    def get_summary_evaluators(self) -> List:
        """Get evaluators that run on the entire experiment."""
        return [
            overall_accuracy,
            average_f1_score,
        ]

    def get_all_evaluators(self) -> List:
        """Get all evaluators."""
        return self.get_row_evaluators()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _extract_tools_from_state(state: Dict) -> List[str]:
    """Extract tool names from workflow state."""
    tools = []

    # Check api_data keys
    api_data = state.get("api_data", {})
    if api_data:
        for key in api_data.keys():
            if api_data.get(key):
                tools.append(f"fetch_{key}")

    # Check search_data
    if state.get("search_data"):
        tools.append("web_search")

    # Check task_plan
    task_plan = state.get("task_plan", [])
    for task in task_plan:
        action = task.get("action", "").lower()
        if "sec" in action:
            tools.append("fetch_sec_data")
        elif "finnhub" in action or "market" in action:
            tools.append("fetch_market_data")
        elif "court" in action or "legal" in action:
            tools.append("fetch_legal_data")
        elif "search" in action:
            tools.append("web_search")

    return list(set(tools))


def _extract_trajectory(state: Dict) -> List[str]:
    """Extract node trajectory from workflow state."""
    trajectory = []

    # Check for explicit trajectory
    if "trajectory" in state:
        return state["trajectory"]

    # Infer from status changes
    status = state.get("status", "")
    if status:
        # Common workflow stages
        stages = ["input_parsed", "validated", "plan_created", "api_data_fetched",
                  "search_complete", "synthesized", "complete"]
        for stage in stages:
            if stage in status or status == stage:
                trajectory.append(stage)
                break

    # Infer from present data
    if state.get("company_info"):
        trajectory.append("parse_input")
    if state.get("task_plan"):
        trajectory.append("create_plan")
    if state.get("api_data"):
        trajectory.append("fetch_api_data")
    if state.get("search_data"):
        trajectory.append("search_web")
    if state.get("assessment"):
        trajectory.append("synthesize")
    if state.get("evaluation"):
        trajectory.append("evaluate")

    return trajectory
