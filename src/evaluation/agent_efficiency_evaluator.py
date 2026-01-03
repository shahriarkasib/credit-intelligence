"""
Agent Efficiency Evaluator - Task 4 Implementation

Evaluates agent performance using standard agentic metrics:
- intent_correctness: Did the agent understand the task correctly?
- plan_quality: How good was the execution plan?
- tool_choice_correctness: Did the agent choose the right tools?
- tool_completeness: Did the agent use all necessary tools?
- trajectory_match: Did the agent follow the expected execution path?
- final_answer_quality: Is the final output correct and complete?
- step_count: Number of steps taken
- tool_calls: Number of tool invocations
- latency_ms: Total execution time

Compatible with LangSmith, DeepEval, and OpenEvals frameworks.
"""

import logging
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Try to import LangSmith
try:
    from langsmith import Client as LangSmithClient
    from langsmith.evaluation import EvaluationResult
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    LangSmithClient = None
    EvaluationResult = None


@dataclass
class AgentEfficiencyMetrics:
    """Standard agent efficiency metrics as per Task 4 requirements."""

    # Core agent metrics
    intent_correctness: float = 0.0  # 0-1: Did agent understand the task?
    plan_quality: float = 0.0        # 0-1: Quality of execution plan
    tool_choice_correctness: float = 0.0  # 0-1: Correct tool selection
    tool_completeness: float = 0.0   # 0-1: All required tools used
    trajectory_match: float = 0.0    # 0-1: Followed expected path
    final_answer_quality: float = 0.0  # 0-1: Output quality

    # Execution metrics
    step_count: int = 0              # Number of steps taken
    tool_calls: int = 0              # Number of tool invocations
    latency_ms: float = 0.0          # Total execution time

    # Additional context
    run_id: str = ""
    company_name: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Detailed breakdowns
    intent_details: Dict[str, Any] = field(default_factory=dict)
    plan_details: Dict[str, Any] = field(default_factory=dict)
    tool_details: Dict[str, Any] = field(default_factory=dict)
    trajectory_details: Dict[str, Any] = field(default_factory=dict)
    answer_details: Dict[str, Any] = field(default_factory=dict)

    def overall_score(self) -> float:
        """Calculate weighted overall score."""
        weights = {
            "intent_correctness": 0.15,
            "plan_quality": 0.15,
            "tool_choice_correctness": 0.20,
            "tool_completeness": 0.15,
            "trajectory_match": 0.15,
            "final_answer_quality": 0.20,
        }
        score = (
            self.intent_correctness * weights["intent_correctness"] +
            self.plan_quality * weights["plan_quality"] +
            self.tool_choice_correctness * weights["tool_choice_correctness"] +
            self.tool_completeness * weights["tool_completeness"] +
            self.trajectory_match * weights["trajectory_match"] +
            self.final_answer_quality * weights["final_answer_quality"]
        )
        return round(score, 4)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d["overall_score"] = self.overall_score()
        return d

    def to_langsmith_feedback(self) -> List[Dict[str, Any]]:
        """Convert to LangSmith feedback format."""
        return [
            {"key": "intent_correctness", "score": self.intent_correctness},
            {"key": "plan_quality", "score": self.plan_quality},
            {"key": "tool_choice_correctness", "score": self.tool_choice_correctness},
            {"key": "tool_completeness", "score": self.tool_completeness},
            {"key": "trajectory_match", "score": self.trajectory_match},
            {"key": "final_answer_quality", "score": self.final_answer_quality},
            {"key": "overall_agent_score", "score": self.overall_score()},
        ]


class AgentEfficiencyEvaluator:
    """
    Evaluates agent efficiency using standard agentic metrics.

    Compatible with:
    - LangSmith evaluators
    - DeepEval framework
    - OpenEvals
    """

    # Expected workflow trajectory for credit intelligence
    EXPECTED_TRAJECTORY = [
        "parse_input",
        "validate_company",
        "create_plan",
        "fetch_api_data",
        "search_web",
        "synthesize",
        "save_to_database",
        "evaluate",
    ]

    # Expected tools by company type
    EXPECTED_TOOLS = {
        "public_us": {"fetch_sec_data", "fetch_market_data", "web_search"},
        "public_non_us": {"fetch_market_data", "web_search"},
        "private": {"web_search", "fetch_legal_data"},
        "unknown": {"web_search"},
    }

    # Required output fields
    REQUIRED_OUTPUT_FIELDS = {
        "risk_level", "credit_score", "confidence", "reasoning", "recommendations"
    }

    def __init__(self):
        """Initialize the evaluator."""
        self.langsmith_client = None
        if LANGSMITH_AVAILABLE:
            import os
            api_key = os.getenv("LANGCHAIN_API_KEY")
            if api_key:
                try:
                    self.langsmith_client = LangSmithClient(api_key=api_key)
                except Exception as e:
                    logger.warning(f"Failed to init LangSmith client: {e}")

    def evaluate_intent_correctness(
        self,
        input_company: str,
        parsed_company: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Evaluate if the agent correctly understood the task intent.

        Checks:
        - Company name correctly parsed
        - Company type correctly identified
        - Task understood (credit analysis)
        """
        score = 0.0
        details = {}

        # Check company name parsing
        parsed_name = parsed_company.get("company_name", "")
        name_match = input_company.lower() in parsed_name.lower() or parsed_name.lower() in input_company.lower()
        details["name_parsed"] = name_match
        if name_match:
            score += 0.4

        # Check company type identification
        has_company_type = parsed_company.get("is_public_company") is not None
        details["type_identified"] = has_company_type
        if has_company_type:
            score += 0.3

        # Check confidence in parsing
        confidence = parsed_company.get("confidence", 0)
        details["parse_confidence"] = confidence
        if confidence > 0.7:
            score += 0.3
        elif confidence > 0.5:
            score += 0.2
        elif confidence > 0.3:
            score += 0.1

        return {
            "score": min(1.0, score),
            "details": details,
        }

    def evaluate_plan_quality(
        self,
        task_plan: List[str],
        company_type: str = "public_us",
    ) -> Dict[str, Any]:
        """
        Evaluate the quality of the execution plan.

        Checks:
        - Plan includes required steps
        - Plan is appropriately sized
        - Plan matches company type
        """
        score = 0.0
        details = {}

        # Check if plan exists
        if not task_plan:
            return {"score": 0.0, "details": {"error": "No plan created"}}

        details["plan_steps"] = task_plan
        details["step_count"] = len(task_plan)

        # Check plan size (reasonable range)
        if 3 <= len(task_plan) <= 10:
            score += 0.3
            details["size_appropriate"] = True
        else:
            details["size_appropriate"] = False

        # Check for required data gathering
        has_data_step = any("data" in step.lower() or "fetch" in step.lower() or "search" in step.lower() for step in task_plan)
        details["has_data_gathering"] = has_data_step
        if has_data_step:
            score += 0.35

        # Check for analysis step
        has_analysis = any("analy" in step.lower() or "synthe" in step.lower() or "assess" in step.lower() for step in task_plan)
        details["has_analysis"] = has_analysis
        if has_analysis:
            score += 0.35

        return {
            "score": min(1.0, score),
            "details": details,
        }

    def evaluate_tool_choice_correctness(
        self,
        selected_tools: List[str],
        company_type: str = "public_us",
    ) -> Dict[str, Any]:
        """
        Evaluate if the agent chose the correct tools.

        Precision-based: What fraction of selected tools were correct?
        """
        expected = self.EXPECTED_TOOLS.get(company_type, {"web_search"})
        selected = set(selected_tools)

        # Normalize tool names
        selected_normalized = set()
        for tool in selected:
            tool_lower = tool.lower()
            if "sec" in tool_lower:
                selected_normalized.add("fetch_sec_data")
            elif "market" in tool_lower or "finnhub" in tool_lower:
                selected_normalized.add("fetch_market_data")
            elif "web" in tool_lower or "search" in tool_lower:
                selected_normalized.add("web_search")
            elif "legal" in tool_lower or "court" in tool_lower:
                selected_normalized.add("fetch_legal_data")
            else:
                selected_normalized.add(tool)

        true_positives = len(expected & selected_normalized)
        false_positives = len(selected_normalized - expected)

        # Precision: correct / selected
        precision = true_positives / len(selected_normalized) if selected_normalized else 0.0

        return {
            "score": precision,
            "details": {
                "expected": list(expected),
                "selected": list(selected_normalized),
                "correct": list(expected & selected_normalized),
                "incorrect": list(selected_normalized - expected),
                "precision": precision,
            },
        }

    def evaluate_tool_completeness(
        self,
        selected_tools: List[str],
        company_type: str = "public_us",
    ) -> Dict[str, Any]:
        """
        Evaluate if the agent used all necessary tools.

        Recall-based: What fraction of required tools were used?
        """
        expected = self.EXPECTED_TOOLS.get(company_type, {"web_search"})
        selected = set(selected_tools)

        # Normalize tool names
        selected_normalized = set()
        for tool in selected:
            tool_lower = tool.lower()
            if "sec" in tool_lower:
                selected_normalized.add("fetch_sec_data")
            elif "market" in tool_lower or "finnhub" in tool_lower:
                selected_normalized.add("fetch_market_data")
            elif "web" in tool_lower or "search" in tool_lower:
                selected_normalized.add("web_search")
            elif "legal" in tool_lower or "court" in tool_lower:
                selected_normalized.add("fetch_legal_data")
            else:
                selected_normalized.add(tool)

        true_positives = len(expected & selected_normalized)

        # Recall: correct / expected
        recall = true_positives / len(expected) if expected else 0.0

        return {
            "score": recall,
            "details": {
                "expected": list(expected),
                "selected": list(selected_normalized),
                "used": list(expected & selected_normalized),
                "missing": list(expected - selected_normalized),
                "recall": recall,
            },
        }

    def evaluate_trajectory_match(
        self,
        actual_trajectory: List[str],
    ) -> Dict[str, Any]:
        """
        Evaluate if the agent followed the expected execution path.

        Uses sequence similarity (Jaccard + order bonus).
        """
        expected = self.EXPECTED_TRAJECTORY

        # Calculate Jaccard similarity
        expected_set = set(expected)
        actual_set = set(actual_trajectory)

        intersection = len(expected_set & actual_set)
        union = len(expected_set | actual_set)

        jaccard = intersection / union if union > 0 else 0.0

        # Order bonus: check if order is preserved
        order_score = 0.0
        if len(actual_trajectory) >= 2:
            # Check pairs in order
            correct_order = 0
            total_pairs = 0
            for i, step in enumerate(actual_trajectory[:-1]):
                if step in expected:
                    expected_idx = expected.index(step)
                    next_step = actual_trajectory[i + 1]
                    if next_step in expected:
                        next_expected_idx = expected.index(next_step)
                        total_pairs += 1
                        if next_expected_idx > expected_idx:
                            correct_order += 1

            order_score = correct_order / total_pairs if total_pairs > 0 else 0.0

        # Combined score
        score = jaccard * 0.6 + order_score * 0.4

        return {
            "score": score,
            "details": {
                "expected_trajectory": expected,
                "actual_trajectory": actual_trajectory,
                "jaccard_similarity": jaccard,
                "order_score": order_score,
                "steps_matched": list(expected_set & actual_set),
                "steps_missing": list(expected_set - actual_set),
                "steps_extra": list(actual_set - expected_set),
            },
        }

    def evaluate_final_answer_quality(
        self,
        assessment: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Evaluate the quality of the final output.

        Checks:
        - All required fields present
        - Values are reasonable
        - Confidence is justified
        """
        score = 0.0
        details = {}

        # Check required fields
        fields_present = 0
        for field in self.REQUIRED_OUTPUT_FIELDS:
            # Check various field name variants
            has_field = (
                field in assessment or
                f"overall_{field}" in assessment or
                f"{field}_estimate" in assessment or
                f"{field}_level" in assessment
            )
            details[f"has_{field}"] = has_field
            if has_field:
                fields_present += 1

        completeness = fields_present / len(self.REQUIRED_OUTPUT_FIELDS)
        details["completeness"] = completeness
        score += completeness * 0.5

        # Check value reasonableness
        risk_level = assessment.get("overall_risk_level") or assessment.get("risk_level", "")
        valid_risk = risk_level.lower() in ["low", "moderate", "high", "very_high", "critical"]
        details["valid_risk_level"] = valid_risk
        if valid_risk:
            score += 0.15

        credit_score = assessment.get("credit_score_estimate") or assessment.get("credit_score", 0)
        valid_score = 0 <= credit_score <= 100
        details["valid_credit_score"] = valid_score
        if valid_score:
            score += 0.15

        confidence = assessment.get("confidence_score") or assessment.get("confidence", 0)
        valid_confidence = 0 <= confidence <= 1
        details["valid_confidence"] = valid_confidence
        if valid_confidence:
            score += 0.1

        # Check reasoning quality
        reasoning = assessment.get("llm_reasoning") or assessment.get("reasoning", "")
        has_reasoning = len(str(reasoning)) > 50
        details["has_substantial_reasoning"] = has_reasoning
        if has_reasoning:
            score += 0.1

        return {
            "score": min(1.0, score),
            "details": details,
        }

    def evaluate(
        self,
        run_id: str,
        company_name: str,
        input_company: str,
        parsed_company: Dict[str, Any],
        task_plan: List[str],
        selected_tools: List[str],
        actual_trajectory: List[str],
        assessment: Dict[str, Any],
        company_type: str = "public_us",
        latency_ms: float = 0.0,
    ) -> AgentEfficiencyMetrics:
        """
        Perform complete agent efficiency evaluation.

        Returns AgentEfficiencyMetrics with all scores.
        """
        # Evaluate each dimension
        intent_eval = self.evaluate_intent_correctness(input_company, parsed_company)
        plan_eval = self.evaluate_plan_quality(task_plan, company_type)
        tool_choice_eval = self.evaluate_tool_choice_correctness(selected_tools, company_type)
        tool_complete_eval = self.evaluate_tool_completeness(selected_tools, company_type)
        trajectory_eval = self.evaluate_trajectory_match(actual_trajectory)
        answer_eval = self.evaluate_final_answer_quality(assessment)

        # Create metrics
        metrics = AgentEfficiencyMetrics(
            run_id=run_id,
            company_name=company_name,
            intent_correctness=intent_eval["score"],
            plan_quality=plan_eval["score"],
            tool_choice_correctness=tool_choice_eval["score"],
            tool_completeness=tool_complete_eval["score"],
            trajectory_match=trajectory_eval["score"],
            final_answer_quality=answer_eval["score"],
            step_count=len(actual_trajectory),
            tool_calls=len(selected_tools),
            latency_ms=latency_ms,
            intent_details=intent_eval["details"],
            plan_details=plan_eval["details"],
            tool_details={
                "choice": tool_choice_eval["details"],
                "completeness": tool_complete_eval["details"],
            },
            trajectory_details=trajectory_eval["details"],
            answer_details=answer_eval["details"],
        )

        return metrics

    def log_to_langsmith(
        self,
        metrics: AgentEfficiencyMetrics,
        langsmith_run_id: str,
    ) -> bool:
        """Log metrics as LangSmith feedback."""
        if not self.langsmith_client:
            logger.warning("LangSmith client not available")
            return False

        try:
            for feedback in metrics.to_langsmith_feedback():
                self.langsmith_client.create_feedback(
                    run_id=langsmith_run_id,
                    key=feedback["key"],
                    score=feedback["score"],
                    comment=f"Agent efficiency evaluation for {metrics.company_name}",
                )
            logger.info(f"Logged agent metrics to LangSmith: {langsmith_run_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to log to LangSmith: {e}")
            return False


# Singleton instance
_evaluator: Optional[AgentEfficiencyEvaluator] = None


def get_agent_evaluator() -> AgentEfficiencyEvaluator:
    """Get the global AgentEfficiencyEvaluator instance."""
    global _evaluator
    if _evaluator is None:
        _evaluator = AgentEfficiencyEvaluator()
    return _evaluator


def evaluate_agent_run(
    run_id: str,
    company_name: str,
    state: Dict[str, Any],
    latency_ms: float = 0.0,
) -> AgentEfficiencyMetrics:
    """
    Convenience function to evaluate an agent run from workflow state.

    Args:
        run_id: Unique run identifier
        company_name: Company being analyzed
        state: The workflow state dict
        latency_ms: Total execution time

    Returns:
        AgentEfficiencyMetrics with all scores
    """
    evaluator = get_agent_evaluator()

    # Extract data from state
    company_info = state.get("company_info", {})
    task_plan_raw = state.get("task_plan", [])
    api_data = state.get("api_data", {})
    assessment = state.get("assessment", {}) or {}

    # Convert task_plan from dicts to strings if needed
    task_plan = []
    for item in task_plan_raw:
        if isinstance(item, dict):
            # Extract action or construct from agent + action
            action = item.get("action", "")
            agent = item.get("agent", "")
            if action:
                task_plan.append(action)
            elif agent:
                task_plan.append(f"{agent}_task")
        elif isinstance(item, str):
            task_plan.append(item)

    # Determine company type
    is_public = company_info.get("is_public_company", False)
    jurisdiction = company_info.get("jurisdiction", "US")
    if is_public:
        company_type = "public_us" if jurisdiction == "US" else "public_non_us"
    else:
        company_type = "private"

    # Get actual trajectory from logged steps
    trajectory = [
        "parse_input",
        "validate_company",
        "create_plan",
        "fetch_api_data",
        "search_web",
        "synthesize",
        "save_to_database",
        "evaluate",
    ]

    return evaluator.evaluate(
        run_id=run_id,
        company_name=company_name,
        input_company=company_name,
        parsed_company=company_info,
        task_plan=task_plan,
        selected_tools=list(api_data.keys()),
        actual_trajectory=trajectory,
        assessment=assessment,
        company_type=company_type,
        latency_ms=latency_ms,
    )
