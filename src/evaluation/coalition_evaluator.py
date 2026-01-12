"""
Coalition Evaluator - Uses multiple evaluators to estimate workflow correctness.

The coalition approach combines results from multiple evaluation methods:
- Agent Efficiency Evaluator (intent, plan, tools, trajectory, answer)
- LLM Judge Evaluator (accuracy, completeness, consistency)
- Tool Selection Evaluator (precision, recall, F1)
- Consistency Scorer (cross-run consistency)

The coalition uses weighted voting to produce a robust correctness estimate
with confidence bounds.
"""

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class EvaluatorVote:
    """A single evaluator's vote on correctness."""
    evaluator_name: str
    score: float  # 0-1
    confidence: float  # 0-1 (how confident this evaluator is)
    weight: float  # Weight in coalition voting
    details: Dict[str, Any] = field(default_factory=dict)

    def weighted_score(self) -> float:
        """Calculate weighted score considering confidence."""
        return self.score * self.confidence * self.weight


@dataclass
class CoalitionResult:
    """Result of coalition evaluation."""
    run_id: str
    company_name: str

    # Overall correctness assessment
    is_correct: bool  # True if score > threshold
    correctness_score: float  # 0-1 aggregated score
    confidence: float  # 0-1 confidence in the assessment
    correctness_category: str  # "high", "medium", "low"

    # Coalition details
    votes: List[Dict[str, Any]]  # Individual evaluator votes
    agreement_score: float  # How much evaluators agree (0-1)
    num_evaluators: int

    # Component scores
    efficiency_score: float  # From Agent Efficiency Evaluator
    quality_score: float  # From LLM Judge
    tool_score: float  # From Tool Selection Evaluator
    consistency_score: float  # From Consistency Scorer

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    evaluation_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class CoalitionEvaluator:
    """
    Combines multiple evaluators to estimate workflow correctness.

    The coalition approach provides:
    1. Robust correctness estimation (not dependent on single evaluator)
    2. Confidence bounds based on evaluator agreement
    3. Identification of which aspects need improvement
    """

    # Weights for different evaluators in the coalition
    EVALUATOR_WEIGHTS = {
        "agent_efficiency": 0.30,  # Agent workflow metrics
        "llm_judge": 0.25,        # LLM-based quality assessment
        "tool_selection": 0.25,    # Tool choice correctness
        "consistency": 0.20,       # Cross-run consistency
    }

    # Thresholds for correctness categories
    CORRECTNESS_THRESHOLDS = {
        "high": 0.80,
        "medium": 0.60,
        "low": 0.0,
    }

    def __init__(self):
        """Initialize the coalition evaluator."""
        self._agent_evaluator = None
        self._tool_evaluator = None

    def _get_agent_evaluator(self):
        """Lazy load agent efficiency evaluator."""
        if self._agent_evaluator is None:
            try:
                from evaluation.agent_efficiency_evaluator import get_agent_evaluator
                self._agent_evaluator = get_agent_evaluator()
            except ImportError:
                logger.warning("Agent efficiency evaluator not available")
        return self._agent_evaluator

    def _get_tool_evaluator(self):
        """Lazy load tool selection evaluator."""
        if self._tool_evaluator is None:
            try:
                from evaluation.tool_selection_evaluator import ToolSelectionEvaluator
                self._tool_evaluator = ToolSelectionEvaluator()
            except ImportError:
                logger.warning("Tool selection evaluator not available")
        return self._tool_evaluator

    def evaluate_agent_efficiency(
        self,
        state: Dict[str, Any],
        run_id: str,
        company_name: str,
    ) -> EvaluatorVote:
        """
        Get vote from Agent Efficiency Evaluator.

        Evaluates:
        - Intent correctness
        - Plan quality
        - Tool choice/completeness
        - Trajectory match
        - Final answer quality
        """
        try:
            evaluator = self._get_agent_evaluator()
            if evaluator is None:
                return EvaluatorVote(
                    evaluator_name="agent_efficiency",
                    score=0.5,
                    confidence=0.3,
                    weight=self.EVALUATOR_WEIGHTS["agent_efficiency"],
                    details={"error": "Evaluator not available"},
                )

            # Extract data from state
            company_info = state.get("company_info", {})
            task_plan = state.get("task_plan", [])
            api_data = state.get("api_data", {})
            assessment = state.get("assessment", {})

            # Determine company type
            is_public = company_info.get("is_public_company", False)
            jurisdiction = company_info.get("jurisdiction", "US")
            company_type = "public_us" if is_public and jurisdiction == "US" else (
                "public_non_us" if is_public else "private"
            )

            # Get trajectory from state
            trajectory = state.get("executed_nodes", [
                "parse_input", "validate_company", "create_plan",
                "fetch_api_data", "search_web", "synthesize",
                "save_to_database", "evaluate"
            ])

            # Convert task_plan to list of strings
            plan_strings = []
            for item in task_plan:
                if isinstance(item, dict):
                    plan_strings.append(item.get("action", item.get("agent", str(item))))
                else:
                    plan_strings.append(str(item))

            # Evaluate
            metrics = evaluator.evaluate(
                run_id=run_id,
                company_name=company_name,
                input_company=company_name,
                parsed_company=company_info,
                task_plan=plan_strings,
                selected_tools=list(api_data.keys()) if api_data else [],
                actual_trajectory=trajectory,
                assessment=assessment,
                company_type=company_type,
            )

            score = metrics.overall_score()

            # Confidence based on how many metrics were evaluated
            num_metrics = sum([
                metrics.intent_correctness > 0,
                metrics.plan_quality > 0,
                metrics.tool_choice_correctness > 0,
                metrics.tool_completeness > 0,
                metrics.trajectory_match > 0,
                metrics.final_answer_quality > 0,
            ])
            confidence = num_metrics / 6.0

            return EvaluatorVote(
                evaluator_name="agent_efficiency",
                score=score,
                confidence=confidence,
                weight=self.EVALUATOR_WEIGHTS["agent_efficiency"],
                details={
                    "intent_correctness": metrics.intent_correctness,
                    "plan_quality": metrics.plan_quality,
                    "tool_choice_correctness": metrics.tool_choice_correctness,
                    "tool_completeness": metrics.tool_completeness,
                    "trajectory_match": metrics.trajectory_match,
                    "final_answer_quality": metrics.final_answer_quality,
                },
            )

        except Exception as e:
            logger.error(f"Agent efficiency evaluation failed: {e}")
            return EvaluatorVote(
                evaluator_name="agent_efficiency",
                score=0.5,
                confidence=0.2,
                weight=self.EVALUATOR_WEIGHTS["agent_efficiency"],
                details={"error": str(e)},
            )

    def evaluate_tool_selection(
        self,
        state: Dict[str, Any],
        run_id: str,
        company_name: str,
    ) -> EvaluatorVote:
        """
        Get vote from Tool Selection Evaluator.

        Evaluates precision, recall, and F1 of tool selection.
        """
        try:
            # Get tools used from state
            api_data = state.get("api_data", {})
            selected_tools = list(api_data.keys()) if api_data else []

            # Get company type to determine expected tools
            company_info = state.get("company_info", {})
            is_public = company_info.get("is_public_company", False)
            jurisdiction = company_info.get("jurisdiction", "US")

            # Define expected tools based on company type
            if is_public and jurisdiction == "US":
                expected_tools = {"sec_edgar", "finnhub", "web_search"}
            elif is_public:
                expected_tools = {"finnhub", "web_search"}
            else:
                expected_tools = {"web_search", "court_listener"}

            # Normalize selected tools
            normalized_selected = set()
            for tool in selected_tools:
                tool_lower = tool.lower()
                if "sec" in tool_lower:
                    normalized_selected.add("sec_edgar")
                elif "finnhub" in tool_lower or "market" in tool_lower:
                    normalized_selected.add("finnhub")
                elif "court" in tool_lower or "legal" in tool_lower:
                    normalized_selected.add("court_listener")
                elif "search" in tool_lower or "web" in tool_lower:
                    normalized_selected.add("web_search")
                else:
                    normalized_selected.add(tool)

            # Calculate precision, recall, F1
            true_positives = len(expected_tools & normalized_selected)
            precision = true_positives / len(normalized_selected) if normalized_selected else 0.0
            recall = true_positives / len(expected_tools) if expected_tools else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            # Confidence based on whether we have data
            confidence = 0.8 if normalized_selected else 0.3

            return EvaluatorVote(
                evaluator_name="tool_selection",
                score=f1,
                confidence=confidence,
                weight=self.EVALUATOR_WEIGHTS["tool_selection"],
                details={
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "expected_tools": list(expected_tools),
                    "selected_tools": list(normalized_selected),
                    "correct_tools": list(expected_tools & normalized_selected),
                    "missing_tools": list(expected_tools - normalized_selected),
                    "extra_tools": list(normalized_selected - expected_tools),
                },
            )

        except Exception as e:
            logger.error(f"Tool selection evaluation failed: {e}")
            return EvaluatorVote(
                evaluator_name="tool_selection",
                score=0.5,
                confidence=0.2,
                weight=self.EVALUATOR_WEIGHTS["tool_selection"],
                details={"error": str(e)},
            )

    def evaluate_llm_quality(
        self,
        state: Dict[str, Any],
        run_id: str,
        company_name: str,
    ) -> EvaluatorVote:
        """
        Get vote from LLM quality assessment.

        Evaluates the quality of the final assessment output.
        """
        try:
            assessment = state.get("assessment", {})

            if not assessment:
                return EvaluatorVote(
                    evaluator_name="llm_judge",
                    score=0.0,
                    confidence=0.3,
                    weight=self.EVALUATOR_WEIGHTS["llm_judge"],
                    details={"error": "No assessment available"},
                )

            # Check completeness
            required_fields = [
                "overall_risk_level", "risk_level",
                "credit_score_estimate", "credit_score",
                "confidence_score", "confidence",
                "reasoning", "llm_reasoning",
            ]

            fields_present = 0
            for field in required_fields:
                if assessment.get(field):
                    fields_present += 1
            completeness = fields_present / 4  # 4 main categories

            # Check value validity
            risk_level = assessment.get("overall_risk_level") or assessment.get("risk_level", "")
            valid_risk = risk_level.lower() in ["low", "medium", "high", "critical", "moderate", "very_high"]

            credit_score = assessment.get("credit_score_estimate") or assessment.get("credit_score", 0)
            valid_score = 0 <= credit_score <= 100

            confidence = assessment.get("confidence_score") or assessment.get("confidence", 0)
            valid_confidence = 0 <= confidence <= 1

            reasoning = assessment.get("llm_reasoning") or assessment.get("reasoning", "")
            has_reasoning = len(str(reasoning)) > 50

            # Calculate quality score
            validity_score = (
                (0.25 if valid_risk else 0) +
                (0.25 if valid_score else 0) +
                (0.25 if valid_confidence else 0) +
                (0.25 if has_reasoning else 0)
            )

            score = (completeness * 0.5) + (validity_score * 0.5)

            return EvaluatorVote(
                evaluator_name="llm_judge",
                score=score,
                confidence=0.7 if assessment else 0.3,
                weight=self.EVALUATOR_WEIGHTS["llm_judge"],
                details={
                    "completeness": completeness,
                    "validity_score": validity_score,
                    "valid_risk_level": valid_risk,
                    "valid_credit_score": valid_score,
                    "valid_confidence": valid_confidence,
                    "has_reasoning": has_reasoning,
                    "fields_present": fields_present,
                },
            )

        except Exception as e:
            logger.error(f"LLM quality evaluation failed: {e}")
            return EvaluatorVote(
                evaluator_name="llm_judge",
                score=0.5,
                confidence=0.2,
                weight=self.EVALUATOR_WEIGHTS["llm_judge"],
                details={"error": str(e)},
            )

    def evaluate_consistency(
        self,
        state: Dict[str, Any],
        run_id: str,
        company_name: str,
        historical_runs: Optional[List[Dict[str, Any]]] = None,
    ) -> EvaluatorVote:
        """
        Get vote from Consistency evaluation.

        Evaluates consistency with historical runs for the same company.
        """
        try:
            assessment = state.get("assessment", {})

            if not historical_runs:
                # No historical data - give neutral score with low confidence
                return EvaluatorVote(
                    evaluator_name="consistency",
                    score=0.7,  # Assume reasonable consistency
                    confidence=0.3,  # But low confidence without history
                    weight=self.EVALUATOR_WEIGHTS["consistency"],
                    details={"note": "No historical runs for comparison"},
                )

            # Compare with historical runs
            current_risk = assessment.get("overall_risk_level") or assessment.get("risk_level", "")
            current_score = assessment.get("credit_score_estimate") or assessment.get("credit_score", 0)

            historical_risks = []
            historical_scores = []

            for run in historical_runs:
                run_assessment = run.get("assessment", {})
                if run_assessment:
                    risk = run_assessment.get("overall_risk_level") or run_assessment.get("risk_level", "")
                    score = run_assessment.get("credit_score_estimate") or run_assessment.get("credit_score", 0)
                    if risk:
                        historical_risks.append(risk.lower())
                    if score:
                        historical_scores.append(score)

            # Calculate consistency scores
            risk_consistency = 0.0
            score_consistency = 0.0

            if historical_risks:
                risk_matches = sum(1 for r in historical_risks if r == current_risk.lower())
                risk_consistency = risk_matches / len(historical_risks)

            if historical_scores and current_score:
                avg_historical = sum(historical_scores) / len(historical_scores)
                score_diff = abs(current_score - avg_historical)
                score_consistency = max(0, 1 - score_diff / 50)  # 50-point tolerance

            overall = (risk_consistency * 0.5) + (score_consistency * 0.5)

            return EvaluatorVote(
                evaluator_name="consistency",
                score=overall,
                confidence=min(1.0, len(historical_runs) / 5),  # More history = more confidence
                weight=self.EVALUATOR_WEIGHTS["consistency"],
                details={
                    "risk_consistency": risk_consistency,
                    "score_consistency": score_consistency,
                    "current_risk": current_risk,
                    "current_score": current_score,
                    "historical_count": len(historical_runs),
                },
            )

        except Exception as e:
            logger.error(f"Consistency evaluation failed: {e}")
            return EvaluatorVote(
                evaluator_name="consistency",
                score=0.5,
                confidence=0.2,
                weight=self.EVALUATOR_WEIGHTS["consistency"],
                details={"error": str(e)},
            )

    def evaluate(
        self,
        run_id: str,
        company_name: str,
        state: Dict[str, Any],
        historical_runs: Optional[List[Dict[str, Any]]] = None,
    ) -> CoalitionResult:
        """
        Perform coalition evaluation using all evaluators.

        Args:
            run_id: Unique run identifier
            company_name: Company being analyzed
            state: The workflow state dictionary
            historical_runs: Optional list of historical runs for consistency check

        Returns:
            CoalitionResult with aggregated correctness assessment
        """
        import time
        start_time = time.time()

        # Collect votes from all evaluators
        votes = [
            self.evaluate_agent_efficiency(state, run_id, company_name),
            self.evaluate_tool_selection(state, run_id, company_name),
            self.evaluate_llm_quality(state, run_id, company_name),
            self.evaluate_consistency(state, run_id, company_name, historical_runs),
        ]

        # Calculate weighted average score
        total_weighted = sum(v.weighted_score() for v in votes)
        total_weight = sum(v.confidence * v.weight for v in votes)

        correctness_score = total_weighted / total_weight if total_weight > 0 else 0.0

        # Calculate agreement score (how much evaluators agree)
        scores = [v.score for v in votes]
        if scores:
            avg_score = sum(scores) / len(scores)
            variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
            agreement_score = 1 - min(1, variance * 4)  # Scale variance to 0-1 agreement
        else:
            agreement_score = 0.0

        # Calculate overall confidence
        confidence = (sum(v.confidence for v in votes) / len(votes)) * agreement_score

        # Determine correctness category
        if correctness_score >= self.CORRECTNESS_THRESHOLDS["high"]:
            category = "high"
        elif correctness_score >= self.CORRECTNESS_THRESHOLDS["medium"]:
            category = "medium"
        else:
            category = "low"

        # Extract component scores for easy access
        efficiency_score = next((v.score for v in votes if v.evaluator_name == "agent_efficiency"), 0.0)
        quality_score = next((v.score for v in votes if v.evaluator_name == "llm_judge"), 0.0)
        tool_score = next((v.score for v in votes if v.evaluator_name == "tool_selection"), 0.0)
        consistency_score = next((v.score for v in votes if v.evaluator_name == "consistency"), 0.0)

        evaluation_time_ms = (time.time() - start_time) * 1000

        return CoalitionResult(
            run_id=run_id,
            company_name=company_name,
            is_correct=correctness_score >= self.CORRECTNESS_THRESHOLDS["medium"],
            correctness_score=round(correctness_score, 4),
            confidence=round(confidence, 4),
            correctness_category=category,
            votes=[{
                "evaluator": v.evaluator_name,
                "score": round(v.score, 4),
                "confidence": round(v.confidence, 4),
                "weight": v.weight,
                "weighted_score": round(v.weighted_score(), 4),
                "details": v.details,
            } for v in votes],
            agreement_score=round(agreement_score, 4),
            num_evaluators=len(votes),
            efficiency_score=round(efficiency_score, 4),
            quality_score=round(quality_score, 4),
            tool_score=round(tool_score, 4),
            consistency_score=round(consistency_score, 4),
            evaluation_time_ms=round(evaluation_time_ms, 2),
        )


# Singleton instance
_coalition_evaluator: Optional[CoalitionEvaluator] = None


def get_coalition_evaluator() -> CoalitionEvaluator:
    """Get the global CoalitionEvaluator instance."""
    global _coalition_evaluator
    if _coalition_evaluator is None:
        _coalition_evaluator = CoalitionEvaluator()
    return _coalition_evaluator


def evaluate_workflow_correctness(
    run_id: str,
    company_name: str,
    state: Dict[str, Any],
    historical_runs: Optional[List[Dict[str, Any]]] = None,
) -> CoalitionResult:
    """
    Convenience function to evaluate workflow correctness using coalition.

    Args:
        run_id: Unique run identifier
        company_name: Company being analyzed
        state: The workflow state dictionary
        historical_runs: Optional historical runs for consistency check

    Returns:
        CoalitionResult with correctness assessment
    """
    evaluator = get_coalition_evaluator()
    return evaluator.evaluate(run_id, company_name, state, historical_runs)
