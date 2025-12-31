"""LangSmith Evaluation Integration - Custom evaluators for credit intelligence."""

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import LangSmith evaluation components
try:
    from langsmith.evaluation import EvaluationResult
    from langsmith import Client as LangSmithClient
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    EvaluationResult = None
    LangSmithClient = None
    logger.warning("LangSmith SDK not installed. Evaluation features limited.")


@dataclass
class CreditEvaluationResult:
    """Result from credit intelligence evaluation."""
    run_id: str
    company_name: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # Tool Selection Metrics
    tool_selection_precision: float = 0.0
    tool_selection_recall: float = 0.0
    tool_selection_f1: float = 0.0

    # Data Quality Metrics
    data_completeness: float = 0.0
    sources_used: List[str] = field(default_factory=list)

    # Synthesis Quality Metrics
    synthesis_score: float = 0.0
    has_risk_level: bool = False
    has_credit_score: bool = False
    has_reasoning: bool = False
    has_recommendations: bool = False

    # LLM Consistency Metrics
    same_model_consistency: float = 0.0
    cross_model_consistency: float = 0.0
    num_llm_calls: int = 0

    # Cost Metrics
    total_tokens: int = 0
    total_cost: float = 0.0

    # Overall Score
    overall_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "company_name": self.company_name,
            "timestamp": self.timestamp,
            "tool_selection": {
                "precision": self.tool_selection_precision,
                "recall": self.tool_selection_recall,
                "f1": self.tool_selection_f1,
            },
            "data_quality": {
                "completeness": self.data_completeness,
                "sources_used": self.sources_used,
            },
            "synthesis": {
                "score": self.synthesis_score,
                "has_risk_level": self.has_risk_level,
                "has_credit_score": self.has_credit_score,
                "has_reasoning": self.has_reasoning,
                "has_recommendations": self.has_recommendations,
            },
            "consistency": {
                "same_model": self.same_model_consistency,
                "cross_model": self.cross_model_consistency,
                "num_calls": self.num_llm_calls,
            },
            "cost": {
                "total_tokens": self.total_tokens,
                "total_cost": self.total_cost,
            },
            "overall_score": self.overall_score,
        }

    def to_langsmith_results(self) -> List[Dict[str, Any]]:
        """Convert to LangSmith EvaluationResult format."""
        results = [
            {"key": "tool_selection_f1", "score": self.tool_selection_f1},
            {"key": "tool_selection_precision", "score": self.tool_selection_precision},
            {"key": "tool_selection_recall", "score": self.tool_selection_recall},
            {"key": "data_completeness", "score": self.data_completeness},
            {"key": "synthesis_score", "score": self.synthesis_score},
            {"key": "same_model_consistency", "score": self.same_model_consistency},
            {"key": "cross_model_consistency", "score": self.cross_model_consistency},
            {"key": "overall_score", "score": self.overall_score},
        ]
        return results


class CreditIntelligenceEvaluator:
    """
    Custom evaluator for Credit Intelligence workflow.

    Evaluates:
    1. Tool Selection Accuracy - Did the LLM choose the right data sources?
    2. Data Quality - How complete is the collected data?
    3. Synthesis Quality - Is the assessment well-formed?
    4. LLM Consistency - Do multiple LLM calls agree?
    """

    # Expected tools by company type
    EXPECTED_TOOLS = {
        "public_us": ["fetch_sec_data", "fetch_market_data"],
        "public_non_us": ["fetch_market_data", "web_search"],
        "private": ["web_search", "fetch_legal_data"],
        "unknown": ["web_search"],
    }

    def __init__(self, client: Optional[Any] = None):
        """Initialize evaluator with optional LangSmith client."""
        self.client = client
        if LANGSMITH_AVAILABLE and client is None:
            import os
            api_key = os.getenv("LANGCHAIN_API_KEY")
            if api_key:
                try:
                    self.client = LangSmithClient(api_key=api_key)
                except Exception as e:
                    logger.warning(f"Failed to initialize LangSmith client: {e}")

    def evaluate_tool_selection(
        self,
        selected_tools: List[str],
        company_type: str = "public_us",
    ) -> Dict[str, float]:
        """
        Evaluate tool selection accuracy.

        Args:
            selected_tools: Tools that were selected
            company_type: Type of company (public_us, public_non_us, private, unknown)

        Returns:
            Dict with precision, recall, f1
        """
        expected = set(self.EXPECTED_TOOLS.get(company_type, ["web_search"]))
        selected = set(selected_tools)

        true_positives = len(expected & selected)
        precision = true_positives / len(selected) if selected else 0.0
        recall = true_positives / len(expected) if expected else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "expected": list(expected),
            "selected": list(selected),
            "true_positives": list(expected & selected),
            "false_positives": list(selected - expected),
            "false_negatives": list(expected - selected),
        }

    def evaluate_data_quality(
        self,
        sources_used: List[str],
        max_sources: int = 4,
    ) -> Dict[str, Any]:
        """
        Evaluate data collection quality.

        Args:
            sources_used: List of data sources that returned data
            max_sources: Maximum expected sources

        Returns:
            Dict with completeness score and details
        """
        completeness = len(sources_used) / max_sources if max_sources > 0 else 0.0

        return {
            "completeness": min(1.0, completeness),
            "sources_used": sources_used,
            "source_count": len(sources_used),
            "max_sources": max_sources,
        }

    def evaluate_synthesis(
        self,
        assessment: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Evaluate synthesis quality.

        Args:
            assessment: The synthesized assessment

        Returns:
            Dict with synthesis quality metrics
        """
        has_risk_level = bool(assessment.get("overall_risk_level") or assessment.get("risk_level"))
        has_credit_score = bool(assessment.get("credit_score_estimate") or assessment.get("credit_score"))
        has_reasoning = bool(assessment.get("llm_reasoning") or assessment.get("reasoning"))
        has_recommendations = bool(assessment.get("recommendations"))

        # Calculate score based on required fields
        fields_present = sum([has_risk_level, has_credit_score, has_reasoning, has_recommendations])
        score = fields_present / 4.0

        return {
            "score": score,
            "has_risk_level": has_risk_level,
            "has_credit_score": has_credit_score,
            "has_reasoning": has_reasoning,
            "has_recommendations": has_recommendations,
            "fields_present": fields_present,
        }

    def evaluate_consistency(
        self,
        llm_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Evaluate LLM consistency across multiple calls.

        Args:
            llm_results: List of LLM result dicts

        Returns:
            Dict with consistency metrics
        """
        if len(llm_results) < 2:
            return {
                "same_model_consistency": 1.0,
                "cross_model_consistency": 1.0,
                "num_calls": len(llm_results),
                "risk_levels": [],
                "credit_scores": [],
            }

        # Extract risk levels and scores
        risk_levels = [r.get("risk_level", "unknown") for r in llm_results if r.get("success", True)]
        credit_scores = [r.get("credit_score", 0) for r in llm_results if r.get("success", True)]

        # Calculate risk level agreement
        unique_risks = set(risk_levels)
        risk_agreement = 1.0 if len(unique_risks) == 1 else 1.0 / len(unique_risks)

        # Calculate score consistency (standard deviation based)
        if len(credit_scores) >= 2:
            mean_score = sum(credit_scores) / len(credit_scores)
            variance = sum((s - mean_score) ** 2 for s in credit_scores) / len(credit_scores)
            std_dev = variance ** 0.5
            # Lower std dev = higher consistency (normalize to 0-1)
            score_consistency = max(0, 1 - (std_dev / 50))  # 50 is max expected std dev
        else:
            score_consistency = 1.0

        overall_consistency = (risk_agreement + score_consistency) / 2

        return {
            "same_model_consistency": overall_consistency,
            "cross_model_consistency": overall_consistency,
            "num_calls": len(llm_results),
            "risk_levels": risk_levels,
            "credit_scores": credit_scores,
            "risk_agreement": risk_agreement,
            "score_consistency": score_consistency,
        }

    def evaluate_run(
        self,
        run_id: str,
        company_name: str,
        selected_tools: List[str],
        sources_used: List[str],
        assessment: Dict[str, Any],
        llm_results: List[Dict[str, Any]] = None,
        company_type: str = "public_us",
        total_tokens: int = 0,
        total_cost: float = 0.0,
    ) -> CreditEvaluationResult:
        """
        Perform complete evaluation of a credit intelligence run.

        Args:
            run_id: Unique run identifier
            company_name: Company being analyzed
            selected_tools: Tools that were selected
            sources_used: Data sources that returned data
            assessment: The synthesized assessment
            llm_results: Results from LLM consistency checks
            company_type: Type of company
            total_tokens: Total tokens used
            total_cost: Total cost

        Returns:
            CreditEvaluationResult with all metrics
        """
        llm_results = llm_results or []

        # Evaluate each dimension
        tool_eval = self.evaluate_tool_selection(selected_tools, company_type)
        data_eval = self.evaluate_data_quality(sources_used)
        synthesis_eval = self.evaluate_synthesis(assessment)
        consistency_eval = self.evaluate_consistency(llm_results)

        # Calculate overall score (weighted average)
        overall_score = (
            tool_eval["f1"] * 0.25 +
            data_eval["completeness"] * 0.25 +
            synthesis_eval["score"] * 0.25 +
            consistency_eval["same_model_consistency"] * 0.25
        )

        result = CreditEvaluationResult(
            run_id=run_id,
            company_name=company_name,
            tool_selection_precision=tool_eval["precision"],
            tool_selection_recall=tool_eval["recall"],
            tool_selection_f1=tool_eval["f1"],
            data_completeness=data_eval["completeness"],
            sources_used=sources_used,
            synthesis_score=synthesis_eval["score"],
            has_risk_level=synthesis_eval["has_risk_level"],
            has_credit_score=synthesis_eval["has_credit_score"],
            has_reasoning=synthesis_eval["has_reasoning"],
            has_recommendations=synthesis_eval["has_recommendations"],
            same_model_consistency=consistency_eval["same_model_consistency"],
            cross_model_consistency=consistency_eval["cross_model_consistency"],
            num_llm_calls=consistency_eval["num_calls"],
            total_tokens=total_tokens,
            total_cost=total_cost,
            overall_score=overall_score,
        )

        return result

    def log_to_langsmith(
        self,
        result: CreditEvaluationResult,
        run_id: str,
    ) -> bool:
        """
        Log evaluation results to LangSmith as feedback.

        Args:
            result: The evaluation result
            run_id: LangSmith run ID to attach feedback to

        Returns:
            True if successful
        """
        if not LANGSMITH_AVAILABLE or not self.client:
            logger.warning("LangSmith not available for feedback logging")
            return False

        try:
            # Log each metric as feedback
            for metric in result.to_langsmith_results():
                self.client.create_feedback(
                    run_id=run_id,
                    key=metric["key"],
                    score=metric["score"],
                    comment=f"Automated evaluation for {result.company_name}",
                )

            logger.info(f"Logged evaluation feedback to LangSmith for run {run_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to log to LangSmith: {e}")
            return False


# Singleton instance
_evaluator: Optional[CreditIntelligenceEvaluator] = None


def get_credit_evaluator() -> CreditIntelligenceEvaluator:
    """Get the global CreditIntelligenceEvaluator instance."""
    global _evaluator
    if _evaluator is None:
        _evaluator = CreditIntelligenceEvaluator()
    return _evaluator
