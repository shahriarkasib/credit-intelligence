"""Workflow Evaluator - Evaluates complete credit assessment workflow."""

import logging
import json
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from .tool_selection_evaluator import ToolSelectionEvaluator, ToolSelectionResult
from .consistency_scorer import ConsistencyScorer, ConsistencyResult
from .execution_wrapper import ExecutionWrapper
from .analyzer import CorrelationAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class WorkflowEvaluationResult:
    """Complete workflow evaluation result."""
    run_id: str
    company_name: str

    # Tool Selection Evaluation
    tool_selection_score: float
    tool_selection_details: Dict[str, Any]

    # Data Quality Evaluation
    data_completeness: float
    data_sources_used: List[str]

    # Synthesis Evaluation
    synthesis_consistency: float
    synthesis_details: Dict[str, Any]

    # Overall Metrics
    total_execution_time_ms: float
    total_tokens_used: int
    llm_calls: int

    # Consistency Evaluation (multi-run)
    multi_run_consistency: Optional[float] = None
    consistency_details: Optional[Dict[str, Any]] = None

    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "company_name": self.company_name,
            "tool_selection_score": self.tool_selection_score,
            "tool_selection_details": self.tool_selection_details,
            "data_completeness": self.data_completeness,
            "data_sources_used": self.data_sources_used,
            "synthesis_consistency": self.synthesis_consistency,
            "synthesis_details": self.synthesis_details,
            "total_execution_time_ms": self.total_execution_time_ms,
            "total_tokens_used": self.total_tokens_used,
            "llm_calls": self.llm_calls,
            "multi_run_consistency": self.multi_run_consistency,
            "consistency_details": self.consistency_details,
            "timestamp": self.timestamp,
        }


class WorkflowEvaluator:
    """
    Comprehensive evaluator for the credit intelligence workflow.

    Evaluates:
    1. Tool Selection - Did the LLM choose appropriate tools?
    2. Data Quality - How complete is the collected data?
    3. Synthesis Quality - Is the assessment accurate and well-reasoned?
    4. Consistency - Are results consistent across multiple runs?
    """

    def __init__(self):
        self.tool_evaluator = ToolSelectionEvaluator()
        self.consistency_scorer = ConsistencyScorer(method="semantic_similarity")
        self.results: List[WorkflowEvaluationResult] = []

    def evaluate_single_run(
        self,
        run_id: str,
        company_name: str,
        tool_selection: Dict[str, Any],
        tool_results: Dict[str, Any],
        assessment: Dict[str, Any],
        execution_metrics: Dict[str, Any],
    ) -> WorkflowEvaluationResult:
        """
        Evaluate a single workflow run.

        Args:
            run_id: Unique run identifier
            company_name: Company name
            tool_selection: Tool selection result from ToolSupervisor
            tool_results: Results from tool execution
            assessment: Final assessment from synthesis
            execution_metrics: Timing and token metrics

        Returns:
            WorkflowEvaluationResult
        """
        # 1. Evaluate Tool Selection
        selection = tool_selection.get("selection", {})
        selected_tools = [t.get("name") for t in selection.get("tools_to_use", [])]

        tool_eval = self.tool_evaluator.evaluate(
            company_name=company_name,
            selected_tools=selected_tools,
            selection_reasoning=selection,
        )

        # 2. Evaluate Data Quality
        data_completeness = self._evaluate_data_quality(tool_results)
        data_sources = list(tool_results.get("results", {}).keys())

        # 3. Evaluate Synthesis Quality
        synthesis_score, synthesis_details = self._evaluate_synthesis(assessment)

        # 4. Calculate metrics
        total_time = execution_metrics.get("total_execution_time_ms", 0)
        total_tokens = self._sum_tokens(tool_selection, assessment)
        llm_calls = 2  # tool_selection + synthesis

        result = WorkflowEvaluationResult(
            run_id=run_id,
            company_name=company_name,
            tool_selection_score=tool_eval.f1_score,
            tool_selection_details=tool_eval.to_dict(),
            data_completeness=data_completeness,
            data_sources_used=data_sources,
            synthesis_consistency=synthesis_score,
            synthesis_details=synthesis_details,
            total_execution_time_ms=total_time,
            total_tokens_used=total_tokens,
            llm_calls=llm_calls,
        )

        self.results.append(result)
        return result

    def evaluate_consistency(
        self,
        company_name: str,
        assessments: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Evaluate consistency across multiple runs for the same company.

        Args:
            company_name: Company name
            assessments: List of assessment results from multiple runs

        Returns:
            Consistency evaluation results
        """
        if len(assessments) < 2:
            return {"error": "Need at least 2 assessments for consistency evaluation"}

        # Extract key fields for comparison
        risk_levels = []
        credit_scores = []
        reasonings = []

        for a in assessments:
            assessment = a.get("assessment", {})
            risk_levels.append(assessment.get("risk_level", "unknown"))
            credit_scores.append(assessment.get("credit_score", 0))
            reasonings.append(assessment.get("reasoning", ""))

        # Evaluate risk level consistency
        unique_risk_levels = set(risk_levels)
        risk_consistency = 1.0 if len(unique_risk_levels) == 1 else 1.0 / len(unique_risk_levels)

        # Evaluate credit score consistency
        if credit_scores:
            score_range = max(credit_scores) - min(credit_scores)
            score_consistency = 1.0 - (score_range / 100.0)  # Normalized
        else:
            score_consistency = 0.0

        # Evaluate reasoning consistency
        if reasonings:
            reasoning_result = self.consistency_scorer.score(reasonings)
            reasoning_consistency = reasoning_result.consistency_score
        else:
            reasoning_consistency = 0.0

        # Overall consistency
        overall = (risk_consistency + score_consistency + reasoning_consistency) / 3

        return {
            "overall_consistency": overall,
            "risk_level_consistency": risk_consistency,
            "credit_score_consistency": score_consistency,
            "reasoning_consistency": reasoning_consistency,
            "risk_levels": risk_levels,
            "credit_scores": credit_scores,
            "score_range": max(credit_scores) - min(credit_scores) if credit_scores else 0,
            "unique_risk_levels": list(unique_risk_levels),
        }

    def evaluate_cross_model_consistency(
        self,
        company_name: str,
        model_assessments: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Evaluate consistency across different models.

        Args:
            company_name: Company name
            model_assessments: Dict mapping model name to assessment

        Returns:
            Cross-model consistency evaluation
        """
        if len(model_assessments) < 2:
            return {"error": "Need at least 2 models for cross-model evaluation"}

        # Extract assessments
        models = list(model_assessments.keys())
        assessments = list(model_assessments.values())

        # Use the same consistency evaluation
        result = self.evaluate_consistency(company_name, assessments)
        result["models_compared"] = models

        return result

    def _evaluate_data_quality(self, tool_results: Dict[str, Any]) -> float:
        """Evaluate data quality/completeness."""
        results = tool_results.get("results", {})
        execution_metrics = tool_results.get("execution_metrics", [])

        if not results:
            return 0.0

        # Count successful and failed tools
        successful = sum(1 for m in execution_metrics if m.get("success"))
        total = len(execution_metrics)

        if total == 0:
            return 0.0

        # Base score from success rate
        success_rate = successful / total

        # Bonus for data richness
        bonus = 0.0
        for tool_name, result in results.items():
            data = result.get("data", {})
            if isinstance(data, dict) and data.get("found"):
                bonus += 0.1

        return min(1.0, success_rate + bonus)

    def _evaluate_synthesis(self, assessment: Dict[str, Any]) -> tuple:
        """Evaluate synthesis quality."""
        details = {}
        scores = []

        assessment_data = assessment.get("assessment", {})

        # Check for required fields
        required_fields = ["risk_level", "credit_score", "reasoning", "risk_factors"]
        present_fields = sum(1 for f in required_fields if assessment_data.get(f))
        completeness = present_fields / len(required_fields)
        scores.append(completeness)
        details["completeness"] = completeness

        # Check credit score validity
        credit_score = assessment_data.get("credit_score", 0)
        if 0 <= credit_score <= 100:
            scores.append(1.0)
            details["valid_credit_score"] = True
        else:
            scores.append(0.0)
            details["valid_credit_score"] = False

        # Check confidence provided
        confidence = assessment_data.get("confidence", 0)
        if 0 <= confidence <= 1:
            scores.append(1.0)
            details["valid_confidence"] = True
        else:
            scores.append(0.5)  # Partial credit
            details["valid_confidence"] = False

        # Check reasoning quality
        reasoning = assessment_data.get("reasoning", "")
        if len(reasoning) > 100:
            scores.append(1.0)
            details["adequate_reasoning"] = True
        elif len(reasoning) > 50:
            scores.append(0.5)
            details["adequate_reasoning"] = False
        else:
            scores.append(0.0)
            details["adequate_reasoning"] = False

        # Check for data quality self-assessment
        dqa = assessment_data.get("data_quality_assessment", {})
        if dqa:
            scores.append(1.0)
            details["has_data_quality_assessment"] = True
        else:
            scores.append(0.0)
            details["has_data_quality_assessment"] = False

        overall = sum(scores) / len(scores) if scores else 0.0
        details["overall_score"] = overall

        return overall, details

    def _sum_tokens(self, tool_selection: Dict[str, Any], assessment: Dict[str, Any]) -> int:
        """Sum total tokens used."""
        total = 0

        # Tool selection tokens
        ts_metrics = tool_selection.get("llm_metrics", {})
        total += ts_metrics.get("total_tokens", 0)

        # Assessment tokens
        a_metrics = assessment.get("llm_metrics", {})
        total += a_metrics.get("total_tokens", 0)

        return total

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluations."""
        if not self.results:
            return {"total_runs": 0}

        total = len(self.results)

        return {
            "total_runs": total,
            "avg_tool_selection_score": sum(r.tool_selection_score for r in self.results) / total,
            "avg_data_completeness": sum(r.data_completeness for r in self.results) / total,
            "avg_synthesis_score": sum(r.synthesis_consistency for r in self.results) / total,
            "avg_execution_time_ms": sum(r.total_execution_time_ms for r in self.results) / total,
            "total_tokens_used": sum(r.total_tokens_used for r in self.results),
            "total_llm_calls": sum(r.llm_calls for r in self.results),
        }

    def export_results(self, filepath: str):
        """Export all results to JSON."""
        data = {
            "summary": self.get_summary(),
            "results": [r.to_dict() for r in self.results],
            "tool_selection_summary": self.tool_evaluator.get_summary(),
            "timestamp": datetime.utcnow().isoformat(),
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Results exported to: {filepath}")

    def clear(self):
        """Clear all results."""
        self.results = []
        self.tool_evaluator.clear()
