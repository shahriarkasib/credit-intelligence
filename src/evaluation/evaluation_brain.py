"""
Evaluation Brain - Task 3 Implementation

Central orchestration layer that coordinates all evaluation components:
- WorkflowEvaluator (tool selection, data quality, synthesis)
- AgentEfficiencyEvaluator (intent, plan, tool choice, trajectory, answer)
- LLMJudgeEvaluator (accuracy, completeness, consistency, actionability)
- ConsistencyScorer (multi-run and cross-model consistency)

Provides unified API for:
1. Single run evaluation
2. Multi-run consistency evaluation
3. Cross-model comparison
4. Benchmark comparison (Coalition)
5. Comprehensive evaluation reports
"""

import logging
import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# Import all evaluators
try:
    from .workflow_evaluator import WorkflowEvaluator, WorkflowEvaluationResult
    from .tool_selection_evaluator import ToolSelectionEvaluator, ToolSelectionResult
    from .consistency_scorer import ConsistencyScorer
    from .agent_efficiency_evaluator import (
        AgentEfficiencyEvaluator,
        AgentEfficiencyMetrics,
        evaluate_agent_run,
    )
    from .llm_judge_evaluator import (
        LLMJudgeEvaluator,
        LLMJudgeResult,
        evaluate_with_llm_judge,
    )
    EVALUATORS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Some evaluators not available: {e}")
    EVALUATORS_AVAILABLE = False


@dataclass
class ComprehensiveEvaluationResult:
    """
    Complete evaluation result from all evaluation components.

    This is the unified output from the Evaluation Brain.
    """

    # Identifiers
    run_id: str
    company_name: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Workflow Evaluation (tool selection, data quality, synthesis)
    tool_selection_score: float = 0.0
    data_quality_score: float = 0.0
    synthesis_score: float = 0.0
    workflow_overall: float = 0.0
    workflow_details: Dict[str, Any] = field(default_factory=dict)

    # Agent Efficiency (Task 4 metrics)
    intent_correctness: float = 0.0
    plan_quality: float = 0.0
    tool_choice_correctness: float = 0.0
    tool_completeness: float = 0.0
    trajectory_match: float = 0.0
    final_answer_quality: float = 0.0
    agent_overall: float = 0.0
    agent_details: Dict[str, Any] = field(default_factory=dict)

    # LLM Judge (Task 21 metrics)
    llm_accuracy: float = 0.0
    llm_completeness: float = 0.0
    llm_consistency: float = 0.0
    llm_actionability: float = 0.0
    llm_data_utilization: float = 0.0
    llm_judge_overall: float = 0.0
    llm_suggestions: List[str] = field(default_factory=list)
    llm_judge_details: Dict[str, Any] = field(default_factory=dict)

    # Consistency Metrics
    intra_model_consistency: float = 0.0
    cross_model_consistency: float = 0.0
    consistency_details: Dict[str, Any] = field(default_factory=dict)

    # Benchmark Comparison
    benchmark_alignment: float = 0.0
    benchmark_comparison: str = ""

    # Execution Metrics
    total_execution_time_ms: float = 0.0
    total_tokens: int = 0
    total_cost: float = 0.0
    llm_calls_count: int = 0

    # Overall Combined Score
    overall_score: float = 0.0
    overall_grade: str = ""  # A, B, C, D, F

    # Errors and Warnings
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def calculate_overall(self, weights: Dict[str, float] = None) -> float:
        """
        Calculate weighted overall score.

        Default weights:
        - Workflow: 30%
        - Agent Efficiency: 30%
        - LLM Judge: 30%
        - Consistency: 10%
        """
        if weights is None:
            weights = {
                "workflow": 0.30,
                "agent": 0.30,
                "llm_judge": 0.30,
                "consistency": 0.10,
            }

        self.overall_score = (
            self.workflow_overall * weights.get("workflow", 0.30) +
            self.agent_overall * weights.get("agent", 0.30) +
            self.llm_judge_overall * weights.get("llm_judge", 0.30) +
            self.intra_model_consistency * weights.get("consistency", 0.10)
        )

        # Assign grade
        if self.overall_score >= 0.9:
            self.overall_grade = "A"
        elif self.overall_score >= 0.8:
            self.overall_grade = "B"
        elif self.overall_score >= 0.7:
            self.overall_grade = "C"
        elif self.overall_score >= 0.6:
            self.overall_grade = "D"
        else:
            self.overall_grade = "F"

        return self.overall_score


class EvaluationBrain:
    """
    Central orchestration layer for all evaluation components.

    The "brain" that coordinates:
    - WorkflowEvaluator
    - AgentEfficiencyEvaluator
    - LLMJudgeEvaluator
    - ConsistencyScorer

    Provides unified evaluation API and comprehensive reporting.
    """

    def __init__(self):
        """Initialize all evaluation components."""
        self.workflow_evaluator = WorkflowEvaluator()
        self.agent_evaluator = AgentEfficiencyEvaluator()
        self.llm_judge = LLMJudgeEvaluator()
        self.consistency_scorer = ConsistencyScorer(method="semantic_similarity")
        self.tool_evaluator = ToolSelectionEvaluator()

        # Store evaluation history
        self.evaluation_history: List[ComprehensiveEvaluationResult] = []

        logger.info("EvaluationBrain initialized with all components")

    def evaluate_run(
        self,
        run_id: str,
        company_name: str,
        state: Dict[str, Any],
        benchmark: Dict[str, Any] = None,
        run_llm_judge: bool = True,
    ) -> ComprehensiveEvaluationResult:
        """
        Run comprehensive evaluation on a workflow execution.

        Args:
            run_id: Unique run identifier
            company_name: Company being evaluated
            state: The workflow state with all data
            benchmark: Optional benchmark (e.g., Coalition) to compare against
            run_llm_judge: Whether to run LLM judge (costs tokens)

        Returns:
            ComprehensiveEvaluationResult with all metrics
        """
        start_time = time.time()
        result = ComprehensiveEvaluationResult(
            run_id=run_id,
            company_name=company_name,
        )

        try:
            # 1. Workflow Evaluation (tool selection, data quality, synthesis)
            self._evaluate_workflow(result, state)

            # 2. Agent Efficiency Evaluation
            self._evaluate_agent_efficiency(result, state)

            # 3. LLM Judge Evaluation (optional - costs tokens)
            if run_llm_judge:
                self._evaluate_with_llm_judge(result, state, benchmark)

            # 4. Calculate overall score
            result.calculate_overall()

            # Store execution time
            result.total_execution_time_ms = (time.time() - start_time) * 1000

            # Add to history
            self.evaluation_history.append(result)

            logger.info(
                f"Comprehensive evaluation complete for {company_name}: "
                f"overall={result.overall_score:.4f} grade={result.overall_grade}"
            )

        except Exception as e:
            result.errors.append(f"Evaluation error: {str(e)}")
            logger.error(f"Evaluation failed: {e}")

        return result

    def _evaluate_workflow(
        self,
        result: ComprehensiveEvaluationResult,
        state: Dict[str, Any],
    ):
        """Evaluate workflow components."""
        try:
            api_data = state.get("api_data", {})
            assessment = state.get("assessment", {}) or {}
            company_info = state.get("company_info", {})

            # Tool Selection Score
            selected_tools = list(api_data.keys())
            is_public = company_info.get("is_public_company", False)

            # Expected tools based on company type
            if is_public:
                expected_tools = ["sec_edgar", "finnhub", "parallel_ai"]
            else:
                expected_tools = ["parallel_ai", "web_search"]

            # Calculate precision/recall
            correct = set(selected_tools) & set(expected_tools)
            precision = len(correct) / len(selected_tools) if selected_tools else 0
            recall = len(correct) / len(expected_tools) if expected_tools else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            result.tool_selection_score = f1
            result.workflow_details["tool_selection"] = {
                "selected": selected_tools,
                "expected": expected_tools,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

            # Data Quality Score
            total_sources = len(api_data)
            non_empty_sources = sum(1 for v in api_data.values() if v)
            result.data_quality_score = non_empty_sources / total_sources if total_sources > 0 else 0
            result.workflow_details["data_quality"] = {
                "total_sources": total_sources,
                "non_empty_sources": non_empty_sources,
            }

            # Synthesis Score
            has_risk_level = bool(assessment.get("overall_risk_level"))
            has_score = bool(assessment.get("credit_score_estimate"))
            has_reasoning = len(assessment.get("llm_reasoning", "")) > 50
            has_recommendations = bool(assessment.get("recommendations"))

            synthesis_components = [has_risk_level, has_score, has_reasoning, has_recommendations]
            result.synthesis_score = sum(synthesis_components) / len(synthesis_components)
            result.workflow_details["synthesis"] = {
                "has_risk_level": has_risk_level,
                "has_score": has_score,
                "has_reasoning": has_reasoning,
                "has_recommendations": has_recommendations,
            }

            # Overall workflow score
            result.workflow_overall = (
                result.tool_selection_score * 0.4 +
                result.data_quality_score * 0.3 +
                result.synthesis_score * 0.3
            )

        except Exception as e:
            result.warnings.append(f"Workflow evaluation warning: {str(e)}")

    def _evaluate_agent_efficiency(
        self,
        result: ComprehensiveEvaluationResult,
        state: Dict[str, Any],
    ):
        """Evaluate agent efficiency metrics."""
        try:
            metrics = evaluate_agent_run(
                run_id=result.run_id,
                company_name=result.company_name,
                state=state,
                latency_ms=result.total_execution_time_ms,
            )

            result.intent_correctness = metrics.intent_correctness
            result.plan_quality = metrics.plan_quality
            result.tool_choice_correctness = metrics.tool_choice_correctness
            result.tool_completeness = metrics.tool_completeness
            result.trajectory_match = metrics.trajectory_match
            result.final_answer_quality = metrics.final_answer_quality
            result.agent_overall = metrics.overall_score()

            result.agent_details = {
                "intent": metrics.intent_details,
                "plan": metrics.plan_details,
                "tool": metrics.tool_details,
                "trajectory": metrics.trajectory_details,
                "answer": metrics.answer_details,
                "step_count": metrics.step_count,
                "tool_calls": metrics.tool_calls,
            }

        except Exception as e:
            result.warnings.append(f"Agent efficiency evaluation warning: {str(e)}")

    def _evaluate_with_llm_judge(
        self,
        result: ComprehensiveEvaluationResult,
        state: Dict[str, Any],
        benchmark: Dict[str, Any] = None,
    ):
        """Evaluate using LLM-as-a-judge."""
        try:
            assessment = state.get("assessment", {}) or {}
            api_data = state.get("api_data", {})

            judge_result = evaluate_with_llm_judge(
                run_id=result.run_id,
                company_name=result.company_name,
                assessment=assessment,
                api_data=api_data,
                benchmark=benchmark,
            )

            result.llm_accuracy = judge_result.accuracy_score
            result.llm_completeness = judge_result.completeness_score
            result.llm_consistency = judge_result.consistency_score
            result.llm_actionability = judge_result.actionability_score
            result.llm_data_utilization = judge_result.data_utilization_score
            result.llm_judge_overall = judge_result.overall_score
            result.llm_suggestions = judge_result.suggestions

            result.llm_judge_details = {
                "accuracy_reasoning": judge_result.accuracy_reasoning,
                "completeness_reasoning": judge_result.completeness_reasoning,
                "consistency_reasoning": judge_result.consistency_reasoning,
                "actionability_reasoning": judge_result.actionability_reasoning,
                "data_utilization_reasoning": judge_result.data_utilization_reasoning,
                "overall_reasoning": judge_result.overall_reasoning,
                "tokens_used": judge_result.tokens_used,
                "cost": judge_result.evaluation_cost,
            }

            # Benchmark comparison
            if benchmark:
                result.benchmark_alignment = judge_result.benchmark_alignment
                result.benchmark_comparison = judge_result.benchmark_comparison

            # Add to total tokens/cost
            result.total_tokens += judge_result.tokens_used
            result.total_cost += judge_result.evaluation_cost

        except Exception as e:
            result.warnings.append(f"LLM Judge evaluation warning: {str(e)}")

    def evaluate_consistency(
        self,
        company_name: str,
        runs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Evaluate consistency across multiple runs for the same company.

        Args:
            company_name: Company name
            runs: List of workflow states from multiple runs

        Returns:
            Consistency evaluation results
        """
        if len(runs) < 2:
            return {"error": "Need at least 2 runs for consistency evaluation"}

        risk_levels = []
        credit_scores = []
        reasonings = []

        for run in runs:
            assessment = run.get("assessment", {}) or {}
            risk_levels.append(assessment.get("overall_risk_level", "unknown"))
            credit_scores.append(assessment.get("credit_score_estimate", 0))
            reasonings.append(assessment.get("llm_reasoning", ""))

        # Risk level consistency
        unique_risk = set(risk_levels)
        risk_consistency = 1.0 if len(unique_risk) == 1 else 1.0 / len(unique_risk)

        # Score consistency
        if credit_scores:
            score_range = max(credit_scores) - min(credit_scores)
            score_consistency = max(0, 1.0 - (score_range / 50.0))  # 50 point range = 0
        else:
            score_consistency = 0.0

        # Reasoning consistency
        if reasonings and all(reasonings):
            try:
                reasoning_result = self.consistency_scorer.score(reasonings)
                reasoning_consistency = reasoning_result.consistency_score
            except:
                reasoning_consistency = 0.5
        else:
            reasoning_consistency = 0.5

        overall = (risk_consistency + score_consistency + reasoning_consistency) / 3

        return {
            "overall_consistency": overall,
            "risk_level_consistency": risk_consistency,
            "score_consistency": score_consistency,
            "reasoning_consistency": reasoning_consistency,
            "risk_levels": risk_levels,
            "credit_scores": credit_scores,
            "score_range": max(credit_scores) - min(credit_scores) if credit_scores else 0,
            "num_runs": len(runs),
        }

    def compare_with_benchmark(
        self,
        run_id: str,
        company_name: str,
        our_assessment: Dict[str, Any],
        benchmark_assessment: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compare our assessment with an external benchmark (e.g., Coalition).

        Args:
            run_id: Unique run identifier
            company_name: Company name
            our_assessment: Our credit assessment
            benchmark_assessment: External benchmark assessment

        Returns:
            Comparison results
        """
        # Use LLM judge to compare
        judge_result = evaluate_with_llm_judge(
            run_id=run_id,
            company_name=company_name,
            assessment=our_assessment,
            benchmark=benchmark_assessment,
        )

        # Calculate alignment
        our_risk = our_assessment.get("overall_risk_level", "unknown")
        bench_risk = benchmark_assessment.get("risk_level", "unknown")

        our_score = our_assessment.get("credit_score_estimate", 0)
        bench_score = benchmark_assessment.get("credit_score", 0)

        risk_match = our_risk.lower() == bench_risk.lower()
        score_diff = abs(our_score - bench_score)

        return {
            "risk_level_match": risk_match,
            "our_risk_level": our_risk,
            "benchmark_risk_level": bench_risk,
            "our_credit_score": our_score,
            "benchmark_credit_score": bench_score,
            "score_difference": score_diff,
            "llm_alignment_score": judge_result.benchmark_alignment,
            "llm_comparison": judge_result.benchmark_comparison,
            "suggestions": judge_result.suggestions,
        }

    def generate_report(
        self,
        result: ComprehensiveEvaluationResult,
        format: str = "dict",
    ) -> Any:
        """
        Generate evaluation report.

        Args:
            result: Evaluation result
            format: "dict", "json", or "markdown"

        Returns:
            Report in requested format
        """
        report = {
            "summary": {
                "run_id": result.run_id,
                "company": result.company_name,
                "overall_score": round(result.overall_score, 4),
                "grade": result.overall_grade,
                "timestamp": result.timestamp,
            },
            "scores": {
                "workflow": {
                    "tool_selection": round(result.tool_selection_score, 4),
                    "data_quality": round(result.data_quality_score, 4),
                    "synthesis": round(result.synthesis_score, 4),
                    "overall": round(result.workflow_overall, 4),
                },
                "agent_efficiency": {
                    "intent_correctness": round(result.intent_correctness, 4),
                    "plan_quality": round(result.plan_quality, 4),
                    "tool_choice_correctness": round(result.tool_choice_correctness, 4),
                    "tool_completeness": round(result.tool_completeness, 4),
                    "trajectory_match": round(result.trajectory_match, 4),
                    "final_answer_quality": round(result.final_answer_quality, 4),
                    "overall": round(result.agent_overall, 4),
                },
                "llm_judge": {
                    "accuracy": round(result.llm_accuracy, 4),
                    "completeness": round(result.llm_completeness, 4),
                    "consistency": round(result.llm_consistency, 4),
                    "actionability": round(result.llm_actionability, 4),
                    "data_utilization": round(result.llm_data_utilization, 4),
                    "overall": round(result.llm_judge_overall, 4),
                },
            },
            "suggestions": result.llm_suggestions,
            "errors": result.errors,
            "warnings": result.warnings,
            "execution": {
                "time_ms": round(result.total_execution_time_ms, 2),
                "tokens": result.total_tokens,
                "cost": round(result.total_cost, 6),
            },
        }

        if format == "json":
            return json.dumps(report, indent=2)
        elif format == "markdown":
            return self._to_markdown(report)
        else:
            return report

    def _to_markdown(self, report: Dict[str, Any]) -> str:
        """Convert report to markdown format."""
        md = []
        md.append(f"# Evaluation Report: {report['summary']['company']}")
        md.append(f"\n**Run ID:** {report['summary']['run_id']}")
        md.append(f"**Overall Score:** {report['summary']['overall_score']} ({report['summary']['grade']})")
        md.append(f"**Timestamp:** {report['summary']['timestamp']}")

        md.append("\n## Workflow Scores")
        for k, v in report['scores']['workflow'].items():
            md.append(f"- **{k.replace('_', ' ').title()}:** {v}")

        md.append("\n## Agent Efficiency Scores")
        for k, v in report['scores']['agent_efficiency'].items():
            md.append(f"- **{k.replace('_', ' ').title()}:** {v}")

        md.append("\n## LLM Judge Scores")
        for k, v in report['scores']['llm_judge'].items():
            md.append(f"- **{k.replace('_', ' ').title()}:** {v}")

        if report['suggestions']:
            md.append("\n## Suggestions for Improvement")
            for s in report['suggestions']:
                md.append(f"- {s}")

        if report['errors']:
            md.append("\n## Errors")
            for e in report['errors']:
                md.append(f"- {e}")

        md.append("\n## Execution Metrics")
        md.append(f"- **Time:** {report['execution']['time_ms']}ms")
        md.append(f"- **Tokens:** {report['execution']['tokens']}")
        md.append(f"- **Cost:** ${report['execution']['cost']}")

        return "\n".join(md)

    def get_history_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluations in history."""
        if not self.evaluation_history:
            return {"total_evaluations": 0}

        total = len(self.evaluation_history)

        return {
            "total_evaluations": total,
            "avg_overall_score": sum(r.overall_score for r in self.evaluation_history) / total,
            "avg_workflow_score": sum(r.workflow_overall for r in self.evaluation_history) / total,
            "avg_agent_score": sum(r.agent_overall for r in self.evaluation_history) / total,
            "avg_llm_judge_score": sum(r.llm_judge_overall for r in self.evaluation_history) / total,
            "grade_distribution": self._get_grade_distribution(),
            "companies_evaluated": list(set(r.company_name for r in self.evaluation_history)),
        }

    def _get_grade_distribution(self) -> Dict[str, int]:
        """Get distribution of grades."""
        grades = {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0}
        for r in self.evaluation_history:
            if r.overall_grade in grades:
                grades[r.overall_grade] += 1
        return grades

    def export_history(self, filepath: str):
        """Export evaluation history to JSON file."""
        data = {
            "summary": self.get_history_summary(),
            "evaluations": [r.to_dict() for r in self.evaluation_history],
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Evaluation history exported to: {filepath}")

    def clear_history(self):
        """Clear evaluation history."""
        self.evaluation_history = []


# Singleton instance
_evaluation_brain: Optional[EvaluationBrain] = None


def get_evaluation_brain() -> EvaluationBrain:
    """Get the global EvaluationBrain instance."""
    global _evaluation_brain
    if _evaluation_brain is None:
        _evaluation_brain = EvaluationBrain()
    return _evaluation_brain


def evaluate_comprehensive(
    run_id: str,
    company_name: str,
    state: Dict[str, Any],
    benchmark: Dict[str, Any] = None,
    run_llm_judge: bool = True,
) -> ComprehensiveEvaluationResult:
    """
    Convenience function for comprehensive evaluation.

    Args:
        run_id: Unique run identifier
        company_name: Company being evaluated
        state: Workflow state with all data
        benchmark: Optional benchmark to compare against
        run_llm_judge: Whether to run LLM judge

    Returns:
        ComprehensiveEvaluationResult
    """
    brain = get_evaluation_brain()
    return brain.evaluate_run(
        run_id=run_id,
        company_name=company_name,
        state=state,
        benchmark=benchmark,
        run_llm_judge=run_llm_judge,
    )
