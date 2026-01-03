"""
LLM-as-a-Judge Evaluator - Task 21 Implementation

Uses LLM to evaluate and compare credit assessments:
- Judge quality of credit risk assessments
- Compare our assessments with external benchmarks (e.g., Coalition)
- Provide structured evaluation with reasoning

Evaluation dimensions:
1. Accuracy - Is the risk assessment reasonable given the data?
2. Completeness - Does the assessment cover all relevant factors?
3. Consistency - Is the reasoning consistent with the conclusion?
4. Actionability - Are the recommendations actionable?
5. Data Utilization - Was the available data well utilized?
"""

import os
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Import Groq for LLM calls
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    Groq = None


@dataclass
class LLMJudgeResult:
    """Result from LLM-as-a-judge evaluation."""

    # Overall scores (0-1)
    accuracy_score: float = 0.0
    completeness_score: float = 0.0
    consistency_score: float = 0.0
    actionability_score: float = 0.0
    data_utilization_score: float = 0.0
    overall_score: float = 0.0

    # Reasoning for each dimension
    accuracy_reasoning: str = ""
    completeness_reasoning: str = ""
    consistency_reasoning: str = ""
    actionability_reasoning: str = ""
    data_utilization_reasoning: str = ""
    overall_reasoning: str = ""

    # Comparison with benchmark (if provided)
    benchmark_alignment: float = 0.0  # How well does it align with benchmark?
    benchmark_comparison: str = ""  # Detailed comparison

    # Suggestions for improvement
    suggestions: List[str] = field(default_factory=list)

    # Metadata
    run_id: str = ""
    company_name: str = ""
    model_used: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    tokens_used: int = 0
    evaluation_cost: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class LLMJudgeEvaluator:
    """
    LLM-as-a-Judge evaluator for credit assessments.

    Uses an LLM to evaluate the quality of credit risk assessments
    and compare them with external benchmarks.
    """

    # Evaluation prompt template
    EVALUATION_PROMPT = """You are an expert credit risk analyst evaluating the quality of a credit assessment.

## Assessment to Evaluate

**Company:** {company_name}

**Assessment Details:**
- Risk Level: {risk_level}
- Credit Score: {credit_score}/100
- Confidence: {confidence}%

**Reasoning:**
{reasoning}

**Key Findings:**
{key_findings}

**Risk Factors:**
{risk_factors}

**Positive Factors:**
{positive_factors}

**Recommendations:**
{recommendations}

## Data Sources Used
{data_sources}

## Your Task

Evaluate this credit assessment across the following dimensions (score each 0-100):

1. **Accuracy** - Is the risk level and credit score reasonable given the available data?
2. **Completeness** - Does the assessment cover all relevant risk factors?
3. **Consistency** - Is the reasoning consistent with the final risk level and score?
4. **Actionability** - Are the recommendations clear and actionable?
5. **Data Utilization** - Was the available data effectively used in the analysis?

{benchmark_section}

Respond in JSON format:
```json
{{
    "accuracy_score": <0-100>,
    "accuracy_reasoning": "<reasoning>",
    "completeness_score": <0-100>,
    "completeness_reasoning": "<reasoning>",
    "consistency_score": <0-100>,
    "consistency_reasoning": "<reasoning>",
    "actionability_score": <0-100>,
    "actionability_reasoning": "<reasoning>",
    "data_utilization_score": <0-100>,
    "data_utilization_reasoning": "<reasoning>",
    "overall_score": <0-100>,
    "overall_reasoning": "<summary reasoning>",
    "benchmark_alignment": <0-100 if benchmark provided, else null>,
    "benchmark_comparison": "<comparison if benchmark provided>",
    "suggestions": ["<improvement 1>", "<improvement 2>", ...]
}}
```"""

    BENCHMARK_SECTION = """
## Benchmark Comparison

Compare the assessment with this external benchmark:

**Benchmark Risk Level:** {benchmark_risk_level}
**Benchmark Credit Score:** {benchmark_score}
**Benchmark Reasoning:** {benchmark_reasoning}

Evaluate how well our assessment aligns with the benchmark.
"""

    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        """Initialize the LLM judge evaluator."""
        self.model = model
        self.client = None
        self.api_key = os.getenv("GROQ_API_KEY")

        if GROQ_AVAILABLE and self.api_key:
            try:
                self.client = Groq(api_key=self.api_key)
                logger.info(f"LLM Judge initialized with model: {model}")
            except Exception as e:
                logger.error(f"Failed to initialize Groq client: {e}")
        else:
            logger.warning("Groq not available or GROQ_API_KEY not set")

    def is_available(self) -> bool:
        """Check if the evaluator is available."""
        return self.client is not None

    def _format_data_sources(self, api_data: Dict[str, Any]) -> str:
        """Format data sources for the prompt."""
        if not api_data:
            return "No external data sources used."

        lines = []
        for source, data in api_data.items():
            if data:
                if isinstance(data, dict):
                    record_count = len(data.get("filings", [])) or len(data.get("results", [])) or 1
                    lines.append(f"- {source}: {record_count} records")
                else:
                    lines.append(f"- {source}: Available")
            else:
                lines.append(f"- {source}: No data")

        return "\n".join(lines) if lines else "No external data sources used."

    def _format_list(self, items: List[str]) -> str:
        """Format a list of items."""
        if not items:
            return "None provided"
        return "\n".join(f"- {item}" for item in items)

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the LLM response JSON."""
        try:
            # Try to extract JSON from the response
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start != -1 and end > start:
                json_str = response_text[start:end]
                return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response: {e}")

        # Return default structure if parsing fails
        return {
            "accuracy_score": 50,
            "accuracy_reasoning": "Could not parse response",
            "completeness_score": 50,
            "completeness_reasoning": "Could not parse response",
            "consistency_score": 50,
            "consistency_reasoning": "Could not parse response",
            "actionability_score": 50,
            "actionability_reasoning": "Could not parse response",
            "data_utilization_score": 50,
            "data_utilization_reasoning": "Could not parse response",
            "overall_score": 50,
            "overall_reasoning": response_text[:500],
            "suggestions": ["Evaluation could not be fully completed"],
        }

    def evaluate(
        self,
        run_id: str,
        company_name: str,
        assessment: Dict[str, Any],
        api_data: Dict[str, Any] = None,
        benchmark: Dict[str, Any] = None,
    ) -> LLMJudgeResult:
        """
        Evaluate a credit assessment using LLM-as-a-judge.

        Args:
            run_id: Unique run identifier
            company_name: Company being evaluated
            assessment: The credit assessment to evaluate
            api_data: Data sources used (for context)
            benchmark: Optional benchmark to compare against (e.g., Coalition assessment)

        Returns:
            LLMJudgeResult with scores and reasoning
        """
        if not self.is_available():
            logger.warning("LLM Judge not available")
            return LLMJudgeResult(
                run_id=run_id,
                company_name=company_name,
                overall_reasoning="LLM Judge not available - evaluation skipped",
            )

        # Format the prompt
        benchmark_section = ""
        if benchmark:
            benchmark_section = self.BENCHMARK_SECTION.format(
                benchmark_risk_level=benchmark.get("risk_level", "N/A"),
                benchmark_score=benchmark.get("credit_score", "N/A"),
                benchmark_reasoning=benchmark.get("reasoning", "N/A"),
            )

        prompt = self.EVALUATION_PROMPT.format(
            company_name=company_name,
            risk_level=assessment.get("overall_risk_level", "unknown"),
            credit_score=assessment.get("credit_score_estimate", 0),
            confidence=int(assessment.get("confidence_score", 0) * 100),
            reasoning=assessment.get("llm_reasoning", "No reasoning provided"),
            key_findings=self._format_list(assessment.get("key_findings", [])),
            risk_factors=self._format_list(assessment.get("risk_factors", [])),
            positive_factors=self._format_list(assessment.get("positive_factors", [])),
            recommendations=self._format_list(assessment.get("recommendations", [])),
            data_sources=self._format_data_sources(api_data or {}),
            benchmark_section=benchmark_section,
        )

        try:
            # Make LLM call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert credit risk evaluator. Always respond in valid JSON format."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,  # Lower temperature for more consistent evaluation
                max_tokens=2000,
            )

            response_text = response.choices[0].message.content
            parsed = self._parse_response(response_text)

            # Calculate token usage and cost
            tokens_used = response.usage.total_tokens if response.usage else 0
            # Groq pricing (approximate)
            cost_per_token = 0.27 / 1_000_000  # $0.27 per 1M tokens for llama-3.3-70b
            evaluation_cost = tokens_used * cost_per_token

            # Build result
            result = LLMJudgeResult(
                run_id=run_id,
                company_name=company_name,
                model_used=self.model,
                tokens_used=tokens_used,
                evaluation_cost=evaluation_cost,
                accuracy_score=parsed.get("accuracy_score", 0) / 100,
                accuracy_reasoning=parsed.get("accuracy_reasoning", ""),
                completeness_score=parsed.get("completeness_score", 0) / 100,
                completeness_reasoning=parsed.get("completeness_reasoning", ""),
                consistency_score=parsed.get("consistency_score", 0) / 100,
                consistency_reasoning=parsed.get("consistency_reasoning", ""),
                actionability_score=parsed.get("actionability_score", 0) / 100,
                actionability_reasoning=parsed.get("actionability_reasoning", ""),
                data_utilization_score=parsed.get("data_utilization_score", 0) / 100,
                data_utilization_reasoning=parsed.get("data_utilization_reasoning", ""),
                overall_score=parsed.get("overall_score", 0) / 100,
                overall_reasoning=parsed.get("overall_reasoning", ""),
                benchmark_alignment=parsed.get("benchmark_alignment", 0) / 100 if parsed.get("benchmark_alignment") else 0,
                benchmark_comparison=parsed.get("benchmark_comparison", ""),
                suggestions=parsed.get("suggestions", []),
            )

            logger.info(f"LLM Judge evaluation complete for {company_name}: overall_score={result.overall_score:.2f}")
            return result

        except Exception as e:
            logger.error(f"LLM Judge evaluation failed: {e}")
            return LLMJudgeResult(
                run_id=run_id,
                company_name=company_name,
                model_used=self.model,
                overall_reasoning=f"Evaluation failed: {str(e)}",
            )

    def compare_assessments(
        self,
        run_id: str,
        company_name: str,
        our_assessment: Dict[str, Any],
        coalition_assessment: Dict[str, Any],
        api_data: Dict[str, Any] = None,
    ) -> Tuple[LLMJudgeResult, LLMJudgeResult]:
        """
        Compare our assessment with Coalition's assessment.

        Args:
            run_id: Unique run identifier
            company_name: Company being evaluated
            our_assessment: Our credit assessment
            coalition_assessment: Coalition's credit assessment
            api_data: Data sources used

        Returns:
            Tuple of (our_evaluation, coalition_evaluation)
        """
        # Evaluate our assessment with Coalition as benchmark
        our_eval = self.evaluate(
            run_id=f"{run_id}_ours",
            company_name=company_name,
            assessment=our_assessment,
            api_data=api_data,
            benchmark=coalition_assessment,
        )

        # Evaluate Coalition assessment with ours as benchmark
        coalition_eval = self.evaluate(
            run_id=f"{run_id}_coalition",
            company_name=company_name,
            assessment=coalition_assessment,
            api_data=api_data,
            benchmark=our_assessment,
        )

        return our_eval, coalition_eval


# Singleton instance
_judge_evaluator: Optional[LLMJudgeEvaluator] = None


def get_llm_judge() -> LLMJudgeEvaluator:
    """Get the global LLMJudgeEvaluator instance."""
    global _judge_evaluator
    if _judge_evaluator is None:
        _judge_evaluator = LLMJudgeEvaluator()
    return _judge_evaluator


def evaluate_with_llm_judge(
    run_id: str,
    company_name: str,
    assessment: Dict[str, Any],
    api_data: Dict[str, Any] = None,
    benchmark: Dict[str, Any] = None,
) -> LLMJudgeResult:
    """
    Convenience function to evaluate an assessment with LLM-as-a-judge.

    Args:
        run_id: Unique run identifier
        company_name: Company being evaluated
        assessment: The credit assessment to evaluate
        api_data: Data sources used
        benchmark: Optional benchmark assessment (e.g., Coalition)

    Returns:
        LLMJudgeResult with evaluation scores
    """
    judge = get_llm_judge()
    return judge.evaluate(
        run_id=run_id,
        company_name=company_name,
        assessment=assessment,
        api_data=api_data,
        benchmark=benchmark,
    )
