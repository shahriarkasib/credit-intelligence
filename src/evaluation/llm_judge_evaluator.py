"""
LLM-as-a-Judge Evaluator - Task 21 Implementation (Enhanced)

Uses LLM to evaluate and compare credit assessments:
- Judge quality of credit risk assessments
- Compare our assessments with external benchmarks (e.g., Coalition)
- Evaluate model consistency (same model, multiple runs)
- Compare accuracy across different models
- Provide structured evaluation with reasoning

Evaluation dimensions:
1. Accuracy - Is the risk assessment reasonable given the data?
2. Completeness - Does the assessment cover all relevant factors?
3. Consistency - Is the reasoning consistent with the conclusion?
4. Actionability - Are the recommendations actionable?
5. Data Utilization - Was the available data well utilized?

Additional Evaluation Modes:
- Consistency Mode: Run same company multiple times, measure output stability
- Cross-Model Mode: Compare assessments from different models for same company
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


@dataclass
class ConsistencyEvalResult:
    """Result from model consistency evaluation (same model, multiple runs)."""

    company_name: str = ""
    model_used: str = ""
    num_runs: int = 0

    # Core consistency metrics
    risk_level_consistency: float = 0.0  # How often same risk level was predicted
    credit_score_variance: float = 0.0  # Variance in credit scores
    credit_score_mean: float = 0.0
    credit_score_std: float = 0.0
    confidence_variance: float = 0.0  # Variance in confidence scores

    # Semantic consistency
    reasoning_similarity: float = 0.0  # Semantic similarity of reasoning texts
    risk_factors_overlap: float = 0.0  # Overlap in identified risk factors
    recommendations_overlap: float = 0.0  # Overlap in recommendations

    # Overall consistency score
    overall_consistency: float = 0.0
    is_consistent: bool = False
    consistency_grade: str = ""  # A, B, C, D, F

    # Individual run details
    run_details: List[Dict[str, Any]] = field(default_factory=list)

    # LLM Judge analysis
    llm_judge_analysis: str = ""
    llm_judge_concerns: List[str] = field(default_factory=list)

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class CrossModelEvalResult:
    """Result from cross-model accuracy evaluation."""

    company_name: str = ""
    models_compared: List[str] = field(default_factory=list)
    num_models: int = 0

    # Cross-model agreement metrics
    risk_level_agreement: float = 0.0  # Agreement on risk level across models
    credit_score_range: float = 0.0  # Range of credit scores
    credit_score_mean: float = 0.0
    credit_score_std: float = 0.0
    confidence_agreement: float = 0.0  # Agreement on confidence levels

    # Per-model results
    model_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Pairwise model comparisons
    pairwise_comparisons: List[Dict[str, Any]] = field(default_factory=list)

    # Best performing model (based on internal consistency)
    best_model: str = ""
    best_model_reasoning: str = ""

    # Overall cross-model score
    cross_model_agreement: float = 0.0

    # LLM Judge analysis
    llm_judge_analysis: str = ""
    model_recommendations: List[str] = field(default_factory=list)

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

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

    CONSISTENCY_EVALUATION_PROMPT = """You are an expert credit analyst evaluating the CONSISTENCY of multiple credit assessments.

The same company was assessed {num_runs} times using the same model ({model_name}).
Your task is to evaluate how consistent the model's outputs are across runs.

## Company: {company_name}

## Assessment Results Across Runs:

{run_summaries}

## Consistency Metrics:
- Risk Level Distribution: {risk_level_distribution}
- Credit Score Range: {score_min}-{score_max} (Mean: {score_mean:.1f}, Std: {score_std:.1f})
- Confidence Range: {confidence_min:.2f}-{confidence_max:.2f}

## Your Evaluation Task:

Analyze the consistency of these assessments and respond in JSON:
```json
{{
    "overall_consistency_score": <0-100>,
    "is_acceptable_consistency": true/false,
    "consistency_grade": "A" | "B" | "C" | "D" | "F",
    "analysis": "<detailed analysis of consistency patterns>",
    "concerns": ["<concern 1>", "<concern 2>", ...],
    "risk_level_consistency_score": <0-100>,
    "reasoning_consistency_score": <0-100>,
    "recommendations": ["<recommendation for improving consistency>", ...]
}}
```

Grading:
- A (90-100): Highly consistent, minor variations in reasoning only
- B (75-89): Good consistency, same risk level, small score variations
- C (60-74): Moderate consistency, some disagreement on risk factors
- D (40-59): Poor consistency, different risk levels across runs
- F (<40): Unacceptable inconsistency, unreliable model"""

    CROSS_MODEL_EVALUATION_PROMPT = """You are an expert credit analyst comparing credit assessments from DIFFERENT models.

The same company was assessed by {num_models} different models.
Your task is to evaluate which model provides the best assessment and analyze the differences.

## Company: {company_name}

## Model Assessments:

{model_summaries}

## Cross-Model Metrics:
- Risk Level Agreement: {risk_level_agreement}%
- Credit Score Range: {score_min}-{score_max} (Spread: {score_spread})
- Models Assessed: {model_names}

## Your Evaluation Task:

Compare these assessments and respond in JSON:
```json
{{
    "cross_model_agreement_score": <0-100>,
    "best_model": "<model name>",
    "best_model_reasoning": "<why this model is best>",
    "analysis": "<detailed comparison of model outputs>",
    "model_rankings": [
        {{"model": "<name>", "score": <0-100>, "strengths": ["..."], "weaknesses": ["..."]}},
        ...
    ],
    "recommendations": ["<which model to use for production>", "<when to use each model>", ...],
    "confidence_in_ranking": <0-100>
}}
```

Consider:
- Quality and depth of reasoning
- Appropriate use of available data
- Realistic risk assessments
- Actionable recommendations
- Internal consistency within each assessment"""

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

    def evaluate_consistency(
        self,
        company_name: str,
        assessments: List[Dict[str, Any]],
        model_name: str = "unknown",
    ) -> ConsistencyEvalResult:
        """
        Evaluate consistency of multiple assessments for the same company.

        This measures how stable/consistent a model is when run multiple times
        on the same company.

        Args:
            company_name: Name of the company assessed
            assessments: List of assessment dicts from multiple runs
            model_name: Name of the model used

        Returns:
            ConsistencyEvalResult with consistency metrics
        """
        if len(assessments) < 2:
            return ConsistencyEvalResult(
                company_name=company_name,
                model_used=model_name,
                num_runs=len(assessments),
                overall_consistency=1.0,
                is_consistent=True,
                consistency_grade="A",
                llm_judge_analysis="Only one assessment provided - cannot evaluate consistency",
            )

        # Extract metrics from each assessment
        risk_levels = []
        credit_scores = []
        confidences = []
        reasonings = []
        risk_factors_sets = []
        recommendations_sets = []
        run_details = []

        for i, assessment in enumerate(assessments):
            risk_level = assessment.get("risk_level") or assessment.get("overall_risk_level", "unknown")
            credit_score = assessment.get("credit_score") or assessment.get("credit_score_estimate", 0)
            confidence = assessment.get("confidence") or assessment.get("confidence_score", 0)
            reasoning = assessment.get("reasoning") or assessment.get("llm_reasoning", "")
            risk_factors = set(assessment.get("risk_factors", []))
            recommendations = set(assessment.get("recommendations", []))

            risk_levels.append(str(risk_level).lower())
            credit_scores.append(float(credit_score) if credit_score else 0)
            confidences.append(float(confidence) if confidence else 0)
            reasonings.append(str(reasoning))
            risk_factors_sets.append(risk_factors)
            recommendations_sets.append(recommendations)

            run_details.append({
                "run": i + 1,
                "risk_level": risk_level,
                "credit_score": credit_score,
                "confidence": confidence,
                "num_risk_factors": len(risk_factors),
                "num_recommendations": len(recommendations),
            })

        # Calculate metrics
        from collections import Counter
        import statistics

        # Risk level consistency
        risk_counter = Counter(risk_levels)
        most_common_risk, most_common_count = risk_counter.most_common(1)[0]
        risk_level_consistency = most_common_count / len(risk_levels)

        # Credit score statistics
        score_mean = statistics.mean(credit_scores) if credit_scores else 0
        score_std = statistics.stdev(credit_scores) if len(credit_scores) > 1 else 0
        score_variance = statistics.variance(credit_scores) if len(credit_scores) > 1 else 0

        # Confidence statistics
        conf_variance = statistics.variance(confidences) if len(confidences) > 1 else 0

        # Risk factors overlap (Jaccard similarity)
        risk_factors_overlap = self._calculate_set_overlap(risk_factors_sets)

        # Recommendations overlap
        recommendations_overlap = self._calculate_set_overlap(recommendations_sets)

        # Build run summaries for LLM
        run_summaries = ""
        for i, detail in enumerate(run_details):
            run_summaries += f"\n### Run {i + 1}:\n"
            run_summaries += f"- Risk Level: {detail['risk_level']}\n"
            run_summaries += f"- Credit Score: {detail['credit_score']}\n"
            run_summaries += f"- Confidence: {detail['confidence']:.2f}\n"
            run_summaries += f"- Risk Factors Identified: {detail['num_risk_factors']}\n"
            run_summaries += f"- Reasoning: {reasonings[i][:300]}...\n" if len(reasonings[i]) > 300 else f"- Reasoning: {reasonings[i]}\n"

        # Get LLM analysis if available
        llm_analysis = ""
        concerns = []
        reasoning_consistency = 0.0
        overall_consistency = (
            risk_level_consistency * 0.4 +
            max(0, 1 - (score_std / 20)) * 0.3 +  # Normalize std, lower is better
            risk_factors_overlap * 0.15 +
            recommendations_overlap * 0.15
        )

        if self.is_available():
            try:
                prompt = self.CONSISTENCY_EVALUATION_PROMPT.format(
                    num_runs=len(assessments),
                    model_name=model_name,
                    company_name=company_name,
                    run_summaries=run_summaries,
                    risk_level_distribution=dict(risk_counter),
                    score_min=min(credit_scores),
                    score_max=max(credit_scores),
                    score_mean=score_mean,
                    score_std=score_std,
                    confidence_min=min(confidences),
                    confidence_max=max(confidences),
                )

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert credit analyst evaluating model consistency. Respond in valid JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3,
                    max_tokens=1500,
                )

                parsed = self._parse_response(response.choices[0].message.content)
                llm_analysis = parsed.get("analysis", "")
                concerns = parsed.get("concerns", [])
                reasoning_consistency = parsed.get("reasoning_consistency_score", 50) / 100
                overall_consistency = parsed.get("overall_consistency_score", overall_consistency * 100) / 100

            except Exception as e:
                logger.warning(f"LLM consistency analysis failed: {e}")

        # Determine grade
        if overall_consistency >= 0.9:
            grade = "A"
        elif overall_consistency >= 0.75:
            grade = "B"
        elif overall_consistency >= 0.6:
            grade = "C"
        elif overall_consistency >= 0.4:
            grade = "D"
        else:
            grade = "F"

        result = ConsistencyEvalResult(
            company_name=company_name,
            model_used=model_name,
            num_runs=len(assessments),
            risk_level_consistency=risk_level_consistency,
            credit_score_variance=score_variance,
            credit_score_mean=score_mean,
            credit_score_std=score_std,
            confidence_variance=conf_variance,
            reasoning_similarity=reasoning_consistency,
            risk_factors_overlap=risk_factors_overlap,
            recommendations_overlap=recommendations_overlap,
            overall_consistency=overall_consistency,
            is_consistent=overall_consistency >= 0.75,
            consistency_grade=grade,
            run_details=run_details,
            llm_judge_analysis=llm_analysis,
            llm_judge_concerns=concerns,
        )

        logger.info(f"Consistency evaluation for {company_name}: {grade} ({overall_consistency:.2f})")
        return result

    def evaluate_cross_model(
        self,
        company_name: str,
        model_assessments: Dict[str, Dict[str, Any]],
    ) -> CrossModelEvalResult:
        """
        Evaluate and compare assessments from different models.

        This measures how different models agree/disagree on the same company.

        Args:
            company_name: Name of the company assessed
            model_assessments: Dict mapping model_name -> assessment

        Returns:
            CrossModelEvalResult with cross-model comparison
        """
        if len(model_assessments) < 2:
            return CrossModelEvalResult(
                company_name=company_name,
                models_compared=list(model_assessments.keys()),
                num_models=len(model_assessments),
                cross_model_agreement=1.0,
                llm_judge_analysis="Only one model provided - cannot compare",
            )

        models = list(model_assessments.keys())
        assessments = list(model_assessments.values())

        # Extract metrics from each model's assessment
        risk_levels = []
        credit_scores = []
        confidences = []
        model_results = {}

        for model_name, assessment in model_assessments.items():
            risk_level = assessment.get("risk_level") or assessment.get("overall_risk_level", "unknown")
            credit_score = assessment.get("credit_score") or assessment.get("credit_score_estimate", 0)
            confidence = assessment.get("confidence") or assessment.get("confidence_score", 0)

            risk_levels.append(str(risk_level).lower())
            credit_scores.append(float(credit_score) if credit_score else 0)
            confidences.append(float(confidence) if confidence else 0)

            model_results[model_name] = {
                "risk_level": risk_level,
                "credit_score": credit_score,
                "confidence": confidence,
                "reasoning": (assessment.get("reasoning") or assessment.get("llm_reasoning", ""))[:500],
                "risk_factors": assessment.get("risk_factors", [])[:5],
                "recommendations": assessment.get("recommendations", [])[:3],
            }

        # Calculate agreement metrics
        from collections import Counter
        import statistics

        risk_counter = Counter(risk_levels)
        most_common_risk, most_common_count = risk_counter.most_common(1)[0]
        risk_level_agreement = most_common_count / len(risk_levels)

        score_mean = statistics.mean(credit_scores)
        score_std = statistics.stdev(credit_scores) if len(credit_scores) > 1 else 0
        score_range = max(credit_scores) - min(credit_scores)

        conf_mean = statistics.mean(confidences)

        # Build pairwise comparisons
        pairwise_comparisons = []
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                score_diff = abs(credit_scores[i] - credit_scores[j])
                same_risk = risk_levels[i] == risk_levels[j]
                pairwise_comparisons.append({
                    "models": (models[i], models[j]),
                    "same_risk_level": same_risk,
                    "score_difference": score_diff,
                    "risk_levels": (risk_levels[i], risk_levels[j]),
                    "scores": (credit_scores[i], credit_scores[j]),
                })

        # Build model summaries for LLM
        model_summaries = ""
        for model_name, result in model_results.items():
            model_summaries += f"\n### {model_name}:\n"
            model_summaries += f"- Risk Level: {result['risk_level']}\n"
            model_summaries += f"- Credit Score: {result['credit_score']}\n"
            model_summaries += f"- Confidence: {result['confidence']:.2f}\n"
            model_summaries += f"- Key Risk Factors: {', '.join(result['risk_factors'][:3])}\n"
            model_summaries += f"- Reasoning: {result['reasoning'][:300]}...\n" if len(result['reasoning']) > 300 else f"- Reasoning: {result['reasoning']}\n"

        # Get LLM analysis if available
        best_model = ""
        best_model_reasoning = ""
        llm_analysis = ""
        model_recommendations = []
        cross_model_agreement = risk_level_agreement * 0.5 + max(0, 1 - score_range / 30) * 0.5

        if self.is_available():
            try:
                prompt = self.CROSS_MODEL_EVALUATION_PROMPT.format(
                    num_models=len(models),
                    company_name=company_name,
                    model_summaries=model_summaries,
                    risk_level_agreement=int(risk_level_agreement * 100),
                    score_min=min(credit_scores),
                    score_max=max(credit_scores),
                    score_spread=score_range,
                    model_names=", ".join(models),
                )

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert credit analyst comparing model outputs. Respond in valid JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3,
                    max_tokens=1500,
                )

                parsed = self._parse_response(response.choices[0].message.content)
                best_model = parsed.get("best_model", "")
                best_model_reasoning = parsed.get("best_model_reasoning", "")
                llm_analysis = parsed.get("analysis", "")
                model_recommendations = parsed.get("recommendations", [])
                cross_model_agreement = parsed.get("cross_model_agreement_score", cross_model_agreement * 100) / 100

            except Exception as e:
                logger.warning(f"LLM cross-model analysis failed: {e}")

        result = CrossModelEvalResult(
            company_name=company_name,
            models_compared=models,
            num_models=len(models),
            risk_level_agreement=risk_level_agreement,
            credit_score_range=score_range,
            credit_score_mean=score_mean,
            credit_score_std=score_std,
            confidence_agreement=1 - statistics.stdev(confidences) if len(confidences) > 1 else 1.0,
            model_results=model_results,
            pairwise_comparisons=pairwise_comparisons,
            best_model=best_model,
            best_model_reasoning=best_model_reasoning,
            cross_model_agreement=cross_model_agreement,
            llm_judge_analysis=llm_analysis,
            model_recommendations=model_recommendations,
        )

        logger.info(f"Cross-model evaluation for {company_name}: agreement={cross_model_agreement:.2f}, best={best_model}")
        return result

    def _calculate_set_overlap(self, sets: List[set]) -> float:
        """Calculate average Jaccard similarity across all set pairs."""
        if len(sets) < 2:
            return 1.0

        similarities = []
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                intersection = sets[i] & sets[j]
                union = sets[i] | sets[j]
                if union:
                    similarities.append(len(intersection) / len(union))
                else:
                    similarities.append(1.0)  # Both empty = perfect agreement

        return sum(similarities) / len(similarities) if similarities else 1.0


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
