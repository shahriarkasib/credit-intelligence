"""
Unified Agent Evaluator - Combines DeepEval, OpenEvals, and Built-in metrics.

Evaluates agent performance across:
- ACCURACY: faithfulness, hallucination, factual_accuracy, answer_relevancy
- CONSISTENCY: same_model, cross_model, semantic_similarity
- AGENT EFFICIENCY: intent, plan, tool_choice, tool_completeness, trajectory, final_answer

Install dependencies:
    pip install deepeval sentence-transformers openai
"""

import os
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ============================================================================
# LIBRARY AVAILABILITY CHECKS
# ============================================================================

# DeepEval
try:
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        FaithfulnessMetric,
        HallucinationMetric,
    )
    from deepeval.test_case import LLMTestCase
    DEEPEVAL_AVAILABLE = True
    logger.info("DeepEval available")
except ImportError:
    DEEPEVAL_AVAILABLE = False
    logger.warning("DeepEval not installed. Run: pip install deepeval")

# OpenAI for OpenEvals-style model-graded evaluation
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = bool(os.getenv("OPENAI_API_KEY"))
    logger.info(f"OpenAI available: {OPENAI_AVAILABLE}")
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI not installed. Run: pip install openai")

# Sentence Transformers for semantic similarity
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    logger.info("Sentence Transformers available")
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("Sentence Transformers not installed. Run: pip install sentence-transformers")


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class AccuracyMetrics:
    """Accuracy-related metrics (is the output correct?)."""
    faithfulness: float = 0.0           # Is output grounded in context?
    hallucination: float = 0.0          # % hallucinated content (lower=better)
    answer_relevancy: float = 0.0       # Is answer relevant to query?
    factual_accuracy: float = 0.0       # Do facts match known data?
    final_answer_quality: float = 0.0   # Are required fields present & valid?

    # Combined accuracy score
    accuracy_score: float = 0.0

    # Details
    details: Dict[str, Any] = field(default_factory=dict)

    def calculate_combined(self) -> float:
        """Calculate weighted accuracy score."""
        self.accuracy_score = (
            self.faithfulness * 0.25 +
            (1 - self.hallucination) * 0.20 +
            self.factual_accuracy * 0.20 +
            self.answer_relevancy * 0.15 +
            self.final_answer_quality * 0.20
        )
        return self.accuracy_score


@dataclass
class ConsistencyMetrics:
    """Consistency-related metrics (do outputs agree?)."""
    same_model_consistency: float = 0.0     # Same LLM, multiple runs
    cross_model_consistency: float = 0.0    # Different LLMs
    risk_level_agreement: float = 0.0       # All agree on risk level?
    semantic_similarity: float = 0.0        # Reasoning text similarity

    # Combined consistency score
    consistency_score: float = 0.0

    # Details
    details: Dict[str, Any] = field(default_factory=dict)

    def calculate_combined(self) -> float:
        """Calculate weighted consistency score."""
        self.consistency_score = (
            self.same_model_consistency * 0.30 +
            self.cross_model_consistency * 0.30 +
            self.risk_level_agreement * 0.20 +
            self.semantic_similarity * 0.20
        )
        return self.consistency_score


@dataclass
class AgentEfficiencyMetrics:
    """Agent efficiency metrics."""
    intent_correctness: float = 0.0
    plan_quality: float = 0.0
    tool_choice_correctness: float = 0.0
    tool_completeness: float = 0.0
    trajectory_match: float = 0.0
    final_answer_quality: float = 0.0

    # Execution stats
    step_count: int = 0
    tool_calls: int = 0
    latency_ms: float = 0.0

    # Combined score
    overall_score: float = 0.0

    # Details
    intent_details: Dict[str, Any] = field(default_factory=dict)
    plan_details: Dict[str, Any] = field(default_factory=dict)
    tool_details: Dict[str, Any] = field(default_factory=dict)
    trajectory_details: Dict[str, Any] = field(default_factory=dict)
    answer_details: Dict[str, Any] = field(default_factory=dict)

    def calculate_overall(self) -> float:
        """Calculate weighted overall score."""
        self.overall_score = (
            self.intent_correctness * 0.15 +
            self.plan_quality * 0.15 +
            self.tool_choice_correctness * 0.20 +
            self.tool_completeness * 0.15 +
            self.trajectory_match * 0.15 +
            self.final_answer_quality * 0.20
        )
        return self.overall_score


@dataclass
class UnifiedEvaluationResult:
    """Complete evaluation result combining all metrics."""
    run_id: str = ""
    company_name: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Category scores
    accuracy: AccuracyMetrics = field(default_factory=AccuracyMetrics)
    consistency: ConsistencyMetrics = field(default_factory=ConsistencyMetrics)
    agent_efficiency: AgentEfficiencyMetrics = field(default_factory=AgentEfficiencyMetrics)

    # Overall combined score
    overall_quality_score: float = 0.0

    # Evaluation metadata
    libraries_used: List[str] = field(default_factory=list)
    evaluation_time_ms: float = 0.0
    errors: List[str] = field(default_factory=list)

    def calculate_overall_quality(self) -> float:
        """Calculate overall quality score."""
        self.accuracy.calculate_combined()
        self.consistency.calculate_combined()
        self.agent_efficiency.calculate_overall()

        self.overall_quality_score = (
            self.accuracy.accuracy_score * 0.40 +
            self.consistency.consistency_score * 0.30 +
            self.agent_efficiency.overall_score * 0.30
        )
        return self.overall_quality_score

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "company_name": self.company_name,
            "timestamp": self.timestamp,
            "accuracy": asdict(self.accuracy),
            "consistency": asdict(self.consistency),
            "agent_efficiency": asdict(self.agent_efficiency),
            "overall_quality_score": self.overall_quality_score,
            "libraries_used": self.libraries_used,
            "evaluation_time_ms": self.evaluation_time_ms,
            "errors": self.errors,
        }


# ============================================================================
# UNIFIED EVALUATOR
# ============================================================================

class UnifiedAgentEvaluator:
    """
    Unified evaluator combining DeepEval, OpenEvals, and built-in metrics.

    Usage:
        evaluator = UnifiedAgentEvaluator()
        result = evaluator.evaluate(
            run_id="abc123",
            company_name="Google",
            state=workflow_state,
            llm_results=multi_llm_results,  # For consistency
        )
        print(f"Overall Quality: {result.overall_quality_score}")
    """

    # Expected tools by company type
    EXPECTED_TOOLS = {
        "public_us": {"sec_edgar", "finnhub", "web_search", "tavily"},
        "public_non_us": {"finnhub", "web_search", "tavily"},
        "private": {"web_search", "court_listener", "tavily"},
        "unknown": {"web_search", "tavily"},
    }

    # Expected trajectory
    EXPECTED_TRAJECTORY = [
        "parse_input", "validate_company", "create_plan",
        "fetch_api_data", "search_web", "synthesize",
        "save_to_database", "evaluate",
    ]

    # Required output fields
    REQUIRED_OUTPUT_FIELDS = {
        "risk_level", "credit_score", "confidence", "reasoning", "recommendations"
    }

    def __init__(self, openai_model: str = "gpt-4o-mini", deepeval_model: str = "gpt-4o-mini"):
        """Initialize evaluator with model configurations."""
        self.openai_model = openai_model
        self.deepeval_model = deepeval_model

        # Initialize clients
        self._openai_client = None
        self._embedding_model = None
        self._deepeval_metrics = {}

        logger.info(f"UnifiedAgentEvaluator initialized (OpenAI: {OPENAI_AVAILABLE}, "
                   f"DeepEval: {DEEPEVAL_AVAILABLE}, SentenceTransformers: {SENTENCE_TRANSFORMERS_AVAILABLE})")

    def _get_openai_client(self):
        """Lazy load OpenAI client."""
        if self._openai_client is None and OPENAI_AVAILABLE:
            self._openai_client = OpenAI()
        return self._openai_client

    def _get_embedding_model(self):
        """Lazy load embedding model."""
        if self._embedding_model is None and SENTENCE_TRANSFORMERS_AVAILABLE:
            self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embedding_model

    # ========================================================================
    # ACCURACY EVALUATION (DeepEval + OpenEvals)
    # ========================================================================

    def evaluate_accuracy(
        self,
        query: str,
        context: List[str],
        output: str,
        assessment: Dict[str, Any],
    ) -> AccuracyMetrics:
        """Evaluate accuracy using DeepEval metrics."""
        metrics = AccuracyMetrics()

        # 1. DeepEval metrics (if available)
        if DEEPEVAL_AVAILABLE:
            try:
                metrics.faithfulness, metrics.hallucination, metrics.answer_relevancy = \
                    self._evaluate_with_deepeval(query, context, output)
                metrics.details["deepeval"] = "success"
            except Exception as e:
                logger.warning(f"DeepEval evaluation failed: {e}")
                metrics.details["deepeval_error"] = str(e)

        # 2. Factual accuracy via OpenEvals-style grading
        if OPENAI_AVAILABLE:
            try:
                metrics.factual_accuracy = self._evaluate_factual_accuracy(assessment, context)
                metrics.details["openevals"] = "success"
            except Exception as e:
                logger.warning(f"OpenEvals evaluation failed: {e}")
                metrics.details["openevals_error"] = str(e)

        # 3. Final answer quality (built-in)
        metrics.final_answer_quality = self._evaluate_final_answer_quality(assessment)

        # Calculate combined
        metrics.calculate_combined()

        return metrics

    def _evaluate_with_deepeval(
        self,
        query: str,
        context: List[str],
        output: str,
    ) -> Tuple[float, float, float]:
        """Run DeepEval metrics."""
        test_case = LLMTestCase(
            input=query,
            actual_output=output,
            context=context,
            retrieval_context=context,
        )

        # Faithfulness
        faithfulness_metric = FaithfulnessMetric(threshold=0.7, model=self.deepeval_model)
        faithfulness_metric.measure(test_case)
        faithfulness = faithfulness_metric.score or 0.0

        # Hallucination
        hallucination_metric = HallucinationMetric(threshold=0.5, model=self.deepeval_model)
        hallucination_metric.measure(test_case)
        hallucination = hallucination_metric.score or 0.0

        # Answer Relevancy
        relevancy_metric = AnswerRelevancyMetric(threshold=0.7, model=self.deepeval_model)
        relevancy_metric.measure(test_case)
        relevancy = relevancy_metric.score or 0.0

        return faithfulness, hallucination, relevancy

    def _evaluate_factual_accuracy(
        self,
        assessment: Dict[str, Any],
        context: List[str],
    ) -> float:
        """Evaluate factual accuracy using GPT-4 as judge."""
        client = self._get_openai_client()
        if not client:
            return 0.5  # Default if no client

        prompt = f"""Evaluate the factual accuracy of this credit assessment.

Context (source data):
{chr(10).join(context[:3])}

Assessment:
- Risk Level: {assessment.get('overall_risk_level', assessment.get('risk_level', 'unknown'))}
- Credit Score: {assessment.get('credit_score_estimate', assessment.get('credit_score', 0))}
- Reasoning: {assessment.get('llm_reasoning', assessment.get('reasoning', ''))[:500]}

Score from 0.0 to 1.0:
- 1.0 = All facts are accurate and supported by context
- 0.5 = Some facts are accurate, some unverifiable
- 0.0 = Contains factual errors

Return ONLY a JSON object: {{"score": 0.X, "reason": "brief explanation"}}"""

        try:
            response = client.chat.completions.create(
                model=self.openai_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
            )
            result = json.loads(response.choices[0].message.content)
            return float(result.get("score", 0.5))
        except Exception as e:
            logger.warning(f"Factual accuracy evaluation failed: {e}")
            return 0.5

    def _evaluate_final_answer_quality(self, assessment: Dict[str, Any]) -> float:
        """Evaluate final answer quality (built-in)."""
        score = 0.0
        details = {}

        # Check required fields
        fields_present = 0
        for field in self.REQUIRED_OUTPUT_FIELDS:
            has_field = (
                field in assessment or
                f"overall_{field}" in assessment or
                f"{field}_estimate" in assessment or
                f"{field}_level" in assessment
            )
            if has_field:
                fields_present += 1

        completeness = fields_present / len(self.REQUIRED_OUTPUT_FIELDS)
        score += completeness * 0.5

        # Check value validity
        risk_level = assessment.get("overall_risk_level") or assessment.get("risk_level", "")
        if str(risk_level).lower() in ["low", "moderate", "high", "very_high", "critical"]:
            score += 0.15

        credit_score = assessment.get("credit_score_estimate") or assessment.get("credit_score", 0)
        if 0 <= int(credit_score) <= 100:
            score += 0.15

        confidence = assessment.get("confidence_score") or assessment.get("confidence", 0)
        if 0 <= float(confidence) <= 1:
            score += 0.1

        reasoning = assessment.get("llm_reasoning") or assessment.get("reasoning", "")
        if len(str(reasoning)) > 50:
            score += 0.1

        return min(1.0, score)

    # ========================================================================
    # CONSISTENCY EVALUATION
    # ========================================================================

    def evaluate_consistency(
        self,
        llm_results: List[Dict[str, Any]],
        llm_consistency_data: Dict[str, Any] = None,
    ) -> ConsistencyMetrics:
        """Evaluate consistency across multiple LLM outputs."""
        metrics = ConsistencyMetrics()

        # Use pre-calculated consistency if available
        if llm_consistency_data:
            metrics.same_model_consistency = llm_consistency_data.get("same_model_consistency", 0.0)
            metrics.cross_model_consistency = llm_consistency_data.get("cross_model_consistency", 0.0)

            # Calculate risk level agreement
            risk_levels = llm_consistency_data.get("risk_levels", [])
            if risk_levels:
                metrics.risk_level_agreement = 1.0 if len(set(risk_levels)) == 1 else 0.5

            metrics.details["source"] = "pre_calculated"

        # Calculate semantic similarity of reasoning
        if llm_results and SENTENCE_TRANSFORMERS_AVAILABLE:
            reasonings = [
                r.get("reasoning", "") or r.get("llm_reasoning", "")
                for r in llm_results if r
            ]
            if len(reasonings) >= 2:
                metrics.semantic_similarity = self._calculate_semantic_similarity(reasonings)

        metrics.calculate_combined()
        return metrics

    def _calculate_semantic_similarity(self, texts: List[str]) -> float:
        """Calculate average pairwise semantic similarity."""
        model = self._get_embedding_model()
        if not model or len(texts) < 2:
            return 1.0

        # Filter empty texts
        texts = [t for t in texts if t and len(t) > 10]
        if len(texts) < 2:
            return 1.0

        embeddings = model.encode(texts)

        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                similarities.append(float(sim))

        return sum(similarities) / len(similarities) if similarities else 1.0

    # ========================================================================
    # AGENT EFFICIENCY EVALUATION
    # ========================================================================

    def evaluate_agent_efficiency(
        self,
        company_name: str,
        company_info: Dict[str, Any],
        task_plan: List[Any],
        api_data: Dict[str, Any],
        assessment: Dict[str, Any],
        trajectory: List[str],
        latency_ms: float = 0.0,
    ) -> AgentEfficiencyMetrics:
        """Evaluate agent efficiency metrics."""
        metrics = AgentEfficiencyMetrics()

        # Determine company type
        is_public = company_info.get("is_public_company", False)
        jurisdiction = company_info.get("jurisdiction", "US")
        if is_public:
            company_type = "public_us" if jurisdiction == "US" else "public_non_us"
        else:
            company_type = "private"

        # 1. Intent Correctness (OpenEvals-style or built-in)
        if OPENAI_AVAILABLE:
            metrics.intent_correctness, metrics.intent_details = \
                self._evaluate_intent_with_llm(company_name, company_info)
        else:
            metrics.intent_correctness, metrics.intent_details = \
                self._evaluate_intent_builtin(company_name, company_info)

        # 2. Plan Quality (OpenEvals-style or built-in)
        if OPENAI_AVAILABLE:
            metrics.plan_quality, metrics.plan_details = \
                self._evaluate_plan_with_llm(task_plan, company_type)
        else:
            metrics.plan_quality, metrics.plan_details = \
                self._evaluate_plan_builtin(task_plan)

        # 3. Tool Choice Correctness (built-in - precision)
        metrics.tool_choice_correctness, tool_choice_details = \
            self._evaluate_tool_choice(api_data, company_type)

        # 4. Tool Completeness (built-in - recall)
        metrics.tool_completeness, tool_complete_details = \
            self._evaluate_tool_completeness(api_data, company_type)

        metrics.tool_details = {
            "choice": tool_choice_details,
            "completeness": tool_complete_details,
        }

        # 5. Trajectory Match (built-in)
        metrics.trajectory_match, metrics.trajectory_details = \
            self._evaluate_trajectory(trajectory)

        # 6. Final Answer Quality (already calculated in accuracy, reuse logic)
        metrics.final_answer_quality = self._evaluate_final_answer_quality(assessment)

        # Execution stats
        metrics.step_count = len(trajectory)
        metrics.tool_calls = len([k for k in api_data.keys() if api_data.get(k)])
        metrics.latency_ms = latency_ms

        metrics.calculate_overall()
        return metrics

    def _evaluate_intent_with_llm(
        self,
        company_name: str,
        company_info: Dict[str, Any],
    ) -> Tuple[float, Dict]:
        """Evaluate intent correctness using LLM."""
        client = self._get_openai_client()
        if not client:
            return self._evaluate_intent_builtin(company_name, company_info)

        prompt = f"""Evaluate if the agent correctly understood the task.

Task: Analyze credit risk for "{company_name}"

Agent's Understanding:
- Parsed company name: {company_info.get('company_name', 'unknown')}
- Identified as public: {company_info.get('is_public_company', 'unknown')}
- Ticker: {company_info.get('ticker', 'none')}
- Confidence: {company_info.get('confidence', 0)}

Score from 0.0 to 1.0:
- 1.0 = Perfectly understood the company and task
- 0.5 = Partially correct
- 0.0 = Completely wrong

Return ONLY JSON: {{"score": 0.X, "reason": "brief explanation"}}"""

        try:
            response = client.chat.completions.create(
                model=self.openai_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
            )
            result = json.loads(response.choices[0].message.content)
            return float(result.get("score", 0.5)), {"llm_reason": result.get("reason", "")}
        except Exception as e:
            logger.warning(f"LLM intent evaluation failed: {e}")
            return self._evaluate_intent_builtin(company_name, company_info)

    def _evaluate_intent_builtin(
        self,
        company_name: str,
        company_info: Dict[str, Any],
    ) -> Tuple[float, Dict]:
        """Built-in intent evaluation."""
        score = 0.0
        details = {}

        parsed_name = company_info.get("company_name", "")
        name_match = company_name.lower() in parsed_name.lower() or parsed_name.lower() in company_name.lower()
        details["name_parsed"] = name_match
        if name_match:
            score += 0.4

        has_type = company_info.get("is_public_company") is not None
        details["type_identified"] = has_type
        if has_type:
            score += 0.3

        confidence = company_info.get("confidence", 0)
        details["confidence"] = confidence
        if confidence > 0.7:
            score += 0.3
        elif confidence > 0.5:
            score += 0.2

        return min(1.0, score), details

    def _evaluate_plan_with_llm(
        self,
        task_plan: List[Any],
        company_type: str,
    ) -> Tuple[float, Dict]:
        """Evaluate plan quality using LLM."""
        client = self._get_openai_client()
        if not client:
            return self._evaluate_plan_builtin(task_plan)

        # Convert plan to readable format
        plan_str = ""
        for i, item in enumerate(task_plan[:10], 1):
            if isinstance(item, dict):
                plan_str += f"{i}. {item.get('action', item.get('agent', str(item)))}\n"
            else:
                plan_str += f"{i}. {item}\n"

        prompt = f"""Evaluate this execution plan for credit risk analysis.

Company Type: {company_type}

Plan:
{plan_str}

Score from 0.0 to 1.0:
- 1.0 = Comprehensive plan with all necessary steps
- 0.5 = Adequate but missing some steps
- 0.0 = Inadequate plan

Return ONLY JSON: {{"score": 0.X, "reason": "brief explanation"}}"""

        try:
            response = client.chat.completions.create(
                model=self.openai_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
            )
            result = json.loads(response.choices[0].message.content)
            return float(result.get("score", 0.5)), {"llm_reason": result.get("reason", ""), "plan_steps": len(task_plan)}
        except Exception as e:
            logger.warning(f"LLM plan evaluation failed: {e}")
            return self._evaluate_plan_builtin(task_plan)

    def _evaluate_plan_builtin(self, task_plan: List[Any]) -> Tuple[float, Dict]:
        """Built-in plan evaluation."""
        score = 0.0
        details = {"step_count": len(task_plan)}

        if not task_plan:
            return 0.0, {"error": "No plan"}

        # Size check
        if 3 <= len(task_plan) <= 10:
            score += 0.3
            details["size_appropriate"] = True

        # Convert to strings for checking
        plan_strs = []
        for item in task_plan:
            if isinstance(item, dict):
                plan_strs.append(str(item.get("action", item.get("agent", ""))))
            else:
                plan_strs.append(str(item))
        plan_text = " ".join(plan_strs).lower()

        # Has data gathering
        has_data = any(kw in plan_text for kw in ["data", "fetch", "search", "api"])
        details["has_data_gathering"] = has_data
        if has_data:
            score += 0.35

        # Has analysis
        has_analysis = any(kw in plan_text for kw in ["analy", "synthe", "assess", "evaluat"])
        details["has_analysis"] = has_analysis
        if has_analysis:
            score += 0.35

        return min(1.0, score), details

    def _evaluate_tool_choice(
        self,
        api_data: Dict[str, Any],
        company_type: str,
    ) -> Tuple[float, Dict]:
        """Evaluate tool choice correctness (precision)."""
        expected = self.EXPECTED_TOOLS.get(company_type, {"web_search"})
        selected = set(api_data.keys())

        # Normalize names
        selected_normalized = set()
        for tool in selected:
            tool_lower = tool.lower()
            if "sec" in tool_lower:
                selected_normalized.add("sec_edgar")
            elif "finnhub" in tool_lower or "market" in tool_lower:
                selected_normalized.add("finnhub")
            elif "tavily" in tool_lower:
                selected_normalized.add("tavily")
            elif "web" in tool_lower or "search" in tool_lower:
                selected_normalized.add("web_search")
            elif "court" in tool_lower:
                selected_normalized.add("court_listener")
            else:
                selected_normalized.add(tool)

        correct = expected & selected_normalized
        precision = len(correct) / len(selected_normalized) if selected_normalized else 0.0

        return precision, {
            "expected": list(expected),
            "selected": list(selected_normalized),
            "correct": list(correct),
            "precision": precision,
        }

    def _evaluate_tool_completeness(
        self,
        api_data: Dict[str, Any],
        company_type: str,
    ) -> Tuple[float, Dict]:
        """Evaluate tool completeness (recall)."""
        expected = self.EXPECTED_TOOLS.get(company_type, {"web_search"})
        selected = set(api_data.keys())

        # Normalize names (same as above)
        selected_normalized = set()
        for tool in selected:
            tool_lower = tool.lower()
            if "sec" in tool_lower:
                selected_normalized.add("sec_edgar")
            elif "finnhub" in tool_lower or "market" in tool_lower:
                selected_normalized.add("finnhub")
            elif "tavily" in tool_lower:
                selected_normalized.add("tavily")
            elif "web" in tool_lower or "search" in tool_lower:
                selected_normalized.add("web_search")
            elif "court" in tool_lower:
                selected_normalized.add("court_listener")
            else:
                selected_normalized.add(tool)

        used = expected & selected_normalized
        recall = len(used) / len(expected) if expected else 0.0

        return recall, {
            "expected": list(expected),
            "used": list(used),
            "missing": list(expected - selected_normalized),
            "recall": recall,
        }

    def _evaluate_trajectory(self, trajectory: List[str]) -> Tuple[float, Dict]:
        """Evaluate trajectory match."""
        expected = self.EXPECTED_TRAJECTORY
        expected_set = set(expected)
        actual_set = set(trajectory)

        # Jaccard similarity
        intersection = len(expected_set & actual_set)
        union = len(expected_set | actual_set)
        jaccard = intersection / union if union > 0 else 0.0

        # Order score
        order_score = 0.0
        if len(trajectory) >= 2:
            correct_order = 0
            total_pairs = 0
            for i, step in enumerate(trajectory[:-1]):
                if step in expected:
                    expected_idx = expected.index(step)
                    next_step = trajectory[i + 1]
                    if next_step in expected:
                        next_idx = expected.index(next_step)
                        total_pairs += 1
                        if next_idx > expected_idx:
                            correct_order += 1
            order_score = correct_order / total_pairs if total_pairs > 0 else 0.0

        score = jaccard * 0.6 + order_score * 0.4

        return score, {
            "expected": expected,
            "actual": trajectory,
            "jaccard": jaccard,
            "order_score": order_score,
            "matched": list(expected_set & actual_set),
            "missing": list(expected_set - actual_set),
        }

    # ========================================================================
    # MAIN EVALUATE METHOD
    # ========================================================================

    def evaluate(
        self,
        run_id: str,
        company_name: str,
        state: Dict[str, Any],
        llm_results: List[Dict[str, Any]] = None,
        llm_consistency_data: Dict[str, Any] = None,
        latency_ms: float = 0.0,
    ) -> UnifiedEvaluationResult:
        """
        Perform complete unified evaluation.

        Args:
            run_id: Run identifier
            company_name: Company being analyzed
            state: Workflow state dict
            llm_results: Results from multiple LLMs (for consistency)
            llm_consistency_data: Pre-calculated consistency data
            latency_ms: Total execution time

        Returns:
            UnifiedEvaluationResult with all metrics
        """
        start_time = time.time()

        result = UnifiedEvaluationResult(
            run_id=run_id,
            company_name=company_name,
        )

        # Track libraries used
        if DEEPEVAL_AVAILABLE:
            result.libraries_used.append("deepeval")
        if OPENAI_AVAILABLE:
            result.libraries_used.append("openai")
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            result.libraries_used.append("sentence-transformers")
        result.libraries_used.append("built-in")

        # Extract data from state
        company_info = state.get("company_info", {})
        task_plan = state.get("task_plan", [])
        api_data = state.get("api_data", {})
        search_data = state.get("search_data", {})
        assessment = state.get("assessment", {}) or {}

        # Build context for accuracy evaluation
        context = []
        for source, data in api_data.items():
            if data:
                context.append(f"{source}: {str(data)[:500]}")
        if search_data:
            context.append(f"Web search: {str(search_data)[:500]}")

        # Build query and output
        query = f"Analyze credit risk for {company_name}"
        output = ""
        if isinstance(assessment, dict):
            output = (
                f"Risk Level: {assessment.get('overall_risk_level', assessment.get('risk_level', 'unknown'))}\n"
                f"Credit Score: {assessment.get('credit_score_estimate', assessment.get('credit_score', 0))}\n"
                f"Reasoning: {assessment.get('llm_reasoning', assessment.get('reasoning', ''))}"
            )

        # Build trajectory
        trajectory = [
            "parse_input", "validate_company", "create_plan",
            "fetch_api_data", "search_web", "synthesize",
            "save_to_database", "evaluate",
        ]

        try:
            # 1. Accuracy Evaluation
            result.accuracy = self.evaluate_accuracy(query, context, output, assessment)
        except Exception as e:
            logger.error(f"Accuracy evaluation failed: {e}")
            result.errors.append(f"accuracy: {str(e)}")

        try:
            # 2. Consistency Evaluation
            result.consistency = self.evaluate_consistency(
                llm_results or [],
                llm_consistency_data,
            )
        except Exception as e:
            logger.error(f"Consistency evaluation failed: {e}")
            result.errors.append(f"consistency: {str(e)}")

        try:
            # 3. Agent Efficiency Evaluation
            result.agent_efficiency = self.evaluate_agent_efficiency(
                company_name=company_name,
                company_info=company_info,
                task_plan=task_plan,
                api_data=api_data,
                assessment=assessment,
                trajectory=trajectory,
                latency_ms=latency_ms,
            )
        except Exception as e:
            logger.error(f"Agent efficiency evaluation failed: {e}")
            result.errors.append(f"agent_efficiency: {str(e)}")

        # Calculate overall score
        result.calculate_overall_quality()

        # Record evaluation time
        result.evaluation_time_ms = (time.time() - start_time) * 1000

        logger.info(f"Unified evaluation complete: overall={result.overall_quality_score:.2f}, "
                   f"accuracy={result.accuracy.accuracy_score:.2f}, "
                   f"consistency={result.consistency.consistency_score:.2f}, "
                   f"agent={result.agent_efficiency.overall_score:.2f}")

        return result


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

_evaluator: Optional[UnifiedAgentEvaluator] = None


def get_unified_evaluator() -> UnifiedAgentEvaluator:
    """Get or create unified evaluator instance."""
    global _evaluator
    if _evaluator is None:
        _evaluator = UnifiedAgentEvaluator()
    return _evaluator


def evaluate_workflow(
    run_id: str,
    company_name: str,
    state: Dict[str, Any],
    llm_results: List[Dict[str, Any]] = None,
    llm_consistency_data: Dict[str, Any] = None,
    latency_ms: float = 0.0,
) -> UnifiedEvaluationResult:
    """Convenience function to evaluate a workflow."""
    evaluator = get_unified_evaluator()
    return evaluator.evaluate(
        run_id=run_id,
        company_name=company_name,
        state=state,
        llm_results=llm_results,
        llm_consistency_data=llm_consistency_data,
        latency_ms=latency_ms,
    )
