"""
DeepEval Integration for Credit Intelligence

DeepEval provides LLM-powered evaluation metrics:
- Hallucination detection
- Answer relevancy
- Faithfulness
- Contextual relevancy
- Bias detection

Install: pip install deepeval litellm
Docs: https://docs.confident-ai.com/

Now supports Groq (free!) via LiteLLM integration.
"""

import os
import logging
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Import sheets logger for logging to dedicated sheets
try:
    from run_logging.sheets_logger import get_sheets_logger
    SHEETS_LOGGER_AVAILABLE = True
except ImportError:
    SHEETS_LOGGER_AVAILABLE = False

# Try to import DeepEval
DEEPEVAL_AVAILABLE = False
LITELLM_AVAILABLE = False

try:
    from deepeval import evaluate
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        FaithfulnessMetric,
        ContextualRelevancyMetric,
        HallucinationMetric,
        BiasMetric,
    )
    from deepeval.test_case import LLMTestCase
    DEEPEVAL_AVAILABLE = True
except ImportError:
    logger.warning("DeepEval not installed. Run: pip install deepeval")

# Try to import LiteLLM for Groq support
try:
    from deepeval.models import DeepEvalBaseLLM
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:
    logger.warning("LiteLLM not installed. Run: pip install litellm")


class GroqModel(DeepEvalBaseLLM):
    """Custom Groq model for DeepEval using LiteLLM."""

    def __init__(self, model: str = "groq/llama-3.3-70b-versatile"):
        self.model_name = model

    def load_model(self):
        return self.model_name

    def generate(self, prompt: str) -> str:
        """Generate response using Groq via LiteLLM."""
        try:
            response = litellm.completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,  # Deterministic for evaluation
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq generation failed: {e}")
            return ""

    async def a_generate(self, prompt: str) -> str:
        """Async generate - falls back to sync for now."""
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return self.model_name


@dataclass
class DeepEvalResult:
    """Result from DeepEval evaluation."""
    answer_relevancy: float = 0.0
    faithfulness: float = 0.0
    hallucination: float = 0.0  # Lower is better
    contextual_relevancy: float = 0.0
    bias: float = 0.0  # Lower is better
    toxicity: float = 0.0  # Lower is better
    overall_score: float = 0.0
    details: Dict[str, Any] = None
    # Metadata for logging
    run_id: str = ""
    company_name: str = ""
    evaluation_time_ms: float = 0.0


class DeepEvalEvaluator:
    """
    Evaluates credit assessments using DeepEval metrics.

    Now supports Groq (free!) via LiteLLM - no OpenAI key required!

    Usage:
        # Use Groq (default, free!)
        evaluator = DeepEvalEvaluator(provider="groq")

        # Or use OpenAI if you have a key
        evaluator = DeepEvalEvaluator(provider="openai", model="gpt-4")

        result = evaluator.evaluate_assessment(
            input_query="Analyze Google's credit risk",
            context=["SEC filings show...", "Market data indicates..."],
            assessment="Google has moderate risk...",
        )
        print(f"Score: {result.overall_score}")
    """

    def __init__(
        self,
        provider: str = "groq",  # "groq" (free!) or "openai"
        model: str = None,
    ):
        """
        Initialize DeepEval evaluator.

        Args:
            provider: LLM provider - "groq" (free!) or "openai"
            model: Model name. Defaults based on provider:
                   - groq: "groq/llama-3.3-70b-versatile"
                   - openai: "gpt-4"
        """
        self.provider = provider
        self.metrics_initialized = False

        # Set default model based on provider
        if model is None:
            if provider == "groq":
                model = "groq/llama-3.3-70b-versatile"
            else:
                model = "gpt-4"

        self.model = model

        if not DEEPEVAL_AVAILABLE:
            logger.error("DeepEval not available. Install with: pip install deepeval")
            return

        # Initialize the LLM model
        try:
            if provider == "groq":
                if not LITELLM_AVAILABLE:
                    logger.error("LiteLLM not available for Groq. Install with: pip install litellm")
                    return

                # Check for Groq API key
                if not os.getenv("GROQ_API_KEY"):
                    logger.error("GROQ_API_KEY not set")
                    return

                self.llm_model = GroqModel(model=model)
                logger.info(f"Using Groq model: {model}")
            else:
                # Use OpenAI (default DeepEval behavior)
                if not os.getenv("OPENAI_API_KEY"):
                    logger.error("OPENAI_API_KEY not set for OpenAI provider")
                    return
                self.llm_model = model  # DeepEval uses string for OpenAI
                logger.info(f"Using OpenAI model: {model}")

            # Initialize metrics with the model
            self.answer_relevancy = AnswerRelevancyMetric(
                threshold=0.7,
                model=self.llm_model,
            )
            self.faithfulness = FaithfulnessMetric(
                threshold=0.7,
                model=self.llm_model,
            )
            self.hallucination = HallucinationMetric(
                threshold=0.5,  # Lower threshold = stricter
                model=self.llm_model,
            )
            self.contextual_relevancy = ContextualRelevancyMetric(
                threshold=0.7,
                model=self.llm_model,
            )
            self.bias = BiasMetric(
                threshold=0.5,
                model=self.llm_model,
            )
            self.metrics_initialized = True
            logger.info(f"DeepEval metrics initialized with {provider}")
        except Exception as e:
            logger.error(f"Failed to initialize DeepEval metrics: {e}")

    def evaluate_assessment(
        self,
        input_query: str,
        context: List[str],
        assessment: str,
        expected_output: Optional[str] = None,
    ) -> DeepEvalResult:
        """
        Evaluate a credit assessment using DeepEval metrics.

        Args:
            input_query: The original query (e.g., "Analyze Google's credit risk")
            context: List of context strings (data sources used)
            assessment: The generated assessment text
            expected_output: Optional expected/reference output

        Returns:
            DeepEvalResult with all metric scores
        """
        if not DEEPEVAL_AVAILABLE or not self.metrics_initialized:
            return DeepEvalResult(details={"error": "DeepEval not available or not initialized"})

        try:
            # Create test case
            test_case = LLMTestCase(
                input=input_query,
                actual_output=assessment,
                expected_output=expected_output,
                context=context,
                retrieval_context=context,  # For faithfulness
            )

            # Run individual metrics
            results = {}

            # Answer Relevancy: Is the answer relevant to the question?
            logger.debug("Running answer_relevancy metric...")
            self.answer_relevancy.measure(test_case)
            results["answer_relevancy"] = self.answer_relevancy.score or 0.0

            # Faithfulness: Is the answer grounded in the context?
            logger.debug("Running faithfulness metric...")
            self.faithfulness.measure(test_case)
            results["faithfulness"] = self.faithfulness.score or 0.0

            # Hallucination: Does the answer contain hallucinations?
            logger.debug("Running hallucination metric...")
            self.hallucination.measure(test_case)
            results["hallucination"] = self.hallucination.score or 0.0

            # Contextual Relevancy: Is the context relevant?
            logger.debug("Running contextual_relevancy metric...")
            self.contextual_relevancy.measure(test_case)
            results["contextual_relevancy"] = self.contextual_relevancy.score or 0.0

            # Bias: Is there bias in the answer?
            logger.debug("Running bias metric...")
            self.bias.measure(test_case)
            results["bias"] = self.bias.score or 0.0

            # Calculate overall score
            # Higher is better for relevancy/faithfulness, lower for hallucination/bias
            overall = (
                results["answer_relevancy"] * 0.25 +
                results["faithfulness"] * 0.30 +
                (1 - results["hallucination"]) * 0.25 +
                results["contextual_relevancy"] * 0.10 +
                (1 - results["bias"]) * 0.10
            )

            return DeepEvalResult(
                answer_relevancy=results["answer_relevancy"],
                faithfulness=results["faithfulness"],
                hallucination=results["hallucination"],
                contextual_relevancy=results["contextual_relevancy"],
                bias=results["bias"],
                overall_score=overall,
                details={
                    "model": self.model,
                    "provider": self.provider,
                    "reasons": {
                        "answer_relevancy": getattr(self.answer_relevancy, 'reason', ''),
                        "faithfulness": getattr(self.faithfulness, 'reason', ''),
                        "hallucination": getattr(self.hallucination, 'reason', ''),
                    }
                }
            )

        except Exception as e:
            logger.error(f"DeepEval evaluation failed: {e}")
            return DeepEvalResult(details={"error": str(e)})

    def evaluate_from_state(
        self,
        state: Dict[str, Any],
        run_id: str = "",
        log_to_sheets: bool = True,
    ) -> DeepEvalResult:
        """
        Evaluate from workflow state.

        Args:
            state: The workflow state dict
            run_id: Optional run identifier for logging
            log_to_sheets: Whether to log results to Google Sheets

        Returns:
            DeepEvalResult
        """
        start_time = time.time()
        company_name = state.get("company_name", "Unknown")
        assessment = state.get("assessment", {})
        api_data = state.get("api_data", {})
        search_data = state.get("search_data", {})

        # Build context from data sources
        context = []
        for source, data in api_data.items():
            if data:
                context.append(f"{source}: {str(data)[:500]}")
        if search_data:
            context.append(f"Web search: {str(search_data)[:500]}")

        # Build assessment text
        assessment_text = ""
        if isinstance(assessment, dict):
            assessment_text = (
                f"Risk Level: {assessment.get('overall_risk_level', 'unknown')}\n"
                f"Credit Score: {assessment.get('credit_score_estimate', 0)}\n"
                f"Reasoning: {assessment.get('llm_reasoning', '')}\n"
            )
        else:
            assessment_text = str(assessment)

        input_query = f"Analyze credit risk for {company_name}"

        result = self.evaluate_assessment(
            input_query=input_query,
            context=context,
            assessment=assessment_text,
        )

        # Add metadata
        result.run_id = run_id
        result.company_name = company_name
        result.evaluation_time_ms = (time.time() - start_time) * 1000

        # Log to dedicated sheets
        if log_to_sheets and SHEETS_LOGGER_AVAILABLE and result.overall_score > 0:
            try:
                sheets_logger = get_sheets_logger()
                sheets_logger.log_deepeval_metrics(
                    run_id=run_id,
                    company_name=company_name,
                    model_used=self.model,
                    node="evaluate",
                    node_type="agent",
                    agent_name="deepeval_evaluator",
                    # Core metrics
                    answer_relevancy=result.answer_relevancy,
                    faithfulness=result.faithfulness,
                    hallucination=result.hallucination,
                    contextual_relevancy=result.contextual_relevancy,
                    bias=result.bias,
                    toxicity=result.toxicity,
                    overall_score=result.overall_score,
                    # Reasoning
                    answer_relevancy_reason=result.details.get("reasons", {}).get("answer_relevancy", "") if result.details else "",
                    faithfulness_reason=result.details.get("reasons", {}).get("faithfulness", "") if result.details else "",
                    hallucination_reason=result.details.get("reasons", {}).get("hallucination", "") if result.details else "",
                    # Input/output
                    input_query=input_query,
                    context_summary=str(context)[:5000],
                    assessment_summary=assessment_text[:5000],
                    # Metadata
                    evaluation_model=self.model,
                    evaluation_time_ms=result.evaluation_time_ms,
                    status="ok",
                )
                logger.debug(f"Logged DeepEval metrics to sheets for run: {run_id}")
            except Exception as e:
                logger.warning(f"Failed to log DeepEval metrics to sheets: {e}")

        return result


# Singleton instance
_deepeval_evaluator: Optional[DeepEvalEvaluator] = None


def get_deepeval_evaluator(provider: str = "groq", model: str = None) -> DeepEvalEvaluator:
    """
    Get or create DeepEval evaluator.

    Args:
        provider: "groq" (free!) or "openai"
        model: Model name (defaults based on provider)
    """
    global _deepeval_evaluator
    if _deepeval_evaluator is None:
        _deepeval_evaluator = DeepEvalEvaluator(provider=provider, model=model)
    return _deepeval_evaluator


def evaluate_with_deepeval(
    state: Dict[str, Any],
    run_id: str = "",
    log_to_sheets: bool = True,
    provider: str = "groq",
) -> DeepEvalResult:
    """
    Convenience function to evaluate workflow state with DeepEval.

    Args:
        state: Workflow state dict
        run_id: Optional run identifier for logging
        log_to_sheets: Whether to log results to Google Sheets
        provider: "groq" (free!) or "openai"

    Returns:
        DeepEvalResult with all metrics
    """
    evaluator = get_deepeval_evaluator(provider=provider)
    return evaluator.evaluate_from_state(state, run_id=run_id, log_to_sheets=log_to_sheets)
