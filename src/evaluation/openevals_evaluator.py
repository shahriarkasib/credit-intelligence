"""
OpenEvals Integration for Credit Intelligence

OpenEvals (by LangChain) provides ready-made LLM-as-judge evaluators:
- Correctness: Is the output correct given expected output?
- Helpfulness: Is the output helpful for the user?
- Coherence: Is the output coherent and well-structured?
- Conciseness: Is the output concise without unnecessary info?
- Hallucination: Does the output contain hallucinations?

Install: pip install openevals langchain-openai
Docs: https://github.com/langchain-ai/openevals

Requires OpenAI API key (uses GPT-4o-mini by default).
"""

import os
import logging
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Import sheets logger for logging to dedicated sheets
try:
    from run_logging.sheets_logger import get_sheets_logger
    SHEETS_LOGGER_AVAILABLE = True
except ImportError:
    SHEETS_LOGGER_AVAILABLE = False

# Try to import OpenEvals
OPENEVALS_AVAILABLE = False

try:
    from openevals.llm import create_llm_as_judge
    from openevals.prompts import CORRECTNESS_PROMPT, HELPFULNESS_PROMPT
    OPENEVALS_AVAILABLE = True
except ImportError:
    logger.warning("OpenEvals not installed. Run: pip install openevals")
    create_llm_as_judge = None

# Try to import LangChain OpenAI for the judge model
try:
    from langchain_openai import ChatOpenAI
    LANGCHAIN_OPENAI_AVAILABLE = True
except ImportError:
    LANGCHAIN_OPENAI_AVAILABLE = False
    ChatOpenAI = None


@dataclass
class OpenEvalsResult:
    """Result from OpenEvals evaluation."""
    correctness: float = 0.0
    helpfulness: float = 0.0
    coherence: float = 0.0
    conciseness: float = 0.0
    relevance: float = 0.0
    overall_score: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    # Metadata for logging
    run_id: str = ""
    company_name: str = ""
    evaluation_time_ms: float = 0.0


class OpenEvalsEvaluator:
    """
    Evaluates credit assessments using OpenEvals (LangChain's evaluation framework).

    OpenEvals uses LLM-as-judge pattern with GPT-4o-mini by default.

    Usage:
        evaluator = OpenEvalsEvaluator()

        result = evaluator.evaluate_assessment(
            input_query="Analyze Google's credit risk",
            output="Google has moderate risk...",
            expected_output="Expected assessment...",  # Optional
            context=["SEC filings show...", "Market data indicates..."],
        )
        print(f"Score: {result.overall_score}")
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
    ):
        """
        Initialize OpenEvals evaluator.

        Args:
            model: OpenAI model to use as judge (default: gpt-4o-mini)
            temperature: Temperature for judge model (0.0 for deterministic)
        """
        self.model = model
        self.temperature = temperature
        self.evaluators_initialized = False
        self._judge_llm = None

        if not OPENEVALS_AVAILABLE:
            logger.error("OpenEvals not available. Install with: pip install openevals")
            return

        if not LANGCHAIN_OPENAI_AVAILABLE:
            logger.error("LangChain OpenAI not available. Install with: pip install langchain-openai")
            return

        # Check for OpenAI API key
        if not os.getenv("OPENAI_API_KEY"):
            logger.error("OPENAI_API_KEY not set - required for OpenEvals")
            return

        try:
            # Initialize judge LLM
            self._judge_llm = ChatOpenAI(
                model=model,
                temperature=temperature,
            )

            # Create evaluators
            self._correctness_evaluator = create_llm_as_judge(
                prompt=CORRECTNESS_PROMPT,
                model=self._judge_llm,
            )

            self._helpfulness_evaluator = create_llm_as_judge(
                prompt=HELPFULNESS_PROMPT,
                model=self._judge_llm,
            )

            # Custom prompts for additional criteria
            self._coherence_evaluator = create_llm_as_judge(
                prompt="""You are evaluating the coherence of a response.

Input: {input}
Output: {output}

Rate the coherence of the output on a scale of 0 to 1, where:
- 0: Completely incoherent, disorganized, contradictory
- 0.5: Somewhat coherent but with some issues
- 1: Highly coherent, well-structured, logical flow

Consider:
1. Is the information presented in a logical order?
2. Are there contradictions in the response?
3. Does each part connect well to the next?

Respond with just a number between 0 and 1.""",
                model=self._judge_llm,
            )

            self._relevance_evaluator = create_llm_as_judge(
                prompt="""You are evaluating the relevance of a credit risk assessment.

Input Query: {input}
Assessment Output: {output}
Context Data: {context}

Rate how relevant the assessment is to the input query on a scale of 0 to 1, where:
- 0: Completely irrelevant, doesn't address the query
- 0.5: Partially relevant but missing key aspects
- 1: Highly relevant, directly addresses all aspects of the query

Consider:
1. Does the assessment address the specific company mentioned?
2. Does it cover credit risk factors?
3. Is the assessment grounded in the provided context?

Respond with just a number between 0 and 1.""",
                model=self._judge_llm,
            )

            self.evaluators_initialized = True
            logger.info(f"OpenEvals initialized with {model}")

        except Exception as e:
            logger.error(f"Failed to initialize OpenEvals: {e}")

    def evaluate_assessment(
        self,
        input_query: str,
        output: str,
        expected_output: Optional[str] = None,
        context: Optional[List[str]] = None,
    ) -> OpenEvalsResult:
        """
        Evaluate a credit assessment using OpenEvals.

        Args:
            input_query: The original query (e.g., "Analyze Google's credit risk")
            output: The generated assessment text
            expected_output: Optional expected/reference output for correctness
            context: List of context strings (data sources used)

        Returns:
            OpenEvalsResult with all metric scores
        """
        if not OPENEVALS_AVAILABLE or not self.evaluators_initialized:
            return OpenEvalsResult(details={"error": "OpenEvals not available or not initialized"})

        results = {}
        context_str = "\n".join(context) if context else ""

        try:
            # Correctness (if expected output provided)
            if expected_output:
                correctness_result = self._correctness_evaluator.invoke({
                    "input": input_query,
                    "output": output,
                    "expected": expected_output,
                })
                results["correctness"] = self._parse_score(correctness_result)
            else:
                results["correctness"] = 0.0  # Can't evaluate without expected

            # Helpfulness
            helpfulness_result = self._helpfulness_evaluator.invoke({
                "input": input_query,
                "output": output,
            })
            results["helpfulness"] = self._parse_score(helpfulness_result)

            # Coherence
            coherence_result = self._coherence_evaluator.invoke({
                "input": input_query,
                "output": output,
            })
            results["coherence"] = self._parse_score(coherence_result)

            # Relevance (with context)
            relevance_result = self._relevance_evaluator.invoke({
                "input": input_query,
                "output": output,
                "context": context_str[:5000],  # Truncate context
            })
            results["relevance"] = self._parse_score(relevance_result)

            # Calculate overall score
            weights = {
                "correctness": 0.25 if expected_output else 0.0,
                "helpfulness": 0.30 if expected_output else 0.35,
                "coherence": 0.20 if expected_output else 0.30,
                "relevance": 0.25 if expected_output else 0.35,
            }

            overall = sum(results[k] * v for k, v in weights.items())

            return OpenEvalsResult(
                correctness=results.get("correctness", 0.0),
                helpfulness=results.get("helpfulness", 0.0),
                coherence=results.get("coherence", 0.0),
                relevance=results.get("relevance", 0.0),
                overall_score=overall,
                details={
                    "model": self.model,
                    "has_expected_output": expected_output is not None,
                    "context_length": len(context_str),
                }
            )

        except Exception as e:
            logger.error(f"OpenEvals evaluation failed: {e}")
            return OpenEvalsResult(details={"error": str(e)})

    def _parse_score(self, result: Any) -> float:
        """Parse score from evaluator result."""
        try:
            if hasattr(result, 'score'):
                return float(result.score)
            elif hasattr(result, 'content'):
                # Try to extract number from content
                content = str(result.content).strip()
                return float(content)
            elif isinstance(result, (int, float)):
                return float(result)
            elif isinstance(result, str):
                return float(result.strip())
            elif isinstance(result, dict):
                return float(result.get('score', result.get('value', 0.0)))
            else:
                logger.warning(f"Unknown result type: {type(result)}")
                return 0.0
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to parse score: {e}")
            return 0.0

    def evaluate_from_state(
        self,
        state: Dict[str, Any],
        run_id: str = "",
        log_to_sheets: bool = True,
    ) -> OpenEvalsResult:
        """
        Evaluate from workflow state.

        Args:
            state: The workflow state dict
            run_id: Optional run identifier for logging
            log_to_sheets: Whether to log results to Google Sheets

        Returns:
            OpenEvalsResult
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
            output=assessment_text,
            context=context,
        )

        # Add metadata
        result.run_id = run_id
        result.company_name = company_name
        result.evaluation_time_ms = (time.time() - start_time) * 1000

        # Log to Google Sheets
        if log_to_sheets and SHEETS_LOGGER_AVAILABLE and result.overall_score > 0:
            try:
                sheets_logger = get_sheets_logger()
                # Use a generic evaluation log method
                sheets_logger.log_evaluation(
                    run_id=run_id,
                    company_name=company_name,
                    evaluator_type="openevals",
                    metric_name="overall_score",
                    metric_value=result.overall_score,
                    details={
                        "correctness": result.correctness,
                        "helpfulness": result.helpfulness,
                        "coherence": result.coherence,
                        "relevance": result.relevance,
                        "model": self.model,
                    },
                )
                logger.debug(f"Logged OpenEvals metrics to sheets for run: {run_id}")
            except Exception as e:
                logger.warning(f"Failed to log OpenEvals metrics to sheets: {e}")

        return result


# Singleton instance
_openevals_evaluator: Optional[OpenEvalsEvaluator] = None


def get_openevals_evaluator(model: str = "gpt-4o-mini") -> OpenEvalsEvaluator:
    """
    Get or create OpenEvals evaluator.

    Args:
        model: OpenAI model to use as judge
    """
    global _openevals_evaluator
    if _openevals_evaluator is None:
        _openevals_evaluator = OpenEvalsEvaluator(model=model)
    return _openevals_evaluator


def evaluate_with_openevals(
    state: Dict[str, Any],
    run_id: str = "",
    log_to_sheets: bool = True,
) -> OpenEvalsResult:
    """
    Convenience function to evaluate workflow state with OpenEvals.

    Args:
        state: Workflow state dict
        run_id: Optional run identifier for logging
        log_to_sheets: Whether to log results to Google Sheets

    Returns:
        OpenEvalsResult with all metrics
    """
    evaluator = get_openevals_evaluator()
    return evaluator.evaluate_from_state(state, run_id=run_id, log_to_sheets=log_to_sheets)
