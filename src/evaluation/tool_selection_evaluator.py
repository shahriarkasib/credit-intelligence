"""Tool Selection Evaluator - Evaluates whether LLM chose the right tools."""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

# Import LangChain ChatGroq for LLM-as-judge evaluation
try:
    from config.langchain_llm import get_chat_llm, get_chat_groq, is_langchain_groq_available
    from config.langchain_callbacks import CostTrackerCallback
    from langchain_core.messages import HumanMessage
    LANGCHAIN_GROQ_AVAILABLE = is_langchain_groq_available()
except ImportError:
    LANGCHAIN_GROQ_AVAILABLE = False

# Import prompts
try:
    from config.prompts import get_prompt_text
    PROMPTS_AVAILABLE = True
except ImportError:
    PROMPTS_AVAILABLE = False

# Import cost tracker
try:
    from config.cost_tracker import get_cost_tracker
    COST_TRACKER_AVAILABLE = True
except ImportError:
    COST_TRACKER_AVAILABLE = False


@dataclass
class ToolSelectionResult:
    """Result of tool selection evaluation."""
    company_name: str
    expected_tools: List[str]
    selected_tools: List[str]
    precision: float  # Of selected tools, how many were correct?
    recall: float  # Of expected tools, how many were selected?
    f1_score: float
    is_correct: bool
    reasoning_quality: float  # 0-1 score for reasoning
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "company_name": self.company_name,
            "expected_tools": self.expected_tools,
            "selected_tools": self.selected_tools,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "is_correct": self.is_correct,
            "reasoning_quality": self.reasoning_quality,
            "details": self.details,
            "timestamp": self.timestamp,
        }


class ToolSelectionEvaluator:
    """
    Evaluates tool selection accuracy.

    Measures:
    - Precision: Of selected tools, how many were appropriate?
    - Recall: Of appropriate tools, how many were selected?
    - F1 Score: Harmonic mean of precision and recall
    - Reasoning Quality: How good was the reasoning?
    """

    # Ground truth: what tools SHOULD be selected for different company types
    # Use actual tool names for consistency across all sheets
    EXPECTED_TOOLS = {
        # Public US companies - SEC + Finnhub + CourtListener + Search
        "public_us": ["fetch_sec_edgar", "fetch_finnhub", "fetch_court_listener", "web_search"],
        # Public non-US - Finnhub + Search
        "public_non_us": ["fetch_finnhub", "web_search"],
        # Private companies - Web Search + CourtListener
        "private": ["web_search", "fetch_court_listener"],
        # Unknown - Enhanced Web Search to gather info
        "unknown": ["web_search_enhanced"],
    }

    # Known public companies (simplified - in production, this would be a database)
    PUBLIC_US_COMPANIES = {
        "apple", "microsoft", "google", "alphabet", "amazon", "meta", "facebook",
        "nvidia", "tesla", "berkshire", "jpmorgan", "johnson & johnson",
        "visa", "mastercard", "walmart", "procter & gamble", "coca-cola",
        "bank of america", "exxon", "chevron", "pfizer", "home depot",
        "disney", "intel", "cisco", "verizon", "at&t", "boeing", "nike",
        "mcdonald's", "starbucks", "salesforce", "adobe", "netflix",
        "paypal", "oracle", "ibm", "general electric", "goldman sachs",
    }

    PUBLIC_NON_US_COMPANIES = {
        "samsung", "toyota", "nestle", "lvmh", "shell", "bp", "hsbc",
        "novartis", "roche", "unilever", "alibaba", "tencent", "sony",
        "siemens", "volkswagen", "bmw", "mercedes", "louis vuitton",
        "total", "bnp paribas", "softbank", "mitsubishi", "honda",
    }

    def __init__(self):
        self.evaluations: List[ToolSelectionResult] = []

    def classify_company(self, company_name: str, context: Dict[str, Any] = None) -> str:
        """
        Classify company type based on context and name.

        Priority order (most reliable to least reliable):
        1. Explicit context flags (is_public, is_private, jurisdiction)
        2. Ticker symbol presence in context
        3. Known company name matching (exact word boundaries)
        4. Suffix indicators (as fallback only)

        Args:
            company_name: Name of the company
            context: Optional context dict with is_public, ticker, jurisdiction, etc.

        Returns:
            Company type: "public_us", "public_non_us", "private", or "unknown"

        Note: Threshold for known company matching uses exact word boundary matching
        to avoid false positives (e.g., "meta" matching "metadata company").
        """
        context = context or {}
        name_lower = company_name.lower()

        # Priority 1: Use explicit context signals (most reliable)
        if context.get("is_public") is True:
            jurisdiction = context.get("jurisdiction", "US").upper()
            if jurisdiction == "US":
                return "public_us"
            else:
                return "public_non_us"

        if context.get("is_private") is True:
            return "private"

        # Priority 2: Ticker symbol indicates public company
        if context.get("ticker"):
            # Has ticker, likely public - check jurisdiction
            jurisdiction = context.get("jurisdiction", "US").upper()
            if jurisdiction == "US":
                return "public_us"
            else:
                return "public_non_us"

        # Priority 3: Check known companies with word boundary matching
        # This prevents false positives like "meta" matching "metadata inc"
        import re

        for known in self.PUBLIC_US_COMPANIES:
            # Use word boundary matching for reliability
            # Match "apple" in "Apple Inc" but not "pineapple inc"
            pattern = r'\b' + re.escape(known) + r'\b'
            if re.search(pattern, name_lower):
                return "public_us"

        for known in self.PUBLIC_NON_US_COMPANIES:
            pattern = r'\b' + re.escape(known) + r'\b'
            if re.search(pattern, name_lower):
                return "public_non_us"

        # Priority 4: Check SEC EDGAR CIK presence in context
        if context.get("cik") or context.get("has_sec_filings"):
            return "public_us"

        # Priority 5: Suffix indicators (least reliable - return "unknown" for safety)
        # These suffixes are common for both public and private companies
        # so we return "unknown" to trigger web search for determination
        public_suffixes = ["inc", "inc.", "corp", "corp.", "corporation", "plc", "n.v."]
        private_suffixes = ["llc", "l.l.c.", "llp", "gmbh", "pte", "pvt"]

        if any(name_lower.endswith(f" {suffix}") for suffix in private_suffixes):
            return "private"

        if any(name_lower.endswith(f" {suffix}") for suffix in public_suffixes):
            # Could be public or private - needs verification
            return "unknown"

        # Default: unknown for safety (will trigger web search)
        return "unknown"

    def get_expected_tools(self, company_name: str, context: Dict[str, Any] = None) -> List[str]:
        """
        Get expected tools for a company.

        Uses classify_company() with context for robust classification,
        then returns the appropriate tool set.

        Args:
            company_name: Name of the company
            context: Optional context with is_public, ticker, jurisdiction, etc.

        Returns:
            List of expected tool names for this company type
        """
        context = context or {}

        # Use the robust classify_company method with full context
        company_type = self.classify_company(company_name, context)
        return self.EXPECTED_TOOLS.get(company_type, self.EXPECTED_TOOLS["unknown"])

    def evaluate(
        self,
        company_name: str,
        selected_tools: List[str],
        selection_reasoning: Dict[str, Any] = None,
        context: Dict[str, Any] = None,
    ) -> ToolSelectionResult:
        """
        Evaluate tool selection for a company.

        Args:
            company_name: Name of the company
            selected_tools: Tools the LLM chose
            selection_reasoning: LLM's reasoning for selection
            context: Additional context (is_public, ticker, jurisdiction, etc.)

        Returns:
            ToolSelectionResult with evaluation metrics

        Evaluation Metrics:
            - precision: Of selected tools, how many were appropriate? (0.0-1.0)
            - recall: Of appropriate tools, how many were selected? (0.0-1.0)
            - f1_score: Harmonic mean of precision and recall (0.0-1.0)

        Thresholds for is_correct:
            - recall >= 0.5: At least half of expected tools must be selected
            - precision >= 0.5: At least half of selected tools must be correct

            These thresholds allow flexibility (e.g., selecting extra tools for
            thoroughness) while ensuring core tools aren't missed.

            Examples:
              - Expected: [SEC, Finnhub], Selected: [SEC, Finnhub, WebSearch]
                precision=0.67, recall=1.0, is_correct=True (extra tool OK)
              - Expected: [SEC, Finnhub], Selected: [SEC]
                precision=1.0, recall=0.5, is_correct=True (borderline)
              - Expected: [SEC, Finnhub], Selected: [WebSearch]
                precision=0.0, recall=0.0, is_correct=False (wrong tools)
        """
        expected_tools = self.get_expected_tools(company_name, context)

        # Calculate precision and recall
        expected_set = set(expected_tools)
        selected_set = set(selected_tools)

        true_positives = len(expected_set & selected_set)
        false_positives = len(selected_set - expected_set)
        false_negatives = len(expected_set - selected_set)

        precision = true_positives / len(selected_set) if selected_set else 0.0
        recall = true_positives / len(expected_set) if expected_set else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Evaluate reasoning quality
        reasoning_quality = self._evaluate_reasoning(selection_reasoning)

        # Is it correct? We allow some flexibility
        is_correct = (
            recall >= 0.5  # At least half of expected tools selected
            and precision >= 0.5  # At least half of selected tools are correct
        )

        result = ToolSelectionResult(
            company_name=company_name,
            expected_tools=expected_tools,
            selected_tools=selected_tools,
            precision=precision,
            recall=recall,
            f1_score=f1,
            is_correct=is_correct,
            reasoning_quality=reasoning_quality,
            details={
                "true_positives": list(expected_set & selected_set),
                "false_positives": list(selected_set - expected_set),
                "false_negatives": list(expected_set - selected_set),
                "company_type": self.classify_company(company_name, context),
            },
        )

        self.evaluations.append(result)
        return result

    def _evaluate_reasoning(self, reasoning: Dict[str, Any]) -> float:
        """
        Evaluate quality of tool selection reasoning.

        Scoring Criteria (max_score = 3.0, normalized to 0.0-1.0):

        1. Company Analysis (1.0 points):
           - Has company_analysis section in reasoning
           - Bonus +0.5 if explicitly determines is_likely_public

        2. Tool Justification (0.0-1.0 points):
           - Each tool should have a "reason" field
           - Score = tools_with_reasons / total_tools
           - Encourages explaining why each tool was selected

        3. Execution Order (0.5 points):
           - Has execution_order_reasoning field
           - Explains why tools should run in that order

        Final score = min(1.0, total_points / 3.0)

        Threshold Interpretation:
            - 0.8-1.0: Excellent reasoning (full analysis + all tools justified)
            - 0.5-0.8: Good reasoning (most criteria met)
            - 0.3-0.5: Basic reasoning (minimal justification)
            - 0.0-0.3: Poor reasoning (missing key elements)

        Args:
            reasoning: The tool selection reasoning dict from LLM

        Returns:
            Normalized score from 0.0 to 1.0
        """
        if not reasoning:
            return 0.0

        score = 0.0
        max_score = 3.0

        # Criterion 1: Company analysis (1.0 + 0.5 bonus)
        company_analysis = reasoning.get("company_analysis", {})
        if company_analysis:
            score += 1.0
            # Bonus for explicit public/private determination
            if "is_likely_public" in company_analysis:
                score += 0.5

        # Criterion 2: Tool justification (0.0-1.0)
        tools_to_use = reasoning.get("tools_to_use", [])
        if tools_to_use:
            tools_with_reasons = sum(1 for t in tools_to_use if t.get("reason"))
            if tools_with_reasons > 0:
                score += min(1.0, tools_with_reasons / len(tools_to_use))

        # Criterion 3: Execution order reasoning (0.5)
        if reasoning.get("execution_order_reasoning"):
            score += 0.5

        return min(1.0, score / max_score)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluations."""
        if not self.evaluations:
            return {"total": 0}

        total = len(self.evaluations)
        correct = sum(1 for e in self.evaluations if e.is_correct)
        avg_precision = sum(e.precision for e in self.evaluations) / total
        avg_recall = sum(e.recall for e in self.evaluations) / total
        avg_f1 = sum(e.f1_score for e in self.evaluations) / total
        avg_reasoning = sum(e.reasoning_quality for e in self.evaluations) / total

        return {
            "total": total,
            "correct": correct,
            "accuracy": correct / total,
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "avg_f1": avg_f1,
            "avg_reasoning_quality": avg_reasoning,
        }

    def evaluate_with_llm(
        self,
        company_name: str,
        selected_tools: List[str],
        tool_selection_reasoning: Dict[str, Any] = None,
        actual_data_results: Dict[str, Any] = None,
        model: str = "llama-3.3-70b-versatile",
    ) -> Tuple[ToolSelectionResult, Dict[str, Any]]:
        """
        Use LLM-as-judge to evaluate tool selection quality.

        Instead of comparing against hardcoded expected tools, the LLM evaluates:
        - Was the company type correctly identified?
        - Were the selected tools appropriate?
        - Were any important tools missed?
        - Were any unnecessary tools selected?

        Args:
            company_name: Name of the company
            selected_tools: Tools that were selected
            tool_selection_reasoning: The reasoning provided for tool selection
            actual_data_results: Results from running the tools (optional)
            model: LLM model to use for evaluation

        Returns:
            Tuple of (ToolSelectionResult, llm_metrics)
        """
        if not LANGCHAIN_GROQ_AVAILABLE:
            logger.warning("LLM not available for evaluation, falling back to rule-based")
            return self.evaluate(company_name, selected_tools, tool_selection_reasoning), {}

        start_time = time.time()

        # Build prompt
        if PROMPTS_AVAILABLE:
            try:
                system_prompt, user_prompt = get_prompt_text(
                    "tool_selection_evaluation",
                    company_name=company_name,
                    selected_tools=json.dumps(selected_tools, indent=2),
                    selection_reasoning=json.dumps(tool_selection_reasoning or {}, indent=2),
                    data_results=json.dumps(actual_data_results or {}, indent=2, default=str)[:2000],  # Truncate
                )
                prompt = f"{system_prompt}\n\n{user_prompt}"
            except Exception as e:
                logger.warning(f"Failed to get prompt from registry: {e}")
                prompt = self._get_fallback_evaluation_prompt(
                    company_name, selected_tools, tool_selection_reasoning, actual_data_results
                )
        else:
            prompt = self._get_fallback_evaluation_prompt(
                company_name, selected_tools, tool_selection_reasoning, actual_data_results
            )

        # Setup callbacks for cost tracking
        callbacks = []
        if COST_TRACKER_AVAILABLE:
            try:
                tracker = get_cost_tracker()
                callbacks.append(CostTrackerCallback(tracker=tracker, call_type="tool_selection_evaluation"))
            except Exception:
                pass

        # Call LLM
        try:
            llm = get_chat_llm(model=model, temperature=0.1, callbacks=callbacks)
            if not llm:
                raise RuntimeError("Failed to create LLM instance")

            response = llm.invoke([HumanMessage(content=prompt)])
            execution_time_ms = (time.time() - start_time) * 1000

            # Extract metrics
            usage_metadata = getattr(response, 'usage_metadata', {}) or {}
            llm_metrics = {
                "model": model,
                "execution_time_ms": round(execution_time_ms, 2),
                "prompt_tokens": usage_metadata.get('input_tokens', 0),
                "completion_tokens": usage_metadata.get('output_tokens', 0),
                "total_tokens": usage_metadata.get('total_tokens', 0),
            }

            # Parse response
            evaluation = self._parse_llm_evaluation(response.content)

            # Build result - use our predefined expected_tools for consistency
            # Don't rely on LLM to define expected tools - use our EXPECTED_TOOLS
            expected_tools = self.get_expected_tools(company_name)

            result = ToolSelectionResult(
                company_name=company_name,
                expected_tools=expected_tools,
                selected_tools=selected_tools,
                precision=evaluation.get("precision", 0.0),
                recall=evaluation.get("recall", 0.0),
                f1_score=evaluation.get("f1_score", 0.0),
                is_correct=evaluation.get("overall_quality", "poor") in ["excellent", "good"],
                reasoning_quality=self._quality_to_score(evaluation.get("overall_quality", "poor")),
                details={
                    "llm_evaluation": evaluation,
                    "company_type_correct": evaluation.get("company_type_correct", False),
                    "company_type_reasoning": evaluation.get("company_type_reasoning", ""),
                    "appropriate_tools": evaluation.get("appropriate_tools", []),
                    "missing_tools": evaluation.get("missing_tools", []),
                    "unnecessary_tools": evaluation.get("unnecessary_tools", []),
                    "suggestions": evaluation.get("suggestions", []),
                    "execution_order_quality": evaluation.get("execution_order_quality", "unknown"),
                    "evaluation_reasoning": evaluation.get("reasoning", ""),
                },
            )

            self.evaluations.append(result)
            return result, llm_metrics

        except Exception as e:
            logger.error(f"LLM evaluation failed: {e}")
            # Fall back to rule-based evaluation
            return self.evaluate(company_name, selected_tools, tool_selection_reasoning), {}

    def _get_fallback_evaluation_prompt(
        self,
        company_name: str,
        selected_tools: List[str],
        tool_selection_reasoning: Dict[str, Any],
        actual_data_results: Dict[str, Any],
    ) -> str:
        """Generate fallback evaluation prompt if registry prompt unavailable."""
        return f"""Evaluate this tool selection for credit assessment.

Company: {company_name}
Selected Tools: {json.dumps(selected_tools)}
Selection Reasoning: {json.dumps(tool_selection_reasoning or {})}
Data Results: {json.dumps(actual_data_results or {}, default=str)[:1000]}

Evaluate:
1. Was the company type (public/private) correctly identified?
2. Were the selected tools appropriate for this company?
3. Were any important tools missed?
4. Were any unnecessary tools selected?

Return JSON:
{{
    "company_type_correct": true/false,
    "company_type_reasoning": "explanation",
    "appropriate_tools": ["list"],
    "missing_tools": ["list"],
    "unnecessary_tools": ["list"],
    "precision": 0.0-1.0,
    "recall": 0.0-1.0,
    "f1_score": 0.0-1.0,
    "execution_order_quality": "good" | "acceptable" | "poor",
    "overall_quality": "excellent" | "good" | "acceptable" | "poor",
    "reasoning": "explanation",
    "suggestions": ["list"]
}}
"""

    def _parse_llm_evaluation(self, response: str) -> Dict[str, Any]:
        """Parse LLM evaluation response."""
        try:
            # Try to extract JSON from response
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_str = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                json_str = response[start:end].strip()
            else:
                # Try to find JSON object
                start = response.find("{")
                end = response.rfind("}") + 1
                json_str = response[start:end]

            return json.loads(json_str)
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse LLM evaluation: {e}")
            return {
                "company_type_correct": False,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "overall_quality": "poor",
                "reasoning": f"Failed to parse LLM response: {str(e)}",
            }

    def _quality_to_score(self, quality: str) -> float:
        """
        Convert semantic quality label to numeric score.

        Semantic to Numeric Mapping:
            - "excellent" -> 1.0: Optimal selection, comprehensive reasoning
            - "good"      -> 0.75: Correct selection with minor issues
            - "acceptable"-> 0.5: Functional but could be improved
            - "poor"      -> 0.25: Significant issues in selection

        This mapping allows LLM-as-judge to provide qualitative assessments
        that integrate with our quantitative evaluation metrics.

        Threshold for is_correct (LLM evaluation):
            quality in ["excellent", "good"] -> is_correct = True
            quality in ["acceptable", "poor"] -> is_correct = False

        Args:
            quality: Semantic quality label from LLM evaluation

        Returns:
            Numeric score from 0.0 to 1.0
        """
        return {
            "excellent": 1.0,
            "good": 0.75,
            "acceptable": 0.5,
            "poor": 0.25,
        }.get(quality.lower(), 0.25)

    def clear(self):
        """Clear evaluations."""
        self.evaluations = []
