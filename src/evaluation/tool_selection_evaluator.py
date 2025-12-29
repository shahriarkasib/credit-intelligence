"""Tool Selection Evaluator - Evaluates whether LLM chose the right tools."""

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


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
    EXPECTED_TOOLS = {
        # Public US companies - SEC + Market Data
        "public_us": ["fetch_sec_data", "fetch_market_data"],
        # Public non-US - Market Data + Web Search
        "public_non_us": ["fetch_market_data", "web_search"],
        # Private companies - Web Search + Legal
        "private": ["web_search", "fetch_legal_data"],
        # Unknown - Web Search to determine
        "unknown": ["web_search"],
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

    def classify_company(self, company_name: str) -> str:
        """Classify company type based on name."""
        name_lower = company_name.lower()

        # Check known public US companies
        for known in self.PUBLIC_US_COMPANIES:
            if known in name_lower:
                return "public_us"

        # Check known non-US public companies
        for known in self.PUBLIC_NON_US_COMPANIES:
            if known in name_lower:
                return "public_non_us"

        # Check for common indicators
        if any(ind in name_lower for ind in ["inc", "corp", "ltd", "llc", "co."]):
            # Could be public or private - need more investigation
            return "unknown"

        return "private"

    def get_expected_tools(self, company_name: str, context: Dict[str, Any] = None) -> List[str]:
        """Get expected tools for a company."""
        context = context or {}

        # Check if context provides hints
        if context.get("is_public"):
            if context.get("jurisdiction", "US").upper() == "US":
                return self.EXPECTED_TOOLS["public_us"]
            else:
                return self.EXPECTED_TOOLS["public_non_us"]

        if context.get("is_private"):
            return self.EXPECTED_TOOLS["private"]

        # Classify based on company name
        company_type = self.classify_company(company_name)
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
            context: Additional context

        Returns:
            ToolSelectionResult with evaluation metrics
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
                "company_type": self.classify_company(company_name),
            },
        )

        self.evaluations.append(result)
        return result

    def _evaluate_reasoning(self, reasoning: Dict[str, Any]) -> float:
        """Evaluate quality of tool selection reasoning."""
        if not reasoning:
            return 0.0

        score = 0.0
        max_score = 3.0

        # Check if reasoning includes company analysis
        company_analysis = reasoning.get("company_analysis", {})
        if company_analysis:
            score += 1.0
            # Bonus for explicit public/private determination
            if "is_likely_public" in company_analysis:
                score += 0.5

        # Check if tools have reasons
        tools_to_use = reasoning.get("tools_to_use", [])
        if tools_to_use:
            tools_with_reasons = sum(1 for t in tools_to_use if t.get("reason"))
            if tools_with_reasons > 0:
                score += min(1.0, tools_with_reasons / len(tools_to_use))

        # Check for execution order reasoning
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

    def clear(self):
        """Clear evaluations."""
        self.evaluations = []
