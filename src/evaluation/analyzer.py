"""Correlation Analyzer - Analyzes relationship between consistency and correctness."""

import logging
import json
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import os

logger = logging.getLogger(__name__)

try:
    import numpy as np
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available. Some statistical functions will be limited.")


@dataclass
class EvaluationRecord:
    """Record of a single evaluation."""
    prompt_id: str
    prompt: str
    golden_answer: str
    model_outputs: List[Dict[str, Any]]
    consistency_result: Dict[str, Any]
    correctness_result: Dict[str, Any]
    is_consistent: bool
    is_correct: bool
    consistency_score: float
    correctness_score: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "prompt": self.prompt,
            "golden_answer": self.golden_answer,
            "model_outputs": self.model_outputs,
            "consistency_result": self.consistency_result,
            "correctness_result": self.correctness_result,
            "is_consistent": self.is_consistent,
            "is_correct": self.is_correct,
            "consistency_score": self.consistency_score,
            "correctness_score": self.correctness_score,
            "timestamp": self.timestamp,
        }


@dataclass
class CorrelationReport:
    """Final correlation analysis report."""
    total_samples: int
    consistent_count: int
    correct_count: int
    consistent_and_correct: int
    consistent_but_incorrect: int
    inconsistent_but_correct: int
    inconsistent_and_incorrect: int

    # Key metrics
    correlation_coefficient: float
    precision: float  # When we say consistent, how often is it correct?
    recall: float  # Of all correct answers, how many were consistent?
    f1_score: float
    accuracy: float

    # Statistical significance
    p_value: Optional[float] = None

    # Detailed breakdown
    records: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "consistent_count": self.consistent_count,
            "correct_count": self.correct_count,
            "consistent_and_correct": self.consistent_and_correct,
            "consistent_but_incorrect": self.consistent_but_incorrect,
            "inconsistent_but_correct": self.inconsistent_but_correct,
            "inconsistent_and_incorrect": self.inconsistent_and_incorrect,
            "correlation_coefficient": self.correlation_coefficient,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "accuracy": self.accuracy,
            "p_value": self.p_value,
            "timestamp": self.timestamp,
        }

    def get_summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            "=" * 50,
            "CONSISTENCY-CORRECTNESS CORRELATION REPORT",
            "=" * 50,
            f"Total samples: {self.total_samples}",
            f"Consistent: {self.consistent_count} ({self.consistent_count/self.total_samples*100:.1f}%)",
            f"Correct: {self.correct_count} ({self.correct_count/self.total_samples*100:.1f}%)",
            "",
            "CONFUSION MATRIX:",
            "-" * 30,
            f"Consistent & Correct:     {self.consistent_and_correct}",
            f"Consistent & Incorrect:   {self.consistent_but_incorrect}",
            f"Inconsistent & Correct:   {self.inconsistent_but_correct}",
            f"Inconsistent & Incorrect: {self.inconsistent_and_incorrect}",
            "",
            "KEY METRICS:",
            "-" * 30,
            f"Correlation Coefficient: {self.correlation_coefficient:.4f}",
            f"Precision: {self.precision:.4f} ({self.precision*100:.1f}%)",
            f"Recall: {self.recall:.4f} ({self.recall*100:.1f}%)",
            f"F1 Score: {self.f1_score:.4f}",
            f"Accuracy: {self.accuracy:.4f} ({self.accuracy*100:.1f}%)",
        ]

        if self.p_value is not None:
            lines.append(f"P-value: {self.p_value:.6f}")
            if self.p_value < 0.05:
                lines.append("Statistical significance: YES (p < 0.05)")
            else:
                lines.append("Statistical significance: NO (p >= 0.05)")

        lines.extend([
            "",
            "HYPOTHESIS VALIDATION:",
            "-" * 30,
        ])

        if self.correlation_coefficient >= 0.85:
            lines.append("SUCCESS: Correlation >= 85% threshold")
        else:
            lines.append(f"BELOW TARGET: Correlation {self.correlation_coefficient:.1%} < 85%")

        if self.precision >= 0.95:
            lines.append("SUCCESS: Near-100% precision achieved (>= 95%)")
        else:
            lines.append(f"BELOW TARGET: Precision {self.precision:.1%} < 95%")

        lines.append("=" * 50)

        return "\n".join(lines)


class CorrelationAnalyzer:
    """
    Analyzes correlation between model consistency and output correctness.

    Measures:
    - Correlation coefficient between consistency and correctness
    - Precision: P(correct | consistent)
    - Recall: P(consistent | correct)
    - F1 Score
    """

    def __init__(self, results_dir: Optional[str] = None):
        self.results_dir = results_dir or "data/results"
        self.records: List[EvaluationRecord] = []

    def add_record(self, record: EvaluationRecord):
        """Add an evaluation record."""
        self.records.append(record)

    def add_result(
        self,
        prompt_id: str,
        prompt: str,
        golden_answer: str,
        model_outputs: List[Dict[str, Any]],
        consistency_result: Dict[str, Any],
        correctness_result: Dict[str, Any],
    ):
        """Add evaluation result from components."""
        record = EvaluationRecord(
            prompt_id=prompt_id,
            prompt=prompt,
            golden_answer=golden_answer,
            model_outputs=model_outputs,
            consistency_result=consistency_result,
            correctness_result=correctness_result,
            is_consistent=consistency_result.get("is_consistent", False),
            is_correct=correctness_result.get("is_correct", False),
            consistency_score=consistency_result.get("consistency_score", 0.0),
            correctness_score=correctness_result.get("correctness_score", 0.0),
        )
        self.records.append(record)

    def analyze(self) -> CorrelationReport:
        """
        Analyze correlation between consistency and correctness.

        Returns:
            CorrelationReport with full analysis
        """
        if not self.records:
            return CorrelationReport(
                total_samples=0,
                consistent_count=0,
                correct_count=0,
                consistent_and_correct=0,
                consistent_but_incorrect=0,
                inconsistent_but_correct=0,
                inconsistent_and_incorrect=0,
                correlation_coefficient=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                accuracy=0.0,
            )

        # Count categories
        total = len(self.records)
        consistent = sum(1 for r in self.records if r.is_consistent)
        correct = sum(1 for r in self.records if r.is_correct)

        # Confusion matrix
        consistent_and_correct = sum(1 for r in self.records if r.is_consistent and r.is_correct)
        consistent_but_incorrect = sum(1 for r in self.records if r.is_consistent and not r.is_correct)
        inconsistent_but_correct = sum(1 for r in self.records if not r.is_consistent and r.is_correct)
        inconsistent_and_incorrect = sum(1 for r in self.records if not r.is_consistent and not r.is_correct)

        # Calculate metrics
        # Precision: P(correct | consistent) - when we say consistent, how often correct?
        precision = consistent_and_correct / consistent if consistent > 0 else 0.0

        # Recall: P(consistent | correct) - of all correct, how many were consistent?
        recall = consistent_and_correct / correct if correct > 0 else 0.0

        # F1 Score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Accuracy
        accuracy = (consistent_and_correct + inconsistent_and_incorrect) / total if total > 0 else 0.0

        # Calculate correlation coefficient
        consistency_scores = [r.consistency_score for r in self.records]
        correctness_scores = [r.correctness_score for r in self.records]

        p_value = None
        if SCIPY_AVAILABLE and len(consistency_scores) > 2:
            # Pearson correlation
            corr, p_value = stats.pearsonr(consistency_scores, correctness_scores)
            correlation = corr
        else:
            # Manual correlation calculation
            correlation = self._calculate_correlation(consistency_scores, correctness_scores)

        return CorrelationReport(
            total_samples=total,
            consistent_count=consistent,
            correct_count=correct,
            consistent_and_correct=consistent_and_correct,
            consistent_but_incorrect=consistent_but_incorrect,
            inconsistent_but_correct=inconsistent_but_correct,
            inconsistent_and_incorrect=inconsistent_and_incorrect,
            correlation_coefficient=correlation,
            precision=precision,
            recall=recall,
            f1_score=f1,
            accuracy=accuracy,
            p_value=p_value,
            records=[r.to_dict() for r in self.records],
        )

    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient manually."""
        n = len(x)
        if n == 0:
            return 0.0

        mean_x = sum(x) / n
        mean_y = sum(y) / n

        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        denom_x = sum((x[i] - mean_x) ** 2 for i in range(n)) ** 0.5
        denom_y = sum((y[i] - mean_y) ** 2 for i in range(n)) ** 0.5

        if denom_x == 0 or denom_y == 0:
            return 0.0

        return numerator / (denom_x * denom_y)

    def save_results(self, filename: Optional[str] = None):
        """Save results to JSON file."""
        os.makedirs(self.results_dir, exist_ok=True)

        if not filename:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_results_{timestamp}.json"

        filepath = os.path.join(self.results_dir, filename)

        report = self.analyze()
        with open(filepath, "w") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)

        logger.info(f"Results saved to: {filepath}")
        return filepath

    def load_results(self, filepath: str):
        """Load results from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)

        # Reconstruct records
        self.records = []
        for record_data in data.get("records", []):
            record = EvaluationRecord(
                prompt_id=record_data["prompt_id"],
                prompt=record_data["prompt"],
                golden_answer=record_data["golden_answer"],
                model_outputs=record_data["model_outputs"],
                consistency_result=record_data["consistency_result"],
                correctness_result=record_data["correctness_result"],
                is_consistent=record_data["is_consistent"],
                is_correct=record_data["is_correct"],
                consistency_score=record_data["consistency_score"],
                correctness_score=record_data["correctness_score"],
                timestamp=record_data.get("timestamp", ""),
            )
            self.records.append(record)

        logger.info(f"Loaded {len(self.records)} records from {filepath}")

    def get_failure_analysis(self) -> Dict[str, Any]:
        """Analyze failure cases where consistency didn't predict correctness."""
        failures = {
            "consistent_but_incorrect": [],
            "inconsistent_but_correct": [],
        }

        for record in self.records:
            if record.is_consistent and not record.is_correct:
                failures["consistent_but_incorrect"].append({
                    "prompt_id": record.prompt_id,
                    "prompt": record.prompt[:100],
                    "golden_answer": record.golden_answer,
                    "consistency_score": record.consistency_score,
                    "correctness_score": record.correctness_score,
                })
            elif not record.is_consistent and record.is_correct:
                failures["inconsistent_but_correct"].append({
                    "prompt_id": record.prompt_id,
                    "prompt": record.prompt[:100],
                    "golden_answer": record.golden_answer,
                    "consistency_score": record.consistency_score,
                    "correctness_score": record.correctness_score,
                })

        return {
            "consistent_but_incorrect_count": len(failures["consistent_but_incorrect"]),
            "inconsistent_but_correct_count": len(failures["inconsistent_but_correct"]),
            "samples": failures,
        }

    def clear(self):
        """Clear all records."""
        self.records = []
