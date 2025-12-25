"""Correctness Scorer - Compares model outputs against golden truth."""

import logging
import re
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


@dataclass
class CorrectnessResult:
    """Result of correctness scoring."""
    is_correct: bool
    correctness_score: float  # 0.0 to 1.0
    method: str
    model_outputs: List[Dict[str, Any]]
    golden_answer: str
    best_output: Optional[str] = None
    best_score: float = 0.0
    details: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_correct": self.is_correct,
            "correctness_score": self.correctness_score,
            "method": self.method,
            "model_outputs": self.model_outputs,
            "golden_answer": self.golden_answer,
            "best_output": self.best_output,
            "best_score": self.best_score,
            "details": self.details or {},
        }


class CorrectnessScorer:
    """
    Scores correctness of model outputs against golden truth.

    Methods:
    - exact_match: Exact string matching
    - semantic_similarity: Uses sentence embeddings
    - contains: Checks if golden answer is contained in output
    - classification: For classification tasks with discrete labels
    """

    def __init__(
        self,
        method: str = "semantic_similarity",
        similarity_threshold: float = 0.90,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.method = method
        self.similarity_threshold = similarity_threshold
        self.embedding_model_name = embedding_model
        self._embedding_model = None

        if method == "semantic_similarity" and not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("Sentence transformers not available, falling back to contains")
            self.method = "contains"

    def _get_embedding_model(self):
        """Lazy load embedding model."""
        if self._embedding_model is None and SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
        return self._embedding_model

    def score(
        self,
        outputs: List[str],
        golden_answer: str,
        model_names: Optional[List[str]] = None,
    ) -> CorrectnessResult:
        """
        Score correctness of outputs against golden answer.

        Args:
            outputs: List of model outputs
            golden_answer: Ground truth answer
            model_names: Optional names of models corresponding to outputs

        Returns:
            CorrectnessResult with correctness assessment
        """
        if not outputs:
            return CorrectnessResult(
                is_correct=False,
                correctness_score=0.0,
                method=self.method,
                model_outputs=[],
                golden_answer=golden_answer,
                details={"note": "No outputs to evaluate"},
            )

        model_names = model_names or [f"model_{i}" for i in range(len(outputs))]

        # Clean outputs and golden answer
        cleaned_outputs = [self._clean_text(o) for o in outputs]
        cleaned_golden = self._clean_text(golden_answer)

        # Score based on method
        if self.method == "exact_match":
            return self._score_exact_match(cleaned_outputs, cleaned_golden, model_names)
        elif self.method == "semantic_similarity":
            return self._score_semantic_similarity(cleaned_outputs, cleaned_golden, model_names)
        elif self.method == "contains":
            return self._score_contains(cleaned_outputs, cleaned_golden, model_names)
        elif self.method == "classification":
            return self._score_classification(outputs, golden_answer, model_names)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for comparison."""
        if not text:
            return ""
        # Remove extra whitespace
        text = " ".join(text.split())
        # Remove common prefixes
        prefixes = [
            "The answer is:",
            "Answer:",
            "Based on the information provided,",
            "The correct answer is",
        ]
        for prefix in prefixes:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()
        return text

    def _score_exact_match(
        self,
        outputs: List[str],
        golden: str,
        model_names: List[str],
    ) -> CorrectnessResult:
        """Score using exact string matching."""
        model_outputs = []
        scores = []

        for i, (output, name) in enumerate(zip(outputs, model_names)):
            is_match = output.lower().strip() == golden.lower().strip()
            score = 1.0 if is_match else 0.0
            scores.append(score)
            model_outputs.append({
                "model": name,
                "output": output[:200],
                "score": score,
                "is_correct": is_match,
            })

        best_idx = scores.index(max(scores))
        avg_score = sum(scores) / len(scores) if scores else 0.0
        any_correct = any(s >= self.similarity_threshold for s in scores)

        return CorrectnessResult(
            is_correct=any_correct,
            correctness_score=avg_score,
            method="exact_match",
            model_outputs=model_outputs,
            golden_answer=golden,
            best_output=outputs[best_idx],
            best_score=max(scores),
        )

    def _score_semantic_similarity(
        self,
        outputs: List[str],
        golden: str,
        model_names: List[str],
    ) -> CorrectnessResult:
        """Score using semantic similarity."""
        model = self._get_embedding_model()

        if model is None:
            return self._score_contains(outputs, golden, model_names)

        # Get embeddings
        all_texts = outputs + [golden]
        embeddings = model.encode(all_texts)

        golden_embedding = embeddings[-1]
        output_embeddings = embeddings[:-1]

        model_outputs = []
        scores = []

        for i, (output, name, emb) in enumerate(zip(outputs, model_names, output_embeddings)):
            # Cosine similarity
            sim = np.dot(emb, golden_embedding) / (
                np.linalg.norm(emb) * np.linalg.norm(golden_embedding)
            )
            sim = float(sim)
            scores.append(sim)
            model_outputs.append({
                "model": name,
                "output": output[:200],
                "score": sim,
                "is_correct": sim >= self.similarity_threshold,
            })

        best_idx = scores.index(max(scores))
        avg_score = sum(scores) / len(scores) if scores else 0.0
        any_correct = any(s >= self.similarity_threshold for s in scores)

        return CorrectnessResult(
            is_correct=any_correct,
            correctness_score=avg_score,
            method="semantic_similarity",
            model_outputs=model_outputs,
            golden_answer=golden,
            best_output=outputs[best_idx],
            best_score=max(scores),
            details={
                "threshold": self.similarity_threshold,
                "embedding_model": self.embedding_model_name,
            },
        )

    def _score_contains(
        self,
        outputs: List[str],
        golden: str,
        model_names: List[str],
    ) -> CorrectnessResult:
        """Score by checking if golden answer is contained in output."""
        model_outputs = []
        scores = []

        golden_lower = golden.lower()
        golden_keywords = set(golden_lower.split())

        for i, (output, name) in enumerate(zip(outputs, model_names)):
            output_lower = output.lower()

            # Check exact containment
            if golden_lower in output_lower:
                score = 1.0
            else:
                # Check keyword overlap
                output_words = set(output_lower.split())
                overlap = len(golden_keywords & output_words)
                score = overlap / len(golden_keywords) if golden_keywords else 0.0

            scores.append(score)
            model_outputs.append({
                "model": name,
                "output": output[:200],
                "score": score,
                "is_correct": score >= self.similarity_threshold,
            })

        best_idx = scores.index(max(scores))
        avg_score = sum(scores) / len(scores) if scores else 0.0
        any_correct = any(s >= self.similarity_threshold for s in scores)

        return CorrectnessResult(
            is_correct=any_correct,
            correctness_score=avg_score,
            method="contains",
            model_outputs=model_outputs,
            golden_answer=golden,
            best_output=outputs[best_idx],
            best_score=max(scores),
        )

    def _score_classification(
        self,
        outputs: List[str],
        golden: str,
        model_names: List[str],
    ) -> CorrectnessResult:
        """Score for classification tasks."""
        model_outputs = []
        scores = []

        golden_class = golden.lower().strip()

        for i, (output, name) in enumerate(zip(outputs, model_names)):
            # Extract predicted class
            predicted = self._extract_class_label(output)

            is_correct = predicted is not None and predicted.lower() == golden_class
            score = 1.0 if is_correct else 0.0
            scores.append(score)

            model_outputs.append({
                "model": name,
                "output": output[:200],
                "predicted_class": predicted,
                "score": score,
                "is_correct": is_correct,
            })

        best_idx = scores.index(max(scores)) if scores else 0
        avg_score = sum(scores) / len(scores) if scores else 0.0
        any_correct = any(s > 0 for s in scores)

        return CorrectnessResult(
            is_correct=any_correct,
            correctness_score=avg_score,
            method="classification",
            model_outputs=model_outputs,
            golden_answer=golden,
            best_output=outputs[best_idx] if outputs else None,
            best_score=max(scores) if scores else 0.0,
            details={
                "golden_class": golden_class,
            },
        )

    def _extract_class_label(self, output: str) -> Optional[str]:
        """Extract class label from output text."""
        output = output.strip()

        # Check for common patterns
        patterns = [
            r"(?:answer|class|label|prediction)[:\s]+(\w+)",
            r"^(\w+)$",
            r"^(\w+)[.\s]",
        ]

        for pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                return match.group(1)

        # If short output, treat whole thing as label
        if len(output.split()) <= 3:
            return output.split()[0] if output.split() else None

        return None

    def score_numeric(
        self,
        outputs: List[str],
        golden_value: float,
        tolerance: float = 0.1,
        model_names: Optional[List[str]] = None,
    ) -> CorrectnessResult:
        """
        Score numeric outputs against a golden value.

        Args:
            outputs: List of model outputs
            golden_value: Ground truth numeric value
            tolerance: Relative tolerance for correctness (0.1 = 10%)
            model_names: Optional names of models

        Returns:
            CorrectnessResult
        """
        model_names = model_names or [f"model_{i}" for i in range(len(outputs))]
        model_outputs = []
        scores = []

        for i, (output, name) in enumerate(zip(outputs, model_names)):
            # Extract number from output
            extracted = self._extract_number(output)

            if extracted is not None:
                # Calculate relative error
                rel_error = abs(extracted - golden_value) / abs(golden_value) if golden_value != 0 else abs(extracted)
                is_correct = rel_error <= tolerance
                score = max(0, 1 - rel_error)
            else:
                is_correct = False
                score = 0.0
                extracted = None

            scores.append(score)
            model_outputs.append({
                "model": name,
                "output": output[:200],
                "extracted_value": extracted,
                "score": score,
                "is_correct": is_correct,
            })

        best_idx = scores.index(max(scores)) if scores else 0
        avg_score = sum(scores) / len(scores) if scores else 0.0
        any_correct = any(s >= 1 - tolerance for s in scores)

        return CorrectnessResult(
            is_correct=any_correct,
            correctness_score=avg_score,
            method="numeric",
            model_outputs=model_outputs,
            golden_answer=str(golden_value),
            best_output=outputs[best_idx] if outputs else None,
            best_score=max(scores) if scores else 0.0,
            details={
                "golden_value": golden_value,
                "tolerance": tolerance,
            },
        )

    def _extract_number(self, text: str) -> Optional[float]:
        """Extract a number from text."""
        # Remove commas and common formatting
        text = text.replace(",", "").replace("$", "").replace("%", "")

        # Find all numbers
        numbers = re.findall(r'-?\d+\.?\d*', text)

        if numbers:
            try:
                return float(numbers[0])
            except ValueError:
                return None

        return None
