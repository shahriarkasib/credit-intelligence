"""Consistency Scorer - Measures agreement across multiple LLM outputs."""

import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import re

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not installed. Using fallback similarity.")


@dataclass
class ConsistencyResult:
    """Result of consistency scoring."""
    is_consistent: bool
    consistency_score: float  # 0.0 to 1.0
    method: str
    pairwise_scores: List[Dict[str, Any]]
    majority_answer: Optional[str] = None
    agreement_ratio: float = 0.0
    details: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_consistent": self.is_consistent,
            "consistency_score": self.consistency_score,
            "method": self.method,
            "pairwise_scores": self.pairwise_scores,
            "majority_answer": self.majority_answer,
            "agreement_ratio": self.agreement_ratio,
            "details": self.details or {},
        }


class ConsistencyScorer:
    """
    Scores consistency of outputs across multiple LLMs.

    Methods:
    - exact_match: Simple string equality
    - semantic_similarity: Uses sentence embeddings (Sentence-BERT)
    - keyword_overlap: Jaccard similarity of keywords
    - llm_judge: Uses an LLM to judge similarity (future)
    """

    def __init__(
        self,
        method: str = "semantic_similarity",
        similarity_threshold: float = 0.85,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.method = method
        self.similarity_threshold = similarity_threshold
        self.embedding_model_name = embedding_model
        self._embedding_model = None

        if method == "semantic_similarity" and not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("Sentence transformers not available, falling back to keyword_overlap")
            self.method = "keyword_overlap"

    def _get_embedding_model(self):
        """Lazy load embedding model."""
        if self._embedding_model is None and SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
        return self._embedding_model

    def score(self, outputs: List[str]) -> ConsistencyResult:
        """
        Score consistency across multiple outputs.

        Args:
            outputs: List of output strings from different models

        Returns:
            ConsistencyResult with consistency assessment
        """
        if len(outputs) < 2:
            return ConsistencyResult(
                is_consistent=True,
                consistency_score=1.0,
                method=self.method,
                pairwise_scores=[],
                agreement_ratio=1.0,
                details={"note": "Less than 2 outputs to compare"},
            )

        # Clean outputs
        cleaned_outputs = [self._clean_output(o) for o in outputs]

        # Score based on method
        if self.method == "exact_match":
            return self._score_exact_match(cleaned_outputs)
        elif self.method == "semantic_similarity":
            return self._score_semantic_similarity(cleaned_outputs)
        elif self.method == "keyword_overlap":
            return self._score_keyword_overlap(cleaned_outputs)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _clean_output(self, output: str) -> str:
        """Clean and normalize output for comparison."""
        if not output:
            return ""
        # Remove extra whitespace
        output = " ".join(output.split())
        # Remove common prefixes
        prefixes_to_remove = [
            "The answer is:",
            "Answer:",
            "Based on the information provided,",
            "In conclusion,",
        ]
        for prefix in prefixes_to_remove:
            if output.lower().startswith(prefix.lower()):
                output = output[len(prefix):].strip()
        return output

    def _score_exact_match(self, outputs: List[str]) -> ConsistencyResult:
        """Score using exact string matching."""
        pairwise_scores = []
        matches = 0
        total_pairs = 0

        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                is_match = outputs[i].lower() == outputs[j].lower()
                score = 1.0 if is_match else 0.0
                pairwise_scores.append({
                    "pair": (i, j),
                    "score": score,
                    "match": is_match,
                })
                if is_match:
                    matches += 1
                total_pairs += 1

        agreement_ratio = matches / total_pairs if total_pairs > 0 else 1.0
        avg_score = sum(p["score"] for p in pairwise_scores) / len(pairwise_scores) if pairwise_scores else 1.0

        # Find majority answer
        from collections import Counter
        counter = Counter(o.lower() for o in outputs)
        majority_answer, majority_count = counter.most_common(1)[0]

        return ConsistencyResult(
            is_consistent=avg_score >= self.similarity_threshold,
            consistency_score=avg_score,
            method="exact_match",
            pairwise_scores=pairwise_scores,
            majority_answer=majority_answer,
            agreement_ratio=agreement_ratio,
        )

    def _score_semantic_similarity(self, outputs: List[str]) -> ConsistencyResult:
        """Score using semantic similarity with sentence embeddings."""
        model = self._get_embedding_model()

        if model is None:
            # Fallback to keyword overlap
            return self._score_keyword_overlap(outputs)

        # Get embeddings
        embeddings = model.encode(outputs)

        # Calculate pairwise cosine similarity
        pairwise_scores = []
        similarities = []

        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                # Cosine similarity
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                sim = float(sim)
                similarities.append(sim)
                pairwise_scores.append({
                    "pair": (i, j),
                    "score": sim,
                    "above_threshold": sim >= self.similarity_threshold,
                })

        avg_score = sum(similarities) / len(similarities) if similarities else 1.0
        agreement_ratio = sum(1 for s in similarities if s >= self.similarity_threshold) / len(similarities) if similarities else 1.0

        # Find most central output (highest avg similarity to others)
        avg_sims = []
        for i in range(len(embeddings)):
            sims = []
            for j in range(len(embeddings)):
                if i != j:
                    sim = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                    )
                    sims.append(float(sim))
            avg_sims.append(sum(sims) / len(sims) if sims else 0)

        best_idx = avg_sims.index(max(avg_sims))
        majority_answer = outputs[best_idx]

        return ConsistencyResult(
            is_consistent=avg_score >= self.similarity_threshold,
            consistency_score=avg_score,
            method="semantic_similarity",
            pairwise_scores=pairwise_scores,
            majority_answer=majority_answer,
            agreement_ratio=agreement_ratio,
            details={
                "embedding_model": self.embedding_model_name,
                "threshold": self.similarity_threshold,
            },
        )

    def _score_keyword_overlap(self, outputs: List[str]) -> ConsistencyResult:
        """Score using keyword overlap (Jaccard similarity)."""
        # Extract keywords from each output
        keyword_sets = [self._extract_keywords(o) for o in outputs]

        pairwise_scores = []
        similarities = []

        for i in range(len(keyword_sets)):
            for j in range(i + 1, len(keyword_sets)):
                # Jaccard similarity
                intersection = keyword_sets[i] & keyword_sets[j]
                union = keyword_sets[i] | keyword_sets[j]
                sim = len(intersection) / len(union) if union else 1.0
                similarities.append(sim)
                pairwise_scores.append({
                    "pair": (i, j),
                    "score": sim,
                    "shared_keywords": list(intersection)[:10],
                })

        avg_score = sum(similarities) / len(similarities) if similarities else 1.0
        agreement_ratio = sum(1 for s in similarities if s >= self.similarity_threshold) / len(similarities) if similarities else 1.0

        # Find output with most keyword overlap with others
        overlap_counts = []
        for i, ks in enumerate(keyword_sets):
            overlap = sum(len(ks & other) for j, other in enumerate(keyword_sets) if i != j)
            overlap_counts.append(overlap)

        best_idx = overlap_counts.index(max(overlap_counts))
        majority_answer = outputs[best_idx]

        return ConsistencyResult(
            is_consistent=avg_score >= self.similarity_threshold,
            consistency_score=avg_score,
            method="keyword_overlap",
            pairwise_scores=pairwise_scores,
            majority_answer=majority_answer,
            agreement_ratio=agreement_ratio,
        )

    def _extract_keywords(self, text: str) -> set:
        """Extract keywords from text."""
        # Simple keyword extraction
        # Remove punctuation and split
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = text.split()

        # Remove stopwords (simplified list)
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
            'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
            'from', 'as', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'under', 'again', 'further', 'then',
            'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each',
            'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
            'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'and',
            'but', 'if', 'or', 'because', 'until', 'while', 'this', 'that',
            'these', 'those', 'it', 'its', 'i', 'me', 'my', 'we', 'our', 'you',
            'your', 'he', 'him', 'his', 'she', 'her', 'they', 'them', 'their',
            'what', 'which', 'who', 'whom', 'also', 'however', 'therefore',
        }

        keywords = {w for w in words if w not in stopwords and len(w) > 2}
        return keywords

    def score_classification(self, outputs: List[str], valid_classes: List[str]) -> ConsistencyResult:
        """
        Score consistency for classification tasks.

        Args:
            outputs: List of classification outputs
            valid_classes: List of valid class labels

        Returns:
            ConsistencyResult with class-specific analysis
        """
        # Extract predicted classes from outputs
        predicted_classes = []
        for output in outputs:
            predicted = self._extract_class(output, valid_classes)
            predicted_classes.append(predicted)

        # Count agreement
        from collections import Counter
        counter = Counter(predicted_classes)
        majority_class, majority_count = counter.most_common(1)[0] if counter else (None, 0)

        agreement_ratio = majority_count / len(predicted_classes) if predicted_classes else 0.0

        # Calculate pairwise agreement
        pairwise_scores = []
        for i in range(len(predicted_classes)):
            for j in range(i + 1, len(predicted_classes)):
                is_match = predicted_classes[i] == predicted_classes[j]
                pairwise_scores.append({
                    "pair": (i, j),
                    "score": 1.0 if is_match else 0.0,
                    "classes": (predicted_classes[i], predicted_classes[j]),
                })

        avg_score = sum(p["score"] for p in pairwise_scores) / len(pairwise_scores) if pairwise_scores else 1.0

        return ConsistencyResult(
            is_consistent=agreement_ratio >= 0.66,  # Majority agrees
            consistency_score=avg_score,
            method="classification",
            pairwise_scores=pairwise_scores,
            majority_answer=majority_class,
            agreement_ratio=agreement_ratio,
            details={
                "predicted_classes": predicted_classes,
                "class_distribution": dict(counter),
            },
        )

    def _extract_class(self, output: str, valid_classes: List[str]) -> Optional[str]:
        """Extract predicted class from output text."""
        output_lower = output.lower()

        # Direct match
        for cls in valid_classes:
            if cls.lower() in output_lower:
                return cls

        # Check for common patterns
        patterns = [
            r"answer[:\s]+(\w+)",
            r"class[:\s]+(\w+)",
            r"prediction[:\s]+(\w+)",
            r"^(\w+)$",
        ]

        for pattern in patterns:
            match = re.search(pattern, output_lower)
            if match:
                extracted = match.group(1)
                for cls in valid_classes:
                    if cls.lower() == extracted or cls.lower() in extracted:
                        return cls

        return None
