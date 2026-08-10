"""
RAG evaluation using RAGAS metrics.
"""

import logging
from typing import List, Dict, Any, Optional
from datasets import Dataset

from src.config import settings

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """Evaluate RAG system quality using RAGAS."""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.RAGEvaluator")
        self._ragas_available = False

        try:
            from ragas import evaluate
            from ragas.metrics import (
                context_precision,
                context_recall,
                faithfulness,
                answer_relevancy,
            )
            self.evaluate_fn = evaluate
            self.metrics = [
                context_precision,
                context_recall,
                faithfulness,
                answer_relevancy,
            ]
            self._ragas_available = True
            self.logger.info("✅ RAGAS available")
        except ImportError as e:
            self.logger.warning(f"⚠️ RAGAS not available: {e}")
            self.logger.info("💡 Install: pip install ragas")

    def is_available(self) -> bool:
        """Check if RAGAS is available."""
        return self._ragas_available

    def prepare_dataset(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str],
    ) -> Dataset:
        """Prepare dataset for RAGAS evaluation."""
        if not self._ragas_available:
            self.logger.error("❌ RAGAS not available")
            return None

        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }

        return Dataset.from_dict(data)

    def evaluate(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str],
    ) -> Dict[str, float]:
        """
        Evaluate RAG system using RAGAS metrics.

        Args:
            questions: List of questions.
            answers: List of generated answers.
            contexts: List of context lists for each answer.
            ground_truths: List of ground truth answers.

        Returns:
            Dictionary of metric names to scores.
        """
        if not self._ragas_available:
            self.logger.error("❌ RAGAS not available")
            return {}

        try:
            dataset = self.prepare_dataset(
                questions, answers, contexts, ground_truths
            )

            result = self.evaluate_fn(
                dataset=dataset,
                metrics=self.metrics,
            )

            scores = {}
            for metric in self.metrics:
                metric_name = metric.__name__
                if metric_name in result:
                    scores[metric_name] = result[metric_name]

            self.logger.info(f"✅ Evaluation complete: {scores}")
            return scores

        except Exception as e:
            self.logger.error(f"❌ Evaluation failed: {e}")
            return {}

    def check_quality_gate(
        self,
        scores: Dict[str, float],
        thresholds: Dict[str, float] = None,
    ) -> tuple:
        """
        Check if scores pass quality gate.

        Args:
            scores: Metric scores from evaluation.
            thresholds: Dict of metric_name -> threshold.

        Returns:
            (passed: bool, details: dict)
        """
        if thresholds is None:
            thresholds = {
                "faithfulness": 0.7,
                "answer_relevancy": 0.7,
                "context_recall": 0.6,
                "context_precision": 0.6,
            }

        details = {}
        passed = True

        for metric, score in scores.items():
            threshold = thresholds.get(metric, 0.5)
            details[metric] = {
                "score": score,
                "threshold": threshold,
                "passed": score >= threshold,
            }
            if score < threshold:
                passed = False

        return passed, details