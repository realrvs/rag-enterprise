"""
Observability Module — Logging, Metrics, Tracing, Evaluation.
"""

print("🔍🔍🔍 OBSERVABILITY INIT LOADED 🔍🔍🔍")

from src.observability.logger import setup_logging
from src.observability.metrics import (
    MetricsCollector,
    setup_metrics,
    get_metrics,
)
from src.observability.tracing import (
    Tracer,
    setup_tracing,
    get_tracer,
)
from src.observability.evaluator import RAGEvaluator

__all__ = [
    "setup_logging",
    "MetricsCollector",
    "setup_metrics",
    "get_metrics",
    "Tracer",
    "setup_tracing",
    "get_tracer",
    "RAGEvaluator",
]

print("🔍 OBSERVABILITY INIT COMPLETE")