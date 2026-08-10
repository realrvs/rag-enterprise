"""
OpenTelemetry tracing for distributed tracing.
"""

print("🔍🔍🔍 TRACING.PY LOADED 🔍🔍🔍")

import logging
from contextlib import contextmanager
from typing import Optional, Dict, Any

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import SpanKind

from src.config import settings

logger = logging.getLogger(__name__)


class Tracer:
    """OpenTelemetry tracer wrapper."""

    def __init__(self, service_name: str = "rag-enterprise"):
        print(f"🔍 Tracer.__init__({service_name})")
        self.service_name = service_name
        self._tracer = None
        self._initialized = False

    def setup(self):
        """Setup OpenTelemetry tracer."""
        print(f"🔍 Tracer.setup() called, _initialized={self._initialized}")
        if self._initialized:
            print("🔍 Already initialized, returning")
            return

        print("🔍 TRACER: Starting setup...")

        try:
            # Create resource
            resource = Resource.create({
                "service.name": self.service_name,
                "service.version": "2.0.0",
                "deployment.environment": getattr(settings, "environment", "development"),
            })
            print("✅ TRACER: Resource created")

            # Create tracer provider
            provider = TracerProvider(resource=resource)
            print("✅ TRACER: Provider created")

            # Console exporter for development
            console_exporter = ConsoleSpanExporter()
            provider.add_span_processor(BatchSpanProcessor(console_exporter))
            print("✅ TRACER: Console exporter added")

            # Set global tracer provider
            trace.set_tracer_provider(provider)
            print("✅ TRACER: Provider set globally")

            self._tracer = trace.get_tracer(__name__)
            self._initialized = True
            print(f"✅ TRACER: Initialized: {self.service_name}")

        except Exception as e:
            print(f"❌ TRACER: Failed: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to no-op tracer
            trace.set_tracer_provider(TracerProvider())
            self._tracer = trace.get_tracer(__name__)
            self._initialized = True

    def get_tracer(self):
        """Get the tracer instance."""
        print(f"🔍 get_tracer() called, _initialized={self._initialized}")
        if not self._initialized:
            self.setup()
        return self._tracer

    @contextmanager
    def start_span(self, name: str, attributes: Optional[Dict[str, Any]] = None, kind: SpanKind = SpanKind.INTERNAL):
        """Start a new span with logging."""
        print(f"🔍 Starting span: {name}")
        tracer = self.get_tracer()
        with tracer.start_as_current_span(name, kind=kind) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, str(value))
                    print(f"   {key}: {value}")
            yield span
        print(f"✅ Span completed: {name}")

    @contextmanager
    def trace_request(self, endpoint: str, question: str):
        """Trace a request through the system."""
        with self.start_span(
            "rag_request",
            attributes={
                "endpoint": endpoint,
                "question": question[:100],
                "environment": getattr(settings, "environment", "development"),
            },
            kind=SpanKind.SERVER,
        ) as span:
            yield span

    @contextmanager
    def trace_search(self, query: str, top_k: int):
        """Trace search operation."""
        with self.start_span(
            "search",
            attributes={
                "query": query[:100],
                "top_k": top_k,
            },
        ) as span:
            yield span

    @contextmanager
    def trace_llm(self, model: str, prompt_length: int):
        """Trace LLM generation."""
        with self.start_span(
            "llm_generation",
            attributes={
                "model": model,
                "prompt_length": prompt_length,
            },
        ) as span:
            yield span

    @contextmanager
    def trace_retrieval(self, collection: str):
        """Trace retrieval from vector store."""
        with self.start_span(
            "vector_store_retrieval",
            attributes={
                "collection": collection,
            },
        ) as span:
            yield span


# Global tracer instance
_tracer: Optional[Tracer] = None


def setup_tracing(service_name: str = "rag-enterprise") -> Tracer:
    """Setup and return tracer."""
    print(f"🔍 setup_tracing({service_name}) called")
    global _tracer
    if _tracer is None:
        print("🔍 Creating new tracer instance...")
        _tracer = Tracer(service_name=service_name)
        _tracer.setup()
    else:
        print("🔍 Returning existing tracer instance")
    return _tracer


def get_tracer() -> Optional[Tracer]:
    """Get the global tracer."""
    print(f"🔍 get_tracer() called, _tracer={_tracer}")
    return _tracer