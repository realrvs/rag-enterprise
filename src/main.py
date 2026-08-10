"""
Enterprise RAG Platform — FastAPI Application Entry Point.
"""

import os
import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from src.config import settings

# ==========================================
# Offline mode for Hugging Face
# ==========================================
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ==========================================
# Observability imports
# ==========================================
from src.observability import (
    setup_logging,
    setup_metrics,
    setup_tracing,
    get_metrics,
    get_tracer,
)
from src.observability.metrics import MetricsCollector

# ==========================================
# Logging Configuration
# ==========================================
setup_logging()
logger = logging.getLogger(__name__)


# ==========================================
# Middleware for Request Logging
# ==========================================
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all incoming requests and responses."""

    async def dispatch(self, request: Request, call_next):
        logger.info(f"📨 {request.method} {request.url.path} - Request received")
        start_time = time.time()

        response = await call_next(request)

        process_time = (time.time() - start_time) * 1000
        logger.info(
            f"📨 {request.method} {request.url.path} - "
            f"Response {response.status_code} ({process_time:.2f}ms)"
        )

        return response


class CharsetMiddleware(BaseHTTPMiddleware):
    """Ensure UTF-8 encoding for all responses."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["Content-Type"] = "application/json; charset=utf-8"
        return response


# ==========================================
# Pydantic Models
# ==========================================
class QueryRequest(BaseModel):
    """Request model for /query endpoint."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="User question for the RAG system"
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of contexts to retrieve"
    )
    use_hybrid: bool = Field(
        default=False,
        description="Use hybrid search (dense + sparse)"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="LLM temperature"
    )
    max_tokens: int = Field(
        default=512,
        ge=1,
        le=2048,
        description="Maximum tokens to generate"
    )


class QueryResponse(BaseModel):
    """Response model for /query endpoint."""

    answer: str = Field(..., description="Generated answer")
    contexts: list = Field(default_factory=list, description="Retrieved contexts")
    latency_ms: float = Field(default=0.0, description="Total latency in milliseconds")
    from_cache: bool = Field(default=False, description="Whether response came from cache")
    status: str = Field(default="success", description="Response status")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class HealthResponse(BaseModel):
    """Response model for /health endpoint."""

    status: str
    environment: str
    version: str
    pipeline_ready: bool
    components: Dict[str, bool]


# ==========================================
# Global Pipeline Instance
# ==========================================
_pipeline_instance = None


def get_pipeline():
    """
    Get or create RAGPipeline instance.
    Uses lazy initialization on first request.
    """
    global _pipeline_instance

    if _pipeline_instance is not None:
        return _pipeline_instance

    logger.info("🔍 Creating RAGPipeline instance (lazy initialization)...")
    try:
        from src.pipeline.rag_pipeline import RAGPipeline

        _pipeline_instance = RAGPipeline(
            model_name=settings.llm_model,
            use_ollama=settings.llm_use_ollama,
        )
        logger.info("✅ RAGPipeline created successfully")

        # Update metrics
        metrics = get_metrics()
        if metrics:
            metrics.update_pipeline_status(True)

        return _pipeline_instance
    except Exception as e:
        logger.error(f"❌ Failed to create pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None


# ==========================================
# FastAPI Application Lifespan
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Initializes resources on startup and cleans up on shutdown.
    """
    global _pipeline_instance

    logger.info("=" * 60)
    logger.info("🚀 Starting Enterprise RAG Platform")
    logger.info("=" * 60)
    logger.info(f"📌 Environment: {settings.environment}")
    logger.info(f"📌 Qdrant URL: {settings.qdrant_url}")
    logger.info(f"📌 Collection: {settings.qdrant_collection}")
    logger.info(f"📌 LLM Model: {settings.llm_model}")
    logger.info(f"📌 LLM Use Ollama: {settings.llm_use_ollama}")

    # Setup observability
    try:
        # Metrics
        metrics = setup_metrics(port=9091)
        metrics.start_server()
        logger.info("✅ Metrics server started on port 9091")

        # Tracing — ЯВНАЯ ИНИЦИАЛИЗАЦИЯ С ПРОВЕРКОЙ
        print("🔍🔍🔍 CALLING setup_tracing() FROM LIFESPAN 🔍🔍🔍")
        from src.observability import setup_tracing
        tracer = setup_tracing(service_name="rag-enterprise")
        print(f"🔍🔍🔍 TRACER CREATED: {tracer} 🔍🔍🔍")
        app.state.tracer = tracer
        logger.info(f"✅ Tracer initialized: {tracer}")

        # Update pipeline status (not ready yet)
        metrics.update_pipeline_status(False)

        # Get vector store stats
        try:
            from src.retrieval import QdrantClientWrapper
            qdrant = QdrantClientWrapper()
            count = qdrant.count_points()
            metrics.update_vector_store_points(count)
            logger.info(f"📊 Vector store points: {count}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to get vector store stats: {e}")

    except Exception as e:
        logger.warning(f"⚠️ Observability setup warning: {e}")

    # Pre-initialize pipeline in background
    logger.info("⏳ Initializing RAGPipeline in background thread...")
    try:
        from src.pipeline.rag_pipeline import RAGPipeline

        _pipeline_instance = await asyncio.to_thread(
            RAGPipeline,
            model_name=settings.llm_model,
            use_ollama=settings.llm_use_ollama,
        )
        app.state.pipeline = _pipeline_instance

        # Update metrics
        metrics = get_metrics()
        if metrics:
            metrics.update_pipeline_status(True)

        logger.info("✅ RAGPipeline successfully loaded in memory and ready!")

    except Exception as e:
        logger.error(f"❌ Failed to initialize pipeline: {e}", exc_info=True)
        app.state.pipeline = None
        _pipeline_instance = None

    yield

    logger.info("🛑 Shutting down Enterprise RAG Platform...")
    _pipeline_instance = None


# ==========================================
# FastAPI Application
# ==========================================
app = FastAPI(
    title="Enterprise RAG Platform",
    description="Production RAG with SPLADE, Hybrid Search, and Quality Gate",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ==========================================
# Middleware Configuration
# ==========================================

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.is_development else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Logging
app.add_middleware(RequestLoggingMiddleware)

# Charset
app.add_middleware(CharsetMiddleware)

# Trusted Host (for production)
if settings.is_production:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"],
    )


# ==========================================
# Health Check Endpoint
# ==========================================
@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check endpoint",
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint with component status.
    """
    pipeline_ready = _pipeline_instance is not None

    # Check Qdrant
    qdrant_ok = False
    try:
        from src.retrieval import QdrantClientWrapper
        qdrant = QdrantClientWrapper()
        count = qdrant.count_points()
        qdrant_ok = True
    except Exception:
        qdrant_ok = False

    # Check LLM
    llm_ok = False
    if _pipeline_instance:
        try:
            llm_ok = _pipeline_instance.llm.is_available() if _pipeline_instance.llm else False
        except Exception:
            llm_ok = False

    return HealthResponse(
        status="healthy",
        environment=settings.environment,
        version="2.0.0",
        pipeline_ready=pipeline_ready,
        components={
            "qdrant": qdrant_ok,
            "llm": llm_ok,
            "pipeline": pipeline_ready,
        }
    )


# ==========================================
# Metrics Endpoint (for Prometheus)
# ==========================================
@app.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint.
    """
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


# ==========================================
# Test Trace Endpoint
# ==========================================
@app.get("/test-trace")
async def test_trace():
    """
    Test endpoint for tracing without LLM.
    """
    print("🔍🔍🔍 /test-trace CALLED 🔍🔍🔍")
    
    # Get tracer from app.state
    tracer = getattr(app.state, "tracer", None)
    print(f"🔍 tracer from app.state: {tracer}")

    # If not found — try to create
    if tracer is None:
        print("🔍 tracer is None, creating new one...")
        try:
            from src.observability import setup_tracing
            tracer = setup_tracing(service_name="rag-enterprise")
            app.state.tracer = tracer
            print(f"✅ Tracer created in test endpoint: {tracer}")
        except Exception as e:
            print(f"❌ Failed to create tracer: {e}")
            import traceback
            traceback.print_exc()
            return {"error": f"Tracer not available: {str(e)}"}

    if tracer is None:
        return {"error": "Tracer not available"}

    try:
        print("🔍 Starting trace...")
        with tracer.trace_request("test", "test question"):
            print("📝 Inside request span")
            time.sleep(0.1)

            with tracer.trace_search("test query", 3):
                print("📝 Inside search span")
                time.sleep(0.05)

                with tracer.trace_retrieval("test_collection"):
                    print("📝 Inside retrieval span")
                    time.sleep(0.05)

            with tracer.trace_llm("test_model", 100):
                print("📝 Inside LLM span")
                time.sleep(0.1)

        print("✅ Trace completed!")
        return {
            "status": "trace completed",
            "tracer_available": True,
            "message": "Check console for trace output"
        }
    except Exception as e:
        print(f"❌ Test trace error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "tracer_available": False}


# ==========================================
# Query Endpoint
# ==========================================
@app.post(
    "/query",
    response_model=QueryResponse,
    tags=["RAG"],
    summary="Query the RAG system",
    description="Send a question to the RAG pipeline and get an answer with context.",
)
async def query(request: QueryRequest) -> QueryResponse:
    """
    Process a user query through the RAG pipeline.
    """
    start_time = time.time()

    logger.info("=" * 50)
    logger.info(f"📝 Query: {request.question[:100]}...")
    logger.info("=" * 50)

    # Get metrics
    metrics = get_metrics()

    # Get pipeline
    pipeline = getattr(app.state, "pipeline", None)
    if pipeline is None:
        logger.error("❌ Pipeline is None in query endpoint")
        return QueryResponse(
            answer="Pipeline not available. Please check logs.",
            contexts=[],
            latency_ms=0.0,
            status="error",
            metadata={"error": "pipeline_not_available"}
        )

    # Execute query with metrics
    try:
        if metrics:
            with metrics.measure_request("query"):
                response = pipeline.query(
                    question=request.question,
                    top_k=request.top_k,
                    use_hybrid=request.use_hybrid,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                )
        else:
            response = pipeline.query(
                question=request.question,
                top_k=request.top_k,
                use_hybrid=request.use_hybrid,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )

        # Record metrics
        if metrics:
            metrics.record_search_results(len(response.get("contexts", [])))
            metrics.record_contexts(len(response.get("contexts", [])))

        latency_ms = (time.time() - start_time) * 1000

        logger.info(f"✅ Query successful: {len(response.get('contexts', []))} contexts")
        logger.info(f"⏱️ Latency: {latency_ms:.2f}ms")

        return QueryResponse(
            answer=response.get("answer", "No answer generated."),
            contexts=response.get("contexts", []),
            latency_ms=latency_ms,
            from_cache=response.get("from_cache", False),
            status=response.get("status", "success"),
            metadata=response.get("metadata", {}),
        )

    except Exception as e:
        logger.error(f"❌ Query error: {e}", exc_info=True)

        return QueryResponse(
            answer=f"Error processing query: {str(e)}",
            contexts=[],
            latency_ms=(time.time() - start_time) * 1000,
            status="error",
            metadata={"error": str(e)},
        )


# ==========================================
# Error Handlers
# ==========================================
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unexpected errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred. Please try again later."}
    )


# ==========================================
# Root and Utility Endpoints
# ==========================================
@app.get("/")
async def root():
    """Root endpoint with basic information."""
    return {
        "service": "Enterprise RAG Platform",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }


@app.get("/ping")
async def ping():
    """Simple ping endpoint for testing."""
    return {"pong": True, "timestamp": time.time()}


@app.get("/routes")
async def list_routes():
    """List all registered routes."""
    routes = []
    for route in app.routes:
        routes.append({
            "path": route.path,
            "methods": list(route.methods) if hasattr(route, "methods") else [],
        })
    return {"routes": sorted(routes, key=lambda x: x["path"])}