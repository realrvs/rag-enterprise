"""
Enterprise RAG Platform — FastAPI Application Entry Point.
"""

import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import settings

# ==========================================
#  Logging Configuration
# ==========================================
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ==========================================
#  Pydantic Models
# ==========================================
class QueryRequest(BaseModel):
    """Request model for /query endpoint."""
    question: str = Field(..., description="User question", min_length=1, max_length=10000)
    context_filter: Dict[str, Any] = Field(default_factory=dict, description="Optional metadata filter")


class QueryResponse(BaseModel):
    """Response model for /query endpoint."""
    answer: str = Field(..., description="Generated answer")
    contexts: list = Field(default_factory=list, description="Retrieved contexts")
    latency_ms: float = Field(..., description="Total latency in milliseconds")
    from_cache: bool = Field(default=False, description="Whether response came from cache")
    status: str = Field(default="success", description="Response status")


# ==========================================
#  Lifespan Manager
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Initializes resources on startup and cleans up on shutdown.
    """
    logger.info("🚀 Starting Enterprise RAG Platform...")
    logger.info(f"📌 Environment: {settings.environment}")
    logger.info(f"📌 Log Level: {settings.log_level}")
    
    # TODO: Initialize RAG Pipeline here
    # app.state.pipeline = RAGPipeline()
    
    yield
    
    logger.info("🛑 Shutting down Enterprise RAG Platform...")


# ==========================================
#  FastAPI Application
# ==========================================
app = FastAPI(
    title="Enterprise RAG Platform",
    description="Production RAG with SPLADE, Hybrid Search, and Quality Gate",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# ==========================================
#  CORS Middleware
# ==========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.is_development else ["https://your-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
#  Health Check
# ==========================================
@app.get(
    "/health",
    tags=["System"],
    summary="Health check endpoint",
    description="Returns the current health status of the application."
)
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint.
    
    Returns:
        Dict[str, str]: Health status and environment information.
    """
    return {
        "status": "healthy",
        "environment": settings.environment,
        "version": "2.0.0"
    }


# ==========================================
#  Query Endpoint
# ==========================================
@app.post(
    "/query",
    response_model=QueryResponse,
    tags=["RAG"],
    summary="Query the RAG system",
    description="Send a question to the RAG pipeline and get an answer with context."
)
async def query(request: QueryRequest) -> QueryResponse:
    """
    Process a user query through the RAG pipeline.
    
    Args:
        request: QueryRequest with question and optional filter.
        
    Returns:
        QueryResponse: Generated answer with contexts and metadata.
    """
    # TODO: Implement RAG pipeline querying
    # response = await app.state.pipeline.query(
    #     question=request.question,
    #     filters=request.context_filter
    # )
    
    # Placeholder response
    return QueryResponse(
        answer="RAG pipeline not yet implemented.",
        contexts=[],
        latency_ms=0.0,
        from_cache=False,
        status="success"
    )


# ==========================================
#  Error Handlers
# ==========================================
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unexpected errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    raise HTTPException(
        status_code=500,
        detail="An internal error occurred. Please try again later."
    )