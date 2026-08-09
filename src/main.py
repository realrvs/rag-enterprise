import os

# 1. Отключаем сетевые проверки Hugging Face (быстрая загрузка из кэша)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Глобальный синглтон пайплайна (гарантирует доступ из любого потока)
_pipeline = None


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str = ""
    contexts: List[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    from_cache: bool = False
    status: str = "success"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    logger.info("=" * 60)
    logger.info("🚀 Starting Enterprise RAG Platform")
    logger.info("=" * 60)
    
    # Выводим конфигурацию, если переменные заданы в settings
    if hasattr(settings, "ENVIRONMENT"):
        logger.info(f"📌 Environment: {settings.ENVIRONMENT}")
    if hasattr(settings, "QDRANT_URL"):
        logger.info(f"📌 Qdrant URL: {settings.QDRANT_URL}")
    if hasattr(settings, "QDRANT_COLLECTION_NAME"):
        logger.info(f"📌 Collection: {settings.QDRANT_COLLECTION_NAME}")

    logger.info("⏳ Initializing RAGPipeline in background thread...")
    try:
        from src.pipeline.rag_pipeline import RAGPipeline
        # Загрузка тяжёлой модели в отдельном потоке, чтобы не блокировать Event Loop
        _pipeline = await asyncio.to_thread(RAGPipeline)
        logger.info("✅ RAGPipeline successfully loaded in memory and ready!")
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAGPipeline: {e}", exc_info=True)
        _pipeline = None
        
    yield
    
    logger.info("🛑 Shutting down server...")
    _pipeline = None


app = FastAPI(
    title="Enterprise RAG Platform",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "pipeline_ready": _pipeline is not None
    }


@app.get("/ping")
async def ping():
    return {"pong": True, "time": time.time()}


@app.get("/routes")
async def list_routes():
    return {"routes": [{"path": r.path, "methods": list(r.methods)} for r in app.routes]}


@app.post("/query")
def query(request: QueryRequest) -> QueryResponse:
    """
    Синхронный эндпоинт (def вместо async def).
    FastAPI автоматически запускает его в ThreadPool, не блокируя веб-сервер.
    """
    global _pipeline
    
    logger.info("=" * 60)
    logger.info("🔍 QUERY CALLED!")
    logger.info(f"📝 Question: {request.question}")
    logger.info("=" * 60)
    
    start = time.time()
    
    if _pipeline is None:
        logger.error("❌ Query received, but _pipeline is None!")
        return QueryResponse(
            answer="Pipeline not available. Please check server logs.",
            status="error"
        )
    
    try:
        response = _pipeline.query(question=request.question, top_k=3)
        return QueryResponse(
            answer=response.get("answer", ""),
            contexts=response.get("contexts", []),
            latency_ms=(time.time() - start) * 1000,
            from_cache=response.get("from_cache", False),
            status=response.get("status", "success")
        )
    except Exception as e:
        logger.error(f"❌ Error processing query: {e}", exc_info=True)
        return QueryResponse(
            answer=f"Error: {str(e)}",
            status="error"
        )