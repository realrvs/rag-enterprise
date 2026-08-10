"""
Main RAG Pipeline — orchestrates retrieval and generation.
"""

import logging
import time
from typing import List, Dict, Any, Optional

from src.retrieval import QdrantClientWrapper
from src.llm import LocalLLM, PromptBuilder
from src.config import settings
from src.observability import get_metrics, get_tracer

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Main RAG pipeline orchestrating search and generation.
    
    Features:
    - Hybrid search (dense + sparse)
    - Local LLM generation (Ollama / HuggingFace)
    - Prompt engineering with context
    - Observability integration (metrics, tracing)
    - Fallback to context-only answer if LLM unavailable
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        use_ollama: Optional[bool] = None,
    ):
        """
        Initialize RAG Pipeline.

        Args:
            model_name: Name of the LLM model to use.
            use_ollama: If True, use Ollama. Otherwise, use HuggingFace.
        """
        self.logger = logging.getLogger(f"{__name__}.RAGPipeline")

        # Use settings if not provided
        if model_name is None:
            model_name = getattr(settings, "llm_model", "qwen2.5-coder:7b")
        if use_ollama is None:
            use_ollama = getattr(settings, "llm_use_ollama", True)

        self.model_name = model_name
        self.use_ollama = use_ollama

        # Initialize retriever
        self.retriever = None
        try:
            self.retriever = QdrantClientWrapper()
            self.logger.info("✅ Retriever initialized")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize retriever: {e}")
            self.retriever = None

        # Initialize LLM
        self.llm = None
        try:
            self.llm = LocalLLM(
                model_name=model_name,
                use_ollama=use_ollama,
            )
            self.logger.info(f"✅ LLM initialized: {model_name}")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize LLM: {e}")
            self.llm = None

        # Initialize Prompt Builder
        try:
            self.prompt_builder = PromptBuilder()
            self.logger.info("✅ Prompt Builder initialized")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Prompt Builder: {e}")
            self.prompt_builder = None

        # Check if pipeline is ready
        self.is_ready = (
            self.retriever is not None and
            self.llm is not None and
            self.prompt_builder is not None
        )

        if self.is_ready:
            self.logger.info("✅ RAG Pipeline ready")
        else:
            self.logger.warning("⚠️ RAG Pipeline not fully ready")

        # Update metrics
        metrics = get_metrics()
        if metrics:
            metrics.update_pipeline_status(self.is_ready)

    def search(
        self,
        query: str,
        top_k: int = 3,
        use_hybrid: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents from Qdrant.

        Args:
            query: Search query.
            top_k: Number of results to return.
            use_hybrid: If True, use hybrid search (dense + sparse).

        Returns:
            List of search results with text, score, and metadata.
        """
        if self.retriever is None:
            self.logger.error("❌ Retriever not available")
            return []

        tracer = get_tracer()
        metrics = get_metrics()

        try:
            # Trace search
            if tracer:
                with tracer.trace_search(query, top_k):
                    if metrics:
                        with metrics.measure_vector_store("search"):
                            results = self.retriever.search(
                                query=query,
                                top_k=top_k,
                                use_hybrid=use_hybrid,
                            )
                    else:
                        results = self.retriever.search(
                            query=query,
                            top_k=top_k,
                            use_hybrid=use_hybrid,
                        )
            else:
                results = self.retriever.search(
                    query=query,
                    top_k=top_k,
                    use_hybrid=use_hybrid,
                )

            # Record metrics
            if metrics:
                metrics.record_search_results(len(results))

            self.logger.info(f"🔍 Search returned {len(results)} results")
            return results

        except Exception as e:
            self.logger.error(f"❌ Search failed: {e}")
            import traceback
            traceback.print_exc()
            return []

    def generate(
        self,
        query: str,
        contexts: List[str],
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> str:
        """
        Generate answer using LLM with context.

        Args:
            query: User question.
            contexts: Retrieved context strings.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            Generated answer.
        """
        # If no contexts, return helpful message
        if not contexts:
            return "Не удалось найти информацию по вашему запросу."

        # If LLM not available, return context-based answer
        if self.llm is None or not self.llm.is_available():
            self.logger.warning("⚠️ LLM not available, using fallback")
            fallback = "На основе найденных документов:\n\n"
            fallback += "\n\n".join(contexts[:3])
            return fallback

        # If Prompt Builder not available
        if self.prompt_builder is None:
            self.logger.warning("⚠️ Prompt Builder not available, using fallback")
            fallback = "На основе найденных документов:\n\n"
            fallback += "\n\n".join(contexts[:3])
            return fallback

        tracer = get_tracer()
        metrics = get_metrics()

        try:
            # Build prompt
            prompt = self.prompt_builder.build(
                question=query,
                contexts=contexts,
                include_system=True,
            )

            # Trace LLM generation
            if tracer:
                with tracer.trace_llm(self.model_name, len(prompt)):
                    if metrics:
                        with metrics.measure_llm(self.model_name):
                            answer = self.llm.generate(
                                prompt=prompt,
                                temperature=temperature,
                                max_tokens=max_tokens,
                            )
                    else:
                        answer = self.llm.generate(
                            prompt=prompt,
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )
            else:
                answer = self.llm.generate(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

            # Record metrics
            if metrics:
                metrics.record_tokens(len(answer.split()), self.model_name)

            # Fix encoding for Russian text
            answer = self._fix_encoding(answer)

            # If LLM returned error, use fallback
            if answer.startswith("Error:"):
                self.logger.warning(f"⚠️ LLM returned error: {answer[:50]}...")
                fallback = "На основе найденных документов:\n\n"
                fallback += "\n\n".join(contexts[:3])
                return fallback

            return answer.strip()

        except Exception as e:
            self.logger.error(f"❌ Generation failed: {e}")
            import traceback
            traceback.print_exc()
            # Fallback
            fallback = "На основе найденных документов:\n\n"
            fallback += "\n\n".join(contexts[:3])
            return fallback

    def query(
        self,
        question: str,
        top_k: int = 3,
        use_hybrid: bool = False,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> Dict[str, Any]:
        """
        Full RAG pipeline: retrieve → generate.

        Args:
            question: User question.
            top_k: Number of contexts to retrieve.
            use_hybrid: If True, use hybrid search.
            temperature: LLM temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            Dict with answer, contexts, latency, and metadata.
        """
        start_time = time.time()

        self.logger.info(f"📝 Processing query: {question[:100]}...")

        # Get tracer and metrics
        tracer = get_tracer()
        metrics = get_metrics()

        # Check if pipeline is ready
        if not self.is_ready:
            self.logger.warning("⚠️ Pipeline not ready, attempting to reinitialize...")
            self._reinitialize()

        # Start trace
        if tracer:
            with tracer.trace_request("query", question):
                result = self._query_impl(
                    question, top_k, use_hybrid, temperature, max_tokens
                )
        else:
            result = self._query_impl(
                question, top_k, use_hybrid, temperature, max_tokens
            )

        # Add latency
        result["latency_ms"] = (time.time() - start_time) * 1000

        self.logger.info(
            f"✅ Query complete: {len(result.get('contexts', []))} contexts, "
            f"{result['latency_ms']:.0f}ms"
        )

        return result

    def _query_impl(
        self,
        question: str,
        top_k: int,
        use_hybrid: bool,
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        """
        Internal query implementation.
        """
        # 1. Retrieve relevant documents
        results = self.search(
            query=question,
            top_k=top_k,
            use_hybrid=use_hybrid,
        )

        contexts = [r["text"] for r in results]

        # Record metrics
        metrics = get_metrics()
        if metrics:
            metrics.record_contexts(len(contexts))

        # 2. Generate answer
        answer = self.generate(
            query=question,
            contexts=contexts,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return {
            "answer": answer,
            "contexts": contexts,
            "from_cache": False,
            "status": "success" if contexts else "no_results",
            "metadata": {
                "num_contexts": len(contexts),
                "top_k": top_k,
                "use_hybrid": use_hybrid,
                "model": self.model_name,
                "llm_available": self.llm is not None and self.llm.is_available(),
                "retriever_available": self.retriever is not None,
            },
        }

    def _fix_encoding(self, text: str) -> str:
        """Fix encoding issues for Russian text."""
        if not text:
            return text

        try:
            # Try to decode from latin-1 to utf-8
            return text.encode('latin-1').decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            try:
                # Fallback: try windows-1251
                return text.encode('windows-1251').decode('utf-8')
            except (UnicodeEncodeError, UnicodeDecodeError):
                try:
                    # Another fallback
                    return text.encode('cp1251').decode('utf-8')
                except (UnicodeEncodeError, UnicodeDecodeError):
                    return text

    def _reinitialize(self) -> bool:
        """
        Reinitialize pipeline components if they failed.
        Returns True if successful.
        """
        self.logger.info("🔄 Reinitializing pipeline components...")

        # Reinitialize retriever
        if self.retriever is None:
            try:
                self.retriever = QdrantClientWrapper()
                self.logger.info("✅ Retriever reinitialized")
            except Exception as e:
                self.logger.error(f"❌ Failed to reinitialize retriever: {e}")

        # Reinitialize LLM
        if self.llm is None or not self.llm.is_available():
            try:
                self.llm = LocalLLM(
                    model_name=self.model_name,
                    use_ollama=self.use_ollama,
                )
                self.logger.info("✅ LLM reinitialized")
            except Exception as e:
                self.logger.error(f"❌ Failed to reinitialize LLM: {e}")

        # Reinitialize Prompt Builder
        if self.prompt_builder is None:
            try:
                self.prompt_builder = PromptBuilder()
                self.logger.info("✅ Prompt Builder reinitialized")
            except Exception as e:
                self.logger.error(f"❌ Failed to reinitialize Prompt Builder: {e}")

        # Update ready status
        self.is_ready = (
            self.retriever is not None and
            self.llm is not None and
            self.llm.is_available() and
            self.prompt_builder is not None
        )

        # Update metrics
        metrics = get_metrics()
        if metrics:
            metrics.update_pipeline_status(self.is_ready)

        return self.is_ready

    def get_status(self) -> Dict[str, Any]:
        """
        Get pipeline status information.

        Returns:
            Dict with component status information.
        """
        return {
            "ready": self.is_ready,
            "components": {
                "retriever": self.retriever is not None,
                "llm": self.llm is not None,
                "llm_available": self.llm is not None and self.llm.is_available(),
                "prompt_builder": self.prompt_builder is not None,
            },
            "model": {
                "name": self.model_name,
                "use_ollama": self.use_ollama,
            },
        }

    def get_stats(self) -> Dict[str, Any]:
        """
        Get pipeline statistics.

        Returns:
            Dict with statistics.
        """
        stats = {
            "status": self.get_status(),
        }

        # Get vector store stats
        if self.retriever is not None:
            try:
                count = self.retriever.count_points()
                stats["vector_store"] = {
                    "points_count": count,
                    "collection": self.retriever.collection_name,
                }
            except Exception as e:
                stats["vector_store"] = {"error": str(e)}

        return stats