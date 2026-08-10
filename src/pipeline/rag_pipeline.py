"""
Main RAG Pipeline — orchestrates retrieval and generation.
"""

import logging
import time
from typing import List, Dict, Any, Optional

from src.retrieval import QdrantClientWrapper
from src.llm import LocalLLM, PromptBuilder
from src.config import settings

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Main RAG pipeline orchestrating search and generation."""

    def __init__(
        self,
        model_name: str = "qwen2.5-coder:7b",
        use_ollama: bool = True,
    ):
        self.logger = logging.getLogger(f"{__name__}.RAGPipeline")

        # Initialize retriever
        try:
            self.retriever = QdrantClientWrapper()
            self.logger.info("✅ Retriever initialized")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize retriever: {e}")
            self.retriever = None

        # Initialize LLM
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
        self.prompt_builder = PromptBuilder()
        self.logger.info("✅ Prompt Builder initialized")

    def search(
        self,
        query: str,
        top_k: int = 3,
        use_hybrid: bool = False,
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant documents from Qdrant."""
        if self.retriever is None:
            self.logger.error("Retriever not available")
            return []

        try:
            return self.retriever.search(
                query=query,
                top_k=top_k,
                use_hybrid=use_hybrid,
            )
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
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
        if self.llm is None:
            # Fallback: return context-based answer without LLM
            if contexts:
                return "На основе найденных документов:\n\n" + "\n\n".join(contexts[:3])
            return "Не удалось найти информацию по вашему запросу."

        try:
            # Build prompt
            prompt = self.prompt_builder.build(
                question=query,
                contexts=contexts,
                include_system=True,
            )

            # Generate answer
            answer = self.llm.generate(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Post-process answer
            if answer.startswith("Error:"):
                # Fallback if LLM failed
                if contexts:
                    return "На основе найденных документов:\n\n" + "\n\n".join(contexts[:3])
                return "Не удалось найти информацию по вашему запросу."

            return answer

        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            # Fallback
            if contexts:
                return "На основе найденных документов:\n\n" + "\n\n".join(contexts[:3])
            return "Не удалось найти информацию по вашему запросу."

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

        Returns:
            Dict with answer, contexts, latency, metadata.
        """
        start_time = time.time()

        # 1. Retrieve relevant documents
        results = self.search(
            query=question,
            top_k=top_k,
            use_hybrid=use_hybrid,
        )

        contexts = [r["text"] for r in results]

        # 2. Generate answer
        answer = self.generate(
            query=question,
            contexts=contexts,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        latency_ms = (time.time() - start_time) * 1000

        return {
            "answer": answer,
            "contexts": contexts,
            "latency_ms": latency_ms,
            "from_cache": False,
            "status": "success" if contexts else "no_results",
            "metadata": {
                "num_contexts": len(contexts),
                "llm_available": self.llm is not None and self.llm.is_available(),
            },
        }