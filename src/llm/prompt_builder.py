"""
Prompt builder for RAG generation.
Creates structured prompts with context for LLM.
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class PromptBuilder:
    """
    Builds prompts for LLM generation with context from retrieved documents.
    """

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        max_context_length: int = 4000,
    ):
        """
        Initialize PromptBuilder.

        Args:
            system_prompt: Custom system prompt.
            max_context_length: Maximum context characters to include.
        """
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.max_context_length = max_context_length
        self.logger = logging.getLogger(f"{__name__}.PromptBuilder")

    def _default_system_prompt(self) -> str:
        """Default system prompt for RAG."""
        return """Ты — полезный ассистент, который отвечает на вопросы, используя только предоставленный контекст.

Правила:
1. Отвечай только на основе предоставленного контекста.
2. Если в контексте нет информации для ответа, честно скажи об этом.
3. Не выдумывай факты и не используй свои собственные знания.
4. Если информация есть в контексте — приведи её чётко и структурированно.
5. Отвечай на русском языке, если вопрос задан на русском.
6. Будь вежливым и профессиональным.
"""

    def _truncate_context(self, contexts: List[str]) -> str:
        """Truncate contexts to max length."""
        combined = "\n\n---\n\n".join(contexts)

        if len(combined) <= self.max_context_length:
            return combined

        # Truncate keeping complete chunks
        truncated = []
        current_len = 0

        for ctx in contexts:
            if current_len + len(ctx) + 4 <= self.max_context_length:
                truncated.append(ctx)
                current_len += len(ctx) + 4
            else:
                break

        if not truncated:
            # If even first chunk is too long, truncate it
            return contexts[0][:self.max_context_length]

        return "\n\n---\n\n".join(truncated)

    def build(
        self,
        question: str,
        contexts: List[str],
        include_system: bool = True,
    ) -> str:
        """
        Build a prompt for LLM generation.

        Args:
            question: User question.
            contexts: Retrieved context strings.
            include_system: Whether to include system prompt.

        Returns:
            Formatted prompt string.
        """
        # Truncate contexts
        context_text = self._truncate_context(contexts)

        # Build prompt parts
        parts = []

        if include_system:
            parts.append(f"[INST] {self.system_prompt} [/INST]")

        if context_text:
            parts.append(f"Контекст:\n{context_text}")

        parts.append(f"Вопрос: {question}")

        parts.append(
            "Ответ (на основе контекста, если информации нет — честно скажи об этом):"
        )

        prompt = "\n\n".join(parts)

        self.logger.debug(f"Built prompt: {len(prompt)} chars")
        return prompt

    def build_chat_prompt(
        self,
        question: str,
        contexts: List[str],
        chat_history: Optional[List[dict]] = None,
    ) -> str:
        """
        Build a chat-style prompt with history.

        Args:
            question: User question.
            contexts: Retrieved context strings.
            chat_history: List of previous messages [{"role": "user/assistant", "content": "..."}].

        Returns:
            Formatted chat prompt.
        """
        # Truncate contexts
        context_text = self._truncate_context(contexts)

        # Build chat prompt
        messages = []

        # System message
        messages.append(f"[SYSTEM] {self.system_prompt}")

        if context_text:
            messages.append(f"[CONTEXT]\n{context_text}")

        # Chat history
        if chat_history:
            for msg in chat_history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                messages.append(f"[{role.upper()}] {content}")

        # Current question
        messages.append(f"[USER] {question}")
        messages.append("[ASSISTANT]")

        prompt = "\n\n".join(messages)

        self.logger.debug(f"Built chat prompt: {len(prompt)} chars")
        return prompt

    def get_context_stats(self, contexts: List[str]) -> dict:
        """Get statistics about context."""
        total_chars = sum(len(c) for c in contexts)
        return {
            "num_contexts": len(contexts),
            "total_chars": total_chars,
            "avg_chars": total_chars / len(contexts) if contexts else 0,
        }