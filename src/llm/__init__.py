"""
LLM Module — Local LLM integration for response generation.
"""

from src.llm.local_llm import LocalLLM
from src.llm.prompt_builder import PromptBuilder

__all__ = ["LocalLLM", "PromptBuilder"]