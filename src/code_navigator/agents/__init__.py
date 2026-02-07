"""
Agents Module - LLM-powered code understanding.

This module provides:
- LLMClient: Provider-agnostic LLM wrapper (Gemini, Ollama, Anthropic, OpenAI)
- CodeNavigator: RAG agent combining retrieval + LLM

Usage:
    >>> from code_navigator.agents import CodeNavigator
    >>> navigator = CodeNavigator()
    >>> answer = navigator.ask("What does this function do?")
"""

from .llm import LLMClient, TokenUsage, get_llm
from .navigator import CodeNavigator, Message, get_code_navigator

# Backward compatibility
GeminiClient = LLMClient


def get_gemini_client() -> LLMClient:
    """Backward compatibility wrapper."""
    return LLMClient()


__all__ = [
    # LLM
    "LLMClient",
    "GeminiClient",  # Backward compat
    "TokenUsage",
    "get_llm",
    "get_gemini_client",  # Backward compat
    # Navigator
    "CodeNavigator",
    "Message",
    "get_code_navigator",
]
