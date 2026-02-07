"""
Agents Module - LLM-powered code understanding.

This module handles:
- Gemini API integration
- RAG-based code Q&A
- Conversation memory
- Response generation with citations

Why an "agent" instead of just prompts?
- Agents can retrieve relevant context before answering
- They maintain conversation state for follow-ups
- They can use tools (search, filter) to find information

Usage:
    >>> from code_navigator.agents import CodeNavigator
    >>> nav = CodeNavigator()
    >>> answer = nav.ask("What does the parse_file method do?")
"""

from .llm import GeminiClient, TokenUsage, get_gemini_client
from .navigator import CodeNavigator, Message, get_code_navigator

__all__ = [
    # LLM
    "GeminiClient",
    "TokenUsage",
    "get_gemini_client",
    # Navigator
    "CodeNavigator",
    "Message",
    "get_code_navigator",
]
