"""
Provider-agnostic LLM factory using LangChain.

Why LangChain?
- Unified interface across providers (Gemini, Ollama, Anthropic, OpenAI)
- Easy model switching via environment variables
- Built-in streaming and async support

Supported Providers:
- gemini: Google Gemini (requires GOOGLE_API_KEY)
- ollama: Local Ollama (requires OLLAMA_BASE_URL)
- anthropic: Claude (requires ANTHROPIC_API_KEY)
- openai: GPT-4 (requires OPENAI_API_KEY)

Usage:
    >>> from code_navigator.agents.llm import get_llm
    >>> llm = get_llm()  # Uses LLM_PROVIDER from env
    >>> response = llm.invoke("Explain this code...")
"""

import os
from collections.abc import Iterator
from dataclasses import dataclass, field

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from rich.console import Console

console = Console()

# Default models per provider
DEFAULT_MODELS = {
    "gemini": "gemini-2.0-flash",
    "ollama": "llama3.2",
    "anthropic": "claude-3-5-sonnet-latest",
    "openai": "gpt-4o-mini",
}


@dataclass
class TokenUsage:
    """Tracks token usage for cost monitoring."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, prompt: int, completion: int) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += prompt + completion


def _create_gemini_llm(model: str, temperature: float) -> BaseChatModel:
    """Create a Google Gemini LLM."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set")

    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        google_api_key=api_key,
    )


def _create_ollama_llm(model: str, temperature: float) -> BaseChatModel:
    """Create a local Ollama LLM."""
    from langchain_ollama import ChatOllama

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    return ChatOllama(
        model=model,
        temperature=temperature,
        base_url=base_url,
    )


def _create_anthropic_llm(model: str, temperature: float) -> BaseChatModel:
    """Create an Anthropic Claude LLM."""
    from langchain_anthropic import ChatAnthropic

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    return ChatAnthropic(
        model=model,
        temperature=temperature,
        anthropic_api_key=api_key,
    )


def _create_openai_llm(model: str, temperature: float) -> BaseChatModel:
    """Create an OpenAI LLM."""
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        openai_api_key=api_key,
    )


def get_llm(
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.3,
) -> BaseChatModel:
    """Create an LLM instance based on provider configuration.

    Args:
        provider: LLM provider (gemini, ollama, anthropic, openai).
                  Defaults to LLM_PROVIDER env var or 'gemini'.
        model: Model name. Defaults to LLM_MODEL env var or provider default.
        temperature: Generation temperature (0.0-1.0).

    Returns:
        A LangChain BaseChatModel instance.

    Raises:
        ValueError: If provider is not supported or required env vars missing.
    """
    provider = provider or os.getenv("LLM_PROVIDER", "gemini")
    model = model or os.getenv("LLM_MODEL") or DEFAULT_MODELS.get(provider, "")

    console.print(f"[dim]Using LLM: {provider}/{model}[/]")

    match provider.lower():
        case "gemini":
            return _create_gemini_llm(model, temperature)
        case "ollama":
            return _create_ollama_llm(model, temperature)
        case "anthropic":
            return _create_anthropic_llm(model, temperature)
        case "openai":
            return _create_openai_llm(model, temperature)
        case _:
            raise ValueError(
                f"Unknown provider: {provider}. "
                f"Supported: gemini, ollama, anthropic, openai"
            )


@dataclass
class LLMClient:
    """Wrapper for LangChain LLMs with usage tracking.

    Provides a consistent interface for all providers with:
    - Token usage tracking
    - Streaming support
    - System prompt handling

    Usage:
        >>> client = LLMClient()
        >>> response = client.generate("Explain this code: ...")
        >>> print(response)
    """

    provider: str | None = None
    model: str | None = None
    temperature: float = 0.3
    usage: TokenUsage = field(default_factory=TokenUsage)
    _llm: BaseChatModel | None = None

    def __post_init__(self):
        """Initialize the LLM."""
        self._llm = get_llm(
            provider=self.provider,
            model=self.model,
            temperature=self.temperature,
        )

    @property
    def llm(self) -> BaseChatModel:
        """Get the underlying LLM."""
        if self._llm is None:
            self._llm = get_llm(
                provider=self.provider,
                model=self.model,
                temperature=self.temperature,
            )
        return self._llm

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generate a response from the model.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt

        Returns:
            Generated text response
        """
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        response = self.llm.invoke(messages)

        # Track usage if available
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            self.usage.add(
                prompt=response.usage_metadata.get("input_tokens", 0),
                completion=response.usage_metadata.get("output_tokens", 0),
            )

        return response.content

    def generate_stream(
        self, prompt: str, system_prompt: str | None = None
    ) -> Iterator[str]:
        """Generate a streaming response.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt

        Yields:
            Text chunks as they are generated
        """
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        for chunk in self.llm.stream(messages):
            if hasattr(chunk, "content") and chunk.content:
                yield chunk.content

    def get_usage_stats(self) -> dict:
        """Get token usage statistics."""
        return {
            "prompt_tokens": self.usage.prompt_tokens,
            "completion_tokens": self.usage.completion_tokens,
            "total_tokens": self.usage.total_tokens,
        }


# Backward compatibility
GeminiClient = LLMClient


def get_gemini_client() -> LLMClient:
    """Get an LLM client (backward compatibility)."""
    return LLMClient()
