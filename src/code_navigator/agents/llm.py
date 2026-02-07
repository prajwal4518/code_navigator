"""
Gemini LLM client for code-related tasks.

Why Gemini?
- Large context window (up to 1M tokens) for full codebase understanding
- Strong code comprehension capabilities
- Cost-effective for development

This module provides a thin wrapper with:
- Environment-based API key management
- Token counting for cost awareness
- Streaming support for real-time responses
"""

import os
from dataclasses import dataclass, field

import google.generativeai as genai
from rich.console import Console

console = Console()


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


@dataclass
class GeminiClient:
    """Client for Gemini API interactions.

    Usage:
        >>> client = GeminiClient()
        >>> response = client.generate("Explain this code: ...")
        >>> print(response)
    """

    model_name: str = "gemini-2.0-flash"
    temperature: float = 0.3  # Lower for more deterministic code explanations
    max_output_tokens: int = 2048
    usage: TokenUsage = field(default_factory=TokenUsage)
    _model: genai.GenerativeModel | None = None

    def __post_init__(self):
        """Configure the Gemini API."""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found in environment. "
                "Please set it in your .env file."
            )
        genai.configure(api_key=api_key)

    @property
    def model(self) -> genai.GenerativeModel:
        """Lazy-load the model."""
        if self._model is None:
            self._model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=genai.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_output_tokens,
                ),
            )
        return self._model

    def generate(self, prompt: str) -> str:
        """Generate a response from the model.

        Args:
            prompt: The input prompt

        Returns:
            Generated text response
        """
        response = self.model.generate_content(prompt)

        # Track token usage if available
        if hasattr(response, "usage_metadata"):
            self.usage.add(
                prompt=response.usage_metadata.prompt_token_count,
                completion=response.usage_metadata.candidates_token_count,
            )

        return response.text

    def generate_stream(self, prompt: str):
        """Generate a streaming response.

        Args:
            prompt: The input prompt

        Yields:
            Text chunks as they are generated
        """
        response = self.model.generate_content(prompt, stream=True)

        for chunk in response:
            if chunk.text:
                yield chunk.text

    def get_usage_stats(self) -> dict:
        """Get token usage statistics."""
        return {
            "prompt_tokens": self.usage.prompt_tokens,
            "completion_tokens": self.usage.completion_tokens,
            "total_tokens": self.usage.total_tokens,
        }


def get_gemini_client() -> GeminiClient:
    """Get the default Gemini client."""
    return GeminiClient()
