"""
CodeNavigator RAG Agent.

This is the core agent that combines:
- Hybrid retrieval (semantic + keyword)
- LLM for synthesis (provider-agnostic via LangChain)
- Conversation memory for context

The RAG pipeline:
1. User asks a question about code
2. Retriever finds relevant code chunks
3. Chunks are formatted as context
4. LLM generates an answer with citations
"""

from dataclasses import dataclass, field

from rich.console import Console

from code_navigator.retrieval import HybridRetriever, SearchMode, SearchResult

from .llm import LLMClient

console = Console()


# System prompt for code understanding
SYSTEM_PROMPT = """You are a code navigation assistant. Your job is to help developers understand codebases.

When answering questions:
1. Base your answers ONLY on the provided code context
2. Cite specific files and line numbers when referencing code
3. If the context doesn't contain relevant information, say so
4. Explain code clearly, focusing on what it does and why
5. Use markdown formatting for code snippets

Format citations as: `file.py:line`"""


@dataclass
class Message:
    """A single message in the conversation."""

    role: str  # "user" or "assistant"
    content: str


@dataclass
class CodeNavigator:
    """RAG agent for navigating and understanding code.

    Usage:
        >>> navigator = CodeNavigator()
        >>> answer = navigator.ask("What does the parse_file method do?")
        >>> print(answer)
    """

    retriever: HybridRetriever | None = None
    llm: LLMClient | None = None
    history: list[Message] = field(default_factory=list)
    max_history: int = 10  # Keep last N exchanges
    num_chunks: int = 5  # Number of chunks to retrieve

    def __post_init__(self):
        """Initialize components."""
        if self.retriever is None:
            self.retriever = HybridRetriever()
        if self.llm is None:
            self.llm = LLMClient()

    def _format_context(self, results: list[SearchResult]) -> str:
        """Format retrieved chunks as context for the LLM.

        Creates a structured context with clear boundaries between chunks.
        """
        if not results:
            return "No relevant code found in the codebase."

        context_parts = []
        for i, result in enumerate(results, 1):
            chunk = result.chunk
            context_parts.append(
                f"--- Code Chunk {i} ---\n"
                f"File: {chunk.file_path}:{chunk.start_line}-{chunk.end_line}\n"
                f"Type: {chunk.chunk_type.value}\n"
                f"Name: {chunk.name}\n"
                f"```python\n{chunk.content}\n```"
            )

        return "\n\n".join(context_parts)

    def _format_history(self) -> str:
        """Format conversation history for context."""
        if not self.history:
            return ""

        history_text = "\n--- Previous Conversation ---\n"
        for msg in self.history[-self.max_history :]:
            role = "User" if msg.role == "user" else "Assistant"
            # Truncate long messages
            content = (
                msg.content[:500] + "..." if len(msg.content) > 500 else msg.content
            )
            history_text += f"{role}: {content}\n"

        return history_text

    def _build_prompt(self, question: str, context: str) -> str:
        """Build the full prompt for the LLM."""
        history = self._format_history()

        prompt = f"""{SYSTEM_PROMPT}

{history}

--- Retrieved Code Context ---
{context}

--- User Question ---
{question}

Please answer the question based on the code context above. Cite specific files and line numbers."""

        return prompt

    def ask(self, question: str, stream: bool = False) -> str:
        """Ask a question about the codebase.

        This is a stateless query - no conversation history is maintained.

        Args:
            question: Natural language question about the code
            stream: If True, print response as it streams

        Returns:
            The assistant's answer
        """
        # Retrieve relevant code
        results = self.retriever.search(
            question, k=self.num_chunks, mode=SearchMode.HYBRID
        )

        # Format context
        context = self._format_context(results)

        # Build prompt
        prompt = self._build_prompt(question, context)

        # Generate response
        if stream:
            response_text = ""
            for chunk in self.llm.generate_stream(prompt):
                console.print(chunk, end="")
                response_text += chunk
            console.print()  # Newline after streaming
            return response_text
        else:
            return self.llm.generate(prompt)

    def chat(self, message: str, stream: bool = True) -> str:
        """Chat with conversation history.

        Maintains context from previous exchanges for follow-up questions.

        Args:
            message: User's message
            stream: If True, stream the response

        Returns:
            The assistant's response
        """
        # Add user message to history
        self.history.append(Message(role="user", content=message))

        # Generate response
        response = self.ask(message, stream=stream)

        # Add assistant response to history
        self.history.append(Message(role="assistant", content=response))

        # Trim history if too long
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-self.max_history * 2 :]

        return response

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.history = []
        console.print("[dim]Conversation history cleared.[/]")

    def get_stats(self) -> dict:
        """Get agent statistics."""
        return {
            "history_length": len(self.history),
            "llm_usage": self.llm.get_usage_stats(),
            "retriever_index_size": self.retriever.bm25_index.size,
        }


def get_code_navigator() -> CodeNavigator:
    """Get the default CodeNavigator instance."""
    return CodeNavigator()
