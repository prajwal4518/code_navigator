#!/usr/bin/env python3
"""
Interactive chat CLI for the Code Navigator.

Usage:
    python scripts/chat.py

Commands:
    /quit, /exit    Exit the chat
    /clear          Clear conversation history
    /stats          Show usage statistics
    /help           Show this help message

Example:
    > What does the parse_file method do?
    > How does BM25 scoring work in this codebase?
    > /quit
"""

import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Load environment variables before importing our modules
load_dotenv()

from code_navigator.agents import CodeNavigator  # noqa: E402

console = Console()


def print_help():
    """Print help message."""
    help_text = """
**Commands:**
- `/quit` or `/exit` — Exit the chat
- `/clear` — Clear conversation history
- `/stats` — Show token usage statistics
- `/help` — Show this message

**Tips:**
- Ask about specific functions, classes, or concepts
- Follow up with "why" or "how" questions
- Reference previous answers for clarification
    """
    console.print(Markdown(help_text))


def main():
    console.print()
    console.print(
        Panel.fit(
            "[bold blue]🧭 Code Navigator[/]\n"
            "[dim]Ask questions about your codebase[/]",
            border_style="blue",
        )
    )
    console.print()
    console.print("[dim]Type /help for commands, /quit to exit[/]")
    console.print()

    # Initialize navigator
    try:
        navigator = CodeNavigator()
    except ValueError as e:
        console.print(f"[red]Error:[/] {e}")
        console.print("[dim]Make sure GOOGLE_API_KEY is set in your .env file[/]")
        sys.exit(1)

    while True:
        try:
            # Get user input
            user_input = console.input("[bold green]>[/] ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() in ("/quit", "/exit", "/q"):
                console.print("[dim]Goodbye![/]")
                break

            if user_input.lower() == "/clear":
                navigator.clear_history()
                continue

            if user_input.lower() == "/stats":
                stats = navigator.get_stats()
                console.print()
                console.print("[bold]Usage Statistics[/]")
                console.print(
                    f"  Conversation length: {stats['history_length']} messages"
                )
                console.print(f"  Prompt tokens: {stats['llm_usage']['prompt_tokens']}")
                console.print(
                    f"  Completion tokens: {stats['llm_usage']['completion_tokens']}"
                )
                console.print(f"  Total tokens: {stats['llm_usage']['total_tokens']}")
                console.print(f"  Index size: {stats['retriever_index_size']} chunks")
                console.print()
                continue

            if user_input.lower() == "/help":
                print_help()
                continue

            if user_input.startswith("/"):
                console.print(f"[yellow]Unknown command: {user_input}[/]")
                console.print("[dim]Type /help for available commands[/]")
                continue

            # Process the question
            console.print()
            console.print("[dim]Thinking...[/]")

            # Stream the response
            navigator.chat(user_input, stream=True)

            console.print()

        except KeyboardInterrupt:
            console.print("\n[dim]Use /quit to exit[/]")
            continue

        except Exception as e:
            console.print(f"[red]Error:[/] {e}")
            continue


if __name__ == "__main__":
    main()
