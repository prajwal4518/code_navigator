#!/usr/bin/env python3
"""
CLI script for searching the code navigator index.

Usage:
    python scripts/search.py "parse Python code" --mode hybrid
    python scripts/search.py "CodeChunk" --mode keyword
    python scripts/search.py "extract functions" --mode semantic

Modes:
- semantic: Vector similarity only (good for concepts)
- keyword: BM25 exact matching (good for function/class names)
- hybrid: Combined with RRF (best of both)
"""

import argparse
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.table import Table

from code_navigator.retrieval import HybridRetriever, SearchMode

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="Search the code navigator index",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/search.py "parse Python code"
  python scripts/search.py "CodeChunk" --mode keyword
  python scripts/search.py "extract docstrings" --mode semantic
  python scripts/search.py "parse_file" --mode hybrid -k 10
        """,
    )
    parser.add_argument(
        "query",
        help="Search query",
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["semantic", "keyword", "hybrid"],
        default="hybrid",
        help="Search mode (default: hybrid)",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=5,
        help="Number of results (default: 5)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed scores",
    )

    args = parser.parse_args()

    # Print header
    console.print()
    console.print("[bold blue]🔍 Code Navigator - Search[/]")
    console.print()

    # Initialize retriever
    retriever = HybridRetriever()

    # Convert mode string to enum
    mode = SearchMode(args.mode)

    # Search
    console.print(f"[dim]Query:[/] [cyan]{args.query}[/]")
    console.print(f"[dim]Mode:[/] [yellow]{mode.value}[/]")
    console.print()

    results = retriever.search(args.query, k=args.k, mode=mode)

    if not results:
        console.print("[yellow]No results found.[/]")
        return

    # Display results
    if args.verbose:
        table = Table(title=f"Search Results ({mode.value})", show_lines=True)
        table.add_column("Score", style="green", width=8)
        table.add_column("Sem", style="blue", width=6)
        table.add_column("KW", style="magenta", width=6)
        table.add_column("Type", style="cyan", width=10)
        table.add_column("Name", style="bold", width=25)
        table.add_column("File", style="dim", width=25)

        for result in results:
            sem_score = f"{result.semantic_score:.2f}" if result.semantic_score else "-"
            kw_score = f"{result.keyword_score:.1f}" if result.keyword_score else "-"

            table.add_row(
                f"{result.combined_score:.3f}",
                sem_score,
                kw_score,
                result.chunk.chunk_type.value,
                result.chunk.name,
                result.chunk.relative_path,
            )
    else:
        table = Table(title=f"Search Results ({mode.value})", show_lines=True)
        table.add_column("Score", style="green", width=8)
        table.add_column("Type", style="cyan", width=10)
        table.add_column("Name", style="bold", width=30)
        table.add_column("File", style="dim", width=25)

        for result in results:
            table.add_row(
                f"{result.combined_score:.3f}",
                result.chunk.chunk_type.value,
                result.chunk.name,
                result.chunk.relative_path,
            )

    console.print(table)

    # Show first result preview
    if results:
        console.print()
        console.print("[bold]Top Result Preview:[/]")
        console.print(
            f"[dim]{results[0].chunk.file_path}:{results[0].chunk.start_line}[/]"
        )
        console.print()

        # Show first few lines of content
        content_lines = results[0].chunk.content.split("\n")[:10]
        for line in content_lines:
            console.print(f"  {line}")
        if len(results[0].chunk.content.split("\n")) > 10:
            console.print("  [dim]...[/]")


if __name__ == "__main__":
    main()
