#!/usr/bin/env python3
"""
CLI script to index/chunk a repository.

Usage:
    python scripts/index_repo.py ./path/to/repo
    python scripts/index_repo.py ./path/to/repo --language python
    python scripts/index_repo.py ./path/to/repo --output chunks.json

This script is for testing the ingestion pipeline during development.
In production, chunking will be part of the full RAG pipeline.
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.table import Table

from code_navigator.ingestion import (
    CodeChunker,
    Language,
)

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="Index a code repository into chunks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/index_repo.py ./my-repo
  python scripts/index_repo.py ./src --language python
  python scripts/index_repo.py ./api --output api_chunks.json
        """,
    )
    parser.add_argument(
        "repo_path",
        help="Path to the repository or directory to index",
    )
    parser.add_argument(
        "--language",
        "-l",
        choices=["python", "javascript", "typescript"],
        help="Filter to a specific language",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file for chunks (JSON format)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=None,
        help="Limit number of chunks to display",
    )

    args = parser.parse_args()

    # Validate repo path
    repo_path = Path(args.repo_path).resolve()
    if not repo_path.exists():
        console.print(f"[red]Error:[/] Path not found: {repo_path}")
        sys.exit(1)

    # Parse language filter
    languages = None
    if args.language:
        languages = [Language(args.language)]

    # Run chunking
    console.print()
    console.print("[bold blue]🔍 Code Navigator - Repository Indexer[/]")
    console.print()

    chunker = CodeChunker(verbose=not args.quiet)
    chunks = chunker.chunk_repository(repo_path, languages=languages)

    if not chunks:
        console.print("[yellow]No chunks extracted.[/]")
        return

    # Display results
    console.print()

    # Create summary table
    table = Table(title="Extracted Chunks", show_lines=True)
    table.add_column("Type", style="cyan", width=10)
    table.add_column("Name", style="green", width=30)
    table.add_column("File", style="dim", width=25)
    table.add_column("Lines", style="magenta", width=10)
    table.add_column("Docstring", style="yellow", width=40)

    display_chunks = chunks[: args.limit] if args.limit else chunks

    for chunk in display_chunks:
        docstring_preview = ""
        if chunk.docstring:
            docstring_preview = (
                chunk.docstring[:37] + "..."
                if len(chunk.docstring) > 40
                else chunk.docstring
            )
            docstring_preview = docstring_preview.replace("\n", " ")

        table.add_row(
            chunk.chunk_type.value,
            chunk.name,
            chunk.relative_path,
            f"{chunk.start_line}-{chunk.end_line}",
            docstring_preview or "-",
        )

    console.print(table)

    if args.limit and len(chunks) > args.limit:
        console.print(f"\n[dim]Showing {args.limit} of {len(chunks)} chunks[/]")

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)

        # Convert to JSON-serializable format
        chunks_data = [
            {
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "chunk_type": chunk.chunk_type.value,
                "name": chunk.name,
                "file_path": chunk.file_path,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "language": chunk.language.value,
                "docstring": chunk.docstring,
                "parent_name": chunk.parent_name,
                "decorators": chunk.decorators,
            }
            for chunk in chunks
        ]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)

        console.print(f"\n[green]✓ Saved {len(chunks)} chunks to:[/] {output_path}")


if __name__ == "__main__":
    main()
