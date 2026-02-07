#!/usr/bin/env python3
"""
CLI script to index a repository into the vector store.

Usage:
    python scripts/index_to_vectordb.py ./path/to/repo
    python scripts/index_to_vectordb.py --url https://github.com/user/repo
    python scripts/index_to_vectordb.py ./path/to/repo --reset
    python scripts/index_to_vectordb.py ./path/to/repo --search "parse code"

This script combines ingestion and vector storage:
1. Chunk the repository using tree-sitter
2. Generate embeddings using sentence-transformers
3. Store in ChromaDB for semantic search
"""

import argparse
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.table import Table

from code_navigator.ingestion import CodeChunker, Language, RepoManager
from code_navigator.vectorstore import VectorStore

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="Index a repository into the vector store",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/index_to_vectordb.py ./my-repo
  python scripts/index_to_vectordb.py --url https://github.com/pallets/flask
  python scripts/index_to_vectordb.py ./src --reset
  python scripts/index_to_vectordb.py ./api --search "authentication"
        """,
    )
    parser.add_argument(
        "repo_path",
        nargs="?",
        help="Path to the repository or directory to index",
    )
    parser.add_argument(
        "--url",
        "-u",
        help="Git repository URL to clone and index",
    )
    parser.add_argument(
        "--branch",
        "-b",
        help="Branch to clone (default: default branch)",
    )
    parser.add_argument(
        "--language",
        "-l",
        choices=["python", "javascript", "typescript"],
        help="Filter to a specific language",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear existing data before indexing",
    )
    parser.add_argument(
        "--search",
        "-s",
        help="Test search query after indexing",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    # Validate inputs - need either repo_path or url
    if not args.repo_path and not args.url:
        console.print("[red]Error:[/] Either repo_path or --url is required")
        parser.print_help()
        sys.exit(1)

    # Handle git URL
    if args.url:
        console.print()
        console.print("[bold blue]🔍 Code Navigator - Vector Store Indexer[/]")
        console.print()

        repo_manager = RepoManager()
        try:
            repo_path = repo_manager.clone_or_update(
                args.url, branch=args.branch, verbose=not args.quiet
            )
        except RuntimeError as e:
            console.print(f"[red]Error:[/] {e}")
            sys.exit(1)
    else:
        # Use local path
        repo_path = Path(args.repo_path).resolve()
        if not repo_path.exists():
            console.print(f"[red]Error:[/] Path not found: {repo_path}")
            sys.exit(1)

        console.print()
        console.print("[bold blue]🔍 Code Navigator - Vector Store Indexer[/]")

    console.print()

    # Parse language filter
    languages = None
    if args.language:
        languages = [Language(args.language)]

    # Initialize components
    chunker = CodeChunker(verbose=not args.quiet)
    store = VectorStore()

    # Clear if requested
    if args.reset:
        store.clear()

    # Show current stats
    stats = store.get_stats()
    console.print(f"[dim]Current store: {stats['chunk_count']} chunks[/]")
    console.print()

    # Chunk the repository
    console.print("[bold]Step 1: Chunking repository...[/]")
    chunks = chunker.chunk_repository(repo_path, languages=languages)

    if not chunks:
        console.print("[yellow]No chunks extracted.[/]")
        return

    # Add to vector store
    console.print()
    console.print("[bold]Step 2: Adding to vector store...[/]")
    store.add_chunks(chunks)

    # Show final stats
    console.print()
    stats = store.get_stats()
    console.print("[bold green]✓ Indexing complete![/]")
    console.print(f"  Total chunks in store: {stats['chunk_count']}")
    console.print(f"  Embedding dimension: {stats['embedding_dimension']}")
    console.print(f"  Persist directory: {stats['persist_dir']}")

    # Test search if requested
    if args.search:
        console.print()
        console.print(f"[bold]Searching for: [cyan]'{args.search}'[/][/]")
        console.print()

        results = store.search(args.search, k=5)

        if results:
            table = Table(title="Search Results", show_lines=True)
            table.add_column("Score", style="green", width=8)
            table.add_column("Type", style="cyan", width=10)
            table.add_column("Name", style="bold", width=25)
            table.add_column("File", style="dim", width=25)

            for chunk, score in results:
                table.add_row(
                    f"{score:.3f}",
                    chunk.chunk_type.value,
                    chunk.name,
                    chunk.relative_path,
                )

            console.print(table)
        else:
            console.print("[yellow]No results found.[/]")


if __name__ == "__main__":
    main()
