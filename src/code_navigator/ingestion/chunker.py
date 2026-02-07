"""
Code chunker - Orchestrates the ingestion pipeline.

This module ties together file discovery and parsing to produce
a complete set of code chunks from a repository.

Why a separate chunker class?
- Single entry point for the ingestion pipeline
- Can add cross-cutting concerns: logging, metrics, caching
- Handles edge cases: empty files, parse errors, encoding issues
"""

from collections.abc import Iterator
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .file_discovery import discover_files
from .models import CodeChunk, FileInfo, Language
from .parser import PythonParser

console = Console()


class CodeChunker:
    """Main orchestrator for code chunking.

    Usage:
        >>> chunker = CodeChunker()
        >>> chunks = chunker.chunk_repository("./my-repo")
        >>> for chunk in chunks:
        ...     print(chunk.name, chunk.chunk_type)
    """

    def __init__(self, verbose: bool = True):
        """Initialize the chunker.

        Args:
            verbose: If True, show progress output
        """
        self.verbose = verbose
        self._python_parser = PythonParser()

        # Statistics for reporting
        self._stats = {
            "files_processed": 0,
            "files_failed": 0,
            "chunks_extracted": 0,
        }

    def chunk_repository(
        self,
        repo_path: str | Path,
        languages: list[Language] | None = None,
    ) -> list[CodeChunk]:
        """Chunk all code files in a repository.

        Args:
            repo_path: Path to the repository root
            languages: Filter to specific languages (None = all supported)

        Returns:
            List of all extracted CodeChunk objects
        """
        all_chunks: list[CodeChunk] = []

        # Reset statistics
        self._stats = {
            "files_processed": 0,
            "files_failed": 0,
            "chunks_extracted": 0,
        }

        for chunk in self.chunk_repository_iter(repo_path, languages):
            all_chunks.append(chunk)

        if self.verbose:
            self._print_summary()

        return all_chunks

    def chunk_repository_iter(
        self,
        repo_path: str | Path,
        languages: list[Language] | None = None,
    ) -> Iterator[CodeChunk]:
        """Iterate over chunks from a repository (memory-efficient).

        Use this for large repositories where you want to process
        chunks one at a time without loading all into memory.
        """
        repo_path = Path(repo_path).resolve()

        # Discover files
        if self.verbose:
            console.print(f"[bold blue]Discovering files in:[/] {repo_path}")

        files = discover_files(repo_path, languages=languages)

        if self.verbose:
            console.print(f"[green]Found {len(files)} files to process[/]")

        # Process each file
        if self.verbose:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Processing...", total=len(files))

                for file_info in files:
                    progress.update(
                        task, description=f"Processing {file_info.relative_name}"
                    )

                    for chunk in self._process_file(file_info):
                        yield chunk

                    progress.advance(task)
        else:
            for file_info in files:
                for chunk in self._process_file(file_info):
                    yield chunk

    def chunk_file(self, file_path: str | Path) -> list[CodeChunk]:
        """Chunk a single file.

        Args:
            file_path: Path to the source file

        Returns:
            List of CodeChunk objects from the file
        """
        file_path = Path(file_path).resolve()

        # Detect language
        from .file_discovery import detect_language

        language = detect_language(file_path)

        if language is None:
            if self.verbose:
                console.print(f"[yellow]Unsupported file type:[/] {file_path}")
            return []

        file_info = FileInfo(
            path=file_path,
            language=language,
            size_bytes=file_path.stat().st_size,
        )

        return list(self._process_file(file_info))

    def _process_file(self, file_info: FileInfo) -> Iterator[CodeChunk]:
        """Process a single file and yield chunks."""
        try:
            if file_info.language == Language.PYTHON:
                chunks = self._python_parser.parse_file(file_info.path)
            else:
                # TODO: Add parsers for other languages
                if self.verbose:
                    console.print(
                        f"[yellow]Parser not implemented for:[/] "
                        f"{file_info.language.value}"
                    )
                return

            self._stats["files_processed"] += 1
            self._stats["chunks_extracted"] += len(chunks)

            yield from chunks

        except Exception as e:
            self._stats["files_failed"] += 1
            if self.verbose:
                console.print(
                    f"[red]Failed to parse:[/] {file_info.path}\n  Error: {e}"
                )

    def _print_summary(self) -> None:
        """Print processing summary."""
        console.print()
        console.print("[bold]Chunking Summary:[/]")
        console.print(f"  Files processed: {self._stats['files_processed']}")
        console.print(f"  Files failed: {self._stats['files_failed']}")
        console.print(f"  Chunks extracted: {self._stats['chunks_extracted']}")

    @property
    def stats(self) -> dict:
        """Get processing statistics."""
        return self._stats.copy()


def chunk_repository(
    repo_path: str | Path,
    languages: list[Language] | None = None,
    verbose: bool = True,
) -> list[CodeChunk]:
    """Convenience function to chunk a repository.

    Example:
        >>> from code_navigator.ingestion import chunk_repository
        >>> chunks = chunk_repository("./my-repo")
    """
    chunker = CodeChunker(verbose=verbose)
    return chunker.chunk_repository(repo_path, languages)
