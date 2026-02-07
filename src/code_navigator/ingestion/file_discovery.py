"""
File discovery module - Traverses repositories to find code files.

Why a dedicated module for file discovery?
- Separation of concerns: discovery vs parsing are different responsibilities
- Respects .gitignore patterns (don't index what's not in version control)
- Handles edge cases: binary files, symlinks, large files
- Easy to extend: add new ignore patterns, file type filters
"""

import fnmatch
from pathlib import Path

from .models import EXTENSION_TO_LANGUAGE, FileInfo, Language

# Directories to always skip (common across all languages)
DEFAULT_IGNORE_DIRS: set[str] = {
    ".git",
    ".svn",
    ".hg",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    ".venv",
    "venv",
    "env",
    ".env",
    "dist",
    "build",
    ".eggs",
    "*.egg-info",
    ".tox",
    ".nox",
    "htmlcov",
    ".coverage",
}

# Files to always skip
DEFAULT_IGNORE_FILES: set[str] = {
    ".DS_Store",
    "Thumbs.db",
    "*.pyc",
    "*.pyo",
    "*.so",
    "*.dylib",
    "*.dll",
    "*.exe",
}

# Maximum file size to process (skip huge generated files)
MAX_FILE_SIZE_BYTES: int = 1_000_000  # 1 MB


def load_gitignore_patterns(repo_root: Path) -> list[str]:
    """Load patterns from .gitignore file.

    Why respect .gitignore?
    - If it's not version-controlled, it shouldn't be indexed
    - Avoids indexing generated files, secrets, build artifacts
    - Aligns with developer expectations
    """
    gitignore_path = repo_root / ".gitignore"
    patterns: list[str] = []

    if gitignore_path.exists():
        with open(gitignore_path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if line and not line.startswith("#"):
                    patterns.append(line)

    return patterns


def should_ignore_path(
    path: Path,
    repo_root: Path,
    gitignore_patterns: list[str],
    ignore_dirs: set[str] = DEFAULT_IGNORE_DIRS,
    ignore_files: set[str] = DEFAULT_IGNORE_FILES,
) -> bool:
    """Check if a path should be ignored.

    Order of checks (short-circuit for performance):
    1. Built-in ignore patterns (fastest)
    2. Gitignore patterns (requires path manipulation)
    """
    name = path.name

    # Check built-in patterns
    if path.is_dir():
        for pattern in ignore_dirs:
            if fnmatch.fnmatch(name, pattern):
                return True
    else:
        for pattern in ignore_files:
            if fnmatch.fnmatch(name, pattern):
                return True

    # Check gitignore patterns
    try:
        relative_path = path.relative_to(repo_root)
        rel_str = str(relative_path)

        for pattern in gitignore_patterns:
            # Handle directory patterns (ending with /)
            if pattern.endswith("/"):
                if path.is_dir() and fnmatch.fnmatch(name, pattern[:-1]):
                    return True
            # Handle full path patterns
            elif "/" in pattern:
                if fnmatch.fnmatch(rel_str, pattern):
                    return True
            # Handle simple patterns
            else:
                if fnmatch.fnmatch(name, pattern):
                    return True
    except ValueError:
        # path is not relative to repo_root
        pass

    return False


def detect_language(file_path: Path) -> Language | None:
    """Detect programming language from file extension.

    Returns None for unsupported languages.

    Why extension-based detection?
    - Fast and reliable for most cases
    - No need to read file content
    - Shebang detection can be added later for edge cases
    """
    suffix = file_path.suffix.lower()
    return EXTENSION_TO_LANGUAGE.get(suffix)


def discover_files(
    repo_path: str | Path,
    languages: list[Language] | None = None,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    max_file_size: int = MAX_FILE_SIZE_BYTES,
) -> list[FileInfo]:
    """Discover all code files in a repository.

    Args:
        repo_path: Path to the repository root
        languages: Filter to specific languages (None = all supported)
        include_patterns: Glob patterns for files to include
        exclude_patterns: Additional glob patterns to exclude
        max_file_size: Skip files larger than this (bytes)

    Returns:
        List of FileInfo objects for discovered files

    Example:
        >>> files = discover_files("./my-repo", languages=[Language.PYTHON])
        >>> for f in files:
        ...     print(f.path, f.language)
    """
    repo_root = Path(repo_path).resolve()

    if not repo_root.exists():
        raise FileNotFoundError(f"Repository path not found: {repo_root}")

    if not repo_root.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {repo_root}")

    # Load gitignore patterns
    gitignore_patterns = load_gitignore_patterns(repo_root)

    # Add user-specified exclude patterns
    if exclude_patterns:
        gitignore_patterns.extend(exclude_patterns)

    # Allowed languages (None means all)
    allowed_languages = set(languages) if languages else None

    discovered: list[FileInfo] = []

    # Walk the directory tree
    for item in repo_root.rglob("*"):
        # Skip directories (we only want files)
        if item.is_dir():
            continue

        # Check if any parent directory should be ignored
        should_skip = False
        for parent in item.relative_to(repo_root).parents:
            parent_path = repo_root / parent
            if should_ignore_path(parent_path, repo_root, gitignore_patterns):
                should_skip = True
                break

        if should_skip:
            continue

        # Check if file itself should be ignored
        if should_ignore_path(item, repo_root, gitignore_patterns):
            continue

        # Detect language
        language = detect_language(item)
        if language is None:
            continue  # Unsupported file type

        # Filter by requested languages
        if allowed_languages and language not in allowed_languages:
            continue

        # Check file size
        try:
            size = item.stat().st_size
            if size > max_file_size:
                continue  # Skip large files
        except OSError:
            continue  # Skip files we can't stat

        # Apply include patterns if specified
        if include_patterns:
            matched = any(
                fnmatch.fnmatch(item.name, pattern) for pattern in include_patterns
            )
            if not matched:
                continue

        discovered.append(
            FileInfo(
                path=item,
                language=language,
                size_bytes=size,
            )
        )

    # Sort by path for consistent ordering
    discovered.sort(key=lambda f: str(f.path))

    return discovered
