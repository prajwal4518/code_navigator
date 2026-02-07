"""
Repository Manager for cloning and caching Git repositories.

Why cache repos?
- Avoid re-cloning on every request
- Support incremental updates (git pull)
- Enable offline access after first clone

Cache structure:
    ~/.cache/code_navigator/repos/
    └── {url_hash}/
        └── {repo_name}/
"""

import hashlib
import subprocess
from pathlib import Path
from urllib.parse import urlparse

from rich.console import Console

console = Console()

# Default cache directory
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "code_navigator" / "repos"


def _hash_url(url: str) -> str:
    """Create a short hash of the URL for cache key."""
    return hashlib.sha256(url.encode()).hexdigest()[:12]


def _extract_repo_name(url: str) -> str:
    """Extract repository name from URL.

    Examples:
        https://github.com/user/repo.git -> repo
        https://github.com/user/repo -> repo
        git@github.com:user/repo.git -> repo
    """
    # Handle SSH URLs
    if url.startswith("git@"):
        path = url.split(":")[-1]
    else:
        parsed = urlparse(url)
        path = parsed.path

    # Remove .git suffix and get last part
    name = path.rstrip("/").split("/")[-1]
    if name.endswith(".git"):
        name = name[:-4]

    return name or "repo"


def _run_git(args: list[str], cwd: Path | None = None) -> tuple[bool, str]:
    """Run a git command and return success status and output."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout for large repos
        )
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, "Git operation timed out"
    except FileNotFoundError:
        return False, "Git is not installed"


class RepoManager:
    """Manages cloning and caching of Git repositories.

    Usage:
        >>> manager = RepoManager()
        >>> repo_path = manager.clone_or_update("https://github.com/user/repo")
        >>> # Now use repo_path with the indexer
    """

    def __init__(self, cache_dir: Path | None = None):
        """Initialize the repo manager.

        Args:
            cache_dir: Directory for caching repos.
                      Defaults to ~/.cache/code_navigator/repos/
        """
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_path(self, url: str) -> Path:
        """Get the cache path for a repository URL.

        Args:
            url: Git repository URL

        Returns:
            Path where the repo is/will be cached
        """
        url_hash = _hash_url(url)
        repo_name = _extract_repo_name(url)
        return self.cache_dir / url_hash / repo_name

    def is_cached(self, url: str) -> bool:
        """Check if a repository is already cached.

        Args:
            url: Git repository URL

        Returns:
            True if the repo exists in cache
        """
        cache_path = self.get_cache_path(url)
        return (cache_path / ".git").exists()

    def clone_or_update(
        self,
        url: str,
        branch: str | None = None,
        depth: int | None = 1,
        verbose: bool = True,
    ) -> Path:
        """Clone a repository or update if already cached.

        Args:
            url: Git repository URL
            branch: Specific branch to clone (default: default branch)
            depth: Clone depth (1 for shallow, None for full history)
            verbose: Show progress output

        Returns:
            Path to the cloned repository

        Raises:
            RuntimeError: If cloning or updating fails
        """
        cache_path = self.get_cache_path(url)

        if self.is_cached(url):
            # Update existing repo
            if verbose:
                console.print(f"[dim]Updating cached repo:[/] {cache_path}")

            success, output = _run_git(["pull", "--ff-only"], cwd=cache_path)
            if not success:
                # If pull fails, try reset
                console.print("[yellow]Pull failed, resetting to origin...[/]")
                _run_git(["fetch", "origin"], cwd=cache_path)
                branch_name = branch or "main"
                _run_git(["reset", "--hard", f"origin/{branch_name}"], cwd=cache_path)

            if verbose:
                console.print("[green]✓ Repository updated[/]")

        else:
            # Clone new repo
            if verbose:
                console.print(f"[blue]Cloning:[/] {url}")

            # Create parent directory
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            # Build clone command
            clone_args = ["clone"]
            if depth:
                clone_args.extend(["--depth", str(depth)])
            if branch:
                clone_args.extend(["--branch", branch])
            clone_args.extend([url, str(cache_path)])

            success, output = _run_git(clone_args)

            if not success:
                raise RuntimeError(f"Failed to clone repository: {output}")

            if verbose:
                console.print(f"[green]✓ Cloned to:[/] {cache_path}")

        return cache_path

    def cleanup(self, url: str) -> bool:
        """Remove a cached repository.

        Args:
            url: Git repository URL

        Returns:
            True if successfully removed
        """
        import shutil

        cache_path = self.get_cache_path(url)
        if cache_path.exists():
            shutil.rmtree(cache_path)
            # Also remove parent hash directory if empty
            if cache_path.parent.exists() and not any(cache_path.parent.iterdir()):
                cache_path.parent.rmdir()
            return True
        return False

    def list_cached(self) -> list[Path]:
        """List all cached repositories.

        Returns:
            List of paths to cached repos
        """
        repos = []
        if self.cache_dir.exists():
            for hash_dir in self.cache_dir.iterdir():
                if hash_dir.is_dir():
                    for repo_dir in hash_dir.iterdir():
                        if (repo_dir / ".git").exists():
                            repos.append(repo_dir)
        return repos


def get_repo_manager() -> RepoManager:
    """Get the default RepoManager instance."""
    return RepoManager()
