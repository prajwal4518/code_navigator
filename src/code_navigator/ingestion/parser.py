"""
Tree-sitter based code parser.

Why tree-sitter?
1. Multi-language support with consistent API
2. Incremental parsing (fast for large files)
3. Error-tolerant (parses broken code gracefully)
4. Produces concrete syntax trees (CST) not just AST

This module wraps tree-sitter to extract semantic code chunks
(functions, classes, methods) with their metadata.
"""

from pathlib import Path

import tree_sitter_python as ts_python
from tree_sitter import Language, Node, Parser

from .models import ChunkType, CodeChunk
from .models import Language as CodeLanguage

# Initialize tree-sitter language
PY_LANGUAGE = Language(ts_python.language())


class PythonParser:
    """Parser for Python source code using tree-sitter.

    Why a class instead of functions?
    - Parser can be reused (avoids re-initialization overhead)
    - Easy to extend with caching, statistics, etc.
    - Clear interface for future multi-language support
    """

    def __init__(self):
        self.parser = Parser(PY_LANGUAGE)

    def parse_file(self, file_path: str | Path) -> list[CodeChunk]:
        """Parse a Python file and extract code chunks.

        Args:
            file_path: Path to the Python source file

        Returns:
            List of CodeChunk objects (functions, classes, methods)
        """
        file_path = Path(file_path).resolve()

        with open(file_path, encoding="utf-8", errors="replace") as f:
            source_code = f.read()

        return self.parse_source(source_code, str(file_path))

    def parse_source(self, source_code: str, file_path: str) -> list[CodeChunk]:
        """Parse Python source code string and extract code chunks.

        Args:
            source_code: The Python source code
            file_path: Path for metadata (doesn't need to exist)

        Returns:
            List of CodeChunk objects
        """
        # Parse to syntax tree
        tree = self.parser.parse(bytes(source_code, "utf-8"))

        # Split into lines for line-based extraction
        lines = source_code.split("\n")

        chunks: list[CodeChunk] = []

        # Walk the tree and extract chunks
        self._extract_chunks(
            node=tree.root_node,
            source_lines=lines,
            file_path=file_path,
            chunks=chunks,
            parent_class=None,
        )

        return chunks

    def _extract_chunks(
        self,
        node: Node,
        source_lines: list[str],
        file_path: str,
        chunks: list[CodeChunk],
        parent_class: str | None,
    ) -> None:
        """Recursively extract code chunks from AST nodes.

        Design decision: We extract at multiple granularities:
        - Classes as chunks (for "what does this class do?")
        - Methods as chunks (for "how does this method work?")
        - Functions as chunks (for top-level functions)

        This means methods are indexed twice (as part of class AND individually).
        This is intentional: different queries need different granularity.
        """
        if node.type == "function_definition":
            chunk = self._extract_function(node, source_lines, file_path, parent_class)
            if chunk:
                chunks.append(chunk)

        elif node.type == "class_definition":
            # Extract the class itself
            chunk = self._extract_class(node, source_lines, file_path)
            if chunk:
                chunks.append(chunk)

                # Extract methods within the class
                class_name = self._get_node_name(node)
                for child in node.children:
                    if child.type == "block":
                        for block_child in child.children:
                            self._extract_chunks(
                                block_child,
                                source_lines,
                                file_path,
                                chunks,
                                parent_class=class_name,
                            )
                return  # Don't recurse further for classes

        # Recurse into children
        for child in node.children:
            self._extract_chunks(child, source_lines, file_path, chunks, parent_class)

    def _extract_function(
        self,
        node: Node,
        source_lines: list[str],
        file_path: str,
        parent_class: str | None,
    ) -> CodeChunk | None:
        """Extract a function/method from an AST node."""
        name = self._get_node_name(node)
        if not name:
            return None

        # Get line range (tree-sitter uses 0-indexed lines)
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        # Extract content
        content = "\n".join(source_lines[start_line - 1 : end_line])

        # Extract docstring
        docstring = self._extract_docstring(node)

        # Extract decorators
        decorators = self._extract_decorators(node, source_lines)

        # Determine chunk type
        chunk_type = ChunkType.METHOD if parent_class else ChunkType.FUNCTION

        return CodeChunk(
            content=content,
            chunk_type=chunk_type,
            name=name,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            language=CodeLanguage.PYTHON,
            docstring=docstring,
            parent_name=parent_class,
            decorators=decorators,
        )

    def _extract_class(
        self,
        node: Node,
        source_lines: list[str],
        file_path: str,
    ) -> CodeChunk | None:
        """Extract a class from an AST node."""
        name = self._get_node_name(node)
        if not name:
            return None

        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        content = "\n".join(source_lines[start_line - 1 : end_line])
        docstring = self._extract_docstring(node)
        decorators = self._extract_decorators(node, source_lines)

        return CodeChunk(
            content=content,
            chunk_type=ChunkType.CLASS,
            name=name,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            language=CodeLanguage.PYTHON,
            docstring=docstring,
            parent_name=None,
            decorators=decorators,
        )

    def _get_node_name(self, node: Node) -> str | None:
        """Get the name of a function or class node."""
        for child in node.children:
            if child.type == "identifier":
                return child.text.decode("utf-8") if child.text else None
        return None

    def _extract_docstring(self, node: Node) -> str | None:
        """Extract docstring from a function or class node.

        Python docstrings are expression statements containing
        a string literal as the first statement in the body.
        """
        for child in node.children:
            if child.type == "block":
                for block_child in child.children:
                    if block_child.type == "expression_statement":
                        for expr_child in block_child.children:
                            if expr_child.type == "string":
                                text = expr_child.text
                                if text:
                                    # Remove quotes and clean up
                                    docstring = text.decode("utf-8")
                                    # Strip triple quotes
                                    if docstring.startswith('"""'):
                                        docstring = docstring[3:-3]
                                    elif docstring.startswith("'''"):
                                        docstring = docstring[3:-3]
                                    elif docstring.startswith('"'):
                                        docstring = docstring[1:-1]
                                    elif docstring.startswith("'"):
                                        docstring = docstring[1:-1]
                                    return docstring.strip()
                    # Only check the first statement
                    break
        return None

    def _extract_decorators(self, node: Node, source_lines: list[str]) -> list[str]:
        """Extract decorator names from a function or class."""
        decorators: list[str] = []

        # Look for decorator nodes before the function/class
        parent = node.parent
        if parent:
            for i, child in enumerate(parent.children):
                if child == node:
                    # Check previous siblings for decorators
                    for j in range(i - 1, -1, -1):
                        sibling = parent.children[j]
                        if sibling.type == "decorator":
                            # Extract decorator name
                            for dec_child in sibling.children:
                                if dec_child.type in ("identifier", "attribute"):
                                    if dec_child.text:
                                        decorators.append(
                                            dec_child.text.decode("utf-8")
                                        )
                                    break
                                elif dec_child.type == "call":
                                    # Decorator with arguments like @lru_cache(maxsize=100)
                                    for call_child in dec_child.children:
                                        if call_child.type in (
                                            "identifier",
                                            "attribute",
                                        ):
                                            if call_child.text:
                                                decorators.append(
                                                    call_child.text.decode("utf-8")
                                                )
                                            break
                                    break
                        else:
                            break  # No more decorators
                    break

        return list(reversed(decorators))  # Preserve original order
