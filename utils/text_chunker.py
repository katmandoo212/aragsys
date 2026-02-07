"""Unified text chunker with format detection."""

from pathlib import Path
from typing import Callable

from utils.pdf_chunker import PDFChunker
from utils.markdown_chunker import MarkdownChunker


class TextChunker:
    """Format-aware text chunker."""

    def __init__(self):
        self._pdf_chunker = PDFChunker()
        self._md_chunker = MarkdownChunker()
        self._handlers: dict[str, Callable[[Path, int], list[tuple[str, dict]]]] = {
            '.txt': self._chunk_txt,
            '.md': self._chunk_md,
            '.markdown': self._chunk_md,
            '.pdf': self._chunk_pdf,
        }

    def chunk_file(self, path: str, max_chunk_size: int) -> list[tuple[str, dict]]:
        """Dispatch to appropriate format handler."""
        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        ext = file_path.suffix.lower()
        handler = self._handlers.get(ext)

        if not handler:
            # Fallback: attempt plain text
            return self._chunk_as_text(file_path, max_chunk_size)

        return handler(file_path, max_chunk_size)

    def _chunk_txt(self, path: Path, max_chunk_size: int) -> list[tuple[str, dict]]:
        """Chunk TXT file by paragraphs."""
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()

        if not content.strip():
            return []

        paragraphs = content.split('\n\n')
        chunks = []

        for idx, para in enumerate(paragraphs):
            if not para.strip():
                continue

            chunks.append((
                para.strip(),
                {"source": str(path), "chunk_index": idx}
            ))

        return chunks

    def _chunk_pdf(self, path: Path, max_chunk_size: int) -> list[tuple[str, dict]]:
        """Chunk PDF file."""
        return self._pdf_chunker.chunk_file(str(path), max_chunk_size)

    def _chunk_md(self, path: Path, max_chunk_size: int) -> list[tuple[str, dict]]:
        """Chunk Markdown file."""
        return self._md_chunker.chunk_file(str(path), max_chunk_size)

    def _chunk_as_text(self, path: Path, max_chunk_size: int) -> list[tuple[str, dict]]:
        """Fallback: read as plain text."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()

            if not content.strip():
                return []

            # Simple paragraph splitting
            paragraphs = content.split('\n\n')
            chunks = []

            for idx, para in enumerate(paragraphs):
                if not para.strip():
                    continue

                chunks.append((
                    para.strip(),
                    {"source": str(path), "chunk_index": idx}
                ))

            return chunks
        except Exception as e:
            # Log and return empty on fallback failure
            print(f"Warning: Could not read {path} as text: {e}")
            return []