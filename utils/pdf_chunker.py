"""PDF chunker with structure preservation for scientific literature."""

from pathlib import Path

import pdfplumber


class PDFChunker:
    """Extract and chunk PDF files preserving structure."""

    def chunk_file(self, path: str, max_chunk_size: int) -> list[tuple[str, dict]]:
        """Extract and chunk PDF with structure preservation.

        Returns list of (content, metadata) tuples.
        Metadata includes: source, chunk_index, page_number, section_title,
        figure_captions, table_data.
        """
        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        chunks = []
        chunk_index = 0

        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                tables = page.extract_tables() or []
                figures = self._extract_figure_captions(text)

                # Simple chunking by page for now
                # Section detection and table preservation in full implementation
                metadata = {
                    "source": path,  # Use original path string to preserve format
                    "chunk_index": chunk_index,
                    "page_number": page_num,
                    "section_title": f"Page {page_num}",
                    "figure_captions": figures,
                    "table_data": tables,
                }

                if text.strip():
                    chunks.append((text.strip(), metadata))
                    chunk_index += 1

        return chunks

    def _extract_figure_captions(self, text: str) -> list[str]:
        """Extract figure captions from text.

        Looks for patterns like "Figure 1:", "Fig. 2", etc.
        """
        import re

        patterns = [
            r"Figure\s+\d+[:.]?\s+[^\n]+",
            r"Fig\.?\s+\d+[:.]?\s+[^\n]+",
        ]

        captions = []
        for pattern in patterns:
            captions.extend(re.findall(pattern, text, re.IGNORECASE))

        return captions