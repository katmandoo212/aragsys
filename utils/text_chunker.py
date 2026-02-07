from pathlib import Path


class TextChunker:
    def chunk_file(self, path: str, max_chunk_size: int) -> list[tuple[str, dict]]:
        """Chunk TXT file by paragraphs.

        Returns list of (content, metadata) tuples.
        Metadata includes: source, chunk_index.
        """
        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        if not content.strip():
            return []

        # Split by double newline (paragraphs)
        paragraphs = content.split('\n\n')
        chunks = []

        for idx, para in enumerate(paragraphs):
            if not para.strip():
                continue

            chunks.append((
                para.strip(),
                {"source": str(file_path), "chunk_index": idx}
            ))

        return chunks