"""Markdown chunker with section-aware splitting."""

from pathlib import Path


class MarkdownChunker:
    """Extract and chunk Markdown files by sections."""

    def chunk_file(self, path: str, max_chunk_size: int) -> list[tuple[str, dict]]:
        """Extract and chunk Markdown by sections.

        Returns list of (content, metadata) tuples.
        Metadata includes: source, chunk_index, heading_path, heading_level.
        """
        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        chunks = []
        chunk_index = 0
        current_content = []
        current_heading_path = []
        current_level = 0

        for line in lines:
            heading_match = self._parse_heading(line)

            if heading_match:
                # Save previous section
                if current_content:
                    content = "\n".join(current_content).strip()
                    if content:
                        metadata = {
                            "source": str(file_path),
                            "chunk_index": chunk_index,
                            "heading_path": list(current_heading_path),
                            "heading_level": current_level,
                        }
                        chunks.append((content, metadata))
                        chunk_index += 1

                    current_content = []

                # Update heading path
                level, title = heading_match
                current_level = level
                # Pop headings at same or deeper level
                # Keep headings at indices < level - 1 (0-indexed)
                current_heading_path = [
                    h for i, h in enumerate(current_heading_path) if i < level - 1
                ]
                current_heading_path.append(title)

            else:
                current_content.append(line.rstrip())

        # Save last section
        if current_content:
            content = "\n".join(current_content).strip()
            if content:
                metadata = {
                    "source": str(file_path),
                    "chunk_index": chunk_index,
                    "heading_path": list(current_heading_path),
                    "heading_level": current_level,
                }
                chunks.append((content, metadata))

        return chunks

    def _parse_heading(self, line: str) -> tuple[int, str] | None:
        """Parse markdown heading line.

        Returns (level, title) or None if not a heading.
        Only parses H1-H3.
        """
        line = line.strip()
        if line.startswith("# "):
            return (1, line[2:].strip())
        elif line.startswith("## "):
            return (2, line[3:].strip())
        elif line.startswith("### "):
            return (3, line[4:].strip())
        return None