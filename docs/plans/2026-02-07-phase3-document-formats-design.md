# Phase 3: Document Formats - Design

**Status:** Implemented
**Date:** 2026-02-07

---

## Overview

Phase 3 extends the TextChunker to support PDF and Markdown formats with advanced structure preservation. This enables ingestion of scientific literature and technical documentation.

**Scope:**
- **PDF:** Advanced chunking preserving tables, figures, page numbers, multi-column layouts
- **Markdown:** Section-aware chunking by H1-H3 headings with heading paths in metadata
- **TXT:** Existing paragraph-based chunking (unchanged)
- **Fallback:** Unsupported formats attempt plain text extraction

**Supported Formats:**
- `.txt` - Plain text (existing)
- `.pdf` - PDF documents with structure
- `.md` - Markdown with heading hierarchy
- `.markdown` - Alternative Markdown extension

**Architecture:**
TextChunker becomes a format-aware chunker that:
1. Detects file type from extension
2. Dispatches to format-specific handler
3. Applies unified chunk size constraints
4. Returns consistent (content, metadata) tuples

---

## PDF Chunking

**PDF Chunker** uses a sophisticated PDF parsing library to extract structured content. The approach preserves document hierarchy for scientific literature.

**Dependencies:**
- `pdfplumber` - Table extraction and layout analysis (recommended)
- Or `pypdf` - Alternative with good section detection

**Chunking Logic:**
1. Parse PDF page by page
2. Detect section headers (larger font, bold)
3. Extract tables into structured metadata
4. Find figure captions (e.g., "Figure 1:", "Fig. 2")
5. Group content by section boundaries
6. Split oversized sections by paragraph boundaries

**Metadata per chunk:**
```python
{
    "source": str,
    "chunk_index": int,
    "page_number": int,
    "section_title": str,
    "figure_captions": list[str],
    "table_data": list[dict]
}
```

**Implementation Approach:**
```python
class PDFChunker:
    def chunk_file(self, path: str, max_chunk_size: int) -> list[tuple[str, dict]]:
        """Extract and chunk PDF with structure preservation."""
        # Open PDF, extract pages with layout analysis
        # Identify sections by headings (font size/weight detection)
        # Extract tables as structured data
        # Extract figure captions from text patterns
        # Group content by sections, split oversized sections by paragraphs
```

---

## Markdown Chunking

**Markdown Chunker** parses Markdown files and chunks by section boundaries (H1-H3). Heading hierarchy is preserved in metadata for context.

**Dependencies:**
- Built-in string parsing (no external lib needed for basic MD)
- Or `markdown-it-py` for robust parsing

**Chunking Logic:**
1. Parse file line by line
2. Detect headings (`# `, `## `, `### `)
3. Maintain heading stack for current path
4. Accumulate content until next heading
5. Split oversized content by paragraphs

**Heading Path Example:**
```markdown
# Introduction
Content here...

## Background
More content...

### Methods
Details...
```

Chunk metadata for "Methods" section:
```python
{
    "source": str,
    "chunk_index": int,
    "heading_path": ["Introduction", "Background", "Methods"],
    "heading_level": 3
}
```

**Implementation Approach:**
```python
class MarkdownChunker:
    def chunk_file(self, path: str, max_chunk_size: int) -> list[tuple[str, dict]]:
        """Extract and chunk Markdown by sections."""
        # Parse Markdown, identify headings H1-H3
        # Track current heading path
        # Extract content between headings
        # Split oversized sections by paragraphs
```

---

## Unified TextChunker API

**TextChunker** refactors to a format-agnostic dispatcher with format-specific handlers.

```python
class TextChunker:
    def __init__(self):
        self._handlers = {
            '.txt': self._chunk_txt,
            '.md': self._chunk_md,
            '.markdown': self._chunk_md,
            '.pdf': self._chunk_pdf,
        }

    def chunk_file(self, path: str, max_chunk_size: int) -> list[tuple[str, dict]]:
        """Dispatch to appropriate format handler."""
        file_path = Path(path)
        ext = file_path.suffix.lower()

        handler = self._handlers.get(ext)
        if not handler:
            # Fallback: attempt plain text
            return self._chunk_as_text(file_path, max_chunk_size)

        return handler(file_path, max_chunk_size)

    def _chunk_txt(self, path: Path, max_size: int) -> list[tuple[str, dict]]:
        # Existing paragraph-based chunking

    def _chunk_md(self, path: Path, max_size: int) -> list[tuple[str, dict]]:
        # Section-aware chunking

    def _chunk_pdf(self, path: Path, max_size: int) -> list[tuple[str, dict]]:
        # Advanced structure-preserving chunking

    def _chunk_as_text(self, path: Path, max_size: int) -> list[tuple[str, dict]]:
        # Fallback: read as plain text
```

---

## Configuration

Extend existing `config/techniques.yaml`:

```yaml
techniques:
  naive_rag:
    enabled: true
    config:
      chunk_size: 500
      top_k: 5
      embedding_model: "bge-m3:latest"
      collection_name: "documents"
      supported_formats: [txt, pdf, md, markdown]
      pdf:
        extract_tables: true
        extract_figures: true
        heading_min_size: 16  # points
```

---

## Error Handling

| Scenario | Handling |
|----------|----------|
| Corrupted PDF | Propagate PDF parsing error |
| Invalid PDF | ValueError with clear message |
| Malformed Markdown | Log warning, continue parsing |
| File not found | FileNotFoundError (existing) |
| Fallback failure | Log warning, return empty |

---

## Testing

**Total:** ~8 new tests (Phase 3 only)

| File | Tests |
|------|-------|
| `tests/test_pdf_chunker.py` | 3 tests |
| `tests/test_markdown_chunker.py` | 3 tests |
| `tests/test_text_chunker.py` (extend) | 2 tests |

**Test Coverage:**
- PDF: basic extraction, section detection, table extraction
- Markdown: heading path extraction, section splitting, nested headings
- Unified: format detection, fallback to plain text

**Sample Tests:**
```python
def test_pdf_chunker_extracts_sections():
    """PDF chunker extracts sections with metadata."""

def test_pdf_chunker_preserves_tables():
    """PDF chunker preserves table structure."""

def test_markdown_chunker_heading_paths():
    """Markdown chunker captures heading hierarchy."""

def test_fallback_to_plain_text():
    """Unsupported format falls back to plain text."""
```

---

## Files to Create

```
utils/
  ├── text_chunker.py (refactor)
  ├── pdf_chunker.py (new)
  └── markdown_chunker.py (new)

tests/
  ├── test_text_chunker.py (extend)
  ├── test_pdf_chunker.py (new)
  └── test_markdown_chunker.py (new)
```

---

## Files to Modify

- `pyproject.toml` - Add `pdfplumber` or `pypdf`
- `config/techniques.yaml` - Add PDF config

---

## Success Criteria

Phase 3 is complete when:
1. All ~8 new tests pass
2. PDF files are chunked with structure preservation
3. Markdown files are chunked by sections with heading paths
4. Unsupported formats fall back to plain text
5. TextChunker dispatches correctly based on file extension

---

## Next Phases

Phase 3 enables:
- Phase 4: Advanced retrieval (HyDE, Multi-Query, Hybrid search)
- Phase 5: Reranking and compression
- Phase 6: Generation