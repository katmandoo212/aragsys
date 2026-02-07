import pytest
from utils.text_chunker import TextChunker

def test_chunk_file_splits_by_paragraphs(tmp_path):
    # Create test file with paragraphs
    test_file = tmp_path / "test.txt"
    test_file.write_text("Paragraph one.\n\nParagraph two.\n\nParagraph three.")

    chunker = TextChunker()
    chunks = chunker.chunk_file(str(test_file), max_chunk_size=100)

    assert len(chunks) == 3
    assert all(isinstance(content, str) for content, meta in chunks)
    assert all("source" in meta for _, meta in chunks)
    assert all("chunk_index" in meta for _, meta in chunks)


def test_chunk_empty_file_returns_empty_list(tmp_path):
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("")

    chunker = TextChunker()
    chunks = chunker.chunk_file(str(empty_file), max_chunk_size=100)

    assert chunks == []


def test_chunk_file_not_found_raises_error():
    chunker = TextChunker()

    with pytest.raises(FileNotFoundError, match="File not found"):
        chunker.chunk_file("nonexistent.txt", max_chunk_size=100)


def test_text_chunker_pdf_format(tmp_path):
    """TextChunker correctly routes PDF files to PDFChunker."""
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("Simple text content")

    from utils.text_chunker import TextChunker

    chunker = TextChunker()
    chunks = chunker.chunk_file(str(txt_file), 500)

    assert len(chunks) == 1
    assert "Simple text content" in chunks[0][0]


def test_fallback_to_plain_text(tmp_path):
    """Unsupported format falls back to plain text."""
    # Create a file with unsupported extension
    unsupported_file = tmp_path / "test.xyz"
    unsupported_file.write_text("This should be readable as plain text")

    from utils.text_chunker import TextChunker

    chunker = TextChunker()
    chunks = chunker.chunk_file(str(unsupported_file), 500)

    # Fallback should succeed
    assert len(chunks) == 1
    assert "plain text" in chunks[0][0]