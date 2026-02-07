"""Tests for PDFChunker."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from utils.pdf_chunker import PDFChunker


class TestPDFChunker:
    """Tests for PDFChunker class."""

    def test_chunk_file_not_found(self):
        """Test that FileNotFoundError is raised for non-existent file."""
        chunker = PDFChunker()

        with pytest.raises(FileNotFoundError, match="File not found"):
            chunker.chunk_file("nonexistent.pdf", max_chunk_size=1000)

    @patch("utils.pdf_chunker.pdfplumber.open")
    @patch("utils.pdf_chunker.Path.exists", return_value=True)
    def test_chunk_file_basic(self, mock_exists, mock_pdf_open):
        """Test basic PDF chunking returns chunks with metadata."""
        # Mock a PDF page
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Test content on page 1"
        mock_page.extract_tables.return_value = []

        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]

        # Set up the pdfplumber.open context manager mock
        mock_pdf_open.return_value.__enter__.return_value = mock_pdf
        mock_pdf_open.return_value.__exit__.return_value = None

        chunker = PDFChunker()

        chunks = chunker.chunk_file("test.pdf", max_chunk_size=1000)

        assert len(chunks) == 1
        content, metadata = chunks[0]

        assert content == "Test content on page 1"
        assert metadata["source"] == "test.pdf"
        assert metadata["chunk_index"] == 0
        assert metadata["page_number"] == 1
        assert metadata["section_title"] == "Page 1"
        assert metadata["figure_captions"] == []
        assert metadata["table_data"] == []

    @patch("utils.pdf_chunker.pdfplumber.open")
    @patch("utils.pdf_chunker.Path.exists", return_value=True)
    def test_chunk_file_multiple_pages(self, mock_exists, mock_pdf_open):
        """Test chunking multiple pages."""
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "Content on page 1"
        mock_page1.extract_tables.return_value = []

        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = "Content on page 2"
        mock_page2.extract_tables.return_value = []

        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page1, mock_page2]

        mock_pdf_open.return_value.__enter__.return_value = mock_pdf
        mock_pdf_open.return_value.__exit__.return_value = None

        chunker = PDFChunker()

        chunks = chunker.chunk_file("test.pdf", max_chunk_size=1000)

        assert len(chunks) == 2

        # Check first chunk
        assert chunks[0][0] == "Content on page 1"
        assert chunks[0][1]["page_number"] == 1
        assert chunks[0][1]["chunk_index"] == 0

        # Check second chunk
        assert chunks[1][0] == "Content on page 2"
        assert chunks[1][1]["page_number"] == 2
        assert chunks[1][1]["chunk_index"] == 1

    @patch("utils.pdf_chunker.pdfplumber.open")
    @patch("utils.pdf_chunker.Path.exists", return_value=True)
    def test_chunk_file_empty_text(self, mock_exists, mock_pdf_open):
        """Test that empty text pages are skipped."""
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = ""
        mock_page1.extract_tables.return_value = []

        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = "Some content"
        mock_page2.extract_tables.return_value = []

        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page1, mock_page2]

        mock_pdf_open.return_value.__enter__.return_value = mock_pdf
        mock_pdf_open.return_value.__exit__.return_value = None

        chunker = PDFChunker()

        chunks = chunker.chunk_file("test.pdf", max_chunk_size=1000)

        assert len(chunks) == 1
        assert chunks[0][0] == "Some content"
        assert chunks[0][1]["chunk_index"] == 0  # Skipped empty page

    @patch("utils.pdf_chunker.pdfplumber.open")
    @patch("utils.pdf_chunker.Path.exists", return_value=True)
    def test_chunk_file_with_tables(self, mock_exists, mock_pdf_open):
        """Test that table data is captured in metadata."""
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Text with table"
        mock_page.extract_tables.return_value = [["Header1", "Header2"], ["Row1", "Row2"]]

        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]

        mock_pdf_open.return_value.__enter__.return_value = mock_pdf
        mock_pdf_open.return_value.__exit__.return_value = None

        chunker = PDFChunker()

        chunks = chunker.chunk_file("test.pdf", max_chunk_size=1000)

        assert len(chunks) == 1
        _, metadata = chunks[0]

        assert metadata["table_data"] == [["Header1", "Header2"], ["Row1", "Row2"]]

    @patch("utils.pdf_chunker.pdfplumber.open")
    @patch("utils.pdf_chunker.Path.exists", return_value=True)
    def test_chunk_file_absolute_path(self, mock_exists, mock_pdf_open):
        """Test that absolute paths are preserved in metadata."""
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Test content"
        mock_page.extract_tables.return_value = []

        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]

        mock_pdf_open.return_value.__enter__.return_value = mock_pdf
        mock_pdf_open.return_value.__exit__.return_value = None

        chunker = PDFChunker()

        chunks = chunker.chunk_file("/path/to/document.pdf", max_chunk_size=1000)

        assert len(chunks) == 1
        _, metadata = chunks[0]

        assert metadata["source"] == "/path/to/document.pdf"

    def test_extract_figure_captions_figure_pattern(self):
        """Test figure caption extraction with 'Figure X:' pattern."""
        chunker = PDFChunker()

        text = "This is a paragraph.\nFigure 1: A nice diagram\nAnother paragraph."

        captions = chunker._extract_figure_captions(text)

        assert captions == ["Figure 1: A nice diagram"]

    def test_extract_figure_captions_fig_pattern(self):
        """Test figure caption extraction with 'Fig. X' pattern."""
        chunker = PDFChunker()

        text = "See Figure 2 for details. Fig. 3 shows the results.\nFig. 4. Something else"

        captions = chunker._extract_figure_captions(text)

        assert len(captions) == 3
        # Note: The regex captures everything after the figure reference
        assert any("Fig. 3 shows the results" in c for c in captions)
        assert "Fig. 4. Something else" in captions

    def test_extract_figure_captions_case_insensitive(self):
        """Test that figure caption extraction is case-insensitive."""
        chunker = PDFChunker()

        text = "figure 5: Lowercase case.\nFIGURE 6: Uppercase case."

        captions = chunker._extract_figure_captions(text)

        assert len(captions) == 2
        # Note: The regex includes trailing punctuation in some cases
        assert any("figure 5: Lowercase case" in c for c in captions)
        assert any("FIGURE 6: Uppercase case" in c for c in captions)

    def test_extract_figure_captions_none_found(self):
        """Test figure caption extraction when none exist."""
        chunker = PDFChunker()

        text = "This text has no figure captions in it."

        captions = chunker._extract_figure_captions(text)

        assert captions == []

    @patch("utils.pdf_chunker.pdfplumber.open")
    @patch("utils.pdf_chunker.Path.exists", return_value=True)
    def test_chunk_file_with_figure_captions(self, mock_exists, mock_pdf_open):
        """Test that figure captions are extracted and stored in metadata."""
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Some text.\nFigure 7: An illustration\nMore text."
        mock_page.extract_tables.return_value = []

        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]

        mock_pdf_open.return_value.__enter__.return_value = mock_pdf
        mock_pdf_open.return_value.__exit__.return_value = None

        chunker = PDFChunker()

        chunks = chunker.chunk_file("test.pdf", max_chunk_size=1000)

        assert len(chunks) == 1
        _, metadata = chunks[0]

        assert "Figure 7: An illustration" in metadata["figure_captions"]