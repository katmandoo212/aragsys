"""Tests for MarkdownChunker."""

import pytest
from utils.markdown_chunker import MarkdownChunker


def test_chunk_file_not_found():
    """Test that FileNotFoundError is raised for non-existent file."""
    chunker = MarkdownChunker()

    with pytest.raises(FileNotFoundError, match="File not found"):
        chunker.chunk_file("nonexistent.md", max_chunk_size=1000)


def test_chunk_file_with_h1_sections(tmp_path):
    """Test chunking by H1 headings."""
    test_file = tmp_path / "test.md"
    test_file.write_text("# Introduction\nSome content.\n\n# Methods\nMore content.")

    chunker = MarkdownChunker()
    chunks = chunker.chunk_file(str(test_file), max_chunk_size=1000)

    assert len(chunks) == 2

    # First chunk
    content1, metadata1 = chunks[0]
    assert content1 == "Some content."
    assert metadata1["source"] == str(test_file)
    assert metadata1["chunk_index"] == 0
    assert metadata1["heading_path"] == ["Introduction"]
    assert metadata1["heading_level"] == 1

    # Second chunk
    content2, metadata2 = chunks[1]
    assert content2 == "More content."
    assert metadata2["chunk_index"] == 1
    assert metadata2["heading_path"] == ["Methods"]
    assert metadata2["heading_level"] == 1


def test_chunk_file_with_nested_headings(tmp_path):
    """Test chunking with nested heading structure (H1 -> H2 -> H3)."""
    test_file = tmp_path / "nested.md"
    test_file.write_text("""# Main Section
Content under main.

## Subsection 1
Content under subsection 1.

### Detail 1
Content under detail 1.

## Subsection 2
Content under subsection 2.""")

    chunker = MarkdownChunker()
    chunks = chunker.chunk_file(str(test_file), max_chunk_size=1000)

    assert len(chunks) == 4

    # Check heading paths
    assert chunks[0][1]["heading_path"] == ["Main Section"]
    assert chunks[0][1]["heading_level"] == 1

    assert chunks[1][1]["heading_path"] == ["Main Section", "Subsection 1"]
    assert chunks[1][1]["heading_level"] == 2

    assert chunks[2][1]["heading_path"] == ["Main Section", "Subsection 1", "Detail 1"]
    assert chunks[2][1]["heading_level"] == 3

    assert chunks[3][1]["heading_path"] == ["Main Section", "Subsection 2"]
    assert chunks[3][1]["heading_level"] == 2


def test_chunk_file_with_h3_only(tmp_path):
    """Test that only H1-H3 are parsed, H4+ are treated as content."""
    test_file = tmp_path / "h4.md"
    test_file.write_text("""# Top
Content under top.

## Mid
Content under mid.

### Lower

#### Too deep - treated as content
This should be part of the previous section.""")

    chunker = MarkdownChunker()
    chunks = chunker.chunk_file(str(test_file), max_chunk_size=1000)

    assert len(chunks) == 3

    # First chunk under H1
    assert chunks[0][0] == "Content under top."
    assert chunks[0][1]["heading_path"] == ["Top"]
    assert chunks[0][1]["heading_level"] == 1

    # Second chunk under H2
    assert chunks[1][0] == "Content under mid."
    assert chunks[1][1]["heading_path"] == ["Top", "Mid"]
    assert chunks[1][1]["heading_level"] == 2

    # Last chunk should include the "#### Too deep" line as content under H3
    assert "Too deep" in chunks[2][0]
    assert "treated as content" in chunks[2][0]
    assert chunks[2][1]["heading_path"] == ["Top", "Mid", "Lower"]
    assert chunks[2][1]["heading_level"] == 3
    assert chunks[2][1]["heading_level"] == 3


def test_chunk_file_with_multiline_content(tmp_path):
    """Test that multi-line content within a section is preserved."""
    test_file = tmp_path / "multiline.md"
    test_file.write_text("""# Section 1
First paragraph.

Second paragraph.
With multiple lines.

# Section 2
More content.""")

    chunker = MarkdownChunker()
    chunks = chunker.chunk_file(str(test_file), max_chunk_size=1000)

    assert len(chunks) == 2

    # First chunk should have all content before H2
    assert "First paragraph." in chunks[0][0]
    assert "Second paragraph." in chunks[0][0]
    assert "With multiple lines" in chunks[0][0]


def test_chunk_empty_content_between_headings(tmp_path):
    """Test that empty sections are skipped."""
    test_file = tmp_path / "empty.md"
    test_file.write_text("""# Section 1
Content here.

# Section 2

# Section 3
Content here.""")

    chunker = MarkdownChunker()
    chunks = chunker.chunk_file(str(test_file), max_chunk_size=1000)

    assert len(chunks) == 2
    assert chunks[0][0] == "Content here."
    assert chunks[1][0] == "Content here."


def test_chunk_file_content_before_first_heading(tmp_path):
    """Test that content before the first heading is captured."""
    test_file = tmp_path / "preamble.md"
    test_file.write_text("""Preamble content
before first heading.

# First Section
Section content.""")

    chunker = MarkdownChunker()
    chunks = chunker.chunk_file(str(test_file), max_chunk_size=1000)

    assert len(chunks) == 2

    # First chunk has preamble
    assert "Preamble content" in chunks[0][0]
    assert chunks[0][1]["heading_path"] == []
    assert chunks[0][1]["heading_level"] == 0

    # Second chunk is the section
    assert chunks[1][0] == "Section content."
    assert chunks[1][1]["heading_path"] == ["First Section"]
    assert chunks[1][1]["heading_level"] == 1


def test_chunk_file_trailing_whitespace(tmp_path):
    """Test that trailing whitespace is stripped from chunks."""
    test_file = tmp_path / "whitespace.md"
    test_file.write_text("""# Section
Content with trailing spaces.

Next line.""")

    chunker = MarkdownChunker()
    chunks = chunker.chunk_file(str(test_file), max_chunk_size=1000)

    assert len(chunks) == 1
    content = chunks[0][0]

    # No trailing whitespace on lines
    assert content.endswith("Next line.")
    assert not content.endswith("   \n")


def test_parse_heading_h1():
    """Test parsing H1 heading."""
    chunker = MarkdownChunker()

    result = chunker._parse_heading("# Title")
    assert result == (1, "Title")

    result = chunker._parse_heading("# Title with spaces  ")
    assert result == (1, "Title with spaces")


def test_parse_heading_h2():
    """Test parsing H2 heading."""
    chunker = MarkdownChunker()

    result = chunker._parse_heading("## Subtitle")
    assert result == (2, "Subtitle")


def test_parse_heading_h3():
    """Test parsing H3 heading."""
    chunker = MarkdownChunker()

    result = chunker._parse_heading("### Sub-subtitle")
    assert result == (3, "Sub-subtitle")


def test_parse_heading_h4_not_parsed():
    """Test that H4+ headings are not parsed."""
    chunker = MarkdownChunker()

    result = chunker._parse_heading("#### Too deep")
    assert result is None


def test_parse_heading_non_heading():
    """Test that non-heading lines return None."""
    chunker = MarkdownChunker()

    assert chunker._parse_heading("Regular text") is None
    assert chunker._parse_heading("#Not a heading") is None  # No space after #
    assert chunker._parse_heading("") is None


def test_chunk_file_absolute_path(tmp_path):
    """Test that absolute paths are preserved in metadata."""
    test_file = tmp_path / "absolute.md"
    test_file.write_text("# Section\nContent.")

    chunker = MarkdownChunker()
    chunks = chunker.chunk_file(str(test_file.absolute()), max_chunk_size=1000)

    assert len(chunks) == 1
    _, metadata = chunks[0]

    assert metadata["source"] == str(test_file.absolute())


def test_heading_path_updates_on_same_level(tmp_path):
    """Test that heading path correctly updates when encountering same-level heading."""
    test_file = tmp_path / "same-level.md"
    test_file.write_text("""# Main 1
Content 1.

## Sub 1
Sub content 1.

## Sub 2
Sub content 2.""")

    chunker = MarkdownChunker()
    chunks = chunker.chunk_file(str(test_file), max_chunk_size=1000)

    assert len(chunks) == 3

    # First sub-section under Main 1, Sub 1
    assert chunks[1][1]["heading_path"] == ["Main 1", "Sub 1"]

    # Second sub-section - Sub 1 should be replaced by Sub 2
    assert chunks[2][1]["heading_path"] == ["Main 1", "Sub 2"]