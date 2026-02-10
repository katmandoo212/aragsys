"""Tests for web document fetching."""

from backend.utils.web_fetcher import WebFetcher, FetchResult


def test_web_fetcher_validates_url():
    """WebFetcher validates URL format."""
    fetcher = WebFetcher()
    assert fetcher.validate_url("https://example.com") is True
    assert fetcher.validate_url("http://example.com") is True
    assert fetcher.validate_url("not-a-url") is False
    assert fetcher.validate_url("ftp://example.com") is False  # Only http/https


def test_web_fetcher_extract_title_from_html():
    """WebFetcher extracts title from HTML."""
    html = "<html><head><title>Test Document</title></head><body>Content here</body></html>"
    fetcher = WebFetcher()
    title = fetcher._extract_title_from_html(html)
    assert title == "Test Document"


def test_web_fetcher_extract_body_from_html():
    """WebFetcher extracts body text from HTML."""
    html = "<html><head><title>Test</title></head><body><p>Paragraph 1</p><p>Paragraph 2</p></body></html>"
    fetcher = WebFetcher()
    body = fetcher._extract_body_from_html(html)
    assert "Paragraph 1" in body
    assert "Paragraph 2" in body


def test_fetch_result_creation():
    """FetchResult can be created with required fields."""
    result = FetchResult(
        url="https://example.com",
        title="Test Doc",
        content="Content",
        content_type="text/html",
        chunk_count=1
    )
    assert result.url == "https://example.com"
    assert result.success is True
