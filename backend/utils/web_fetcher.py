"""Web document fetching service."""

from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Optional
from urllib.parse import urlparse
import httpx
from bs4 import BeautifulSoup


@dataclass
class FetchResult:
    """Result of fetching a web document."""

    url: str
    title: str
    content: str
    content_type: str
    chunk_count: int
    success: bool = True
    error_message: Optional[str] = None
    fetched_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    metadata: dict = field(default_factory=dict)


class WebFetcher:
    """Service for fetching documents from web URLs."""

    ALLOWED_SCHEMES = {"http", "https"}
    MAX_SIZE_BYTES = 5 * 1024 * 1024  # 5MB
    TIMEOUT_SECONDS = 30

    def __init__(self):
        self.client = httpx.AsyncClient(
            timeout=self.TIMEOUT_SECONDS,
            follow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; Aragsys/1.0)"
            }
        )

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    def validate_url(self, url: str) -> bool:
        """Validate URL format and scheme."""
        try:
            parsed = urlparse(url)
            return parsed.scheme in self.ALLOWED_SCHEMES and bool(parsed.netloc)
        except Exception:
            return False

    async def fetch(self, url: str, max_size_mb: int = 5) -> FetchResult:
        """Fetch document from URL."""
        # Validate URL
        if not self.validate_url(url):
            return FetchResult(
                url=url,
                title="",
                content="",
                content_type="",
                chunk_count=0,
                success=False,
                error_message="Invalid URL format"
            )

        try:
            # Fetch content
            response = await self.client.get(url)
            response.raise_for_status()

            # Check content length
            content_length = len(response.content)
            max_size_bytes = max_size_mb * 1024 * 1024
            if content_length > max_size_bytes:
                return FetchResult(
                    url=url,
                    title="",
                    content="",
                    content_type=response.headers.get("content-type", ""),
                    chunk_count=0,
                    success=False,
                    error_message=f"Content exceeds {max_size_mb}MB limit"
                )

            # Process based on content type
            content_type = response.headers.get("content-type", "").lower()

            if "text/html" in content_type:
                return self._process_html(url, response.text)
            elif "application/pdf" in content_type:
                return FetchResult(
                    url=url,
                    title=self._extract_filename(url),
                    content=response.content,
                    content_type=content_type,
                    chunk_count=0,  # Will be chunked separately
                    metadata={"binary": True}
                )
            elif "text/markdown" in content_type or url.endswith(".md"):
                return FetchResult(
                    url=url,
                    title=self._extract_filename(url),
                    content=response.text,
                    content_type=content_type,
                    chunk_count=0,
                    metadata={"filename": self._extract_filename(url)}
                )
            else:
                # Treat as text
                return FetchResult(
                    url=url,
                    title=self._extract_filename(url),
                    content=response.text,
                    content_type=content_type,
                    chunk_count=0
                )

        except httpx.TimeoutException:
            return FetchResult(
                url=url,
                title="",
                content="",
                content_type="",
                chunk_count=0,
                success=False,
                error_message="Request timed out"
            )
        except Exception as e:
            return FetchResult(
                url=url,
                title="",
                content="",
                content_type="",
                chunk_count=0,
                success=False,
                error_message=str(e)
            )

    def _process_html(self, url: str, html: str) -> FetchResult:
        """Process HTML content."""
        soup = BeautifulSoup(html, "html.parser")

        title = self._extract_title_from_html(html)
        body = self._extract_body_from_html(html)

        # Extract metadata
        metadata = {
            "url": url,
            "extracted_at": datetime.now(UTC).isoformat(),
            "content_length": len(body)
        }

        # Extract meta tags
        meta_description = soup.find("meta", attrs={"name": "description"})
        if meta_description:
            metadata["description"] = meta_description.get("content", "")

        return FetchResult(
            url=url,
            title=title,
            content=body,
            content_type="text/html",
            chunk_count=0,  # Will be chunked separately
            metadata=metadata
        )

    def _extract_title_from_html(self, html: str) -> str:
        """Extract title from HTML."""
        try:
            soup = BeautifulSoup(html, "html.parser")
            title_tag = soup.find("title")
            if title_tag:
                return title_tag.get_text().strip()
        except Exception:
            pass
        return "Untitled Document"

    def _extract_body_from_html(self, html: str) -> str:
        """Extract body text from HTML."""
        try:
            soup = BeautifulSoup(html, "html.parser")
            body = soup.find("body")
            if body:
                # Remove script and style elements
                for script in body(["script", "style"]):
                    script.decompose()
                return body.get_text(separator="\n", strip=True)
        except Exception:
            pass
        return html

    def _extract_filename(self, url: str) -> str:
        """Extract filename from URL."""
        path = urlparse(url).path
        filename = path.split("/")[-1]
        if not filename:
            filename = "document"
        return filename