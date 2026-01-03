"""
Web Scraper Tool - Task 11 Implementation

A web scraping tool for extracting company information from web pages:
- Scrapes and parses HTML content
- Extracts text, metadata, and structured data
- Handles common patterns (about pages, press releases, etc.)
- Cleans and normalizes extracted content
"""

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, NavigableString

from .base import BaseDataSource, DataSourceResult, RateLimiter

logger = logging.getLogger(__name__)


@dataclass
class ScrapedContent:
    """Result from a web scraping operation."""

    url: str
    title: str = ""
    description: str = ""
    text_content: str = ""
    headings: List[str] = field(default_factory=list)
    links: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)
    structured_data: Dict[str, Any] = field(default_factory=dict)
    scraped_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "description": self.description,
            "text_content": self.text_content[:5000] if self.text_content else "",
            "headings": self.headings[:20],
            "links": self.links[:50],
            "metadata": self.metadata,
            "structured_data": self.structured_data,
            "scraped_at": self.scraped_at,
            "success": self.success,
            "error": self.error,
        }


class WebScraper(BaseDataSource):
    """
    Web scraper for extracting company information from web pages.

    Features:
    - HTML parsing with BeautifulSoup
    - Text extraction and cleaning
    - Metadata extraction (title, description, keywords)
    - Heading extraction (h1-h6)
    - Link extraction
    - JSON-LD structured data extraction
    - Configurable content limits
    """

    # Common patterns for company pages
    ABOUT_PATTERNS = [
        r"/about",
        r"/company",
        r"/who-we-are",
        r"/our-story",
        r"/corporate",
    ]

    INVESTOR_PATTERNS = [
        r"/investor",
        r"/ir",
        r"/investors",
        r"/financial",
        r"/annual-report",
    ]

    NEWS_PATTERNS = [
        r"/news",
        r"/press",
        r"/media",
        r"/blog",
        r"/announcements",
    ]

    # Elements to exclude from text extraction
    EXCLUDE_TAGS = [
        "script", "style", "nav", "footer", "header",
        "aside", "noscript", "iframe", "svg", "form",
    ]

    def __init__(
        self,
        rate_limit: float = 2.0,  # 2 requests per second
        timeout: int = 30,
        max_content_length: int = 10000,
        user_agent: str = None,
    ):
        """
        Initialize the web scraper.

        Args:
            rate_limit: Maximum requests per second
            timeout: Request timeout in seconds
            max_content_length: Maximum text content to extract
            user_agent: Custom user agent string
        """
        super().__init__(
            name="WebScraper",
            base_url="",  # No base URL for general scraper
            rate_limit=rate_limit,
            timeout=timeout,
        )
        self.max_content_length = max_content_length

        if user_agent:
            self.session.headers["User-Agent"] = user_agent
        else:
            self.session.headers["User-Agent"] = (
                "Mozilla/5.0 (compatible; CreditIntelligence/1.0; +https://example.com)"
            )

    def _setup_session(self):
        """Configure session with headers for scraping."""
        self.session.headers.update({
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        })

    def scrape_url(self, url: str) -> ScrapedContent:
        """
        Scrape a single URL and extract content.

        Args:
            url: The URL to scrape

        Returns:
            ScrapedContent with extracted data
        """
        self.rate_limiter.wait()

        result = ScrapedContent(url=url)

        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()

            # Check content type
            content_type = response.headers.get("Content-Type", "")
            if "text/html" not in content_type and "application/xhtml" not in content_type:
                result.success = False
                result.error = f"Non-HTML content type: {content_type}"
                return result

            # Parse HTML
            soup = BeautifulSoup(response.content, "lxml")

            # Extract title
            result.title = self._extract_title(soup)

            # Extract description
            result.description = self._extract_description(soup)

            # Extract metadata
            result.metadata = self._extract_metadata(soup)

            # Extract headings
            result.headings = self._extract_headings(soup)

            # Extract main text content
            result.text_content = self._extract_text(soup)

            # Extract links
            result.links = self._extract_links(soup, url)

            # Extract JSON-LD structured data
            result.structured_data = self._extract_structured_data(soup)

            logger.info(f"Successfully scraped: {url}")

        except requests.exceptions.Timeout:
            result.success = False
            result.error = "Request timed out"
            logger.warning(f"Timeout scraping {url}")

        except requests.exceptions.RequestException as e:
            result.success = False
            result.error = str(e)
            logger.warning(f"Request error scraping {url}: {e}")

        except Exception as e:
            result.success = False
            result.error = str(e)
            logger.error(f"Error scraping {url}: {e}")

        return result

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title."""
        # Try og:title first
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            return og_title["content"].strip()

        # Try regular title tag
        title_tag = soup.find("title")
        if title_tag and title_tag.string:
            return title_tag.string.strip()

        # Try h1
        h1 = soup.find("h1")
        if h1:
            return h1.get_text(strip=True)

        return ""

    def _extract_description(self, soup: BeautifulSoup) -> str:
        """Extract page description."""
        # Try og:description
        og_desc = soup.find("meta", property="og:description")
        if og_desc and og_desc.get("content"):
            return og_desc["content"].strip()

        # Try meta description
        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc and meta_desc.get("content"):
            return meta_desc["content"].strip()

        return ""

    def _extract_metadata(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract page metadata."""
        metadata = {}

        # Common meta tags
        meta_names = [
            "keywords", "author", "robots", "viewport",
            "generator", "application-name",
        ]

        for name in meta_names:
            tag = soup.find("meta", attrs={"name": name})
            if tag and tag.get("content"):
                metadata[name] = tag["content"]

        # Open Graph tags
        og_properties = [
            "og:type", "og:site_name", "og:locale",
            "og:url", "og:image",
        ]

        for prop in og_properties:
            tag = soup.find("meta", property=prop)
            if tag and tag.get("content"):
                metadata[prop] = tag["content"]

        # Twitter cards
        twitter_names = ["twitter:card", "twitter:site", "twitter:creator"]
        for name in twitter_names:
            tag = soup.find("meta", attrs={"name": name})
            if tag and tag.get("content"):
                metadata[name] = tag["content"]

        # Canonical URL
        canonical = soup.find("link", rel="canonical")
        if canonical and canonical.get("href"):
            metadata["canonical"] = canonical["href"]

        return metadata

    def _extract_headings(self, soup: BeautifulSoup) -> List[str]:
        """Extract headings (h1-h6)."""
        headings = []

        for level in range(1, 7):
            for heading in soup.find_all(f"h{level}"):
                text = heading.get_text(strip=True)
                if text and len(text) < 500:  # Reasonable heading length
                    headings.append(text)

        return headings[:50]  # Limit to 50 headings

    def _extract_text(self, soup: BeautifulSoup) -> str:
        """Extract main text content."""
        # Remove excluded tags
        for tag in soup.find_all(self.EXCLUDE_TAGS):
            tag.decompose()

        # Try to find main content area
        main_content = (
            soup.find("main") or
            soup.find("article") or
            soup.find(id="content") or
            soup.find(class_="content") or
            soup.find("body")
        )

        if not main_content:
            return ""

        # Get text with some formatting
        text_parts = []
        for element in main_content.descendants:
            if isinstance(element, NavigableString):
                text = str(element).strip()
                if text:
                    text_parts.append(text)

        # Join and clean
        full_text = " ".join(text_parts)

        # Clean up whitespace
        full_text = re.sub(r"\s+", " ", full_text)
        full_text = full_text.strip()

        # Truncate if needed
        if len(full_text) > self.max_content_length:
            full_text = full_text[:self.max_content_length] + "..."

        return full_text

    def _extract_links(
        self,
        soup: BeautifulSoup,
        base_url: str,
    ) -> List[Dict[str, str]]:
        """Extract links from the page."""
        links = []
        seen_urls = set()

        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"].strip()

            # Skip empty, javascript, and anchor-only links
            if not href or href.startswith("javascript:") or href == "#":
                continue

            # Make absolute URL
            absolute_url = urljoin(base_url, href)

            # Skip duplicates
            if absolute_url in seen_urls:
                continue
            seen_urls.add(absolute_url)

            # Get link text
            text = a_tag.get_text(strip=True)

            # Classify link type
            link_type = self._classify_link(absolute_url)

            links.append({
                "url": absolute_url,
                "text": text[:200] if text else "",
                "type": link_type,
            })

        return links[:100]  # Limit to 100 links

    def _classify_link(self, url: str) -> str:
        """Classify a link based on URL patterns."""
        url_lower = url.lower()

        for pattern in self.ABOUT_PATTERNS:
            if re.search(pattern, url_lower):
                return "about"

        for pattern in self.INVESTOR_PATTERNS:
            if re.search(pattern, url_lower):
                return "investor"

        for pattern in self.NEWS_PATTERNS:
            if re.search(pattern, url_lower):
                return "news"

        if url_lower.endswith(".pdf"):
            return "pdf"

        return "general"

    def _extract_structured_data(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract JSON-LD structured data."""
        structured = {}

        # Find all JSON-LD scripts
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                import json
                data = json.loads(script.string)

                # Handle single object or array
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "@type" in item:
                            type_name = item["@type"]
                            structured[type_name] = item
                elif isinstance(data, dict):
                    if "@type" in data:
                        type_name = data["@type"]
                        structured[type_name] = data
                    elif "@graph" in data:
                        for item in data["@graph"]:
                            if isinstance(item, dict) and "@type" in item:
                                type_name = item["@type"]
                                structured[type_name] = item

            except (json.JSONDecodeError, TypeError):
                continue

        return structured

    def scrape_company_pages(
        self,
        company_url: str,
        max_pages: int = 5,
    ) -> List[ScrapedContent]:
        """
        Scrape multiple pages from a company website.

        Args:
            company_url: Main company website URL
            max_pages: Maximum pages to scrape

        Returns:
            List of ScrapedContent results
        """
        results = []
        scraped_urls = set()

        # Scrape main page
        main_result = self.scrape_url(company_url)
        results.append(main_result)
        scraped_urls.add(company_url)

        if not main_result.success:
            return results

        # Find relevant pages to scrape
        target_links = []
        for link in main_result.links:
            if link["type"] in ["about", "investor", "news"]:
                if link["url"] not in scraped_urls:
                    target_links.append(link)

        # Scrape additional pages
        for link in target_links[:max_pages - 1]:
            url = link["url"]
            if url in scraped_urls:
                continue

            scraped_urls.add(url)
            page_result = self.scrape_url(url)
            results.append(page_result)

            if len(results) >= max_pages:
                break

        return results

    def search(self, query: str, **kwargs) -> DataSourceResult:
        """
        Search implementation (not applicable for scraper).

        Use scrape_url or scrape_company_pages instead.
        """
        return DataSourceResult(
            source=self.name,
            query=query,
            data={"error": "Use scrape_url() for web scraping"},
            success=False,
            error="Search not implemented for WebScraper",
        )

    def get_company_data(self, identifier: str, **kwargs) -> DataSourceResult:
        """
        Get company data by scraping their website.

        Args:
            identifier: Company website URL

        Returns:
            DataSourceResult with scraped content
        """
        if not identifier.startswith(("http://", "https://")):
            identifier = f"https://{identifier}"

        max_pages = kwargs.get("max_pages", 3)

        try:
            results = self.scrape_company_pages(identifier, max_pages=max_pages)

            # Combine results
            combined_data = {
                "url": identifier,
                "pages_scraped": len(results),
                "pages": [r.to_dict() for r in results],
                "all_headings": [],
                "all_text": "",
            }

            # Aggregate content
            for result in results:
                combined_data["all_headings"].extend(result.headings)
                combined_data["all_text"] += result.text_content + "\n\n"

            success = any(r.success for r in results)

            return DataSourceResult(
                source=self.name,
                query=identifier,
                data=combined_data,
                success=success,
                error=None if success else "All pages failed to scrape",
            )

        except Exception as e:
            logger.error(f"Company data scraping failed: {e}")
            return DataSourceResult(
                source=self.name,
                query=identifier,
                data={},
                success=False,
                error=str(e),
            )

    def health_check(self) -> bool:
        """Check if the scraper is working."""
        try:
            result = self.scrape_url("https://example.com")
            return result.success
        except Exception:
            return False


# Singleton instance
_scraper: Optional[WebScraper] = None


def get_web_scraper() -> WebScraper:
    """Get the global WebScraper instance."""
    global _scraper
    if _scraper is None:
        _scraper = WebScraper()
    return _scraper


def scrape_url(url: str) -> ScrapedContent:
    """Convenience function to scrape a single URL."""
    scraper = get_web_scraper()
    return scraper.scrape_url(url)


def scrape_company_website(url: str, max_pages: int = 3) -> DataSourceResult:
    """Convenience function to scrape a company website."""
    scraper = get_web_scraper()
    return scraper.get_company_data(url, max_pages=max_pages)
