"""Web Search Data Source - DuckDuckGo free web search."""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from .base import BaseDataSource, DataSourceResult

logger = logging.getLogger(__name__)

try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False
    logger.warning("duckduckgo-search not installed. Web search will be unavailable.")


class WebSearchDataSource(BaseDataSource):
    """
    Web Search connector using DuckDuckGo.

    Provides access to:
    - Web search results
    - News search
    - Company information from the web

    Rate Limit: Reasonable use (no strict limit)
    Authentication: None required
    """

    def __init__(self, max_results: int = 10):
        super().__init__(
            name="WebSearch",
            base_url="",  # Not used for DuckDuckGo
            rate_limit=0.5,  # Be conservative
        )
        self.max_results = max_results
        self._ddgs = None

    def _get_ddgs(self):
        """Get or create DDGS instance."""
        if not DDGS_AVAILABLE:
            raise RuntimeError("duckduckgo-search package not installed")
        if self._ddgs is None:
            self._ddgs = DDGS()
        return self._ddgs

    def search(self, query: str, **kwargs) -> DataSourceResult:
        """
        Search the web for information.

        Args:
            query: Search query
            max_results: Maximum number of results (default: 10)
            region: Region for search (default: 'wt-wt' for worldwide)
        """
        if not DDGS_AVAILABLE:
            return DataSourceResult(
                source=self.name,
                query=query,
                data={},
                success=False,
                error="duckduckgo-search package not installed",
            )

        max_results = kwargs.get("max_results", self.max_results)
        region = kwargs.get("region", "wt-wt")

        try:
            self.rate_limiter.wait()
            ddgs = self._get_ddgs()
            results = list(ddgs.text(
                query,
                region=region,
                max_results=max_results,
            ))

            processed = [
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                }
                for r in results
            ]

            return DataSourceResult(
                source=self.name,
                query=query,
                data={
                    "count": len(processed),
                    "results": processed,
                },
                raw_response=results,
            )

        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return DataSourceResult(
                source=self.name,
                query=query,
                data={},
                success=False,
                error=str(e),
            )

    def search_news(self, query: str, **kwargs) -> DataSourceResult:
        """
        Search for news articles.

        Args:
            query: Search query
            max_results: Maximum number of results
            timelimit: Time limit - 'd' (day), 'w' (week), 'm' (month)
        """
        if not DDGS_AVAILABLE:
            return DataSourceResult(
                source=self.name,
                query=query,
                data={},
                success=False,
                error="duckduckgo-search package not installed",
            )

        max_results = kwargs.get("max_results", self.max_results)
        timelimit = kwargs.get("timelimit", "m")  # Default to last month

        try:
            self.rate_limiter.wait()
            ddgs = self._get_ddgs()
            results = list(ddgs.news(
                query,
                timelimit=timelimit,
                max_results=max_results,
            ))

            processed = [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("body", ""),
                    "source": r.get("source", ""),
                    "date": r.get("date", ""),
                }
                for r in results
            ]

            return DataSourceResult(
                source=self.name,
                query=query,
                data={
                    "count": len(processed),
                    "news": processed,
                },
                raw_response=results,
            )

        except Exception as e:
            logger.error(f"News search failed: {e}")
            return DataSourceResult(
                source=self.name,
                query=query,
                data={},
                success=False,
                error=str(e),
            )

    def get_company_data(self, identifier: str, **kwargs) -> DataSourceResult:
        """
        Get web information about a company.

        Args:
            identifier: Company name to search for
        """
        company_name = identifier

        # Perform multiple searches to gather comprehensive data
        general_search = self.search(f"{company_name} company", max_results=5)
        news_search = self.search_news(f"{company_name}", max_results=5)
        financial_search = self.search(f"{company_name} financial news revenue", max_results=5)

        # Combine results
        all_results = []
        if general_search.success:
            all_results.extend(general_search.data.get("results", []))
        if financial_search.success:
            all_results.extend(financial_search.data.get("results", []))

        # Deduplicate by URL
        seen_urls = set()
        unique_results = []
        for r in all_results:
            url = r.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(r)

        # Extract key information
        company_info = self._extract_company_info(unique_results, company_name)

        return DataSourceResult(
            source=self.name,
            query=identifier,
            data={
                "company": company_name,
                "web_results": unique_results[:10],
                "news": news_search.data.get("news", []) if news_search.success else [],
                "extracted_info": company_info,
                "search_timestamp": datetime.utcnow().isoformat(),
            },
        )

    def _extract_company_info(self, results: List[Dict], company_name: str) -> Dict[str, Any]:
        """Extract structured information from search results."""
        info = {
            "mentions_found": len(results),
            "sources": [],
            "key_snippets": [],
        }

        for r in results[:5]:
            source = r.get("url", "")
            if source:
                # Extract domain
                try:
                    from urllib.parse import urlparse
                    domain = urlparse(source).netloc
                    if domain and domain not in info["sources"]:
                        info["sources"].append(domain)
                except Exception:
                    pass

            snippet = r.get("snippet", "")
            if snippet and len(snippet) > 50:
                info["key_snippets"].append(snippet[:200])

        return info

    def analyze_sentiment(self, company_name: str, news_items: List[Dict]) -> Dict[str, Any]:
        """
        Basic sentiment analysis based on keywords.

        Note: For production, use a proper NLP model or LLM for sentiment.
        This is a simple keyword-based approach for demo purposes.
        """
        positive_keywords = [
            "growth", "profit", "success", "increase", "positive",
            "strong", "beat", "exceed", "record", "innovation",
            "expand", "partnership", "award", "leading", "breakthrough"
        ]
        negative_keywords = [
            "loss", "decline", "lawsuit", "investigation", "fraud",
            "bankruptcy", "layoff", "scandal", "fine", "penalty",
            "debt", "default", "miss", "weak", "concern", "fail"
        ]

        positive_count = 0
        negative_count = 0
        neutral_count = 0

        for item in news_items:
            text = (item.get("title", "") + " " + item.get("snippet", "")).lower()

            pos = sum(1 for kw in positive_keywords if kw in text)
            neg = sum(1 for kw in negative_keywords if kw in text)

            if pos > neg:
                positive_count += 1
            elif neg > pos:
                negative_count += 1
            else:
                neutral_count += 1

        total = len(news_items) or 1
        sentiment_score = (positive_count - negative_count) / total

        if sentiment_score > 0.2:
            sentiment = "positive"
        elif sentiment_score < -0.2:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        return {
            "sentiment": sentiment,
            "score": round(sentiment_score, 2),
            "positive_articles": positive_count,
            "negative_articles": negative_count,
            "neutral_articles": neutral_count,
            "total_analyzed": len(news_items),
        }

    def get_supported_fields(self) -> List[str]:
        return [
            "web_results",
            "news_articles",
            "sentiment",
            "company_mentions",
            "key_snippets",
        ]

    def health_check(self) -> bool:
        """Check if web search is working."""
        if not DDGS_AVAILABLE:
            return False
        try:
            result = self.search("test", max_results=1)
            return result.success
        except Exception:
            return False
