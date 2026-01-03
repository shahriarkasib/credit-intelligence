"""
Tavily Search Data Source - Task 12 Implementation

Tavily is an AI-optimized search API designed for LLM applications.
Provides high-quality, relevant search results with AI-powered summarization.

Features:
- Web search with AI-powered result ranking
- Content extraction and summarization
- News search
- Context-aware searching for credit intelligence
"""

import os
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from .base import BaseDataSource, DataSourceResult

logger = logging.getLogger(__name__)

# Import Tavily client
try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False
    TavilyClient = None
    logger.warning("tavily-python not installed. Tavily search will be unavailable.")


class TavilySearchDataSource(BaseDataSource):
    """
    Tavily Search connector for AI-optimized web search.

    Features:
    - AI-powered search results
    - Content extraction and summarization
    - News and general web search
    - Optimized for LLM consumption

    Authentication: Requires TAVILY_API_KEY environment variable
    Rate Limit: Based on plan (default: reasonable use)
    """

    # Search depth options
    SEARCH_DEPTH_BASIC = "basic"
    SEARCH_DEPTH_ADVANCED = "advanced"

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_results: int = 10,
        search_depth: str = "basic",
    ):
        """
        Initialize Tavily search client.

        Args:
            api_key: Tavily API key (defaults to TAVILY_API_KEY env var)
            max_results: Default max results per search
            search_depth: "basic" or "advanced" (advanced = more comprehensive)
        """
        super().__init__(
            name="TavilySearch",
            base_url="https://api.tavily.com",
            api_key=api_key or os.getenv("TAVILY_API_KEY"),
            rate_limit=2.0,  # 2 requests per second
        )
        self.max_results = max_results
        self.search_depth = search_depth
        self._client = None

    def _get_client(self) -> Optional["TavilyClient"]:
        """Get or create Tavily client."""
        if not TAVILY_AVAILABLE:
            return None

        if not self.api_key:
            logger.warning("TAVILY_API_KEY not set")
            return None

        if self._client is None:
            try:
                self._client = TavilyClient(api_key=self.api_key)
            except Exception as e:
                logger.error(f"Failed to initialize Tavily client: {e}")
                return None

        return self._client

    def is_available(self) -> bool:
        """Check if Tavily is available and configured."""
        return TAVILY_AVAILABLE and bool(self.api_key)

    def search(self, query: str, **kwargs) -> DataSourceResult:
        """
        Search the web using Tavily.

        Args:
            query: Search query
            max_results: Maximum number of results
            search_depth: "basic" or "advanced"
            include_answer: Include AI-generated answer (default: True)
            include_raw_content: Include raw page content (default: False)
            include_images: Include images (default: False)

        Returns:
            DataSourceResult with search results
        """
        client = self._get_client()
        if not client:
            return DataSourceResult(
                source=self.name,
                query=query,
                data={},
                success=False,
                error="Tavily client not available. Check TAVILY_API_KEY.",
            )

        max_results = kwargs.get("max_results", self.max_results)
        search_depth = kwargs.get("search_depth", self.search_depth)
        include_answer = kwargs.get("include_answer", True)
        include_raw_content = kwargs.get("include_raw_content", False)
        include_images = kwargs.get("include_images", False)

        try:
            self.rate_limiter.wait()

            response = client.search(
                query=query,
                search_depth=search_depth,
                max_results=max_results,
                include_answer=include_answer,
                include_raw_content=include_raw_content,
                include_images=include_images,
            )

            # Process results
            results = response.get("results", [])
            processed = [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", ""),
                    "score": r.get("score", 0),
                    "raw_content": r.get("raw_content") if include_raw_content else None,
                }
                for r in results
            ]

            return DataSourceResult(
                source=self.name,
                query=query,
                data={
                    "count": len(processed),
                    "results": processed,
                    "answer": response.get("answer"),  # AI-generated answer
                    "query": response.get("query"),
                    "response_time": response.get("response_time"),
                    "images": response.get("images", []) if include_images else [],
                },
                raw_response=response,
            )

        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            return DataSourceResult(
                source=self.name,
                query=query,
                data={},
                success=False,
                error=str(e),
            )

    def search_context(self, query: str, **kwargs) -> DataSourceResult:
        """
        Get search context optimized for LLM consumption.

        This is specifically designed to provide context for AI models,
        returning a concatenated string of relevant content.

        Args:
            query: Search query
            max_results: Maximum results to include
            max_tokens: Approximate max tokens in response (default: 4000)

        Returns:
            DataSourceResult with concatenated context
        """
        client = self._get_client()
        if not client:
            return DataSourceResult(
                source=self.name,
                query=query,
                data={},
                success=False,
                error="Tavily client not available",
            )

        max_results = kwargs.get("max_results", 5)
        max_tokens = kwargs.get("max_tokens", 4000)

        try:
            self.rate_limiter.wait()

            # Use get_search_context for LLM-optimized results
            context = client.get_search_context(
                query=query,
                max_results=max_results,
                max_tokens=max_tokens,
            )

            return DataSourceResult(
                source=self.name,
                query=query,
                data={
                    "context": context,
                    "max_tokens": max_tokens,
                    "query": query,
                },
            )

        except Exception as e:
            logger.error(f"Tavily context search failed: {e}")
            return DataSourceResult(
                source=self.name,
                query=query,
                data={},
                success=False,
                error=str(e),
            )

    def search_qna(self, query: str, **kwargs) -> DataSourceResult:
        """
        Get a direct answer to a question.

        Uses Tavily's Q&A endpoint for direct answers.

        Args:
            query: Question to answer
            search_depth: "basic" or "advanced"

        Returns:
            DataSourceResult with answer
        """
        client = self._get_client()
        if not client:
            return DataSourceResult(
                source=self.name,
                query=query,
                data={},
                success=False,
                error="Tavily client not available",
            )

        search_depth = kwargs.get("search_depth", self.search_depth)

        try:
            self.rate_limiter.wait()

            answer = client.qna_search(
                query=query,
                search_depth=search_depth,
            )

            return DataSourceResult(
                source=self.name,
                query=query,
                data={
                    "answer": answer,
                    "query": query,
                },
            )

        except Exception as e:
            logger.error(f"Tavily Q&A search failed: {e}")
            return DataSourceResult(
                source=self.name,
                query=query,
                data={},
                success=False,
                error=str(e),
            )

    def get_company_data(self, identifier: str, **kwargs) -> DataSourceResult:
        """
        Get comprehensive company information using Tavily.

        Args:
            identifier: Company name to search for

        Returns:
            DataSourceResult with company information
        """
        company_name = identifier

        # Search queries optimized for credit intelligence
        queries = [
            f"{company_name} company overview business",
            f"{company_name} financial performance revenue",
            f"{company_name} news recent developments",
            f"{company_name} risk factors challenges",
        ]

        all_results = []
        answers = []
        errors = []

        for query in queries:
            result = self.search(
                query,
                max_results=3,
                include_answer=True,
                search_depth="basic",
            )

            if result.success:
                all_results.extend(result.data.get("results", []))
                if result.data.get("answer"):
                    answers.append({
                        "query": query,
                        "answer": result.data["answer"],
                    })
            else:
                errors.append(f"{query}: {result.error}")

        # Deduplicate by URL
        seen_urls = set()
        unique_results = []
        for r in all_results:
            url = r.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(r)

        # Sort by score
        unique_results.sort(key=lambda x: x.get("score", 0), reverse=True)

        # Extract key information
        company_info = self._extract_company_info(unique_results, company_name)

        return DataSourceResult(
            source=self.name,
            query=identifier,
            data={
                "company": company_name,
                "results": unique_results[:15],
                "ai_answers": answers,
                "extracted_info": company_info,
                "search_timestamp": datetime.now(timezone.utc).isoformat(),
                "errors": errors if errors else None,
            },
            success=len(unique_results) > 0,
        )

    def _extract_company_info(
        self,
        results: List[Dict],
        company_name: str,
    ) -> Dict[str, Any]:
        """Extract structured information from search results."""
        info = {
            "mentions_found": len(results),
            "sources": [],
            "key_content": [],
            "top_score": 0,
        }

        for r in results[:10]:
            # Track sources
            url = r.get("url", "")
            if url:
                try:
                    from urllib.parse import urlparse
                    domain = urlparse(url).netloc
                    if domain and domain not in info["sources"]:
                        info["sources"].append(domain)
                except Exception:
                    pass

            # Track content
            content = r.get("content", "")
            if content and len(content) > 50:
                info["key_content"].append({
                    "snippet": content[:300],
                    "score": r.get("score", 0),
                    "url": url,
                })

            # Track top score
            score = r.get("score", 0)
            if score > info["top_score"]:
                info["top_score"] = score

        return info

    def get_supported_fields(self) -> List[str]:
        return [
            "web_results",
            "ai_answers",
            "search_context",
            "company_info",
            "news",
        ]

    def health_check(self) -> bool:
        """Check if Tavily is working."""
        if not self.is_available():
            return False
        try:
            result = self.search("test", max_results=1)
            return result.success
        except Exception:
            return False


# Singleton instance
_tavily_search: Optional[TavilySearchDataSource] = None


def get_tavily_search() -> TavilySearchDataSource:
    """Get the global TavilySearchDataSource instance."""
    global _tavily_search
    if _tavily_search is None:
        _tavily_search = TavilySearchDataSource()
    return _tavily_search


def tavily_search(query: str, **kwargs) -> DataSourceResult:
    """Convenience function to search with Tavily."""
    searcher = get_tavily_search()
    return searcher.search(query, **kwargs)


def tavily_company_search(company_name: str) -> DataSourceResult:
    """Convenience function to get company data with Tavily."""
    searcher = get_tavily_search()
    return searcher.get_company_data(company_name)
