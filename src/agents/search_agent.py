"""Search Agent - Gathers public web information about companies."""

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from data_sources import WebSearchDataSource

logger = logging.getLogger(__name__)


@dataclass
class SearchAgentResult:
    """Result from Search Agent data collection."""
    company: str
    web_results: List[Dict[str, Any]] = field(default_factory=list)
    news_articles: List[Dict[str, Any]] = field(default_factory=list)
    sentiment: Dict[str, Any] = field(default_factory=dict)
    key_findings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "company": self.company,
            "web_results": self.web_results,
            "news_articles": self.news_articles,
            "sentiment": self.sentiment,
            "key_findings": self.key_findings,
            "errors": self.errors,
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of search results."""
        return {
            "company": self.company,
            "web_result_count": len(self.web_results),
            "news_article_count": len(self.news_articles),
            "sentiment": self.sentiment.get("sentiment", "unknown"),
            "finding_count": len(self.key_findings),
        }


class SearchAgent:
    """
    Search Agent for gathering public web information.

    Responsible for:
    - Web search for company information
    - News article collection
    - Basic sentiment analysis
    - Key information extraction
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.max_results = self.config.get("max_results", 10)
        self.web_search = WebSearchDataSource(max_results=self.max_results)

    def search_company(self, company_name: str) -> SearchAgentResult:
        """
        Search for comprehensive information about a company.

        Args:
            company_name: Name of the company to search for

        Returns:
            SearchAgentResult with web results, news, and analysis
        """
        result = SearchAgentResult(company=company_name)

        # General web search
        try:
            web_result = self.web_search.search(
                f"{company_name} company information",
                max_results=self.max_results,
            )
            if web_result.success:
                result.web_results = web_result.data.get("results", [])
            else:
                result.errors.append(f"Web search failed: {web_result.error}")
        except Exception as e:
            logger.error(f"Web search error: {e}")
            result.errors.append(f"Web search error: {str(e)}")

        # News search
        try:
            news_result = self.web_search.search_news(
                company_name,
                max_results=self.max_results,
                timelimit="m",  # Last month
            )
            if news_result.success:
                result.news_articles = news_result.data.get("news", [])
            else:
                result.errors.append(f"News search failed: {news_result.error}")
        except Exception as e:
            logger.error(f"News search error: {e}")
            result.errors.append(f"News search error: {str(e)}")

        # Sentiment analysis
        if result.news_articles:
            try:
                result.sentiment = self.web_search.analyze_sentiment(
                    company_name,
                    result.news_articles,
                )
            except Exception as e:
                logger.error(f"Sentiment analysis error: {e}")
                result.sentiment = {"sentiment": "unknown", "error": str(e)}

        # Extract key findings
        result.key_findings = self._extract_key_findings(
            company_name,
            result.web_results,
            result.news_articles,
        )

        return result

    def search_financial_news(self, company_name: str) -> SearchAgentResult:
        """
        Search specifically for financial news about a company.

        Args:
            company_name: Name of the company

        Returns:
            SearchAgentResult focused on financial information
        """
        result = SearchAgentResult(company=company_name)

        # Financial-specific searches
        queries = [
            f"{company_name} financial results",
            f"{company_name} earnings revenue",
            f"{company_name} stock performance",
        ]

        all_results = []
        for query in queries:
            try:
                search_result = self.web_search.search(query, max_results=5)
                if search_result.success:
                    all_results.extend(search_result.data.get("results", []))
            except Exception as e:
                logger.error(f"Financial search error for '{query}': {e}")
                result.errors.append(f"Search '{query}' failed: {str(e)}")

        # Deduplicate by URL
        seen_urls = set()
        unique_results = []
        for r in all_results:
            url = r.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(r)

        result.web_results = unique_results[:self.max_results]

        # Financial news
        try:
            news_result = self.web_search.search_news(
                f"{company_name} financial earnings",
                max_results=self.max_results,
                timelimit="m",
            )
            if news_result.success:
                result.news_articles = news_result.data.get("news", [])
        except Exception as e:
            result.errors.append(f"Financial news search failed: {str(e)}")

        # Sentiment
        if result.news_articles:
            result.sentiment = self.web_search.analyze_sentiment(
                company_name,
                result.news_articles,
            )

        result.key_findings = self._extract_financial_findings(
            company_name,
            result.web_results,
            result.news_articles,
        )

        return result

    def search_legal_news(self, company_name: str) -> SearchAgentResult:
        """
        Search for legal/litigation news about a company.

        Args:
            company_name: Name of the company

        Returns:
            SearchAgentResult focused on legal information
        """
        result = SearchAgentResult(company=company_name)

        # Legal-specific searches
        queries = [
            f"{company_name} lawsuit",
            f"{company_name} legal investigation",
            f"{company_name} regulatory fine penalty",
        ]

        all_results = []
        for query in queries:
            try:
                search_result = self.web_search.search(query, max_results=5)
                if search_result.success:
                    all_results.extend(search_result.data.get("results", []))
            except Exception as e:
                result.errors.append(f"Legal search '{query}' failed: {str(e)}")

        # Deduplicate
        seen_urls = set()
        unique_results = []
        for r in all_results:
            url = r.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(r)

        result.web_results = unique_results[:self.max_results]

        # Legal news
        try:
            news_result = self.web_search.search_news(
                f"{company_name} lawsuit legal",
                max_results=self.max_results,
                timelimit="m",
            )
            if news_result.success:
                result.news_articles = news_result.data.get("news", [])
        except Exception as e:
            result.errors.append(f"Legal news search failed: {str(e)}")

        result.key_findings = self._extract_legal_findings(
            company_name,
            result.web_results,
            result.news_articles,
        )

        return result

    def _extract_key_findings(
        self,
        company_name: str,
        web_results: List[Dict],
        news_articles: List[Dict],
    ) -> List[str]:
        """Extract key findings from search results."""
        findings = []

        # Check for recent news volume
        if len(news_articles) > 5:
            findings.append(f"Active news coverage with {len(news_articles)} recent articles")
        elif len(news_articles) == 0:
            findings.append("Limited recent news coverage")

        # Look for significant keywords in results
        all_text = " ".join([
            r.get("snippet", "") + " " + r.get("title", "")
            for r in web_results + news_articles
        ]).lower()

        # Financial indicators
        if "ipo" in all_text or "public offering" in all_text:
            findings.append("IPO or public offering mentioned")
        if "acquisition" in all_text or "merger" in all_text:
            findings.append("M&A activity mentioned")
        if "layoff" in all_text or "restructuring" in all_text:
            findings.append("Restructuring or layoffs mentioned")

        # Legal/Risk indicators
        if "lawsuit" in all_text or "litigation" in all_text:
            findings.append("Legal proceedings mentioned")
        if "investigation" in all_text:
            findings.append("Investigation mentioned")
        if "bankruptcy" in all_text:
            findings.append("Bankruptcy mentioned - HIGH RISK indicator")

        # Positive indicators
        if "growth" in all_text and "revenue" in all_text:
            findings.append("Revenue growth discussed")
        if "expansion" in all_text or "new market" in all_text:
            findings.append("Business expansion mentioned")

        return findings[:10]  # Limit to top 10 findings

    def _extract_financial_findings(
        self,
        company_name: str,
        web_results: List[Dict],
        news_articles: List[Dict],
    ) -> List[str]:
        """Extract financial-specific findings."""
        findings = []

        all_text = " ".join([
            r.get("snippet", "") + " " + r.get("title", "")
            for r in web_results + news_articles
        ]).lower()

        # Earnings indicators
        if "beat" in all_text and "earnings" in all_text:
            findings.append("Company beat earnings expectations")
        elif "miss" in all_text and "earnings" in all_text:
            findings.append("Company missed earnings expectations")

        # Revenue indicators
        if "revenue growth" in all_text or "revenue increase" in all_text:
            findings.append("Revenue growth reported")
        elif "revenue decline" in all_text or "revenue drop" in all_text:
            findings.append("Revenue decline reported")

        # Profit indicators
        if "profit" in all_text and ("increase" in all_text or "growth" in all_text):
            findings.append("Profit growth reported")
        elif "loss" in all_text:
            findings.append("Loss reported")

        # Market indicators
        if "stock" in all_text and ("surge" in all_text or "rally" in all_text):
            findings.append("Stock price increase noted")
        elif "stock" in all_text and ("drop" in all_text or "fall" in all_text):
            findings.append("Stock price decrease noted")

        return findings

    def _extract_legal_findings(
        self,
        company_name: str,
        web_results: List[Dict],
        news_articles: List[Dict],
    ) -> List[str]:
        """Extract legal-specific findings."""
        findings = []

        all_text = " ".join([
            r.get("snippet", "") + " " + r.get("title", "")
            for r in web_results + news_articles
        ]).lower()

        # Case types
        if "class action" in all_text:
            findings.append("Class action lawsuit mentioned")
        if "securities" in all_text and ("fraud" in all_text or "lawsuit" in all_text):
            findings.append("Securities-related legal issue mentioned")
        if "antitrust" in all_text:
            findings.append("Antitrust matter mentioned")
        if "patent" in all_text and ("infringement" in all_text or "lawsuit" in all_text):
            findings.append("Patent litigation mentioned")

        # Regulatory
        if "sec" in all_text and "investigation" in all_text:
            findings.append("SEC investigation mentioned")
        if "ftc" in all_text or "federal trade" in all_text:
            findings.append("FTC involvement mentioned")
        if "fine" in all_text or "penalty" in all_text:
            findings.append("Fines or penalties mentioned")

        # Settlement
        if "settlement" in all_text:
            findings.append("Legal settlement mentioned")

        return findings

    def health_check(self) -> bool:
        """Check if search functionality is working."""
        return self.web_search.health_check()
