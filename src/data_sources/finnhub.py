"""Finnhub Data Source - Stock and financial market data."""

import os
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta

from .base import BaseDataSource, DataSourceResult

logger = logging.getLogger(__name__)


class FinnhubDataSource(BaseDataSource):
    """
    Finnhub API connector for market and financial data.

    Provides access to:
    - Stock quotes and prices
    - Company profiles
    - Company news
    - Financial metrics

    Rate Limit: 60 API calls/minute (free tier)
    Authentication: API key required
    """

    def __init__(self, api_key: Optional[str] = None):
        api_key = api_key or os.getenv("FINNHUB_API_KEY")
        if not api_key:
            logger.warning("Finnhub API key not provided. Some features may be limited.")

        super().__init__(
            name="Finnhub",
            base_url="https://finnhub.io/api/v1",
            api_key=api_key,
            rate_limit=1.0,  # 1 request per second (60/min)
        )

    def _add_auth(self, params: Dict) -> Dict:
        """Add API key to params."""
        if self.api_key:
            params["token"] = self.api_key
        return params

    def search(self, query: str, **kwargs) -> DataSourceResult:
        """Search for stock symbols by company name."""
        params = self._add_auth({"q": query})
        response = self._make_request("search", params=params)

        if not response:
            return DataSourceResult(
                source=self.name,
                query=query,
                data={},
                success=False,
                error="Search failed",
            )

        results = response.get("result", [])
        processed = [
            {
                "symbol": r.get("symbol", ""),
                "description": r.get("description", ""),
                "type": r.get("type", ""),
                "display_symbol": r.get("displaySymbol", ""),
            }
            for r in results[:10]  # Limit to top 10
        ]

        return DataSourceResult(
            source=self.name,
            query=query,
            data={
                "count": response.get("count", 0),
                "results": processed,
            },
            raw_response=response,
        )

    def get_company_data(self, identifier: str, **kwargs) -> DataSourceResult:
        """
        Get comprehensive company data by stock symbol.

        Args:
            identifier: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        """
        symbol = identifier.upper()

        # Fetch multiple data points in sequence (respecting rate limit)
        profile = self._get_profile(symbol)
        quote = self._get_quote(symbol)
        metrics = self._get_basic_financials(symbol)
        news = self._get_news(symbol, days=30)
        peers = self._get_peers(symbol)

        if not profile and not quote:
            return DataSourceResult(
                source=self.name,
                query=identifier,
                data={},
                success=False,
                error=f"No data found for symbol {symbol}",
            )

        return DataSourceResult(
            source=self.name,
            query=identifier,
            data={
                "symbol": symbol,
                "profile": profile or {},
                "quote": quote or {},
                "metrics": metrics or {},
                "recent_news": news[:5] if news else [],
                "peers": peers or [],
            },
            raw_response={
                "profile": profile,
                "quote": quote,
                "metrics": metrics,
            },
        )

    def _get_profile(self, symbol: str) -> Optional[Dict]:
        """Get company profile."""
        params = self._add_auth({"symbol": symbol})
        response = self._make_request("stock/profile2", params=params)

        if not response or not response.get("name"):
            return None

        return {
            "name": response.get("name", ""),
            "ticker": response.get("ticker", ""),
            "country": response.get("country", ""),
            "currency": response.get("currency", ""),
            "exchange": response.get("exchange", ""),
            "industry": response.get("finnhubIndustry", ""),
            "ipo_date": response.get("ipo"),
            "market_cap": response.get("marketCapitalization"),
            "shares_outstanding": response.get("shareOutstanding"),
            "logo": response.get("logo", ""),
            "website": response.get("weburl", ""),
            "phone": response.get("phone", ""),
        }

    def _get_quote(self, symbol: str) -> Optional[Dict]:
        """Get current stock quote."""
        params = self._add_auth({"symbol": symbol})
        response = self._make_request("quote", params=params)

        if not response or response.get("c") is None:
            return None

        return {
            "current_price": response.get("c"),
            "change": response.get("d"),
            "percent_change": response.get("dp"),
            "high": response.get("h"),
            "low": response.get("l"),
            "open": response.get("o"),
            "previous_close": response.get("pc"),
            "timestamp": response.get("t"),
        }

    def _get_basic_financials(self, symbol: str) -> Optional[Dict]:
        """Get basic financial metrics."""
        params = self._add_auth({"symbol": symbol, "metric": "all"})
        response = self._make_request("stock/metric", params=params)

        if not response or "metric" not in response:
            return None

        metric = response.get("metric", {})
        return {
            "52_week_high": metric.get("52WeekHigh"),
            "52_week_low": metric.get("52WeekLow"),
            "pe_ratio": metric.get("peBasicExclExtraTTM"),
            "pb_ratio": metric.get("pbQuarterly"),
            "ps_ratio": metric.get("psTTM"),
            "dividend_yield": metric.get("dividendYieldIndicatedAnnual"),
            "beta": metric.get("beta"),
            "eps_ttm": metric.get("epsBasicExclExtraItemsTTM"),
            "revenue_per_share": metric.get("revenuePerShareTTM"),
            "net_profit_margin": metric.get("netProfitMarginTTM"),
            "operating_margin": metric.get("operatingMarginTTM"),
            "roe": metric.get("roeTTM"),
            "roa": metric.get("roaTTM"),
            "current_ratio": metric.get("currentRatioQuarterly"),
            "debt_equity": metric.get("totalDebt/totalEquityQuarterly"),
        }

    def _get_news(self, symbol: str, days: int = 7) -> List[Dict]:
        """Get recent company news."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        params = self._add_auth({
            "symbol": symbol,
            "from": start_date.strftime("%Y-%m-%d"),
            "to": end_date.strftime("%Y-%m-%d"),
        })
        response = self._make_request("company-news", params=params)

        if not response:
            return []

        return [
            {
                "headline": item.get("headline", ""),
                "summary": item.get("summary", ""),
                "source": item.get("source", ""),
                "url": item.get("url", ""),
                "datetime": item.get("datetime"),
                "category": item.get("category", ""),
            }
            for item in response[:10]
        ]

    def _get_peers(self, symbol: str) -> List[str]:
        """Get company peers/competitors."""
        params = self._add_auth({"symbol": symbol})
        response = self._make_request("stock/peers", params=params)

        if not response:
            return []

        return response[:10]  # Return top 10 peers

    def get_stock_quote(self, symbol: str) -> DataSourceResult:
        """Get just the stock quote for a symbol."""
        quote = self._get_quote(symbol.upper())

        if not quote:
            return DataSourceResult(
                source=self.name,
                query=symbol,
                data={},
                success=False,
                error=f"No quote data for {symbol}",
            )

        return DataSourceResult(
            source=self.name,
            query=symbol,
            data=quote,
        )

    def get_supported_fields(self) -> List[str]:
        return [
            "company_name",
            "symbol",
            "country",
            "exchange",
            "industry",
            "market_cap",
            "current_price",
            "price_change",
            "52_week_high",
            "52_week_low",
            "pe_ratio",
            "dividend_yield",
            "revenue_per_share",
            "profit_margin",
            "roe",
            "roa",
            "debt_equity",
            "recent_news",
            "peers",
        ]

    def health_check(self) -> bool:
        """Check if Finnhub is accessible."""
        if not self.api_key:
            return False
        # Try to get a quote for a known symbol
        quote = self._get_quote("AAPL")
        return quote is not None
