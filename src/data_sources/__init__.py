"""Data Source Connectors for Credit Intelligence"""

from .base import BaseDataSource
from .sec_edgar import SECEdgarDataSource
from .finnhub import FinnhubDataSource
from .court_listener import CourtListenerDataSource
from .web_search import WebSearchDataSource

# Task 11: Web Scraper
from .web_scraper import (
    WebScraper,
    ScrapedContent,
    get_web_scraper,
    scrape_url,
    scrape_company_website,
)

# Task 12: Tavily Search
from .tavily_search import (
    TavilySearchDataSource,
    get_tavily_search,
    tavily_search,
    tavily_company_search,
)

# Optional sources (may not work without API keys)
try:
    from .opencorporates import OpenCorporatesDataSource
except ImportError:
    OpenCorporatesDataSource = None

try:
    from .opensanctions import OpenSanctionsDataSource
except ImportError:
    OpenSanctionsDataSource = None

__all__ = [
    "BaseDataSource",
    "SECEdgarDataSource",
    "FinnhubDataSource",
    "CourtListenerDataSource",
    "WebSearchDataSource",
    "OpenCorporatesDataSource",
    "OpenSanctionsDataSource",
    # Task 11: Web Scraper
    "WebScraper",
    "ScrapedContent",
    "get_web_scraper",
    "scrape_url",
    "scrape_company_website",
    # Task 12: Tavily Search
    "TavilySearchDataSource",
    "get_tavily_search",
    "tavily_search",
    "tavily_company_search",
]
