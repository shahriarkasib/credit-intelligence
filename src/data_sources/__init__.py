"""Data Source Connectors for Credit Intelligence"""

from .base import BaseDataSource
from .sec_edgar import SECEdgarDataSource
from .finnhub import FinnhubDataSource
from .court_listener import CourtListenerDataSource
from .web_search import WebSearchDataSource

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
]
