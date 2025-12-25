"""Base class for all data source connectors."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging
import time
import requests

logger = logging.getLogger(__name__)


@dataclass
class DataSourceResult:
    """Standardized result from a data source query."""

    source: str
    query: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    success: bool = True
    error: Optional[str] = None
    raw_response: Optional[Dict] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "query": self.query,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "success": self.success,
            "error": self.error,
        }


class RateLimiter:
    """Simple rate limiter for API calls."""

    def __init__(self, calls_per_second: float = 1.0):
        self.calls_per_second = calls_per_second
        self.min_interval = 1.0 / calls_per_second
        self.last_call = 0.0

    def wait(self):
        """Wait if necessary to respect rate limit."""
        elapsed = time.time() - self.last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_call = time.time()


class BaseDataSource(ABC):
    """Abstract base class for data source connectors."""

    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: Optional[str] = None,
        rate_limit: float = 1.0,
        timeout: int = 30,
    ):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.rate_limiter = RateLimiter(rate_limit)
        self.session = requests.Session()
        self._setup_session()

    def _setup_session(self):
        """Configure session with default headers."""
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "CreditIntelligence/1.0 (Demo Project)",
        })

    def _make_request(
        self,
        endpoint: str,
        method: str = "GET",
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        headers: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """Make HTTP request with rate limiting and error handling."""
        self.rate_limiter.wait()

        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        try:
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                json=data,
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {self.name}: {e}")
            return None
        except ValueError as e:
            logger.error(f"JSON decode error for {self.name}: {e}")
            return None

    @abstractmethod
    def search(self, query: str, **kwargs) -> DataSourceResult:
        """Search the data source. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def get_company_data(self, identifier: str, **kwargs) -> DataSourceResult:
        """Get company-specific data. Must be implemented by subclasses."""
        pass

    def health_check(self) -> bool:
        """Check if the data source is accessible."""
        try:
            # Subclasses should override with specific health check
            return True
        except Exception:
            return False

    def get_supported_fields(self) -> List[str]:
        """Return list of fields this data source can provide."""
        return []
