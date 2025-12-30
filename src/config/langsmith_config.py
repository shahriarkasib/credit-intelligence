"""LangSmith Configuration - Enables LangChain/LangGraph execution logging."""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def setup_langsmith(
    project_name: str = "credit-intelligence",
    enabled: bool = True,
) -> bool:
    """
    Configure LangSmith for tracing LangChain/LangGraph executions.

    LangSmith provides:
    - Full execution traces for all LLM calls
    - Token usage tracking
    - Latency metrics
    - Chain/Graph visualization
    - Error debugging

    Args:
        project_name: Name of the LangSmith project
        enabled: Whether to enable tracing

    Returns:
        True if LangSmith was configured successfully

    Environment Variables Required:
        LANGCHAIN_API_KEY: Your LangSmith API key
        LANGCHAIN_TRACING_V2: Set to "true" to enable
        LANGCHAIN_PROJECT: Project name in LangSmith
    """
    if not enabled:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        logger.info("LangSmith tracing disabled")
        return False

    # Check for API key
    api_key = os.getenv("LANGCHAIN_API_KEY")
    if not api_key:
        logger.warning(
            "LANGCHAIN_API_KEY not set. LangSmith tracing disabled. "
            "Get your key at: https://smith.langchain.com/settings"
        )
        return False

    # Configure LangSmith
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = project_name
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

    logger.info(f"LangSmith tracing enabled for project: {project_name}")
    logger.info("View traces at: https://smith.langchain.com")

    return True


def get_langsmith_url(run_id: Optional[str] = None) -> str:
    """
    Get URL to view traces in LangSmith.

    Args:
        run_id: Optional specific run ID

    Returns:
        URL to LangSmith dashboard
    """
    project = os.getenv("LANGCHAIN_PROJECT", "credit-intelligence")
    base_url = f"https://smith.langchain.com/o/default/projects/p/{project}"

    if run_id:
        return f"{base_url}/r/{run_id}"
    return base_url


# Auto-setup on import if environment is configured
_langsmith_enabled = setup_langsmith()


def is_langsmith_enabled() -> bool:
    """Check if LangSmith is currently enabled."""
    return os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"
