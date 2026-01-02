"""LangChain LLM Factory - Creates ChatGroq instances with callbacks.

Provides:
- Centralized LLM creation with consistent configuration
- Automatic callback attachment for token tracking
- LangSmith tracing integration
"""

import os
import logging
from typing import List, Optional, Any

logger = logging.getLogger(__name__)

# Try to import LangChain Groq
try:
    from langchain_groq import ChatGroq
    LANGCHAIN_GROQ_AVAILABLE = True
except ImportError:
    LANGCHAIN_GROQ_AVAILABLE = False
    ChatGroq = None
    logger.warning("langchain-groq not installed. Run: pip install langchain-groq")

# Try to import callback base
try:
    from langchain.callbacks.base import BaseCallbackHandler
    CALLBACKS_AVAILABLE = True
except ImportError:
    CALLBACKS_AVAILABLE = False
    BaseCallbackHandler = None


# Model configurations
MODELS = {
    "primary": "llama-3.3-70b-versatile",
    "fast": "llama-3.1-8b-instant",
    "balanced": "llama3-70b-8192",
    "small": "llama3-8b-8192",
    "mixtral": "mixtral-8x7b-32768",
}

# Default settings
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 2000


def get_chat_groq(
    model: str = "primary",
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    callbacks: Optional[List[Any]] = None,
    api_key: Optional[str] = None,
) -> Optional[Any]:
    """
    Create a ChatGroq instance with optional callbacks.

    Args:
        model: Model name or alias (primary, fast, balanced, small, mixtral)
        temperature: Sampling temperature (0-1)
        max_tokens: Maximum tokens in response
        callbacks: List of LangChain callback handlers
        api_key: Groq API key (defaults to GROQ_API_KEY env var)

    Returns:
        ChatGroq instance or None if not available

    Example:
        from config.langchain_llm import get_chat_groq
        from config.langchain_callbacks import CostTrackerCallback

        llm = get_chat_groq("primary", callbacks=[CostTrackerCallback()])
        response = llm.invoke("Hello, world!")
    """
    if not LANGCHAIN_GROQ_AVAILABLE:
        logger.error("langchain-groq not installed. Cannot create ChatGroq instance.")
        return None

    # Resolve model alias
    model_id = MODELS.get(model, model)

    # Get API key
    groq_api_key = api_key or os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        logger.error("GROQ_API_KEY not set")
        return None

    try:
        llm = ChatGroq(
            model=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            groq_api_key=groq_api_key,
            callbacks=callbacks or [],
        )
        logger.debug(f"Created ChatGroq instance: model={model_id}, temp={temperature}")
        return llm
    except Exception as e:
        logger.error(f"Failed to create ChatGroq: {e}")
        return None


def get_model_id(model: str) -> str:
    """Get the actual model ID from an alias."""
    return MODELS.get(model, model)


def is_langchain_groq_available() -> bool:
    """Check if langchain-groq is available."""
    return LANGCHAIN_GROQ_AVAILABLE
