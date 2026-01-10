"""LangChain LLM Factory - Creates LLM instances with per-prompt configuration.

Provides:
- Centralized LLM creation with consistent configuration
- Per-prompt LLM provider/model configuration
- Support for multiple providers (Groq, OpenAI, Anthropic)
- Automatic callback attachment for token tracking
- LangSmith tracing integration
"""

import os
import logging
from typing import List, Optional, Any, Dict

logger = logging.getLogger(__name__)

# Try to import LangChain Groq
try:
    from langchain_groq import ChatGroq
    LANGCHAIN_GROQ_AVAILABLE = True
except ImportError:
    LANGCHAIN_GROQ_AVAILABLE = False
    ChatGroq = None
    logger.warning("langchain-groq not installed. Run: pip install langchain-groq")

# Try to import LangChain OpenAI
try:
    from langchain_openai import ChatOpenAI
    LANGCHAIN_OPENAI_AVAILABLE = True
except ImportError:
    LANGCHAIN_OPENAI_AVAILABLE = False
    ChatOpenAI = None

# Try to import LangChain Anthropic
try:
    from langchain_anthropic import ChatAnthropic
    LANGCHAIN_ANTHROPIC_AVAILABLE = True
except ImportError:
    LANGCHAIN_ANTHROPIC_AVAILABLE = False
    ChatAnthropic = None

# Try to import callback base
try:
    from langchain.callbacks.base import BaseCallbackHandler
    CALLBACKS_AVAILABLE = True
except ImportError:
    CALLBACKS_AVAILABLE = False
    BaseCallbackHandler = None


# Model configurations per provider
GROQ_MODELS = {
    "primary": "llama-3.3-70b-versatile",
    "fast": "llama-3.1-8b-instant",
    "balanced": "llama3-70b-8192",
    "small": "llama3-8b-8192",
    "mixtral": "mixtral-8x7b-32768",
}

OPENAI_MODELS = {
    "primary": "gpt-4o-mini",
    "fast": "gpt-4o-mini",
    "balanced": "gpt-4o-mini",
    "legacy": "gpt-4-turbo",
}

ANTHROPIC_MODELS = {
    "primary": "claude-3-5-sonnet-20241022",
    "fast": "claude-3-haiku-20240307",
    "balanced": "claude-3-5-sonnet-20241022",
}

# Legacy alias for backward compatibility
MODELS = GROQ_MODELS

# Default settings
DEFAULT_PROVIDER = "groq"
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


def _get_chat_openai(
    model: str = "primary",
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    callbacks: Optional[List[Any]] = None,
    api_key: Optional[str] = None,
) -> Optional[Any]:
    """
    Create a ChatOpenAI instance with optional callbacks.

    Args:
        model: Model name or alias (primary, fast, balanced, legacy)
        temperature: Sampling temperature (0-1)
        max_tokens: Maximum tokens in response
        callbacks: List of LangChain callback handlers
        api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)

    Returns:
        ChatOpenAI instance or None if not available
    """
    if not LANGCHAIN_OPENAI_AVAILABLE:
        logger.warning("langchain-openai not installed. Run: pip install langchain-openai")
        return None

    # Resolve model alias
    model_id = OPENAI_MODELS.get(model, model)

    # Get API key
    openai_api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OPENAI_API_KEY not set")
        return None

    try:
        llm = ChatOpenAI(
            model=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=openai_api_key,
            callbacks=callbacks or [],
        )
        logger.debug(f"Created ChatOpenAI instance: model={model_id}, temp={temperature}")
        return llm
    except Exception as e:
        logger.error(f"Failed to create ChatOpenAI: {e}")
        return None


def _get_chat_anthropic(
    model: str = "primary",
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    callbacks: Optional[List[Any]] = None,
    api_key: Optional[str] = None,
) -> Optional[Any]:
    """
    Create a ChatAnthropic instance with optional callbacks.

    Args:
        model: Model name or alias (primary, fast, balanced)
        temperature: Sampling temperature (0-1)
        max_tokens: Maximum tokens in response
        callbacks: List of LangChain callback handlers
        api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)

    Returns:
        ChatAnthropic instance or None if not available
    """
    if not LANGCHAIN_ANTHROPIC_AVAILABLE:
        logger.warning("langchain-anthropic not installed. Run: pip install langchain-anthropic")
        return None

    # Resolve model alias
    model_id = ANTHROPIC_MODELS.get(model, model)

    # Get API key
    anthropic_api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        logger.error("ANTHROPIC_API_KEY not set")
        return None

    try:
        llm = ChatAnthropic(
            model=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            anthropic_api_key=anthropic_api_key,
            callbacks=callbacks or [],
        )
        logger.debug(f"Created ChatAnthropic instance: model={model_id}, temp={temperature}")
        return llm
    except Exception as e:
        logger.error(f"Failed to create ChatAnthropic: {e}")
        return None


def get_llm_config_for_prompt(prompt_id: str) -> Dict[str, Any]:
    """
    Get the resolved LLM configuration for a prompt.

    Useful for logging and debugging what configuration will be used.

    Args:
        prompt_id: The prompt identifier

    Returns:
        Dict with resolved provider, model, temperature, max_tokens, and source
    """
    try:
        from config.prompts import get_prompt
    except ImportError:
        return {
            "prompt_id": prompt_id,
            "provider": DEFAULT_PROVIDER,
            "model_alias": "primary",
            "model_id": GROQ_MODELS.get("primary", "llama-3.3-70b-versatile"),
            "temperature": DEFAULT_TEMPERATURE,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "source": "defaults",
            "error": "prompts module not available",
        }

    prompt = get_prompt(prompt_id)
    if not prompt:
        return {
            "prompt_id": prompt_id,
            "provider": DEFAULT_PROVIDER,
            "model_alias": "primary",
            "model_id": GROQ_MODELS.get("primary", "llama-3.3-70b-versatile"),
            "temperature": DEFAULT_TEMPERATURE,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "source": "defaults",
            "error": f"prompt '{prompt_id}' not found",
        }

    prompt_llm_config = prompt.get("llm_config", {}) or {}

    # Resolve provider
    provider = prompt_llm_config.get("provider") or DEFAULT_PROVIDER
    model_alias = prompt_llm_config.get("model") or "primary"
    temperature = prompt_llm_config.get("temperature") if prompt_llm_config.get("temperature") is not None else DEFAULT_TEMPERATURE
    max_tokens = prompt_llm_config.get("max_tokens") if prompt_llm_config.get("max_tokens") is not None else DEFAULT_MAX_TOKENS

    # Resolve model alias to actual model ID
    if provider == "openai":
        model_id = OPENAI_MODELS.get(model_alias, model_alias)
    elif provider == "anthropic":
        model_id = ANTHROPIC_MODELS.get(model_alias, model_alias)
    else:  # groq
        model_id = GROQ_MODELS.get(model_alias, model_alias)

    return {
        "prompt_id": prompt_id,
        "provider": provider,
        "model_alias": model_alias,
        "model_id": model_id,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "source": "prompt_config" if prompt_llm_config else "defaults",
    }


def get_llm_for_prompt(
    prompt_id: str,
    callbacks: Optional[List[Any]] = None,
    override_provider: Optional[str] = None,
    override_model: Optional[str] = None,
    override_temperature: Optional[float] = None,
    override_max_tokens: Optional[int] = None,
) -> Optional[Any]:
    """
    Create an LLM instance configured for a specific prompt.

    Resolution order (highest to lowest priority):
    1. Runtime overrides (override_* parameters)
    2. Prompt's llm_config
    3. Global defaults

    Args:
        prompt_id: The prompt identifier
        callbacks: LangChain callback handlers
        override_provider: Override the provider (groq, openai, anthropic)
        override_model: Override the model (primary, fast, balanced, or specific ID)
        override_temperature: Override temperature
        override_max_tokens: Override max tokens

    Returns:
        Configured LLM instance (ChatGroq, ChatOpenAI, or ChatAnthropic)

    Example:
        # Get LLM configured for "company_parser" prompt
        llm = get_llm_for_prompt("company_parser")

        # Override the model at runtime
        llm = get_llm_for_prompt("company_parser", override_model="primary")

        # Use different provider
        llm = get_llm_for_prompt("credit_synthesis", override_provider="openai")
    """
    # Get resolved config for this prompt
    resolved = get_llm_config_for_prompt(prompt_id)

    # Apply overrides
    provider = override_provider or resolved["provider"]
    model = override_model or resolved["model_alias"]
    temperature = override_temperature if override_temperature is not None else resolved["temperature"]
    max_tokens = override_max_tokens if override_max_tokens is not None else resolved["max_tokens"]

    logger.debug(f"Creating LLM for prompt '{prompt_id}': provider={provider}, model={model}, temp={temperature}")

    # Create appropriate LLM based on provider
    if provider == "openai":
        return _get_chat_openai(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=callbacks,
        )
    elif provider == "anthropic":
        return _get_chat_anthropic(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=callbacks,
        )
    else:  # Default to groq
        return get_chat_groq(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=callbacks,
        )
