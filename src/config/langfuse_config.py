"""
Langfuse Integration for Credit Intelligence - Task 22

Langfuse is an open-source LLM observability platform for tracing,
monitoring, and debugging LLM applications.

Features:
- Automatic tracing of LangChain/LangGraph calls
- Token usage and cost tracking
- Latency monitoring
- Session and user tracking
- Score and feedback collection

Setup:
1. Create account at https://cloud.langfuse.com
2. Create a project and get API keys
3. Set environment variables:
   - LANGFUSE_PUBLIC_KEY
   - LANGFUSE_SECRET_KEY
   - LANGFUSE_HOST (optional, defaults to cloud.langfuse.com)
"""

import os
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Check if Langfuse is available
try:
    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
    LANGFUSE_SDK_AVAILABLE = True
except ImportError:
    LANGFUSE_SDK_AVAILABLE = False
    Langfuse = None
    LangfuseCallbackHandler = None
    logger.warning("langfuse not installed. Run: pip install langfuse")


# Singleton instances
_langfuse_client: Optional["Langfuse"] = None
_langfuse_handler: Optional["LangfuseCallbackHandler"] = None


def is_langfuse_available() -> bool:
    """Check if Langfuse SDK is installed and configured."""
    if not LANGFUSE_SDK_AVAILABLE:
        return False

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")

    return bool(public_key and secret_key)


def get_langfuse_client() -> Optional["Langfuse"]:
    """
    Get or create the Langfuse client singleton.

    Returns:
        Langfuse client instance or None if not configured
    """
    global _langfuse_client

    if not is_langfuse_available():
        return None

    if _langfuse_client is None:
        try:
            _langfuse_client = Langfuse(
                public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
                secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
                host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            )
            logger.info("Langfuse client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Langfuse client: {e}")
            return None

    return _langfuse_client


def get_langfuse_handler(
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[list] = None,
) -> Optional["LangfuseCallbackHandler"]:
    """
    Get a Langfuse callback handler for LangChain integration.

    Args:
        session_id: Optional session ID for grouping traces
        user_id: Optional user ID for tracking
        metadata: Optional metadata to attach to traces
        tags: Optional tags for filtering traces

    Returns:
        LangfuseCallbackHandler instance or None if not available

    Usage:
        handler = get_langfuse_handler(session_id="run_123")
        chain.invoke(input, config={"callbacks": [handler]})
    """
    if not is_langfuse_available():
        logger.debug("Langfuse not available, returning None")
        return None

    try:
        handler = LangfuseCallbackHandler(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            session_id=session_id,
            user_id=user_id,
            metadata=metadata or {},
            tags=tags or [],
        )
        return handler
    except Exception as e:
        logger.error(f"Failed to create Langfuse handler: {e}")
        return None


def get_default_langfuse_handler() -> Optional["LangfuseCallbackHandler"]:
    """
    Get a reusable default Langfuse handler singleton.

    Use this for simple cases where you don't need session/user tracking.
    For per-request tracking, use get_langfuse_handler() instead.

    Returns:
        LangfuseCallbackHandler instance or None if not available
    """
    global _langfuse_handler

    if not is_langfuse_available():
        return None

    if _langfuse_handler is None:
        _langfuse_handler = get_langfuse_handler(
            metadata={"source": "credit-intelligence"},
            tags=["credit-intelligence"],
        )

    return _langfuse_handler


def create_trace(
    name: str,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[list] = None,
) -> Optional[Any]:
    """
    Create a new Langfuse trace for manual instrumentation.

    Args:
        name: Name of the trace
        session_id: Optional session ID
        user_id: Optional user ID
        metadata: Optional metadata
        tags: Optional tags

    Returns:
        Langfuse trace object or None

    Usage:
        trace = create_trace("credit_assessment", session_id=run_id)
        if trace:
            span = trace.span(name="data_collection")
            # ... do work ...
            span.end()
            trace.update(output=result)
    """
    client = get_langfuse_client()
    if not client:
        return None

    try:
        trace = client.trace(
            name=name,
            session_id=session_id,
            user_id=user_id,
            metadata=metadata or {},
            tags=tags or [],
        )
        return trace
    except Exception as e:
        logger.error(f"Failed to create Langfuse trace: {e}")
        return None


def log_score(
    trace_id: str,
    name: str,
    value: float,
    comment: Optional[str] = None,
) -> bool:
    """
    Log a score/evaluation to a Langfuse trace.

    Args:
        trace_id: ID of the trace to score
        name: Name of the score (e.g., "accuracy", "relevance")
        value: Score value (0.0 to 1.0)
        comment: Optional comment explaining the score

    Returns:
        True if successful, False otherwise
    """
    client = get_langfuse_client()
    if not client:
        return False

    try:
        client.score(
            trace_id=trace_id,
            name=name,
            value=value,
            comment=comment,
        )
        return True
    except Exception as e:
        logger.error(f"Failed to log Langfuse score: {e}")
        return False


def flush_langfuse():
    """
    Flush any pending Langfuse events.

    Call this before application exit to ensure all traces are sent.
    """
    client = get_langfuse_client()
    if client:
        try:
            client.flush()
            logger.debug("Langfuse events flushed")
        except Exception as e:
            logger.error(f"Failed to flush Langfuse: {e}")


def shutdown_langfuse():
    """
    Shutdown Langfuse client and flush pending events.

    Call this when the application is shutting down.
    """
    global _langfuse_client, _langfuse_handler

    if _langfuse_client:
        try:
            _langfuse_client.flush()
            _langfuse_client.shutdown()
            logger.info("Langfuse client shutdown")
        except Exception as e:
            logger.error(f"Error shutting down Langfuse: {e}")
        finally:
            _langfuse_client = None
            _langfuse_handler = None


class LangfuseTracer:
    """
    Context manager for Langfuse tracing.

    Usage:
        with LangfuseTracer("credit_assessment", run_id=run_id) as tracer:
            # Your workflow code here
            tracer.span("data_collection")
            # ... collect data ...
            tracer.end_span()

            tracer.set_output(result)
            tracer.score("accuracy", 0.95)
    """

    def __init__(
        self,
        name: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[list] = None,
    ):
        self.name = name
        self.session_id = session_id
        self.user_id = user_id
        self.metadata = metadata or {}
        self.tags = tags or []
        self.trace = None
        self._current_span = None

    def __enter__(self):
        self.trace = create_trace(
            name=self.name,
            session_id=self.session_id,
            user_id=self.user_id,
            metadata=self.metadata,
            tags=self.tags,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.trace:
            if exc_type:
                self.trace.update(
                    status_message=str(exc_val),
                    level="ERROR",
                )
            flush_langfuse()
        return False

    def span(self, name: str, metadata: Optional[Dict] = None):
        """Start a new span within the trace."""
        if self.trace:
            self._current_span = self.trace.span(
                name=name,
                metadata=metadata or {},
            )
        return self._current_span

    def end_span(self, output: Any = None):
        """End the current span."""
        if self._current_span:
            self._current_span.end(output=output)
            self._current_span = None

    def set_output(self, output: Any):
        """Set the output of the trace."""
        if self.trace:
            self.trace.update(output=output)

    def score(self, name: str, value: float, comment: Optional[str] = None):
        """Log a score to the trace."""
        if self.trace:
            log_score(
                trace_id=self.trace.id,
                name=name,
                value=value,
                comment=comment,
            )

    def get_handler(self) -> Optional["LangfuseCallbackHandler"]:
        """Get a callback handler linked to this trace."""
        if self.trace:
            return get_langfuse_handler(
                session_id=self.session_id,
                user_id=self.user_id,
                metadata={**self.metadata, "trace_id": self.trace.id},
                tags=self.tags,
            )
        return None


# Export public API
__all__ = [
    "is_langfuse_available",
    "get_langfuse_client",
    "get_langfuse_handler",
    "get_default_langfuse_handler",
    "create_trace",
    "log_score",
    "flush_langfuse",
    "shutdown_langfuse",
    "LangfuseTracer",
]
